"""Join planner (SPEC 5.10): from seed tables to the cheapest set of joins connecting them.

Retrieval picks the tables a question is about; they are usually not directly
joined. The planner finds the cheapest tree through the schema's foreign keys
that connects them, so the tables in between ("bridge" tables) are not missed.

How it works
------------
1. Every relationship becomes two arcs between TABLES: following the foreign key
   (many-to-one, cheap) and walking it backwards (one-to-many, dearer because it
   can multiply rows). Costs come from the weights stored on each edge.
2. Seeds are grouped by connected component; the component holding the most
   seeds is planned, and seeds outside it are reported as unreachable.
3. A Steiner tree is found. Because arc costs depend on direction, the tree is
   rooted at one of the seeds and each arc is costed as walked away from the
   root; every seed is tried as the root and the cheapest wins.
     - up to MAX_EXACT_SEEDS seeds: an exact dynamic programme (Dreyfus-Wagner),
       which finds the cheapest tree. Two seeds is plain shortest path.
     - more seeds, or when asked: NetworkX's approximation (undirected, so it
       costs each edge at its cheaper direction while choosing the tree, then
       re-costs the chosen tree in the walking direction).
4. If the tree needs more than `max_bridge_tables` bridges it is judged
   implausible: the lowest-scoring seed is dropped and the plan redone.
5. The tree's arcs are mapped back to column pairs (a composite foreign key is
   one link with several column pairs), and runner-up trees are found by
   banning each bridge table in turn.

The planner never raises for "no path": it returns status "partial" or "none".
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from schemagraph.graph.queries import ReferenceEdge, get_reference_edges
from schemagraph.models import (
    DatabaseMetadata,
    JoinPlan,
    PlanAlternative,
    PlannedJoin,
    SeedTable,
)

log = logging.getLogger(__name__)

INF = float("inf")
MAX_BRIDGE_TABLES = 4        # more bridges than this and the plan is treated as implausible
MAX_EXACT_SEEDS = 7          # the exact solver costs about 3^k * (tables); beyond this use NetworkX
MAX_ALTERNATIVES = 2

Algorithm = Literal["auto", "exact", "networkx"]


# ---------------------------------------------------------------------------
# Table graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    """One relationship between two tables. A composite foreign key is a single link."""

    from_table: str
    to_table: str
    edges: tuple[ReferenceEdge, ...]  # one per column pair, in stored (FK -> referenced) direction

    @property
    def source(self) -> str:
        return self.edges[0].source

    @property
    def cardinality(self) -> str:
        return self.edges[0].cardinality


@dataclass(frozen=True)
class Arc:
    """A link walked in one direction."""

    src: str
    dst: str
    weight: float
    traversal: str  # "many_to_one" (with the FK) or "one_to_many" (against it, can fan out)
    link: Link

    def rank(self) -> tuple:
        # which of several arcs between the same two tables to prefer: cheapest,
        # then declared over inferred, then a stable name order
        return (self.weight, self.link.source != "declared", self.link.edges[0].from_column)


def edges_from_metadata(meta: DatabaseMetadata) -> list[ReferenceEdge]:
    """ReferenceEdges straight from metadata, with weights, without Neo4j (tests, offline use)."""
    from schemagraph.graph.weights import edge_weights

    table_of = {c.uid: c.table_uid for c in meta.columns}
    out = []
    for r in meta.relationships:
        wf, wr = edge_weights(r.confidence, r.cardinality)
        out.append(
            ReferenceEdge(
                from_column=r.from_column, to_column=r.to_column,
                from_table=table_of[r.from_column], to_table=table_of[r.to_column],
                source=r.source, confidence=r.confidence, cardinality=r.cardinality,
                weight_forward=wf, weight_reverse=wr, constraint_name=r.constraint_name,
            )
        )
    return out


def build_links(edges: list[ReferenceEdge], include_inferred: bool = True) -> list[Link]:
    """Group column pairs into links: pairs of one declared constraint form one link."""
    groups: dict[Any, list[ReferenceEdge]] = {}
    for e in sorted(edges, key=lambda e: (e.from_column, e.to_column)):
        if e.source == "inferred" and not include_inferred:
            continue
        if e.from_table == e.to_table:
            continue  # a self-reference cannot connect two different tables
        key = (e.from_table, e.to_table, e.constraint_name) if e.constraint_name else (e.from_column, e.to_column)
        groups.setdefault(key, []).append(e)
    return [Link(g[0].from_table, g[0].to_table, tuple(g)) for g in groups.values()]


def build_arcs(links: list[Link], weighted: bool = True) -> dict[str, dict[str, Arc]]:
    """src table -> dst table -> the best arc between them.

    `weighted=False` gives every arc cost 1.0 (plain hop count), for the ablation.
    """
    arcs: dict[str, dict[str, Arc]] = {}

    def offer(arc: Arc) -> None:
        current = arcs.setdefault(arc.src, {}).get(arc.dst)
        if current is None or arc.rank() < current.rank():
            arcs[arc.src][arc.dst] = arc

    for link in links:
        first = link.edges[0]
        wf = first.weight_forward if weighted else 1.0
        wr = first.weight_reverse if weighted else 1.0
        offer(Arc(link.from_table, link.to_table, wf, "many_to_one", link))
        # Walking back from the referenced table only fans out for N:1; a 1:1 link yields at most one row.
        back = "one_to_many" if link.cardinality == "N:1" else "many_to_one"
        offer(Arc(link.to_table, link.from_table, wr, back, link))
    return arcs


def _in_arcs(arcs: dict[str, dict[str, Arc]]) -> dict[str, list[Arc]]:
    incoming: dict[str, list[Arc]] = {}
    for src in sorted(arcs):
        for dst in sorted(arcs[src]):
            incoming.setdefault(dst, []).append(arcs[src][dst])
    return incoming


def _components(nodes: set[str], arcs: dict[str, dict[str, Arc]]) -> list[set[str]]:
    """Connected components (arcs always come in pairs, so direction does not matter)."""
    seen: set[str] = set()
    out = []
    for start in sorted(nodes):
        if start in seen:
            continue
        comp, queue = {start}, deque([start])
        while queue:
            for nxt in arcs.get(queue.popleft(), {}):
                if nxt not in comp:
                    comp.add(nxt)
                    queue.append(nxt)
        seen |= comp
        out.append(comp)
    return out


# ---------------------------------------------------------------------------
# Solvers. Both return (root, arcs) or None; `arcs` are oriented away from the root.
# ---------------------------------------------------------------------------


def _restrict(arcs: dict[str, dict[str, Arc]], nodes: set[str], banned: frozenset[str]) -> dict[str, dict[str, Arc]]:
    keep = nodes - banned
    return {
        s: {d: a for d, a in dsts.items() if d in keep}
        for s, dsts in arcs.items()
        if s in keep
    }


def _tidy(root: str, terminals: list[str], chosen: list[Arc]) -> list[Arc]:
    """Make the chosen arcs a proper tree: one way into each table, no dead branches.

    Two branches of the dynamic programme can reach the same table by different
    arcs when costs tie; a shortest-path tree over the chosen arcs picks one,
    then branches that lead to no seed are cut.
    """
    out: dict[str, list[Arc]] = {}
    for a in chosen:
        out.setdefault(a.src, []).append(a)
    dist = {root: 0.0}
    parent: dict[str, Arc] = {}
    heap = [(0.0, root)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, INF):
            continue
        for a in sorted(out.get(u, []), key=lambda a: a.dst):
            nd = d + a.weight
            if nd < dist.get(a.dst, INF) - 1e-12:
                dist[a.dst] = nd
                parent[a.dst] = a
                heapq.heappush(heap, (nd, a.dst))
    needed: set[str] = set()
    for t in terminals:
        node = t
        while node != root and node not in needed:
            needed.add(node)
            node = parent[node].src
    return [parent[n] for n in sorted(needed)]


def solve_exact(terminals: list[str], arcs: dict[str, dict[str, Arc]]) -> tuple[str, list[Arc]] | None:
    """Cheapest directed Steiner tree over `terminals`, trying every terminal as the root.

    Dreyfus-Wagner: dp[S][v] is the cheapest tree rooted at table v that reaches
    every terminal in S, with arcs oriented away from v. One table over all
    subsets of the terminals answers every choice of root at once:
    the tree for root r is dp[all terminals but r][r].
    """
    k = len(terminals)
    nodes = sorted({n for s, d in arcs.items() for n in (s, *d)} | set(terminals))
    incoming = _in_arcs(arcs)
    full = (1 << k) - 1
    dp: dict[int, dict[str, float]] = {}
    how: dict[tuple[int, str], tuple] = {}

    for mask in range(1, full + 1):
        cost: dict[str, float] = {}
        if mask & (mask - 1) == 0:                      # a single terminal
            t = terminals[mask.bit_length() - 1]
            cost[t] = 0.0
            how[(mask, t)] = ("leaf",)
        else:                                           # join two subtrees at a table
            for u in nodes:
                best, best_split = INF, 0
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if sub < other:
                        c = dp[sub].get(u, INF) + dp[other].get(u, INF)
                        if c < best:
                            best, best_split = c, sub
                    sub = (sub - 1) & mask
                if best < INF:
                    cost[u] = best
                    how[(mask, u)] = ("split", best_split)
        # extend each table's cost to the tables that can reach it (reverse Dijkstra)
        heap = [(c, u) for u, c in cost.items()]
        heapq.heapify(heap)
        while heap:
            c, x = heapq.heappop(heap)
            if c > cost.get(x, INF):
                continue
            for arc in incoming.get(x, ()):
                nc = c + arc.weight
                if nc < cost.get(arc.src, INF) - 1e-12:
                    cost[arc.src] = nc
                    how[(mask, arc.src)] = ("arc", arc)
                    heapq.heappush(heap, (nc, arc.src))
        dp[mask] = cost

    if k == 1:
        return terminals[0], []
    best_root, best_cost = None, INF
    for i, r in enumerate(terminals):                   # ties keep the earlier (higher-scoring) seed
        c = dp[full ^ (1 << i)].get(r, INF)
        if c < best_cost - 1e-12:
            best_root, best_cost = r, c
    if best_root is None:
        return None

    chosen: list[Arc] = []

    def build(mask: int, v: str) -> None:
        step = how[(mask, v)]
        if step[0] == "arc":
            chosen.append(step[1])
            build(mask, step[1].dst)
        elif step[0] == "split":
            build(step[1], v)
            build(mask ^ step[1], v)

    build(full ^ (1 << terminals.index(best_root)), best_root)
    return best_root, _tidy(best_root, [t for t in terminals if t != best_root], chosen)


def solve_networkx(terminals: list[str], arcs: dict[str, dict[str, Arc]]) -> tuple[str, list[Arc]] | None:
    """NetworkX's Steiner approximation, re-costed in the direction the tree is walked.

    NetworkX needs an undirected graph, so while choosing the tree each edge costs
    its cheaper direction. The tree is then walked from each seed in turn, every
    edge replaced by the arc walked away from the root, and the cheapest walk wins.
    """
    import networkx as nx
    from networkx.algorithms.approximation import steiner_tree

    g = nx.Graph()
    for s in sorted(arcs):
        for d in sorted(arcs[s]):
            w = min(arcs[s][d].weight, arcs.get(d, {}).get(s, arcs[s][d]).weight)
            g.add_edge(s, d, weight=w)
    if len(terminals) == 1:
        return terminals[0], []
    if not all(t in g for t in terminals):  # e.g. a banned bridge cut a seed off completely
        return None
    try:
        tree = steiner_tree(g, terminals, weight="weight", method="mehlhorn")
    except (nx.NetworkXException, KeyError):
        return None
    if not all(t in tree for t in terminals) or not nx.is_connected(tree):
        return None  # a banned bridge can split the seeds into separate components; the "tree" is then a forest

    def oriented(root: str) -> list[Arc]:
        chosen: list[Arc] = []
        seen, queue = {root}, deque([root])
        while queue:
            u = queue.popleft()
            for v in sorted(tree.neighbors(u)):
                if v not in seen:
                    seen.add(v)
                    chosen.append(arcs[u][v])
                    queue.append(v)
        return chosen

    best: tuple[float, str, list[Arc]] | None = None
    for root in terminals:  # the tree is fixed; walk it from whichever seed is cheapest (ties: better score)
        arcs_from_root = oriented(root)
        cost = tree_cost(arcs_from_root)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, root, arcs_from_root)
    assert best is not None
    return best[1], _tidy(best[1], [t for t in terminals if t != best[1]], best[2])


def _solve(
    terminals: list[str], arcs: dict[str, dict[str, Arc]], algorithm: Algorithm
) -> tuple[str, list[Arc], str] | None:
    """Pick the algorithm, run it, and name it as the JoinPlan contract does."""
    use_nx = algorithm == "networkx" or (algorithm == "auto" and len(terminals) > MAX_EXACT_SEEDS)
    result = solve_networkx(terminals, arcs) if use_nx else solve_exact(terminals, arcs)
    if result is None:
        return None
    name = "networkx_steiner" if use_nx else ("shortest_path" if len(terminals) == 2 else "exact_steiner")
    return result[0], result[1], name


def tree_cost(tree: list[Arc]) -> float:
    return round(sum(a.weight for a in tree), 6)


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------


def _order(root: str, tree: list[Arc]) -> list[Arc]:
    """Arcs in the order a reader would follow them: breadth first from the root."""
    out_of: dict[str, list[Arc]] = {}
    for a in tree:
        out_of.setdefault(a.src, []).append(a)
    ordered, queue = [], deque([root])
    while queue:
        for a in sorted(out_of.get(queue.popleft(), []), key=lambda a: a.dst):
            ordered.append(a)
            queue.append(a.dst)
    return ordered


def _joins(tree: list[Arc], root: str) -> list[PlannedJoin]:
    joins = []
    for arc in _order(root, tree):
        for e in arc.link.edges:  # a composite key contributes every column pair
            joins.append(
                PlannedJoin(
                    from_column=e.from_column, to_column=e.to_column, source=e.source,
                    confidence=e.confidence, cardinality=e.cardinality,
                    traversal=arc.traversal, weight=arc.weight,
                )
            )
    return joins


def _empty(db_id: str, seeds: list[SeedTable], status: str, **extra: Any) -> JoinPlan:
    return JoinPlan(db_id=db_id, status=status, algorithm="none", seed_tables=seeds, **extra)


def plan_from_edges(
    db_id: str,
    seeds: list[SeedTable],
    edges: list[ReferenceEdge],
    *,
    include_inferred: bool = True,
    weighted: bool = True,
    max_bridge_tables: int = MAX_BRIDGE_TABLES,
    algorithm: Algorithm = "auto",
    max_alternatives: int = MAX_ALTERNATIVES,
) -> JoinPlan:
    """The whole planner on in-memory edges. `plan_joins` is this plus reading the edges from Neo4j."""
    by_uid: dict[str, SeedTable] = {}
    for s in seeds:  # duplicates: keep the best score
        if s.uid not in by_uid or s.score > by_uid[s.uid].score:
            by_uid[s.uid] = s
    seeds = sorted(by_uid.values(), key=lambda s: (-s.score, s.uid))  # same output whatever the input order
    if not seeds:
        return _empty(db_id, [], "none")
    if len(seeds) == 1:
        return JoinPlan(db_id=db_id, status="single_table", algorithm="none", seed_tables=seeds, tables=[seeds[0].uid])

    arcs = build_arcs(build_links(edges, include_inferred), weighted)
    nodes = {n for s, d in arcs.items() for n in (s, *d)} | set(by_uid)
    score = {s.uid: s.score for s in seeds}

    # the component holding the most seeds is planned; the others cannot be joined to it
    comps = [(c, [u for u in by_uid if u in c]) for c in _components(nodes, arcs)]
    comps = [(c, members) for c, members in comps if members]
    comps.sort(key=lambda cm: (-len(cm[1]), -round(sum(score[u] for u in cm[1]), 6), min(cm[1])))
    component, members = comps[0]
    ranked = lambda uids: sorted(uids, key=lambda u: (-score[u], u))  # noqa: E731
    unreachable = ranked(u for u in by_uid if u not in component)
    if len(members) < 2:
        return _empty(db_id, seeds, "none", unreachable_seeds=ranked(by_uid))

    terminals = ranked(members)
    dropped: list[str] = []
    local = _restrict(arcs, component, frozenset())
    while True:
        solved = _solve(terminals, local, algorithm)
        if solved is None:  # cannot happen inside one component, but never raise
            return _empty(db_id, seeds, "none", unreachable_seeds=ranked(by_uid))
        root, tree, name = solved
        bridges = {n for a in tree for n in (a.src, a.dst)} - set(terminals)
        if len(bridges) <= max_bridge_tables:
            break
        if len(terminals) <= 2:  # nothing left to drop: even two seeds are too far apart
            return _empty(db_id, seeds, "none", dropped_seeds=ranked(terminals), unreachable_seeds=unreachable)
        victim = min(terminals, key=lambda u: (score[u], u))
        terminals.remove(victim)
        dropped.append(victim)
        log.info("plan for %s needed %d bridges; dropping seed %s", db_id, len(bridges), victim)

    if len(terminals) < 2:  # unreachable in practice: the loop above stops at two seeds
        return _empty(db_id, seeds, "none", dropped_seeds=dropped, unreachable_seeds=unreachable)

    tables_in_tree = {n for a in tree for n in (a.src, a.dst)} | set(terminals)
    bridge_tables = sorted(tables_in_tree - set(terminals))
    status = "full" if not unreachable and not dropped else "partial"

    alternatives: list[PlanAlternative] = []
    seen_sets = {frozenset(tables_in_tree)}
    candidates = []
    for b in bridge_tables:
        alt = _solve(terminals, _restrict(arcs, component, frozenset({b})), algorithm)
        if alt is None:
            continue
        alt_tables = frozenset({n for a in alt[1] for n in (a.src, a.dst)} | set(terminals))
        if alt_tables in seen_sets:
            continue
        seen_sets.add(alt_tables)
        candidates.append((tree_cost(alt[1]), sorted(alt_tables)))
    for cost, tables in sorted(candidates)[:max_alternatives]:
        alternatives.append(PlanAlternative(tables=tables, total_cost=cost))

    return JoinPlan(
        db_id=db_id,
        status=status,
        algorithm=name,
        seed_tables=seeds,
        bridge_tables=bridge_tables,
        tables=sorted(tables_in_tree),
        joins=_joins(tree, root),
        total_cost=tree_cost(tree),
        unreachable_seeds=unreachable,
        dropped_seeds=sorted(dropped, key=lambda u: (-score[u], u)),
        alternatives=alternatives,
    )


def plan_joins(
    db_id: str,
    seeds: list[SeedTable],
    include_inferred: bool = True,
    weighted: bool = True,
    *,
    max_bridge_tables: int = MAX_BRIDGE_TABLES,
    algorithm: Algorithm = "auto",
    session: Any | None = None,
) -> JoinPlan:
    """SPEC 5.10. Reads the relationships from Neo4j, then plans.

    `include_inferred=False` and `weighted=False` are the two ablations.
    """
    edges = get_reference_edges(db_id, include_inferred=include_inferred, session=session)
    return plan_from_edges(
        db_id, seeds, edges,
        include_inferred=include_inferred, weighted=weighted,
        max_bridge_tables=max_bridge_tables, algorithm=algorithm,
    )