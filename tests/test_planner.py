"""graph/planner.py on hand-built table graphs (no Neo4j), plus the AdventureWorks sample."""

import itertools
import json
import random
from pathlib import Path

import pytest

from schemagraph.extraction import extract
from schemagraph.graph import planner as pl
from schemagraph.graph.queries import ReferenceEdge
from schemagraph.graph.weights import edge_weights
from schemagraph.models import JoinPlan, SeedTable

DB = "d"
SAMPLE = Path(__file__).parent.parent / "contracts" / "examples" / "metadata.adventureworks.sample.json"


def t(name):
    return f"{DB}:s.{name}"


def seed(name, score=0.5):
    return SeedTable(uid=t(name), score=score, matched_on="table")


def E(frm, to, *, source="declared", confidence=1.0, cardinality="N:1", constraint="auto", col=None):
    """One relationship; table `frm` holds the FK column `col` pointing at table `to`."""
    col = col or f"{to}id"
    wf, wr = edge_weights(confidence, cardinality)
    if constraint == "auto":
        constraint = f"fk_{frm}_{col}" if source == "declared" else None
    return ReferenceEdge(
        from_column=f"{t(frm)}.{col}", to_column=f"{t(to)}.{to}id",
        from_table=t(frm), to_table=t(to), source=source, confidence=confidence,
        cardinality=cardinality, weight_forward=wf, weight_reverse=wr, constraint_name=constraint,
    )


def plan(seeds, edges, **kw):
    return pl.plan_from_edges(DB, [seed(s, sc) if isinstance(s, str) else s for s, sc in ((x, 0.5) for x in seeds)]
                              if all(isinstance(s, str) for s in seeds) else seeds, edges, **kw)


def names(uids):
    return [u.split(".")[-1] for u in uids]


def join_pairs(p):
    return [(j.from_column.split(".")[-2], j.to_column.split(".")[-2]) for j in p.joins]


# --- basic shapes -------------------------------------------------------------------


def test_no_seeds_is_none():
    p = plan([], [])
    assert (p.status, p.algorithm, p.tables, p.joins) == ("none", "none", [], [])


def test_one_seed_is_single_table():
    p = plan(["a"], [E("a", "b")])
    assert (p.status, p.algorithm, p.tables, p.joins, p.total_cost) == ("single_table", "none", [t("a")], [], 0.0)


def test_direct_join_between_two_seeds():
    p = plan(["orders", "customer"], [E("orders", "customer")])
    assert p.status == "full" and p.algorithm == "shortest_path"
    assert p.bridge_tables == [] and p.total_cost == 1.0
    (j,) = p.joins
    assert (j.from_column, j.to_column) == (f"{t('orders')}.customerid", f"{t('customer')}.customerid")
    assert (j.traversal, j.weight, j.source, j.cardinality) == ("many_to_one", 1.0, "declared", "N:1")


def test_bridge_table_is_found():
    edges = [E("header", "customer"), E("customer", "person")]
    p = plan(["header", "person"], edges)
    assert p.status == "full" and p.bridge_tables == [t("customer")]
    assert p.tables == sorted([t("header"), t("customer"), t("person")])
    assert p.total_cost == 2.0 and join_pairs(p) == [("header", "customer"), ("customer", "person")]


def test_joins_keep_the_stored_direction_when_walked_backwards():
    # seeds: customer and orders. Starting from customer and walking to orders goes against the FK.
    p = plan(["customer", "orders"], [E("orders", "customer")])
    (j,) = p.joins
    assert j.from_column.startswith(t("orders")) and j.to_column.startswith(t("customer"))  # stored direction
    # the planner picks the cheaper root (orders), so the walk is with the FK, not against it
    assert (j.traversal, j.weight, p.total_cost) == ("many_to_one", 1.0, 1.0)


def test_root_choice_makes_fan_out_only_where_unavoidable():
    # three tables hanging off one hub: customer <- orders, customer <- returns. Seeds orders and returns.
    edges = [E("orders", "customer"), E("returns", "customer")]
    p = plan(["orders", "returns"], edges)
    assert p.bridge_tables == [t("customer")]
    # one leg must be walked against its FK (fan-out): cost 1 (forward) + 2 (reverse)
    assert p.total_cost == 3.0
    assert sorted(j.traversal for j in p.joins) == ["many_to_one", "one_to_many"]


def test_one_to_one_link_does_not_fan_out_backwards():
    p = plan(["employee", "person"], [E("employee", "person", cardinality="1:1")])
    assert p.total_cost == 1.0 and [j.traversal for j in p.joins] == ["many_to_one"]
    arcs = pl.build_arcs(pl.build_links([E("employee", "person", cardinality="1:1")]))
    back = arcs[t("person")][t("employee")]
    assert (back.traversal, back.weight) == ("many_to_one", 1.0)


# --- weights and ablations -----------------------------------------------------------


def test_declared_path_beats_a_shorter_inferred_one():
    # A -> C inferred (confidence 0.72, cost 2.4) versus A -> B -> C declared (cost 2.0)
    edges = [E("a", "c", source="inferred", confidence=0.72), E("a", "b"), E("b", "c")]
    p = plan(["a", "c"], edges)
    assert p.bridge_tables == [t("b")] and p.total_cost == 2.0
    assert {j.source for j in p.joins} == {"declared"}


def test_unweighted_ablation_counts_hops_only():
    edges = [E("a", "c", source="inferred", confidence=0.72), E("a", "b"), E("b", "c")]
    p = plan(["a", "c"], edges, weighted=False)
    assert p.bridge_tables == [] and p.total_cost == 1.0 and p.joins[0].source == "inferred"


def test_excluding_inferred_removes_them():
    edges = [E("a", "c", source="inferred", confidence=0.72), E("a", "b"), E("b", "c")]
    p = plan(["a", "c"], edges, include_inferred=False)
    assert {j.source for j in p.joins} == {"declared"}
    only_inferred = plan(["a", "c"], [E("a", "c", source="inferred", confidence=0.72)], include_inferred=False)
    assert only_inferred.status == "none"


def test_cheaper_of_two_foreign_keys_between_the_same_tables():
    edges = [E("a", "b", col="x", source="inferred", confidence=0.6), E("a", "b", col="y")]
    p = plan(["a", "b"], edges)
    assert p.joins[0].from_column.endswith(".y") and p.joins[0].source == "declared"


def test_equal_foreign_keys_choose_deterministically():
    edges = [E("a", "b", col="shipto"), E("a", "b", col="billto")]
    first = plan(["a", "b"], edges)
    assert first.joins[0].from_column.endswith(".billto")           # lowest column uid wins the tie
    assert plan(["a", "b"], list(reversed(edges))).model_dump() == first.model_dump()


def test_composite_foreign_key_is_one_link_with_every_column_pair():
    pair1 = E("a", "b", col="k1", constraint="fk_ab")
    pair2 = E("a", "b", col="k2", constraint="fk_ab")
    p = plan(["a", "b"], [pair1, pair2])
    assert len(p.joins) == 2 and {j.from_column.split(".")[-1] for j in p.joins} == {"k1", "k2"}
    assert p.total_cost == 1.0  # one link, costed once


def test_self_references_are_ignored():
    p = plan(["a", "b"], [E("a", "a", col="parentid"), E("a", "b")])
    assert p.total_cost == 1.0 and len(p.joins) == 1


def test_equal_cost_declared_beats_inferred_between_the_same_tables():
    # an inferred link can cost exactly as much as a declared one (confidence 1.0); declared must win the tie
    edges = [E("a", "b", col="a_inferred", source="inferred", confidence=1.0), E("a", "b", col="z_declared")]
    (j,) = plan(["a", "b"], edges).joins
    assert j.from_column.endswith(".z_declared") and j.source == "declared"


def test_build_links_skips_self_references():
    assert pl.build_links([E("a", "a", col="parentid")]) == []


def test_unweighted_arcs_all_cost_one():
    arcs = pl.build_arcs(pl.build_links([E("a", "b", source="inferred", confidence=0.6), E("b", "c")]), weighted=False)
    assert {a.weight for dsts in arcs.values() for a in dsts.values()} == {1.0}


def test_tidy_keeps_one_way_into_each_table_and_cuts_dead_branches():
    def arc(x, y, w=1.0):
        return pl.Arc(t(x), t(y), w, "many_to_one", pl.Link(t(x), t(y), (E(x, y),)))
    # root r reaches seed s two ways (via p and via q), and also has a dead-end branch r -> junk
    chosen = [arc("r", "p"), arc("r", "q", 3.0), arc("p", "s"), arc("q", "s"), arc("r", "junk")]
    tree = pl._tidy(t("r"), [t("s")], chosen)
    assert sorted((a.src, a.dst) for a in tree) == [(t("p"), t("s")), (t("r"), t("p"))]
    assert pl.tree_cost(tree) == 2.0


def test_networkx_walks_the_tree_from_the_cheapest_seed():
    # orders holds the FK to customer: walking orders -> customer is cheap, the other way fans out
    edges = [E("orders", "customer")]
    for orders_score, customer_score in ((0.9, 0.5), (0.5, 0.9)):
        p = plan([seed("orders", orders_score), seed("customer", customer_score)], edges, algorithm="networkx")
        assert p.total_cost == 1.0 and p.joins[0].traversal == "many_to_one"


# --- reachability and thresholds -------------------------------------------------------


def test_unreachable_seed_makes_a_partial_plan():
    edges = [E("a", "b"), E("x", "y")]
    p = plan([seed("a", 0.9), seed("b", 0.8), seed("x", 0.7)], edges)
    assert p.status == "partial" and p.unreachable_seeds == [t("x")]
    assert p.tables == [t("a"), t("b")] and len(p.joins) == 1


def test_all_seeds_mutually_unreachable_is_none():
    p = plan(["a", "x"], [E("a", "b"), E("x", "y")])
    assert p.status == "none" and set(p.unreachable_seeds) == {t("a"), t("x")}


def test_seed_with_no_relationships_at_all():
    p = plan([seed("a", 0.9), seed("b", 0.8), seed("lonely", 0.7)], [E("a", "b")])
    assert p.status == "partial" and p.unreachable_seeds == [t("lonely")]


def test_main_component_is_the_one_with_more_seeds():
    edges = [E("a", "b"), E("b", "c"), E("x", "y")]
    p = plan([seed("x", 0.99), seed("y", 0.98), seed("a", 0.5), seed("b", 0.4), seed("c", 0.3)], edges)
    assert set(p.tables) == {t("a"), t("b"), t("c")} and set(p.unreachable_seeds) == {t("x"), t("y")}


def chain(n):
    return [E(f"n{i}", f"n{i + 1}") for i in range(n - 1)]


def test_too_many_bridges_drops_the_weakest_seed():
    # n0 - n1 - n2 - n3 - n4 - n5 - n6; seeds n0 (0.9), n6 (0.3), n1 (0.8): the n0..n6 tree needs 4 bridges (n2..n5), over the limit of 3
    p = plan([seed("n0", 0.9), seed("n1", 0.8), seed("n6", 0.3)], chain(7), max_bridge_tables=3)
    assert p.dropped_seeds == [t("n6")] and p.status == "partial"
    assert p.tables == [t("n0"), t("n1")] and p.bridge_tables == []


def test_two_far_apart_seeds_with_nothing_to_drop_is_none():
    p = plan([seed("n0", 0.9), seed("n6", 0.8)], chain(7), max_bridge_tables=4)
    assert p.status == "none" and p.tables == []


def test_bridge_limit_is_inclusive():
    p = plan(["n0", "n5"], chain(6), max_bridge_tables=4)       # 4 bridges
    assert p.status == "full" and len(p.bridge_tables) == 4
    assert plan(["n0", "n5"], chain(6), max_bridge_tables=3).status == "none"


# --- alternatives -----------------------------------------------------------------------


def test_alternatives_use_a_different_bridge():
    # two ways from header to person: via customer (cost 2) or via salesperson then employee (cost 3)
    edges = [E("header", "customer"), E("customer", "person"),
             E("header", "salesperson"), E("salesperson", "employee"), E("employee", "person")]
    p = plan(["header", "person"], edges)
    assert p.bridge_tables == [t("customer")] and p.total_cost == 2.0
    (alt,) = p.alternatives
    assert alt.tables == sorted([t("header"), t("salesperson"), t("employee"), t("person")])
    assert alt.total_cost == 3.0


def test_alternatives_that_use_the_same_tables_are_listed_once():
    # banning b1 or banning b2 leads to the same detour through d: one alternative, not two
    edges = [E("a", "b1"), E("b1", "b2"), E("b2", "z"), E("a", "d", source="inferred", confidence=0.5), E("d", "z")]
    p = plan(["a", "z"], edges)
    assert p.bridge_tables == [t("b1"), t("b2")]
    assert len(p.alternatives) == 1 and p.alternatives[0].tables == sorted([t("a"), t("d"), t("z")])


def test_no_alternatives_without_bridges_or_other_routes():
    assert plan(["a", "b"], [E("a", "b")]).alternatives == []
    assert plan(["a", "c"], [E("a", "b"), E("b", "c")]).alternatives == []


def test_alternatives_are_capped_and_sorted_by_cost():
    # best route a - b1 - b2 - b3 - z (cost 4). Each bridge has a detour that avoids it, at a different price.
    edges = [E("a", "b1"), E("b1", "b2"), E("b2", "b3"), E("b3", "z"),
             E("a", "d1", source="inferred", confidence=0.9), E("d1", "b2"),        # avoids b1: 2.5 instead of 2
             E("b1", "d2", source="inferred", confidence=0.8), E("d2", "b3"),       # avoids b2: 3 instead of 2
             E("b2", "d3", source="inferred", confidence=0.7), E("d3", "z")]        # avoids b3: 3.5 instead of 2
    p = plan(["a", "z"], edges, max_alternatives=2)
    assert p.bridge_tables == [t("b1"), t("b2"), t("b3")] and p.total_cost == 4.0
    assert [a.total_cost for a in p.alternatives] == [4.5, 5.0]
    assert p.alternatives[0].tables == sorted(t(x) for x in ("a", "d1", "b2", "b3", "z"))
    assert len(plan(["a", "z"], edges, max_alternatives=3).alternatives) == 3


# --- exact solver versus brute force, and NetworkX ---------------------------------------


def random_graph(rng, n, extra):
    edges = [E(f"t{i}", f"t{rng.randrange(i)}", confidence=rng.choice([1.0, 1.0, 0.85, 0.7]),
               source="declared", cardinality=rng.choice(["N:1", "N:1", "1:1"])) for i in range(1, n)]
    for _ in range(extra):
        a, b = rng.sample(range(n), 2)
        edges.append(E(f"t{a}", f"t{b}", col=f"x{a}{b}", confidence=rng.choice([1.0, 0.8, 0.6]),
                       source=rng.choice(["declared", "inferred"])))
    return edges


def brute_force(terminals, arcs, nodes):
    """Cheapest directed Steiner tree by trying every root and every choice of parent arc."""
    best = float("inf")
    for root in terminals:
        others = [v for v in nodes if v != root]
        options = [[None] + [a for a in (arcs.get(u, {}).get(v) for u in nodes) if a] for v in others]
        for pick in itertools.product(*options):
            parent = {v: a for v, a in zip(others, pick) if a}
            used, ok = set(), True
            for tm in terminals:
                node, hops = tm, 0
                while node != root:
                    a = parent.get(node)
                    if a is None or hops > len(nodes):
                        ok = False
                        break
                    used.add(node)
                    node, hops = a.src, hops + 1
                if not ok:
                    break
            if ok:
                best = min(best, sum(parent[v].weight for v in used))
    return round(best, 6)


@pytest.mark.parametrize("seed_value", range(12))
def test_exact_solver_matches_brute_force(seed_value):
    rng = random.Random(seed_value)
    n = rng.randint(4, 6)
    edges = random_graph(rng, n, extra=rng.randint(1, 4))
    arcs = pl.build_arcs(pl.build_links(edges))
    nodes = sorted({x for s, d in arcs.items() for x in (s, *d)})
    terminals = [t(f"t{i}") for i in rng.sample(range(n), rng.randint(2, min(4, n)))]
    root, tree = pl.solve_exact(terminals, arcs)
    assert pl.tree_cost(tree) == brute_force(terminals, arcs, nodes)
    reached = {root} | {a.dst for a in tree}
    assert set(terminals) <= reached and len({a.dst for a in tree}) == len(tree)  # connected, one way in


@pytest.mark.parametrize("seed_value", range(12))
def test_networkx_never_beats_exact_and_connects_the_seeds(seed_value):
    rng = random.Random(100 + seed_value)
    n = rng.randint(5, 9)
    arcs = pl.build_arcs(pl.build_links(random_graph(rng, n, extra=rng.randint(2, 6))))
    terminals = [t(f"t{i}") for i in rng.sample(range(n), rng.randint(2, 4))]
    exact_cost = pl.tree_cost(pl.solve_exact(terminals, arcs)[1])
    root, tree = pl.solve_networkx(terminals, arcs)
    assert pl.tree_cost(tree) >= exact_cost - 1e-9
    assert set(terminals) <= {root} | {a.dst for a in tree}


def test_algorithm_names():
    edges = [E("a", "b"), E("b", "c"), E("c", "d")]
    assert plan(["a", "d"], edges).algorithm == "shortest_path"
    assert plan(["a", "b", "d"], edges).algorithm == "exact_steiner"
    assert plan(["a", "b", "d"], edges, algorithm="networkx").algorithm == "networkx_steiner"


def test_many_seeds_fall_back_to_networkx():
    edges = [E(f"n{i}", f"n{i + 1}") for i in range(20)]
    seeds = [seed(f"n{i}", 1 - i / 100) for i in range(pl.MAX_EXACT_SEEDS + 1)]
    p = pl.plan_from_edges(DB, seeds, edges)
    assert p.algorithm == "networkx_steiner" and p.status == "full" and len(p.joins) == pl.MAX_EXACT_SEEDS


def test_exact_and_networkx_agree_on_a_simple_chain():
    edges = chain(5)
    a = plan(["n0", "n2", "n4"], edges)
    b = plan(["n0", "n2", "n4"], edges, algorithm="networkx")
    assert a.tables == b.tables and a.total_cost == b.total_cost


# --- contract and determinism -------------------------------------------------------------


def test_plan_is_a_valid_joinplan_and_round_trips_json():
    p = plan(["header", "person"], [E("header", "customer"), E("customer", "person")])
    assert JoinPlan.model_validate_json(p.model_dump_json(by_alias=True)) == p


def test_result_does_not_depend_on_input_order():
    edges = [E("header", "customer"), E("customer", "person"), E("header", "salesperson"),
             E("salesperson", "employee"), E("employee", "person")]
    reference = plan(["header", "person"], edges).model_dump()
    for _ in range(5):
        shuffled = edges[:]
        random.shuffle(shuffled)
        assert plan(["person", "header"], shuffled).model_dump() == reference


def test_duplicate_seeds_keep_the_best_score():
    p = plan([seed("a", 0.2), seed("a", 0.9), seed("b", 0.5)], [E("a", "b")])
    assert [s.score for s in p.seed_tables if s.uid == t("a")] == [0.9]


def test_the_planner_never_raises_on_odd_input():
    assert plan(["nowhere", "also_nowhere"], []).status == "none"
    assert plan(["a", "b"], [E("a", "b")], max_bridge_tables=0).status == "full"


# --- the AdventureWorks sample --------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_edges():
    return pl.edges_from_metadata(extract.load_metadata(SAMPLE))


def aw(name):
    return f"adventureworks:{name}"


def test_sample_customer_names_for_orders(sample_edges):
    seeds = [SeedTable(uid=aw("sales.salesorderheader"), score=0.84, matched_on="column:totaldue"),
             SeedTable(uid=aw("person.person"), score=0.79, matched_on="column:firstname")]
    p = pl.plan_from_edges("adventureworks", seeds, sample_edges)
    assert p.status == "full" and p.bridge_tables == [aw("sales.customer")] and p.total_cost == 2.0
    assert [(j.from_column.split(".")[-2:], j.traversal) for j in p.joins] == [
        (["salesorderheader", "customerid"], "many_to_one"), (["customer", "personid"], "many_to_one")]


def test_sample_unconnected_tables_are_reported(sample_edges):
    seeds = [SeedTable(uid=aw("sales.salesorderheader"), score=0.9, matched_on="table"),
             SeedTable(uid=aw("sales.customer"), score=0.8, matched_on="table"),
             SeedTable(uid=aw("production.transactionhistoryarchive"), score=0.7, matched_on="table")]
    p = pl.plan_from_edges("adventureworks", seeds, sample_edges)
    assert p.status == "partial" and p.unreachable_seeds == [aw("production.transactionhistoryarchive")]


def test_sample_inferred_edge_is_used_but_can_be_excluded(sample_edges):
    seeds = [SeedTable(uid=aw("production.transactionhistoryarchive"), score=0.9, matched_on="table"),
             SeedTable(uid=aw("production.productsubcategory"), score=0.8, matched_on="table")]
    p = pl.plan_from_edges("adventureworks", seeds, sample_edges)
    assert p.status == "full" and p.bridge_tables == [aw("production.product")]
    assert "inferred" in {j.source for j in p.joins}
    assert pl.plan_from_edges("adventureworks", seeds, sample_edges, include_inferred=False).status == "none"


def test_join_plan_sample_file_matches_what_the_planner_produces(sample_edges):
    seeds = [SeedTable(uid=aw("sales.salesorderheader"), score=0.84, matched_on="column:totaldue"),
             SeedTable(uid=aw("person.person"), score=0.79, matched_on="column:firstname")]
    produced = pl.plan_from_edges("adventureworks", seeds, sample_edges)
    stored = JoinPlan.model_validate_json((SAMPLE.parent / "join_plan.sample.json").read_text())
    assert stored == produced  # regenerate the file if the planner's output changes on purpose


def test_networkx_reports_no_route_when_banning_a_bridge_splits_the_seeds():
    """Two clusters of seeds joined only through `bridge`: without it there is no tree. This used to raise
    KeyError in _tidy (a forest was treated as a tree) whenever the alternatives loop banned the bridge."""
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    edges = [E(a, b) for a, b in zip(left, left[1:], strict=False)] + [E(a, b) for a, b in zip(right, right[1:], strict=False)]
    edges += [E("l0", "bridge"), E("r0", "bridge")]
    seeds = [seed(n, 0.9 - i / 100) for i, n in enumerate(left + right)]
    assert len(seeds) > pl.MAX_EXACT_SEEDS

    arcs = pl.build_arcs(pl.build_links(edges, True), True)
    nodes = {n for src, dsts in arcs.items() for n in (src, *dsts)}
    without = pl._restrict(arcs, nodes, frozenset({t("bridge")}))
    assert pl.solve_networkx([s.uid for s in seeds], without) is None

    p = plan(seeds, edges)  # the whole plan, including its alternatives, must not raise
    assert p.status == "full" and p.bridge_tables == [t("bridge")] and p.alternatives == []
