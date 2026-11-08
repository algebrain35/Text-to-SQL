"""Find the tables most related to a question (SPEC 5.9).

The question is embedded once and searched against two vector indexes: one over
tables and one over columns. A column hit counts for its table, so a table is
found whether the question mentions it or only one of its columns. Each table
keeps its best score; `matched_on` says where that score came from.

The indexes hold every database in one place, so each search asks for more
candidates than needed and keeps only those with this database's db_id.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from schemagraph.graph import vector
from schemagraph.graph.cypher import read, use_session
from schemagraph.models import SeedTable, display_name

DEFAULT_CANDIDATES = 100
COLUMN_FACTOR = 3  # a table has several columns, so over-fetch column hits

TABLE_HITS = (
    "CALL db.index.vector.queryNodes($index, $n, $vector) YIELD node, score "
    "WHERE node.db_id = $db_id "
    "RETURN node.uid AS uid, score"
)
COLUMN_HITS = (
    "CALL db.index.vector.queryNodes($index, $n, $vector) YIELD node, score "
    "WHERE node.db_id = $db_id "
    "MATCH (t:Table)-[:HAS_COLUMN]->(node) "
    "RETURN t.uid AS uid, node.name AS column, score"
)


def rollup(table_hits: list[dict], column_hits: list[dict], k: int) -> list[SeedTable]:
    """Best score per table, ties broken by uid, top k. Pure.

    table_hits: {"uid", "score"}; column_hits: {"uid" (the table's), "column", "score"}.
    A table whose own score is at least as high as every column's is matched on "table".
    """
    if k <= 0:
        return []
    best: dict[str, tuple[float, str]] = {}
    for hit in sorted(table_hits, key=lambda h: (-h["score"], h["uid"])):
        best.setdefault(hit["uid"], (float(hit["score"]), "table"))
    for hit in sorted(column_hits, key=lambda h: (-h["score"], h["column"])):
        score = float(hit["score"])
        if hit["uid"] not in best or score > best[hit["uid"]][0]:
            best[hit["uid"]] = (score, f"column:{hit['column']}")
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [SeedTable(uid=uid, score=round(score, 4), matched_on=on) for uid, (score, on) in ranked[:k]]


def retrieve_seed_tables(
    db_id: str,
    question: str,
    k: int,
    *,
    embedder: vector.Embedder | None = None,
    session: Any | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    min_score: float | None = None,
    name: str | None = None,
) -> list[SeedTable]:
    """The k tables most related to `question`, best first. Fewer if the database has fewer.

    min_score drops weaker hits (Neo4j cosine scores run 0 to 1). The search uses the vectors stored
    under the embedder's `name` (or `name` given here), which must be the model that embedded the nodes. Raises
    EmbeddingError if the embedding service fails; an empty question gives [].
    """
    if k <= 0 or not question.strip():
        return []
    embedder = embedder or vector.get_embedder()
    names = vector.names_for(embedder, name)
    vec = embedder.embed_query(question)
    with use_session(session) as s:
        tables = read(s, TABLE_HITS, index=names.table_index, n=candidates, vector=vec, db_id=db_id)
        columns = read(s, COLUMN_HITS, index=names.column_index, n=candidates * COLUMN_FACTOR, vector=vec, db_id=db_id)
    seeds = rollup(tables, columns, k)
    if min_score is not None:
        seeds = [s_ for s_ in seeds if s_.score >= min_score]
    return seeds


EXPAND_MIN_CONFIDENCE = 0.8  # declared keys are 1.0; weak inferred links would only add noise
EXPAND_SCORE_STEP = 0.05     # an added table scores this much below the seed that reached it, per link


def expand_seeds(
    seeds: list[SeedTable],
    edges: Iterable[Any],
    *,
    max_added: int,
    hops: int = 1,
    min_confidence: float = EXPAND_MIN_CONFIDENCE,
) -> list[SeedTable]:
    """The seeds plus up to `max_added` tables they have foreign keys to, within `hops` links. Pure.

    Retrieval finds the tables a question talks about, but the table holding a needed column is often one the
    question never names: the person behind a salesperson, the department of an employee, the product under a
    category. Those are what the seeds *reference*, so they are the candidates (not the tables that reference
    the seeds, which would multiply with every child table). A candidate is ranked by how many seeds reach it,
    then by fewer links, then by the best score among the seeds that reach it. `edges` are anything with
    from_table, to_table (the foreign-key side first) and confidence, e.g. `queries.ReferenceEdge`.
    An added table's `matched_on` is 'reference_of:<seed>' and its score a little below that seed's.
    """
    if max_added <= 0 or not seeds:
        return list(seeds)
    parents: dict[str, set[str]] = {}
    for e in edges:
        if e.confidence >= min_confidence and e.from_table != e.to_table:
            parents.setdefault(e.from_table, set()).add(e.to_table)

    seed_uids = {s.uid for s in seeds}
    reached: dict[str, dict[str, Any]] = {}  # candidate uid -> {"seeds": set, "hop": int, "score": float, "via": str}
    for seed in sorted(seeds, key=lambda s: (-s.score, s.uid)):
        frontier, seen = {seed.uid}, {seed.uid}
        for hop in range(1, hops + 1):
            frontier = {p for t in frontier for p in parents.get(t, ()) if p not in seen}
            seen |= frontier
            for uid in frontier - seed_uids:
                c = reached.setdefault(uid, {"seeds": set(), "hop": hop, "score": seed.score, "via": seed.uid})
                c["seeds"].add(seed.uid)
                c["hop"] = min(c["hop"], hop)  # score and via keep the first, i.e. best-scoring, seed to reach it
    ranked = sorted(reached.items(), key=lambda kv: (-len(kv[1]["seeds"]), kv[1]["hop"], -kv[1]["score"], kv[0]))
    added = [
        SeedTable(
            uid=uid,
            score=round(max(c["score"] - EXPAND_SCORE_STEP * c["hop"], 0.0), 4),
            matched_on=f"reference_of:{display_name(c['via'])}",
        )
        for uid, c in ranked[:max_added]
    ]
    return list(seeds) + added
