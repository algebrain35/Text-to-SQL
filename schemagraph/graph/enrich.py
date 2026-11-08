"""Write embeddings onto the graph's Table and Column nodes.

`embed_nodes` reads every table and column of one database back from the graph,
builds the text for each (graph/vector.py), embeds it, and stores the vector in
the `embedding` property the vector indexes cover. Re-running it replaces the
vectors, so it is safe after descriptions change. (Generating missing
descriptions with an LLM is a separate step, not written yet.)

Work is saved as it goes and can be resumed. Tables are embedded before columns, one request's
worth of texts at a time, and each chunk is written to Neo4j before the next is requested. Each
stored vector carries a fingerprint of the text and model it came from, so a rerun embeds only
nodes that have no vector, whose text changed or whose model changed (`reembed=True` forces all).
If the embedding service rate-limits for longer than its wait budget, embedding stops with
what was done kept, and the report says what is left.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from schemagraph.graph import queries, vector
from schemagraph.graph.cypher import read, use_session, write

log = logging.getLogger(__name__)

DEFAULT_CHUNK = 64  # texts per request when the embedder does not say


def set_embeddings_query(label: str, prop: str, hash_prop: str | None = None) -> str:
    """`prop` comes from vector.embedding_names, which only produces checked identifiers."""
    query = f"UNWIND $rows AS row MATCH (n:{label} {{uid: row.uid}}) SET n.{prop} = row.vec"
    return query + (f", n.{hash_prop} = row.hash" if hash_prop else "")


def read_hashes_query(label: str, hash_prop: str) -> str:
    return f"MATCH (n:{label} {{db_id: $db_id}}) RETURN n.uid AS uid, n.{hash_prop} AS hash"


@dataclass(frozen=True)
class EmbedReport:
    """What an `embed_nodes` run did. `tables`/`columns` are embedded in this run; `*_current`
    counts every node that has an up-to-date vector afterwards, out of `*_total`."""

    tables: int
    columns: int
    tables_current: int = 0
    columns_current: int = 0
    tables_total: int = 0
    columns_total: int = 0
    stopped: str | None = None  # why it stopped early (rate limiting), else None

    @property
    def complete(self) -> bool:
        return self.stopped is None


def related_tables(details: list) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """For each table uid: the display names of the tables it references and of those that reference it.

    Self-references and weak inferred links are left out. Pure.
    """
    from schemagraph.models import display_name, table_uid_of_column

    known = {d.table.uid for d in details}
    out: dict[str, set[str]] = {uid: set() for uid in known}
    into: dict[str, set[str]] = {uid: set() for uid in known}
    for d in details:
        for rel in d.references:
            if rel.confidence < vector.RELATED_MIN_CONFIDENCE:
                continue
            target = table_uid_of_column(rel.to_column)
            if target == d.table.uid or target not in known:
                continue
            out[d.table.uid].add(display_name(target))
            into[target].add(display_name(d.table.uid))
    return out, into


def embedding_inputs(
    details: list, include_generated: bool = True, include_relationships: bool = False
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(uid, text) pairs for tables and for columns. Pure.

    `include_relationships` adds each table's neighbours to its text (vector.table_text).
    """
    out, into = related_tables(details) if include_relationships else ({}, {})
    tables: list[tuple[str, str]] = []
    columns: list[tuple[str, str]] = []
    for d in details:
        uid = d.table.uid
        text = vector.table_text(
            d.table, include_generated,
            references=sorted(out.get(uid, ())) or None, referenced_by=sorted(into.get(uid, ())) or None,
        )
        tables.append((uid, text))
        for c in d.columns:
            columns.append((c.uid, vector.column_text(d.table, c, include_generated)))
    return tables, columns


def fingerprint(embedder: object, text: str) -> str:
    """Identifies the text and the model setup that produced a vector. A different text, model,
    dimension or provider gives a different value, so the vector is recomputed."""
    config = getattr(embedder, "config", None)
    setup = "|".join(str(getattr(config, key, "")) for key in ("provider", "model", "dimensions", "send_dimensions"))
    return hashlib.sha256(f"{setup}\n{text}".encode()).hexdigest()[:20]


def plan(
    items: list[tuple[str, str]], stored: dict[str, str | None], embedder: object, reembed: bool = False
) -> tuple[list[tuple[str, str, str]], int]:
    """(uid, text, fingerprint) still to embed, and how many nodes are already up to date. Pure."""
    todo: list[tuple[str, str, str]] = []
    for uid, text in items:
        digest = fingerprint(embedder, text)
        if reembed or stored.get(uid) != digest:
            todo.append((uid, text, digest))
    return todo, len(items) - len(todo)


def _store(session: Any, label: str, prop: str, hash_prop: str, chunk: list[tuple[str, str, str]], vectors: list[list[float]], batch_size: int) -> None:
    rows = [{"uid": uid, "vec": vec, "hash": digest} for (uid, _, digest), vec in zip(chunk, vectors, strict=True)]
    write(session, set_embeddings_query(label, prop, hash_prop), rows, batch_size)


def embed_nodes(
    db_id: str,
    embedder: vector.Embedder | None = None,
    session: Any | None = None,
    include_generated: bool = True,
    batch_size: int = 500,
    name: str | None = None,
    reembed: bool = False,
    chunk_size: int | None = None,
    include_relationships: bool | None = None,
) -> EmbedReport:
    """Embed the tables and columns of `db_id` that need it and store the vectors on the nodes.

    Only nodes without an up-to-date vector are embedded (see the module doc); `reembed=True`
    embeds every node. include_relationships (default: the embedding config's setting) adds each table's
    foreign-key neighbours to its text. include_generated=False leaves LLM-written descriptions out of the text,
    to measure what they add to retrieval. Vectors are stored under the embedder's `name`
    (or the `name` given here), so several models can be embedded into one graph.

    Each chunk is saved before the next request. If the service rate-limits past its wait budget
    (vector.RateLimited), the run stops cleanly and the report's `stopped` says why; other
    embedding errors are raised, with the chunks before them kept.
    """
    embedder = embedder or vector.get_embedder()
    names = vector.names_for(embedder, name)
    hash_prop = f"{names.prop}_hash"
    if include_relationships is None:
        include_relationships = bool(getattr(getattr(embedder, "config", None), "include_relationships", False))
    size = chunk_size or getattr(getattr(embedder, "config", None), "batch_size", None) or DEFAULT_CHUNK
    with use_session(session) as s:
        details = queries.get_all_tables(db_id, s)
        inputs = dict(zip(("Table", "Column"), embedding_inputs(details, include_generated, include_relationships), strict=True))
        todo: dict[str, list[tuple[str, str, str]]] = {}
        current: dict[str, int] = {}
        for label, items in inputs.items():
            stored = {} if reembed else {r["uid"]: r["hash"] for r in read(s, read_hashes_query(label, hash_prop), db_id=db_id)}
            todo[label], current[label] = plan(items, stored, embedder, reembed)

        done = {"Table": 0, "Column": 0}
        stopped: str | None = None
        for label in ("Table", "Column"):  # tables first: few, and the most useful if the run is cut short
            pending = todo[label]
            for i in range(0, len(pending), size):
                chunk = pending[i : i + size]
                try:
                    vectors = embedder.embed_documents([text for _, text, _ in chunk])
                except vector.RateLimited as exc:
                    stopped = str(exc)
                    break
                if len(vectors) != len(chunk):
                    raise vector.EmbeddingError(f"embedder returned {len(vectors)} vectors for {len(chunk)} texts")
                _store(s, label, names.prop, hash_prop, chunk, vectors, batch_size)
                done[label] += len(chunk)
            if stopped:
                break

    report = EmbedReport(
        tables=done["Table"],
        columns=done["Column"],
        tables_current=current["Table"] + done["Table"],
        columns_current=current["Column"] + done["Column"],
        tables_total=len(inputs["Table"]),
        columns_total=len(inputs["Column"]),
        stopped=stopped,
    )
    log.info(
        "embedded %d tables and %d columns for %s (%d/%d and %d/%d up to date)%s",
        report.tables, report.columns, db_id, report.tables_current, report.tables_total,
        report.columns_current, report.columns_total, f"; stopped: {stopped}" if stopped else "",
    )
    return report
