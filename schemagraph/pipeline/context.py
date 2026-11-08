"""Building the schema text the LLM sees. The three modes are different in which tables go into the prompt:
    full       every table in the database
    retrieval  the top-k tables vector search found (no join hints)
    graph      those tables + the bridge tables the join planner added,
               plus the exact join conditions

`render_ddl` and `render_joins` are tested without any database.
`build_context` calls the graph functions through their modules
(`queries.get_tables`, ...) so tests can swap them for fakes.
"""

from __future__ import annotations
import re
from typing import Any

from schemagraph.graph import planner, queries, retrieve
from schemagraph.models import (
    JoinPlan,
    Mode,
    RelationshipMeta,
    SchemaContext,
    SeedTable,
    TableDetail,
    display_name,
    split_table_uid,
    table_uid_of_column,
)

MAX_SAMPLE_VALUES = 3     # sample values to help match literal values
MAX_SAMPLE_LENGTH = 40    # longer values barely help
SKIP_SAMPLES_FOR = {"uuid", "bytea"}
_PLAIN_IDENT = re.compile(r"^[a-z_][a-z0-9_$]*$")

def quote_ident(name: str) -> str:
    """Quote a PostgreSQL name only when it needs it (capitals, spaces, symbols)."""
    if _PLAIN_IDENT.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'

def _table_name(table_uid: str) -> str:
    _, schema, table = split_table_uid(table_uid)
    return f"{quote_ident(schema)}.{quote_ident(table)}"

def _column_name(column_uid: str) -> str:
    """'db:schema.table.column' -> 'column'"""
    return quote_ident(column_uid[len(table_uid_of_column(column_uid)) + 1:])

def _comment(text: str | None) -> str:
    return " ".join((text or "").split())

def _fk_groups(references: list[RelationshipMeta]) -> list[list[RelationshipMeta]]:
    """Group column pairs of the same composite foreign key, inferred ones stay single."""
    groups: dict[tuple[str, str], list[RelationshipMeta]] = {}
    for i, rel in enumerate(references):
        key = (rel.constraint_name or f"#{i}", table_uid_of_column(rel.to_column))
        groups.setdefault(key, []).append(rel)
    return list(groups.values())

def render_table(detail: TableDetail) -> str:
    """A CREATE TABLE statement as comments."""
    t = detail.table
    lines: list[str] = []
    if t.description:
        lines.append(f"-- {_comment(t.description)}")

    body: list[tuple[str, str]] = []
    for col in detail.columns:
        line = f"  {quote_ident(col.name)} {col.data_type}"
        if not col.is_nullable:
            line += " NOT NULL"
        notes = []
        if col.description:
            notes.append(_comment(col.description).rstrip(".") + ".")
        samples = [v for v in col.sample_values[:MAX_SAMPLE_VALUES] if len(v) <= MAX_SAMPLE_LENGTH]
        if samples and col.data_type not in SKIP_SAMPLES_FOR:
            notes.append("e.g. " + ", ".join(repr(v) for v in samples))
        body.append((line, " ".join(notes)))

    if t.primary_key:
        body.append((f"  PRIMARY KEY ({', '.join(quote_ident(c) for c in t.primary_key)})", ""))

    for group in _fk_groups(detail.references):
        src = ", ".join(_column_name(r.from_column) for r in group)
        dst = ", ".join(_column_name(r.to_column) for r in group)
        target = _table_name(table_uid_of_column(group[0].to_column))
        note = ""
        if group[0].source == "inferred":
            note = f"inferred, not declared (confidence {group[0].confidence:.2f})"
        body.append((f"  FOREIGN KEY ({src}) REFERENCES {target} ({dst})", note))

    lines.append(f"CREATE TABLE {_table_name(t.uid)} (")
    for i, (text, note) in enumerate(body):
        comma = "," if i < len(body) - 1 else ""
        lines.append(f"{text}{comma}" + (f"  -- {note}" if note else ""))
    lines.append(");")
    return "\n".join(lines)

def render_ddl(tables: list[TableDetail]) -> str:
    """Create table text separated by blank lines."""
    return "\n\n".join(render_table(d) for d in tables)

def render_joins(plan: JoinPlan) -> str:
    """One join condition per line"""
    lines = []
    for j in plan.joins:
        line = f"{display_name(j.from_column)} = {display_name(j.to_column)}"
        notes = []
        if j.source == "inferred":
            notes.append(f"inferred, confidence {j.confidence:.2f}")
        if j.traversal == "one_to_many":
            notes.append("one-to-many: can repeat rows, take care with SUM/COUNT")
        if notes:
            line += "  -- " + "; ".join(notes)
        lines.append(line)
    return "\n".join(lines)

def build_context(
        db_id: str,
        question: str,
        mode: Mode,
        top_k: int,
        *,
        planner_options: dict[str, Any] | None = None,
) -> tuple[SchemaContext, list[SeedTable]]:
    """Returns the prompt context and the seed tables retrieval found.

    the pipeline turns an empty tables list into status "no_schema". `planner_options`
    is passed to plan_joins for the evaluation's ablations.
    """
    notes: list[str] = []

    if mode == "full":
        details = queries.get_all_tables(db_id)
        context = SchemaContext(
            db_id=db_id, mode=mode,
            tables=[d.table.uid for d in details], ddl=render_ddl(details),
        )
        return context, []

    seeds = retrieve.retrieve_seed_tables(db_id, question, top_k)
    if not seeds:
        return SchemaContext(db_id=db_id, mode=mode, tables=[], ddl="", notes=["No tables found."]), []

    seed_uids = [s.uid for s in seeds]
    plan: JoinPlan | None = None
    join_text = ""
    uids = seed_uids

    if mode == "graph":
        plan = planner.plan_joins(db_id, seeds, **(planner_options or {}))
        if plan.status == "none":
            notes.append("No join path was found")
        else:
            # Planned tables first, then seeds the plan could not reach, nothing is lost
            uids = list(plan.tables) + [u for u in plan.unreachable_seeds if u not in plan.tables]
            join_text = render_joins(plan)
            for u in plan.unreachable_seeds:
                notes.append(f"{display_name(u)} has no known join path to the other tables.")

    details = queries.get_tables(db_id, uids)

    context = SchemaContext(
        db_id=db_id,
        mode=mode,
        tables=[d.table.uid for d in details],
        ddl=render_ddl(details),
        join_text=join_text,
        join_plan=plan,
        notes=notes,
    )
    return context, seeds
