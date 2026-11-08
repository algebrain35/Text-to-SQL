"""Schema context construction for full, retrieval, and graph modes."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from schemagraph.graph import planner, queries, retrieve
from schemagraph.models import (
    JoinPlan,
    Mode,
    SchemaContext,
    SeedTable,
    TableDetail,
    display_name,
)


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def render_ddl(tables: list[TableDetail]) -> str:
    """Render table definitions and column constraints as prompt context."""
    identifiers = {
        column.uid: (detail.table.uid, detail.table.schema_, detail.table.name, column.name)
        for detail in tables
        for column in detail.columns
    }
    statements = []
    for detail in tables:
        table = detail.table
        columns = []
        for column in detail.columns:
            definition = f"  {_quote_identifier(column.name)} {column.data_type}"
            if not column.is_nullable:
                definition += " NOT NULL"
            if column.is_unique and not column.is_primary_key:
                definition += " UNIQUE"
            columns.append(definition)
        if table.primary_key:
            columns.append(
                "  PRIMARY KEY ("
                + ", ".join(_quote_identifier(name) for name in table.primary_key)
                + ")"
            )
        relationships: dict[
            tuple[str, str], list[tuple[str, tuple[str, str, str, str]]]
        ] = {}
        for relationship in detail.references:
            target = identifiers.get(relationship.to_column)
            source = identifiers.get(relationship.from_column)
            if target is None or source is None:
                continue
            key = (
                ("constraint", relationship.constraint_name)
                if relationship.constraint_name
                else ("pair", relationship.from_column + "\0" + relationship.to_column)
            )
            relationships.setdefault(key, []).append((source[3], target))
        for grouped in relationships.values():
            source_names = ", ".join(_quote_identifier(source_name) for source_name, _ in grouped)
            _, target_schema, target_table, _ = grouped[0][1]
            target_names = ", ".join(_quote_identifier(target[3]) for _, target in grouped)
            columns.append(
                f"  FOREIGN KEY ({source_names}) REFERENCES "
                f"{_quote_identifier(target_schema)}.{_quote_identifier(target_table)} ({target_names})"
            )
        statement = (
            f"CREATE TABLE {_quote_identifier(table.schema_)}.{_quote_identifier(table.name)} (\n"
            + ",\n".join(columns)
            + "\n);"
        )
        if table.description:
            statement = f"-- {table.description}\n" + statement
        statements.append(statement)
    return "\n\n".join(statements)


def render_joins(plan: JoinPlan) -> str:
    """Render planned column relationships and their provenance for the prompt."""
    lines = []
    for join in plan.joins:
        source = join.source
        confidence = f", confidence {join.confidence:.2f}" if source == "inferred" else ""
        lines.append(
            f"{display_name(join.from_column)} -> {display_name(join.to_column)} "
            f"({source}{confidence}; traversal {join.traversal})"
        )
    return "\n".join(lines)


def expand_options(pipeline_config: Any) -> dict[str, int] | None:
    """The `expand` argument of `build_context` for a pipeline config, or None when expansion is off."""
    if pipeline_config is None or not getattr(pipeline_config, "expand_seeds", 0):
        return None
    return {"max_added": pipeline_config.expand_seeds, "hops": pipeline_config.expand_hops}


def build_context(
    db_id: str,
    question: str,
    mode: Mode,
    top_k: int,
    *,
    planner_options: dict[str, Any] | None = None,
    expand: dict[str, int] | None = None,
    timings: dict[str, int] | None = None,
) -> tuple[SchemaContext, list[SeedTable]]:
    """Build schema context and the retrieval seeds used to explain the answer.

    ``timings`` is an optional mutable accumulator filled with retrieval and
    planning milliseconds for the pipeline metrics.
    """
    elapsed = {"retrieval": 0, "planning": 0}
    seeds: list[SeedTable] = []
    join_plan: JoinPlan | None = None

    if mode == "full":
        started = perf_counter()
        details = queries.get_all_tables(db_id)
        elapsed["retrieval"] = round((perf_counter() - started) * 1000)
    else:
        started = perf_counter()
        seeds = retrieve.retrieve_seed_tables(db_id, question, top_k)
        elapsed["retrieval"] = round((perf_counter() - started) * 1000)
        if mode == "graph" and seeds:
            started = perf_counter()
            if expand and expand.get("max_added", 0) > 0:  # add the tables the seeds reference, which the question may not name (retrieve.expand_seeds)
                edges = queries.get_reference_edges(
                    db_id, include_inferred=(planner_options or {}).get("include_inferred", True)
                )
                seeds = retrieve.expand_seeds(seeds, edges, **expand)
            join_plan = planner.plan_joins(db_id, seeds, **(planner_options or {}))
            elapsed["planning"] = round((perf_counter() - started) * 1000)
            selected = join_plan.tables if join_plan.status != "none" else [s.uid for s in seeds]
            details = queries.get_tables(db_id, selected)
        else:
            details = queries.get_tables(db_id, [s.uid for s in seeds])

    notes = []
    if mode == "graph" and join_plan is not None and join_plan.status == "none":
        notes.append("No join plan connected the retrieved tables; context contains retrieved tables only.")

    context = SchemaContext(
        db_id=db_id,
        mode=mode,
        tables=[detail.table.uid for detail in details],
        ddl=render_ddl(details),
        join_text=render_joins(join_plan) if join_plan is not None else "",
        join_plan=join_plan,
        notes=notes,
    )
    if timings is not None:
        timings.update(elapsed)
    return context, seeds
