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


def render_ddl(tables: list[TableDetail], *, include_inferred: bool = True) -> str:
    """Render table definitions and column constraints as prompt context.

    Only declared foreign keys are written as FOREIGN KEY clauses. An inferred relationship is a guess, so it
    follows the CREATE TABLE as a comment that says so and gives its confidence; with
    `include_inferred=False` (the planner ablation) it is left out, so the model cannot join on it either.
    """
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
        inferred_lines = []
        for relationship in detail.references:
            target = identifiers.get(relationship.to_column)
            source = identifiers.get(relationship.from_column)
            if target is None or source is None:
                continue
            if relationship.source == "inferred":
                if include_inferred:
                    _, target_schema, target_table, target_column = target
                    inferred_lines.append(
                        f"-- Inferred relationship, not a declared foreign key (confidence {relationship.confidence:.2f}): "
                        f"{_quote_identifier(table.schema_)}.{_quote_identifier(table.name)}.{_quote_identifier(source[3])}"
                        f" -> {_quote_identifier(target_schema)}.{_quote_identifier(target_table)}.{_quote_identifier(target_column)}"
                    )
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
        if inferred_lines:
            statement += "\n" + "\n".join(inferred_lines)
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


def planner_defaults(pipeline_config: Any) -> dict[str, Any]:
    """The planner's seed-selection settings from a pipeline config ({} when it uses the defaults)."""
    selection = getattr(pipeline_config, "planner_selection", "all")
    if pipeline_config is None or selection == "all":
        return {}
    return {"selection": selection, "prize_scale": getattr(pipeline_config, "prize_scale", 3.0)}


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
            if join_plan.status == "none":
                selected = [s.uid for s in seeds]
            else:
                # Seeds no relationship reaches stay in the prompt: the question may still need them (in a
                # subquery, or joined on columns inference missed). Dropping them silently lost real tables.
                selected = [*join_plan.tables, *(u for u in join_plan.unreachable_seeds if u not in join_plan.tables)]
            details = queries.get_tables(db_id, selected)
        else:
            details = queries.get_tables(db_id, [s.uid for s in seeds])

    notes = []
    if mode == "graph" and join_plan is not None and join_plan.status == "none":
        notes.append("No join plan connected the retrieved tables; context contains retrieved tables only.")
    elif mode == "graph" and join_plan is not None and join_plan.unreachable_seeds:
        names = ", ".join(display_name(u) for u in join_plan.unreachable_seeds)
        notes.append(
            f"No known relationship connects {names} to the other tables. Use "
            f"{'it' if len(join_plan.unreachable_seeds) == 1 else 'them'} only if the question needs "
            "it, and join only on columns that clearly correspond."
        )

    context = SchemaContext(
        db_id=db_id,
        mode=mode,
        tables=[detail.table.uid for detail in details],
        ddl=render_ddl(details, include_inferred=(planner_options or {}).get("include_inferred", True)),
        join_text=render_joins(join_plan) if join_plan is not None else "",
        join_plan=join_plan,
        notes=notes,
    )
    if timings is not None:
        timings.update(elapsed)
    return context, seeds
