"""How much of the right schema did a mode put in front of the model (SPEC 5.19)?

All three functions work on display names, 'schema.table', which is also what the
evaluation result file stores. Convert UIDs with `models.display_name`.
"""

from __future__ import annotations

from schemagraph.sql_validation import referenced_tables


def tables_in_sql(sql: str, default_schema: str = "public") -> set[str]:
    """Tables a query reads, as 'schema.table'. Unqualified names get `default_schema`.

    PostgreSQL resolves an unqualified name through the search path, which is
    `public` unless configured otherwise, so that is the default here (BIRD
    databases live in `public`; AdventureWorks queries are schema-qualified).
    Tables defined by WITH are not counted, and unparsable SQL gives an empty set.
    """
    return {name if "." in name else f"{default_schema}.{name}" for name in referenced_tables(sql)}


def schema_recall(gold_tables: set[str], context_tables: set[str]) -> float:
    """Share of the tables the gold query needs that were in the context (1.0 if it needs none)."""
    if not gold_tables:
        return 1.0
    return len(gold_tables & context_tables) / len(gold_tables)


def schema_precision(gold_tables: set[str], context_tables: set[str]) -> float | None:
    """Share of the context's tables that the gold query uses (None for an empty context).

    The counterpart of schema_recall: recall says whether the right tables were there,
    precision how much else came with them. Full mode's precision is low by design.
    """
    if not context_tables:
        return None
    return len(gold_tables & context_tables) / len(context_tables)


def bridge_tables_needed(gold_tables: set[str], seed_tables: set[str]) -> set[str]:
    """Gold tables that retrieval did not return, so something else had to supply them.

    In graph mode these are the tables the join planner had to add; in retrieval
    mode nothing adds them, so a model in that mode cannot recover them.
    """
    return gold_tables - seed_tables
