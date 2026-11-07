"""
Everything except `summary` is built from templates, with no LLM call:
    tables          each table the model was given, with its role and why it is there
                    (seed = matched the question, bridge = needed to connect seeds,
                     context = full/retrieval modes)
    join_path       each join in the plan, in words
    interpretation  filled when the planner found other ways to connect the tables
    warnings        inferred joins, unreachable or dropped tables, row fan-out

`summary` is one short LLM call. Pass `summarize=False` to skip it, and the rest is still filled in, because the evaluation reads the table roles.
"""
from __future__ import annotations
import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from schemagraph.models import (
    ExplainJoin,
    ExplainTable,
    Explanation,
    JoinPlan,
    PlannedJoin,
    SchemaContext,
    SeedTable,
    display_name,
    table_uid_of_column,
)
from schemagraph.sql_validation import referenced_tables
log = logging.getLogger(__name__)
SUMMARY_PROMPT = (
    "You explain SQL queries to someone who is new to this database. In one or two short, plain "
    "sentences, say what the query below returns and how it gets there."
)

def _seed_reason(seed: SeedTable) -> str:
    if seed.matched_on.startswith("column:"):
        return f"Matched the question through its column {seed.matched_on.split(':', 1)[1]}."
    return "Matched the question by its name or description."

def _neighbours(plan: JoinPlan) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for j in plan.joins:
        a, b = table_uid_of_column(j.from_column), table_uid_of_column(j.to_column)
        out.setdefault(a, []).append(b)
        out.setdefault(b, []).append(a)
    return out

def _used_in_sql(sql: str) -> set[str]:
    return referenced_tables(sql) if sql else set()

def explain_tables(context: SchemaContext, seeds: list[SeedTable], sql: str = "") -> list[ExplainTable]:
    by_uid = {s.uid: s for s in seeds}
    plan = context.join_plan
    used = _used_in_sql(sql)
    rows: list[ExplainTable] = []

    if context.mode == "graph" and plan is not None and plan.status != "none":
        bridges = set(plan.bridge_tables)
        links = _neighbours(plan)
        for uid in context.tables:
            if uid in bridges:
                between = " and ".join(display_name(n) for n in links.get(uid, [])) or "the other tables"
                rows.append(ExplainTable(uid=uid, display_name=display_name(uid), role="bridge",
                                         reason=f"Not mentioned in the question, but needed to connect {between}."))
            else:
                reason = _seed_reason(by_uid[uid]) if uid in by_uid else "Matched the question."
                if uid in plan.unreachable_seeds:
                    reason += " No join path links it to the other tables."
                rows.append(ExplainTable(uid=uid, display_name=display_name(uid), role="seed", reason=reason))
        return rows

    for uid in context.tables:
        if uid in by_uid:  # retrieval mode, or graph mode without a plan
            rows.append(ExplainTable(uid=uid, display_name=display_name(uid), role="seed",
                                     reason=_seed_reason(by_uid[uid])))
        else:  # full mode
            reason = "Used in the query." if display_name(uid).lower() in used else "Part of the full schema given to the model."
            rows.append(ExplainTable(uid=uid, display_name=display_name(uid), role="context", reason=reason))
    return rows

def join_text(j: PlannedJoin) -> str:
    a = display_name(table_uid_of_column(j.from_column))
    b = display_name(table_uid_of_column(j.to_column))
    if j.cardinality == "1:1":
        text = f"Each {a} row matches at most one {b} row."
    else:
        text = f"Each {a} row points to one {b} row; one {b} row can have many {a} rows."
    if j.source == "inferred":
        text += (f" This link is not declared in the database "
                 f"values (confidence {j.confidence:.2f}).")
    return text

def explain_joins(plan: JoinPlan | None) -> list[ExplainJoin]:
    if plan is None or plan.status == "none":
        return []
    return [
        ExplainJoin(from_=display_name(j.from_column), to=display_name(j.to_column),
                    source=j.source, text=join_text(j))
        for j in plan.joins
    ]

def interpretation(plan: JoinPlan | None) -> str | None:
    """Filled when the planner found other routes, so the question may have been ambiguous."""
    if plan is None or not plan.alternatives:
        return None
    chosen = set(plan.tables)
    others = []
    for alt in plan.alternatives:
        extra = [display_name(t) for t in alt.tables if t not in chosen]
        if extra:
            others.append("through " + ", ".join(extra))
    if not others:
        return None
    return ("The tables were connected the cheapest way. They could also be connected "
            + "; or ".join(others) + ", which may answer a different reading of the question.")

def warnings(context: SchemaContext) -> list[str]:
    plan = context.join_plan
    out = list(context.notes)
    if plan is None:
        return out
    if any(j.source == "inferred" for j in plan.joins):
        out.append("This answer uses a relationship that was inferred, not declared; check it looks right.")
    if any(j.traversal == "one_to_many" for j in plan.joins):
        out.append("A one-to-many join is used, so totals and counts can include repeated rows.")
    for uid in plan.dropped_seeds:
        out.append(f"{display_name(uid)} looked relevant but was left out")
    return out

def summarize_sql(question: str, sql: str, *, llm: Any | None = None) -> str:
    """Returns "" if the call fails, so an answer is never lost."""
    from schemagraph.pipeline.generate import get_chat_model, message_text
    try:
        model = llm if llm is not None else get_chat_model()
        reply = model.invoke([
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nSQL:\n{sql}"),
        ])
        return message_text(reply).strip()
    except Exception as exc:  # noqa: BLE001 - the explanation is optional
        log.warning("summary failed: %s", exc)
        return ""

def build_explanation(
        question: str,
        sql: str,
        context: SchemaContext,
        seeds: list[SeedTable],
        *,
        summarize: bool = True,
        llm: Any | None = None,
) -> Explanation:
    summary = summarize_sql(question, sql, llm=llm) if summarize and sql else ""
    return Explanation(
        summary=summary,
        tables=explain_tables(context, seeds, sql),
        join_path=explain_joins(context.join_plan),
        interpretation=interpretation(context.join_plan),
        warnings=warnings(context),
    )
