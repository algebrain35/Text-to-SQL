"""Build a concise explanation from the selected schema and join plan."""

from __future__ import annotations

from schemagraph.models import (
    Explanation,
    ExplainJoin,
    ExplainTable,
    SchemaContext,
    SeedTable,
    display_name,
)
from schemagraph.pipeline.llm import chat_completion


def build_explanation(
    question: str,
    sql: str,
    context: SchemaContext,
    seeds: list[SeedTable],
    *,
    usage: dict[str, int] | None = None,
) -> Explanation:
    """Create explanation fields from the plan and an LLM-written summary."""
    plan = context.join_plan
    seed_by_uid = {seed.uid: seed for seed in seeds}
    bridge_uids = set(plan.bridge_tables) if plan is not None else set()
    tables = []
    for uid in context.tables:
        seed = seed_by_uid.get(uid)
        if seed is not None:
            role = "seed"
            reason = f"Matched the question via {seed.matched_on} (score {seed.score:.4f})."
        elif uid in bridge_uids:
            role = "bridge"
            reason = "Needed to connect the retrieved tables."
        else:
            role = "context"
            reason = "Included in the schema context."
        tables.append(
            ExplainTable(uid=uid, display_name=display_name(uid), role=role, reason=reason)
        )

    joins = []
    if plan is not None:
        for join in plan.joins:
            joins.append(
                ExplainJoin(
                    **{
                        "from": display_name(join.from_column),
                        "to": display_name(join.to_column),
                        "source": join.source,
                        "text": (
                            "Inferred relationship; confidence "
                            f"{join.confidence:.2f}."
                            if join.source == "inferred"
                            else "Declared foreign-key relationship."
                        ),
                    }
                )
            )

    messages = [
        {
            "role": "system",
            "content": "Explain a SQL query's purpose in one concise sentence. Do not claim results not shown.",
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nSQL:\n{sql}",
        },
    ]
    summary, input_tokens, output_tokens = chat_completion(messages)
    if usage is not None:
        usage["input_tokens"] = input_tokens
        usage["output_tokens"] = output_tokens

    alternatives = plan.alternatives if plan is not None else []
    interpretation = (
        "The join planner found alternative table paths; the SQL chooses one of them."
        if alternatives
        else None
    )
    return Explanation(
        summary=summary,
        tables=tables,
        join_path=joins,
        interpretation=interpretation,
        warnings=list(context.notes),
    )
