"""Prompt construction and SQL generation."""

from __future__ import annotations

import re

from schemagraph.models import (
    Attempt,
    FewShotExample,
    GenerationResult,
    SchemaContext,
)
from schemagraph.pipeline.llm import chat_completion

_FENCE = re.compile(r"^\s*```(?:sql|postgres|postgresql)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def build_prompt(
    question: str,
    context: SchemaContext,
    examples: list[FewShotExample],
    previous_attempts: list[Attempt],
) -> list[dict[str, str]]:
    """Build the same instruction structure for each schema mode."""
    system = (
        "You translate questions into PostgreSQL SQL. Use only the schema supplied. "
        "Return exactly one read-only SELECT statement, with no prose or markdown. "
        "Never use SELECT INTO, locking clauses, or data-modifying CTEs."
    )
    user_parts = [
        f"Database: {context.db_id}",
        f"Question:\n{question}",
        f"Schema:\n{context.ddl}",
    ]
    if context.join_text:
        user_parts.append(f"Available join guidance:\n{context.join_text}")
    if context.notes:
        user_parts.append("Notes:\n" + "\n".join(f"- {note}" for note in context.notes))
    if examples:
        formatted = []
        for item in examples:
            formatted.append(f"Question: {item.question}\nSQL: {item.gold_sql}")
        user_parts.append("Development examples:\n" + "\n\n".join(formatted))
    if previous_attempts:
        formatted_attempts = [
            f"SQL: {attempt.sql}\nFeedback: {attempt.error or 'No database error was reported.'}"
            for attempt in previous_attempts
        ]
        user_parts.append(
            "Previous attempts were unsuccessful. Correct the SQL using this feedback:\n"
            + "\n\n".join(formatted_attempts)
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def extract_sql(llm_output: str) -> str | None:
    """Remove a single surrounding SQL code fence, preserving SQL text otherwise."""
    output = llm_output.strip()
    match = _FENCE.fullmatch(output)
    if match:
        output = match.group(1).strip()
    return output or None


def generate_sql(
    question: str,
    context: SchemaContext,
    examples: list[FewShotExample],
    previous_attempts: list[Attempt],
) -> GenerationResult:
    messages = build_prompt(question, context, examples, previous_attempts)
    raw_output, input_tokens, output_tokens = chat_completion(messages)
    return GenerationResult(
        sql=extract_sql(raw_output),
        raw_output=raw_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
