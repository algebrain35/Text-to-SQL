"""Single entry point for answering natural-language database questions."""

from __future__ import annotations

import time
import uuid

import psycopg

from schemagraph import db
from schemagraph.config import get_config
from schemagraph.models import (
    AskRequest,
    AskResponse,
    Attempt,
    LatencyMs,
    Metrics,
)
from schemagraph.pipeline.context import build_context, expand_options
from schemagraph.pipeline.examples import retrieve_examples
from schemagraph.pipeline.explain import build_explanation, explain_context
from schemagraph.pipeline.generate import generate_sql
from schemagraph.pipeline.validate import validate
from schemagraph.sql_validation import check_select

_MAX_ROW_LIMIT = 100_000


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def answer_question(
    request: AskRequest,
    explain: bool = True,
    *,
    planner_options: dict | None = None,
    expand: dict | None = None,
) -> AskResponse:
    """Answer one question; the evaluation harness uses this same entry point.

    Model-written SQL is parsed, explained, then run only through the read-only
    database API. Service/network failures propagate to the API's error mapping.

    `expand` ({"max_added": N, "hops": H}) overrides the config's `pipeline.expand_seeds` for this
    call (the evaluation sweeps use it); N = 0 turns expansion off. None uses the config.
    """
    started_total = time.perf_counter()
    config = get_config()
    pipeline_config = config.pipeline
    options = request.options
    top_k = options.top_k if options and options.top_k is not None else pipeline_config.top_k
    max_attempts = (
        options.max_attempts if options and options.max_attempts is not None else pipeline_config.max_attempts
    )
    row_limit = options.row_limit if options and options.row_limit is not None else pipeline_config.row_limit
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not 1 <= row_limit <= _MAX_ROW_LIMIT:
        raise ValueError(f"row_limit must be between 1 and {_MAX_ROW_LIMIT}")

    latency = {"retrieval": 0, "planning": 0}
    context, seeds = build_context(
        request.db_id,
        request.question,
        request.mode,
        top_k,
        planner_options=planner_options,
        timings=latency,
        **({"expand": use_expand} if (use_expand := expand if expand is not None else expand_options(pipeline_config)) else {}),
    )
    examples_started = time.perf_counter()
    examples = retrieve_examples(request.db_id, request.question, pipeline_config.examples_k)
    latency["retrieval"] += _elapsed_ms(examples_started)
    attempts: list[Attempt] = []
    input_tokens = 0
    output_tokens = 0
    generation_ms = 0
    validation_ms = 0
    execution_ms = 0
    status = "invalid_sql"
    final_sql: str | None = None
    result = None

    if not context.tables:
        status = "no_schema"
    else:
        for _ in range(max_attempts):
            generation_started = time.perf_counter()
            generated = generate_sql(request.question, context, examples, attempts)
            generation_ms += _elapsed_ms(generation_started)
            input_tokens += generated.input_tokens
            output_tokens += generated.output_tokens
            sql = generated.sql
            final_sql = sql
            if not sql:
                status = "unsafe_sql"
                attempts.append(Attempt(sql="", error="The model did not return SQL."))
                continue

            check = check_select(sql)
            if not check.ok:
                status = "unsafe_sql"
                attempts.append(Attempt(sql=sql, error=check.reason))
                continue

            validation_started = time.perf_counter()
            validation = validate(request.db_id, sql)
            validation_ms += _elapsed_ms(validation_started)
            if not validation.ok:
                status = "invalid_sql"
                attempts.append(Attempt(sql=sql, error=validation.error or "PostgreSQL rejected the query."))
                continue

            execution_started = time.perf_counter()
            try:
                result = db.run_readonly_query(request.db_id, sql, row_limit)
            except psycopg.Error as exc:
                execution_ms += _elapsed_ms(execution_started)
                status = "execution_error"
                attempts.append(Attempt(sql=sql, error=str(exc).strip()))
                continue
            execution_ms += _elapsed_ms(execution_started)
            status = "success"
            attempts.append(Attempt(sql=sql))
            break

    # The structural explanation (tables given, joins, warnings) needs no model call, so it is attached
    # whenever there was a schema context, whatever the status, and whether or not `explain` is set.
    # `explain` only controls the model-written summary, which needs a successful query.
    explanation = explain_context(context, seeds) if context.tables else None
    explanation_ms = 0
    if explain and status == "success" and final_sql is not None:
        explanation_started = time.perf_counter()
        usage: dict[str, int] = {}
        explanation = build_explanation(request.question, final_sql, context, seeds, usage=usage)
        explanation_ms = _elapsed_ms(explanation_started)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    total_ms = _elapsed_ms(started_total)
    return AskResponse(
        request_id=str(uuid.uuid4()),
        question=request.question,
        db_id=request.db_id,
        mode=request.mode,
        status=status,
        sql=final_sql,
        result=result,
        explanation=explanation,
        join_plan=context.join_plan,
        attempts=attempts,
        metrics=Metrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attempt_count=len(attempts),
            latency_ms=LatencyMs(
                retrieval=latency["retrieval"],
                planning=latency["planning"],
                generation=generation_ms,
                validation=validation_ms,
                execution=execution_ms,
                explanation=explanation_ms,
                total=total_ms,
            ),
        ),
    )
