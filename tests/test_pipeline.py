from types import SimpleNamespace

import psycopg
import pytest

from schemagraph.models import (
    AskRequest,
    ExplainResult,
    GenerationResult,
    QueryResult,
    SchemaContext,
)
from schemagraph.pipeline import pipeline


@pytest.fixture
def ask_request():
    return AskRequest(question="count schools", db_id="schools", mode="full")


@pytest.fixture
def fake_config():
    return SimpleNamespace(
        pipeline=SimpleNamespace(top_k=5, max_attempts=2, row_limit=10, examples_k=0)
    )


def context_for(ask_request):
    return SchemaContext(
        db_id=ask_request.db_id,
        mode=ask_request.mode,
        tables=["schools:public.school"],
        ddl='CREATE TABLE "public"."school" ("id" integer);',
    )


def test_answer_question_success_skips_explanation_when_requested(
    monkeypatch, ask_request, fake_config
):
    contexts = []
    queries = []
    monkeypatch.setattr(pipeline, "get_config", lambda: fake_config)

    def build_context(*args, **kwargs):
        contexts.append((args, kwargs))
        kwargs["timings"].update(retrieval=4, planning=2)
        return context_for(ask_request), []

    monkeypatch.setattr(pipeline, "build_context", build_context)
    monkeypatch.setattr(pipeline, "retrieve_examples", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *args: GenerationResult(sql="SELECT COUNT(*) FROM school", raw_output="...", input_tokens=8),
    )
    monkeypatch.setattr(pipeline, "validate", lambda *args: ExplainResult(ok=True))

    def run_query(db_id, sql, row_limit):
        queries.append((db_id, sql, row_limit))
        return QueryResult(columns=["count"], rows=[["2"]], row_count=1)

    monkeypatch.setattr(pipeline.db, "run_readonly_query", run_query)
    monkeypatch.setattr(
        pipeline,
        "build_explanation",
        lambda *args, **kwargs: pytest.fail("explanation should be skipped"),
    )

    response = pipeline.answer_question(
        ask_request, explain=False, planner_options={"weighted": False}
    )

    assert response.status == "success"
    assert response.result.rows == [["2"]]
    assert response.explanation is None
    assert response.metrics.input_tokens == 8
    assert response.metrics.attempt_count == 1
    assert contexts[0][1]["planner_options"] == {"weighted": False}
    assert queries == [("schools", "SELECT COUNT(*) FROM school", 10)]


def test_answer_question_retries_invalid_then_successful_query(monkeypatch, ask_request, fake_config):
    generated = iter(
        [
            GenerationResult(sql="SELECT 1", raw_output="bad"),
            GenerationResult(sql="SELECT 2", raw_output="good"),
        ]
    )
    validations = iter([ExplainResult(ok=False, error="column does not exist"), ExplainResult(ok=True)])
    previous_attempts = []
    monkeypatch.setattr(pipeline, "get_config", lambda: fake_config)
    monkeypatch.setattr(pipeline, "build_context", lambda *a, **k: (context_for(ask_request), []))
    monkeypatch.setattr(pipeline, "retrieve_examples", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda question, context, examples, attempts: (
            previous_attempts.extend(attempts) or next(generated)
        ),
    )
    monkeypatch.setattr(pipeline, "validate", lambda *args: next(validations))
    monkeypatch.setattr(
        pipeline.db,
        "run_readonly_query",
        lambda *args: QueryResult(columns=["n"], rows=[["2"]], row_count=1),
    )

    response = pipeline.answer_question(ask_request, explain=False)

    assert response.status == "success"
    assert response.sql == "SELECT 2"
    assert [attempt.error for attempt in response.attempts] == ["column does not exist", None]
    assert response.metrics.attempt_count == 2


def test_answer_question_rejects_unsafe_sql_without_validation_or_execution(
    monkeypatch, ask_request, fake_config
):
    monkeypatch.setattr(pipeline, "get_config", lambda: fake_config)
    monkeypatch.setattr(pipeline, "build_context", lambda *a, **k: (context_for(ask_request), []))
    monkeypatch.setattr(pipeline, "retrieve_examples", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *args: GenerationResult(sql="DELETE FROM school", raw_output="DELETE FROM school"),
    )
    monkeypatch.setattr(pipeline, "validate", lambda *args: pytest.fail("must reject before validate"))
    monkeypatch.setattr(
        pipeline.db,
        "run_readonly_query",
        lambda *args: pytest.fail("must reject before execution"),
    )

    response = pipeline.answer_question(ask_request, explain=False)

    assert response.status == "unsafe_sql"
    assert response.attempts[0].error
    assert response.result is None


def test_answer_question_returns_no_schema_without_calling_model(monkeypatch, ask_request, fake_config):
    empty_context = SchemaContext(db_id=ask_request.db_id, mode=ask_request.mode, tables=[], ddl="")
    monkeypatch.setattr(pipeline, "get_config", lambda: fake_config)
    monkeypatch.setattr(pipeline, "build_context", lambda *a, **k: (empty_context, []))
    monkeypatch.setattr(pipeline, "retrieve_examples", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *args: pytest.fail("must not call the model without schema"),
    )

    response = pipeline.answer_question(ask_request)

    assert response.status == "no_schema"
    assert response.sql is None
    assert response.attempts == []


def test_answer_question_retries_database_execution_errors(monkeypatch, ask_request, fake_config):
    generated = iter(["SELECT 1", "SELECT 2"])
    execution_count = 0
    monkeypatch.setattr(pipeline, "get_config", lambda: fake_config)
    monkeypatch.setattr(pipeline, "build_context", lambda *a, **k: (context_for(ask_request), []))
    monkeypatch.setattr(pipeline, "retrieve_examples", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda *args: GenerationResult(sql=next(generated), raw_output="query"),
    )
    monkeypatch.setattr(pipeline, "validate", lambda *args: ExplainResult(ok=True))

    def execute(*args):
        nonlocal execution_count
        execution_count += 1
        if execution_count == 1:
            raise psycopg.errors.UndefinedColumn("unknown column")
        return QueryResult(columns=["n"], rows=[["1"]], row_count=1)

    monkeypatch.setattr(pipeline.db, "run_readonly_query", execute)

    response = pipeline.answer_question(ask_request, explain=False)

    assert response.status == "success"
    assert [attempt.error is not None for attempt in response.attempts] == [True, False]
