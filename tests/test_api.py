"""The HTTP layer with every backend faked: routes, status codes (SPEC 3.5), error mapping, CORS, no secrets."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")  # the `api` extra; without it these tests are skipped, not errors

import neo4j.exceptions
import psycopg
from fastapi.testclient import TestClient

from schemagraph.api import main
from schemagraph.graph import queries
from schemagraph.graph.vector import EmbeddingError, RateLimited
from schemagraph.models import (
    AskResponse,
    ColumnMeta,
    JoinPlan,
    SchemaContext,
    SeedTable,
    TableMeta,
    TableSummary,
    TableWithRelationships,
)
from schemagraph.pipeline.llm import LLMError

EXAMPLES = Path(__file__).resolve().parent.parent / "contracts" / "examples"
SAMPLE = json.loads((EXAMPLES / "ask_response.sample.json").read_text(encoding="utf-8"))
PLAN = JoinPlan.model_validate(json.loads((EXAMPLES / "join_plan.sample.json").read_text(encoding="utf-8")))
DB = "adventureworks"
BODY = {"question": "total order value per customer", "db_id": DB, "mode": "graph"}
SECRET = "sk-ant-SECRET-key http://10.0.0.5:7687 password=hunter2"


@pytest.fixture
def api(monkeypatch):
    """A client with the backends replaced; each test overrides what it cares about."""
    state = SimpleNamespace(asked=[], closed=0)
    monkeypatch.setattr(main.db, "list_databases", lambda: [DB, "bird_x"])
    monkeypatch.setattr(main, "get_config", lambda: SimpleNamespace(
        pipeline=SimpleNamespace(top_k=5, expand_seeds=0, expand_hops=1), llm=SimpleNamespace(api_key="k")))

    def answer(request, explain=True):
        state.asked.append(request)
        return AskResponse.model_validate({**SAMPLE, "question": request.question})

    monkeypatch.setattr(main, "answer_question", answer)
    monkeypatch.setattr(main.driver, "close_driver", lambda: setattr(state, "closed", state.closed + 1))
    with TestClient(main.app, raise_server_exceptions=False) as client:
        client.state = state
        yield client


# ------------------------------------------------------------------ /api/ask


def test_ask_returns_the_pipeline_response_unchanged(api):
    r = api.post("/api/ask", json=BODY)
    assert r.status_code == 200
    assert r.json()["sql"] == SAMPLE["sql"] and r.json()["question"] == BODY["question"]
    assert r.json()["explanation"]["join_path"][0]["from"].startswith("sales.")   # aliased field names survive
    assert api.state.asked[0].mode == "graph" and api.state.asked[0].db_id == DB


def test_mode_defaults_to_graph_and_options_are_optional(api):
    assert api.post("/api/ask", json={"question": "q", "db_id": DB}).status_code == 200
    assert api.state.asked[0].mode == "graph" and api.state.asked[0].options is None


def test_a_failed_run_is_still_a_200_so_the_page_can_show_the_attempts(api, monkeypatch):
    failed = AskResponse.model_validate({**SAMPLE, "status": "invalid_sql", "sql": None, "result": None})
    monkeypatch.setattr(main, "answer_question", lambda request, explain=True: failed)
    r = api.post("/api/ask", json=BODY)
    assert r.status_code == 200 and r.json()["status"] == "invalid_sql" and r.json()["attempts"]


@pytest.mark.parametrize("body, fragment", [
    ({**BODY, "question": ""}, "question"),
    ({**BODY, "question": "   \n"}, "Please enter a question"),
    ({"db_id": DB}, "question"),
    ({**BODY, "mode": "turbo"}, "mode"),
    ({**BODY, "db_id": "nope"}, "Unknown db_id"),
    ({**BODY, "db_id": "x'; DROP TABLE y;--"}, "Unknown db_id"),
    ({**BODY, "options": {"top_k": "many"}}, "options"),
    ({**BODY, "surprise": 1}, "surprise"),
])
def test_bad_input_is_a_400_with_a_detail_never_a_422(api, body, fragment):
    r = api.post("/api/ask", json=body)
    assert r.status_code == 400 and fragment in r.json()["detail"]
    assert api.state.asked == []                         # nothing reached the pipeline


def test_the_pipelines_own_option_checks_are_400s(api, monkeypatch):
    def refuse(request, explain=True):
        raise ValueError("top_k must be at least 1")

    monkeypatch.setattr(main, "answer_question", refuse)
    r = api.post("/api/ask", json={**BODY, "options": {"top_k": 0}})
    assert r.status_code == 400 and r.json() == {"detail": "top_k must be at least 1"}


@pytest.mark.parametrize("exc, fragment", [
    (LLMError(SECRET), "language model"),
    (psycopg.OperationalError(SECRET), "PostgreSQL"),
    (neo4j.exceptions.ServiceUnavailable(SECRET), "Neo4j"),
    (EmbeddingError(SECRET), "embedding service failed"),
])
def test_unreachable_services_are_503_and_the_detail_leaks_nothing(api, monkeypatch, exc, fragment):
    def broken(request, explain=True):
        raise exc

    monkeypatch.setattr(main, "answer_question", broken)
    r = api.post("/api/ask", json=BODY)
    assert r.status_code == 503 and fragment in r.json()["detail"]
    assert "sk-ant" not in r.text and "10.0.0.5" not in r.text and "hunter2" not in r.text


def test_rate_limiting_is_a_503_with_retry_after(api, monkeypatch):
    def limited(request, explain=True):
        raise RateLimited("429 for a minute", retry_after=42.0)

    monkeypatch.setattr(main, "answer_question", limited)
    r = api.post("/api/ask", json=BODY)
    assert r.status_code == 503 and r.headers["retry-after"] == "42" and "rate-limited" in r.json()["detail"]


def test_a_bug_is_a_500_not_a_disguised_503(api, monkeypatch):
    def buggy(request, explain=True):
        raise KeyError("oops")

    monkeypatch.setattr(main, "answer_question", buggy)
    assert api.post("/api/ask", json=BODY).status_code == 500


def test_there_is_no_sample_or_guided_mode(api, monkeypatch):
    monkeypatch.setenv("TEXTSQL_MODE", "sample")
    api.post("/api/ask", json={**BODY, "question": "something nobody scripted"})
    assert [r.question for r in api.state.asked] == ["something nobody scripted"]   # always the real pipeline


# ------------------------------------------------------------------ CORS and docs


def test_cors_allows_the_react_dev_server_and_nobody_else(api):
    ask = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = api.options("/api/ask", headers={"Origin": "http://localhost:5173", **ask})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = api.options("/api/ask", headers={"Origin": "http://evil.example", **ask})
    assert "access-control-allow-origin" not in other.headers


def test_openapi_lists_exactly_the_spec_endpoints(api):
    paths = api.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/api/health", "/api/databases", "/api/ask", "/api/plan",
                          "/api/schema/{db_id}/tables", "/api/schema/{db_id}/tables/{table_uid}"}


def test_the_neo4j_driver_is_closed_when_the_app_stops(api):
    assert api.state.closed == 0
    api.__exit__(None, None, None)
    assert api.state.closed == 1


# ------------------------------------------------------------------ /api/plan


def test_plan_runs_retrieval_and_planning_without_the_llm(api, monkeypatch):
    seen = {}
    seeds = [SeedTable(uid=f"{DB}:sales.customer", score=0.84, matched_on="table")]

    def fake_context(db_id, question, mode, top_k, **kw):
        seen.update(db_id=db_id, mode=mode, top_k=top_k, kw=kw)
        return SchemaContext(db_id=db_id, mode=mode, tables=[s.uid for s in seeds], ddl="", join_plan=PLAN), seeds

    monkeypatch.setattr(main, "build_context", fake_context)
    monkeypatch.setattr(main, "answer_question", lambda *a, **k: pytest.fail("the LLM pipeline must not run"))
    r = api.post("/api/plan", json=BODY)
    assert r.status_code == 200
    assert r.json()["seed_tables"][0]["uid"] == f"{DB}:sales.customer" and r.json()["join_plan"]["status"] == PLAN.status
    assert seen["top_k"] == 5 and seen["mode"] == "graph"
    api.post("/api/plan", json={**BODY, "options": {"top_k": 9}})
    assert seen["top_k"] == 9


def test_plan_in_retrieval_mode_has_no_join_plan_and_validates_like_ask(api, monkeypatch):
    monkeypatch.setattr(main, "build_context", lambda db_id, q, mode, k, **kw: (
        SchemaContext(db_id=db_id, mode=mode, tables=[], ddl=""), []))
    r = api.post("/api/plan", json={**BODY, "mode": "retrieval"})
    assert r.json() == {"seed_tables": [], "join_plan": None}
    assert api.post("/api/plan", json={**BODY, "question": " "}).status_code == 400
    assert api.post("/api/plan", json={**BODY, "options": {"top_k": 0}}).status_code == 400


# ------------------------------------------------------------------ /api/databases and /api/schema


def table(name, schema="sales", columns=3):
    return TableSummary(uid=f"{DB}:{schema}.{name}", display_name=f"{schema}.{name}", column_count=columns)


def test_databases_lists_only_those_with_a_graph_and_counts_schemas(api, monkeypatch):
    graph = {DB: [table("customer"), table("order"), table("person", "person")], "bird_x": []}
    monkeypatch.setattr(main.queries, "list_tables", lambda db_id: graph[db_id])
    r = api.get("/api/databases")
    assert r.status_code == 200
    assert r.json() == [{"db_id": DB, "table_count": 3, "schema_count": 2}]


def test_schema_tables_and_one_table(api, monkeypatch):
    monkeypatch.setattr(main.queries, "list_tables", lambda db_id: [table("customer")])
    assert api.get(f"/api/schema/{DB}/tables").json()[0]["display_name"] == "sales.customer"

    meta = TableMeta(uid=f"{DB}:sales.customer", schema="sales", name="customer")
    col = ColumnMeta(uid=f"{DB}:sales.customer.id", table_uid=meta.uid, name="id", data_type="integer",
                     ordinal_position=1, is_nullable=False, is_primary_key=True)

    def detail(db_id, uid):
        if uid != meta.uid:
            raise queries.TableNotFound(f"table {uid!r} not found in the graph for {db_id!r}")
        return TableWithRelationships(table=meta, columns=[col])

    monkeypatch.setattr(main.queries, "get_table_detail", detail)
    ok = api.get(f"/api/schema/{DB}/tables/{DB}:sales.customer")
    assert ok.status_code == 200 and ok.json()["columns"][0]["name"] == "id" and ok.json()["referenced_by"] == []
    missing = api.get(f"/api/schema/{DB}/tables/{DB}:sales.nothing")
    assert missing.status_code == 404 and "not found" in missing.json()["detail"]
    assert api.get("/api/schema/nope/tables").status_code == 400


# ------------------------------------------------------------------ /api/health


def test_health_reports_each_dependency(api, monkeypatch):
    monkeypatch.setattr(main.driver, "check_connection", lambda: None)
    assert api.get("/api/health").json() == {"postgres": "ok", "neo4j": "ok", "llm": "ok"}

    def down():
        raise RuntimeError(SECRET)

    monkeypatch.setattr(main.driver, "check_connection", down)
    monkeypatch.setattr(main.db, "list_databases", down)
    r = api.get("/api/health")
    assert r.status_code == 200 and r.json() == {"postgres": "error", "neo4j": "error", "llm": "ok"}
    assert "hunter2" not in r.text


def test_health_llm_is_configuration_only_unless_deep(api, monkeypatch):
    monkeypatch.setattr(main.driver, "check_connection", lambda: None)
    calls = []
    monkeypatch.setattr(main.llm, "complete", lambda cfg, messages: calls.append(messages) or ("OK", 1, 1))
    assert api.get("/api/health").json()["llm"] == "ok" and calls == []
    assert api.get("/api/health?deep=true").json()["llm"] == "ok" and len(calls) == 1

    def refuse(cfg, messages):
        raise LLMError(SECRET)

    monkeypatch.setattr(main.llm, "complete", refuse)
    assert api.get("/api/health?deep=true").json()["llm"] == "error"


def test_health_says_when_the_llm_is_not_configured(api, monkeypatch):
    monkeypatch.setattr(main.driver, "check_connection", lambda: None)
    monkeypatch.setattr(main, "get_config", lambda: SimpleNamespace(
        pipeline=SimpleNamespace(top_k=5, expand_seeds=0, expand_hops=1), llm=None))
    assert api.get("/api/health").json()["llm"] == "not_configured"


def test_health_says_not_configured_when_the_key_variable_is_unset(api, monkeypatch):
    monkeypatch.setattr(main.driver, "check_connection", lambda: None)

    class NoKey:
        @property
        def api_key(self):
            raise RuntimeError("Environment variable ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(main, "get_config", lambda: SimpleNamespace(
        pipeline=SimpleNamespace(top_k=5, expand_seeds=0, expand_hops=1), llm=NoKey()))
    assert api.get("/api/health").json()["llm"] == "not_configured"


def test_health_survives_a_missing_config_file(api, monkeypatch):
    monkeypatch.setattr(main.driver, "check_connection", lambda: None)

    def no_config():
        raise FileNotFoundError("config.yml")

    monkeypatch.setattr(main, "get_config", no_config)
    r = api.get("/api/health")
    assert r.status_code == 200 and r.json()["llm"] == "error"
