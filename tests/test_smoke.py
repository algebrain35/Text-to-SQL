"""eval/smoke.py with the services replaced by fakes: question choice, per-mode rows, error capture,
gold comparison, output, and the command line's exit codes."""

from types import SimpleNamespace

import pytest

from schemagraph.eval import smoke
from schemagraph.models import (
    AskRequest,
    AskResponse,
    ExplainTable,
    Explanation,
    JoinPlan,
    LatencyMs,
    Metrics,
    QueryResult,
    Question,
    SchemaContext,
    SeedTable,
)

DB = "shop"
ORDERS, ITEMS, CUSTOMERS = f"{DB}:public.orders", f"{DB}:public.items", f"{DB}:public.customers"
GOLD = "SELECT count(*) FROM orders JOIN items ON items.order_id = orders.id"
ONE = QueryResult(columns=["n"], rows=[["1"]], row_count=1)
TWO = QueryResult(columns=["n"], rows=[["2"]], row_count=1)


def question(qid="q1", **kw):
    base = {"id": qid, "db_id": DB, "question": "how many items per order", "gold_sql": GOLD, "tier": 2, "split": "dev"}
    return Question(**(base | kw))


def seeds():
    return [SeedTable(uid=ORDERS, score=0.91, matched_on="table"), SeedTable(uid=ITEMS, score=0.5, matched_on="column:order_id")]


def plan(**kw):
    base = {"db_id": DB, "status": "full", "algorithm": "networkx_steiner", "seed_tables": seeds(),
            "bridge_tables": [CUSTOMERS], "tables": [ORDERS, ITEMS, CUSTOMERS]}
    return JoinPlan(**(base | kw))


def fake_context(db_id, text, mode, top_k):
    tables = {"full": [ORDERS, ITEMS, CUSTOMERS], "retrieval": [ORDERS, ITEMS], "graph": [ORDERS, ITEMS, CUSTOMERS]}[mode]
    ctx = SchemaContext(db_id=db_id, mode=mode, tables=tables, ddl="...", join_plan=plan() if mode == "graph" else None)
    return ctx, ([] if mode == "full" else seeds())


def response(mode, status="success", result=ONE, sql="SELECT 1", tables=None, with_plan=None):
    tables = tables if tables is not None else [ORDERS, ITEMS]
    roles = [ExplainTable(uid=u, display_name=u.split(":")[1], role="seed" if u in (ORDERS, ITEMS) else "context", reason="r") for u in tables]
    join_plan = plan() if (with_plan if with_plan is not None else mode == "graph") else None
    return AskResponse(
        request_id="r", question="q", db_id=DB, mode=mode, status=status, sql=sql,
        result=result if status == "success" else None,
        explanation=Explanation(summary="", tables=roles), join_plan=join_plan,
        metrics=Metrics(input_tokens=30, output_tokens=12, attempt_count=2, latency_ms=LatencyMs(total=500)),
    )


def answer_with(result=ONE, status="success"):
    def answer(request: AskRequest, explain=True):
        assert explain is False
        return response(request.mode, status=status, result=result)

    return answer


def never(*a, **k):
    pytest.fail("must not be called")


# ----------------------------------------------------------------- question choice


def test_pick_questions_precedence_and_filters():
    file_qs = [question("a"), question("b", evidence="a hint"), question("c", db_id="other"), question("d")]
    assert [q.text for q in smoke.pick_questions(DB, ["mine"], file_qs, 5)] == ["mine"]
    chosen = smoke.pick_questions(DB, [], file_qs, 2)
    assert [q.qid for q in chosen] == ["a", "b"]                         # only this db, first n
    assert chosen[1].text.endswith("\nHint: a hint") and chosen[0].gold_sql == GOLD
    assert len(smoke.pick_questions("adventureworks", [], [], 3)) == 3     # built-in list
    assert smoke.pick_questions("unknown_db", [], [], 3) == []


def test_plan_summary():
    assert smoke.plan_summary(None) is None
    text = smoke.plan_summary(plan(unreachable_seeds=[ITEMS], dropped_seeds=[ORDERS]))
    assert "full via networkx_steiner" in text and "bridges: public.customers" in text
    assert "unreachable: public.items" in text and "dropped: public.orders" in text


# ----------------------------------------------------------------- one question, one mode


def test_without_llm_it_shows_retrieval_and_planning_and_scores_recall():
    q = smoke.SmokeQuestion("how many", "q1", GOLD)
    row = smoke.run_one(DB, q, "graph", use_llm=False, top_k=5, build_context=fake_context, answer=never)
    assert row.ok and row.status is None and row.sql is None
    assert row.tables == ["public.customers", "public.items", "public.orders"]
    assert row.seeds[0] == ("public.orders", 0.91, "table") and "bridges" in row.plan
    assert row.recall == 1.0                                              # both gold tables were given
    retrieval = smoke.run_one(DB, q, "retrieval", use_llm=False, top_k=5, build_context=fake_context, answer=never)
    assert retrieval.plan is None
    narrow = smoke.run_one(DB, smoke.SmokeQuestion("x", gold_sql="SELECT * FROM public.customers"), "retrieval",
                           use_llm=False, top_k=5, build_context=fake_context, answer=never)
    assert narrow.recall == 0.0


def test_with_llm_it_runs_the_pipeline_without_explanations_and_grades_against_gold():
    q = smoke.SmokeQuestion("how many", "q1", GOLD)
    good = smoke.run_one(DB, q, "retrieval", use_llm=True, top_k=5, build_context=never, answer=answer_with(), gold=ONE)
    assert good.ok and good.status == "success" and good.correct is True and good.rows == 1
    assert (good.tokens, good.attempts) == (42, 2) and good.recall == 1.0 and good.tables == ["public.items", "public.orders"]
    assert good.seeds == [("public.orders", None, ""), ("public.items", None, "")]   # retrieval mode: names only
    bad = smoke.run_one(DB, q, "retrieval", use_llm=True, top_k=5, build_context=never, answer=answer_with(TWO), gold=ONE)
    assert bad.correct is False
    failed = smoke.run_one(DB, q, "retrieval", use_llm=True, top_k=5, build_context=never,
                           answer=answer_with(status="invalid_sql"), gold=ONE)
    assert failed.status == "invalid_sql" and failed.correct is False and failed.rows is None
    nogold = smoke.run_one(DB, smoke.SmokeQuestion("free text"), "graph", use_llm=True, top_k=5, build_context=never,
                           answer=answer_with())
    assert nogold.correct is None and nogold.recall is None and nogold.seeds[0][1] == 0.91   # graph mode has scores


def test_a_failing_stage_becomes_an_error_on_that_row():
    q = smoke.SmokeQuestion("x")

    def boom(*a, **k):
        raise RuntimeError("neo4j went away")

    row = smoke.run_one(DB, q, "graph", use_llm=False, top_k=5, build_context=boom, answer=never)
    assert not row.ok and row.error == "context: RuntimeError: neo4j went away"
    row = smoke.run_one(DB, q, "graph", use_llm=True, top_k=5, build_context=never, answer=boom)
    assert row.error == "pipeline: RuntimeError: neo4j went away"


# ----------------------------------------------------------------- all questions and modes


def test_gold_runs_once_per_question_and_only_with_an_llm():
    calls = []

    def run_query(db_id, sql, limit):
        calls.append(sql)
        return ONE

    qs = [smoke.SmokeQuestion("a", "q1", GOLD), smoke.SmokeQuestion("b")]
    rows = smoke.run_smoke(DB, qs, ["full", "retrieval", "graph"], use_llm=True, top_k=5, build_context=never,
                           answer=answer_with(), run_query=run_query)
    assert len(rows) == 6 and calls == [GOLD]
    assert [r.correct for r in rows[:3]] == [True, True, True] and rows[3].correct is None
    calls.clear()
    smoke.run_smoke(DB, qs, ["graph"], use_llm=False, top_k=5, build_context=fake_context, answer=never, run_query=run_query)
    assert calls == []


def test_a_broken_gold_query_is_reported_and_the_question_still_runs():
    def run_query(*a):
        raise RuntimeError("relation does not exist")

    rows = smoke.run_smoke(DB, [smoke.SmokeQuestion("a", "q1", GOLD)], ["graph"], use_llm=True, top_k=5,
                           build_context=never, answer=answer_with(), run_query=run_query)
    assert rows[0].error.startswith("gold SQL failed") and rows[1].ok and rows[1].correct is None


# ----------------------------------------------------------------- output


def sample_rows():
    qs = [smoke.SmokeQuestion("how many", "q1", GOLD)]
    return smoke.run_smoke(DB, qs, ["full", "retrieval", "graph"], use_llm=True, top_k=5, build_context=never,
                           answer=answer_with(), run_query=lambda *a: ONE)


def test_format_shows_each_mode_and_a_summary():
    text = smoke.format_rows(sample_rows(), use_llm=True)
    assert "Q q1: how many" in text and "[retrieval]" in text and "[graph    ]" in text
    assert "MATCHES gold" in text and "gold-table recall 100%" in text and "42 tokens" in text
    assert "plan: full via networkx_steiner" in text and "retrieved: public.orders (0.910)" in text
    assert "graph     1 runs, 0 errors, mean gold-table recall 100%, SQL ran 1/1, matches gold 1/1" in text


def test_full_sql_prints_every_line_unclipped():
    sql = "SELECT a,\n       b\nFROM " + "very_long_table_name_" * 12
    rows = [smoke.Row("q", "graph", status="success", sql=sql)]
    clipped = smoke.format_rows(rows, use_llm=True)
    assert "…" in clipped and "very_long_table_name_" * 12 not in clipped
    full = smoke.format_rows(rows, use_llm=True, full_sql=True)
    assert "        SELECT a," in full and "               b" in full and "very_long_table_name_" * 12 in full


def test_cli_full_sql_flag(cli, capsys, monkeypatch):
    from schemagraph.pipeline import pipeline

    long_sql = "SELECT 1 FROM " + "t" * 200
    monkeypatch.setattr(pipeline, "answer_question", lambda request, explain=True: response(request.mode, sql=long_sql))
    assert smoke.main(["--db", DB, "--question", "q", "--mode", "graph"]) == 0
    assert "t" * 200 not in capsys.readouterr().out
    assert smoke.main(["--db", DB, "--question", "q", "--mode", "graph", "--full-sql"]) == 0
    assert "t" * 200 in capsys.readouterr().out


def test_format_marks_errors_and_long_text_is_clipped():
    rows = [smoke.Row("q", "graph", error="context: RuntimeError: x"), smoke.Row("w " * 300, "full", status="success")]
    text = smoke.format_rows(rows, use_llm=True)
    assert "ERROR context: RuntimeError: x" in text and "1 errors" in text
    assert max(len(line) for line in text.splitlines()) < 250


# ----------------------------------------------------------------- command line


@pytest.fixture
def cli(monkeypatch):
    from schemagraph import db
    from schemagraph.config import (
        get_config,  # noqa: F401 - imported so the attribute exists to patch
    )
    from schemagraph.graph import driver
    from schemagraph.pipeline import context, pipeline

    state = SimpleNamespace(llm=object(), closed=0, conn_error=None)
    monkeypatch.setattr("schemagraph.config.get_config", lambda: SimpleNamespace(llm=state.llm))

    def check():
        if state.conn_error:
            raise ConnectionError(state.conn_error)

    monkeypatch.setattr(driver, "check_connection", check)
    monkeypatch.setattr(driver, "close_driver", lambda: setattr(state, "closed", state.closed + 1))
    monkeypatch.setattr(context, "build_context", fake_context)
    monkeypatch.setattr(pipeline, "answer_question", answer_with())
    monkeypatch.setattr(db, "run_readonly_query", lambda *a: ONE)
    return state


def test_cli_runs_and_prints_a_report(cli, capsys):
    assert smoke.main(["--db", DB, "--question", "how many orders", "--mode", "graph"]) == 0
    out = capsys.readouterr().out
    assert "[graph    ]" in out and "[retrieval]" not in out and "SQL ran 1/1" in out and cli.closed == 1


def test_cli_without_an_llm_section_notes_it_and_skips_the_pipeline(cli, capsys, monkeypatch):
    from schemagraph.pipeline import pipeline

    monkeypatch.setattr(pipeline, "answer_question", never)
    cli.llm = None
    assert smoke.main(["--db", DB, "--question", "q"]) == 0
    out = capsys.readouterr().out
    assert "no `llm:` section" in out and "status" not in out
    cli.llm = object()
    assert smoke.main(["--db", DB, "--question", "q", "--no-llm"]) == 0


def test_cli_reads_a_questions_file_with_gold(cli, capsys, tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(question(f"q{i}").model_dump_json() for i in range(4)))
    assert smoke.main(["--db", DB, "--questions-file", str(path), "--n", "2", "--mode", "retrieval"]) == 0
    out = capsys.readouterr().out
    assert "Q q0" in out and "Q q1" in out and "Q q2" not in out and "MATCHES gold" in out


def test_cli_exit_codes(cli, capsys, monkeypatch):
    assert smoke.main(["--db", "no_such_db"]) == 2
    assert "no questions" in capsys.readouterr().out
    cli.conn_error = "neo4j down"
    assert smoke.main(["--db", DB, "--question", "q"]) == 2
    assert "cannot start: ConnectionError: neo4j down" in capsys.readouterr().out and cli.closed == 2
    cli.conn_error = None
    from schemagraph.pipeline import pipeline

    def broken(*a, **k):
        raise RuntimeError("llm 401")

    monkeypatch.setattr(pipeline, "answer_question", broken)
    assert smoke.main(["--db", DB, "--question", "q", "--mode", "graph"]) == 1
    assert "pipeline: RuntimeError: llm 401" in capsys.readouterr().out
