"""run_eval with a fake pipeline and fake gold queries: no database, no model."""

import json
from datetime import datetime, timezone

import pytest

from schemagraph.eval import run_eval as ev
from schemagraph.models import (
    AskResponse,
    Explanation,
    ExplainTable,
    JoinPlan,
    LatencyMs,
    Metrics,
    QueryResult,
    SeedTable,
)

DB = "adventureworks"
HEADER, CUSTOMER, PERSON = (f"{DB}:sales.salesorderheader", f"{DB}:sales.customer", f"{DB}:person.person")
GOLD_SQL = (
    "SELECT p.firstname, SUM(h.totaldue) FROM sales.salesorderheader h "
    "JOIN sales.customer c ON h.customerid = c.customerid "
    "JOIN person.person p ON c.personid = p.businessentityid GROUP BY p.firstname"
)
NOW = datetime(2026, 11, 12, 19, 0, tzinfo=timezone.utc)


def qr(cols, rows, truncated=False):
    return QueryResult(columns=cols, rows=rows, row_count=len(rows), truncated=truncated)


GOLD = qr(["firstname", "total"], [["Jon", "8248.99"], ["Eugene", "6383.66"]])


def question(**over):
    row = {"id": "aw-001", "db_id": DB, "question": "total per person", "gold_sql": GOLD_SQL, "tier": 3,
           "split": "test", "order_matters": False, "evidence": None, "author": "t", "notes": ""}
    row.update(over)
    return row


def write_questions(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def graph_response(mode="graph", result=GOLD, status="success", tables=(HEADER, CUSTOMER, PERSON),
                   seeds=(HEADER, PERSON), plan_status="full", attempts=1, tokens=1800):
    plan = JoinPlan(
        db_id=DB, status=plan_status, algorithm="shortest_path",
        seed_tables=[SeedTable(uid=u, score=0.8, matched_on="table") for u in seeds],
        bridge_tables=[t for t in tables if t not in seeds], tables=list(tables),
    )
    return AskResponse(
        request_id="r", question="q", db_id=DB, mode=mode, status=status, sql="SELECT ...", result=result,
        join_plan=plan if mode == "graph" else None,
        metrics=Metrics(input_tokens=tokens, attempt_count=attempts, latency_ms=LatencyMs(total=900)),
    )


def run(tmp_path, answer, rows=None, modes=("graph",), **kw):
    qfile = write_questions(tmp_path / "q.jsonl", rows or [question()])
    out = tmp_path / "out" / "r.jsonl"
    report = ev.run_eval(
        [qfile], list(modes), out, "run-a", llm_model="m", answer=answer,
        run_query=kw.pop("run_query", lambda db, sql, n: GOLD), now=lambda: NOW, **kw,
    )
    return report, out


def read(out):
    return [json.loads(line) for line in out.read_text().splitlines()]


# ----------------------------------------------------------------- loading


def test_load_questions_filters_split_and_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(question()) + "\n\n" + json.dumps(question(id="aw-002", split="dev")) + "\n")
    assert [q.id for q in ev.load_questions(path)] == ["aw-001"]
    assert [q.id for q in ev.load_questions(path, "dev")] == ["aw-002"]
    assert [q.id for q in ev.load_questions(path, "all")] == ["aw-001", "aw-002"]
    with pytest.raises(ValueError, match="split"):
        ev.load_questions(path, "train")


def test_load_questions_names_the_bad_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(question()) + "\n" + json.dumps(question(tier=7)) + "\n")
    with pytest.raises(ValueError, match=r"q\.jsonl:2"):
        ev.load_questions(path)


def test_duplicate_ids_across_files_are_rejected(tmp_path):
    a = write_questions(tmp_path / "a.jsonl", [question()])
    b = write_questions(tmp_path / "b.jsonl", [question()])
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_question_files([a, b])


# ----------------------------------------------------------------- gold


def test_check_gold_reports_errors_truncation_and_empty(tmp_path):
    qs = ev.load_questions(write_questions(tmp_path / "q.jsonl", [
        question(id="ok"), question(id="boom"), question(id="big"), question(id="none")]))

    answers = {"ok": GOLD, "big": qr(["a"], [["1"]], truncated=True), "none": qr(["a"], [])}
    calls = iter(["ok", "boom", "big", "none"])

    def rq(db, sql, n):
        key = next(calls)
        if key == "boom":
            raise RuntimeError("relation does not exist")
        return answers[key]

    check = ev.check_gold(qs, rq)
    assert set(check.results) == {"ok", "none"}
    assert "relation does not exist" in check.errors["boom"]
    assert "more than" in check.errors["big"]
    assert check.empty == ["none"]


def test_bad_gold_stops_the_run_before_any_model_call(tmp_path):
    def answer(*a, **k):
        pytest.fail("the model must not be called when gold is broken")

    def broken(db, sql, n):
        raise RuntimeError("nope")

    with pytest.raises(ev.GoldError, match="aw-001"):
        run(tmp_path, answer, run_query=broken)


def test_skip_bad_gold_drops_only_those_questions(tmp_path):
    rows = [question(id="good"), question(id="bad", gold_sql="SELECT 1 FROM missing")]

    def run_query(db, sql, n):
        if "missing" in sql:
            raise RuntimeError("no such table")
        return GOLD

    report, out = run(tmp_path, lambda req, **k: graph_response(), rows=rows, skip_bad_gold=True, run_query=run_query)
    assert [r["question_id"] for r in read(out)] == ["good"]
    assert list(report.skipped_bad_gold) == ["bad"]


# ----------------------------------------------------------------- scoring


def test_correct_graph_answer_row(tmp_path):
    report, out = run(tmp_path, lambda req, **k: graph_response())
    (row,) = read(out)
    assert report.written == 1
    assert row["correct"] is True and row["status"] == "success" and row["mode"] == "graph"
    assert row["gold_tables"] == ["person.person", "sales.customer", "sales.salesorderheader"]
    assert row["context_tables"] == row["gold_tables"]
    assert row["schema_recall"] == 1.0
    assert row["bridge_tables_needed"] == ["sales.customer"]
    assert row["bridge_tables_recovered"] == ["sales.customer"]
    assert (row["plan_status"], row["input_tokens"], row["latency_ms"], row["attempt_count"]) == ("full", 1800, 900, 1)
    assert (row["llm_model"], row["run_id"], row["tier"]) == ("m", "run-a", 3)


def test_wrong_rows_and_failed_statuses_are_incorrect(tmp_path):
    wrong = qr(["firstname", "total"], [["Jon", "1.00"]])
    _, out = run(tmp_path, lambda req, **k: graph_response(result=wrong))
    assert read(out)[0]["correct"] is False
    for status in ("invalid_sql", "execution_error", "unsafe_sql", "no_schema"):
        out.unlink()
        _, out = run(tmp_path, lambda req, s=status, **k: graph_response(status=s, result=None))
        assert read(out)[0]["correct"] is False and read(out)[0]["status"] == status


def test_status_success_without_result_is_incorrect(tmp_path):
    _, out = run(tmp_path, lambda req, **k: graph_response(result=None))
    assert read(out)[0]["correct"] is False


def test_missing_bridge_is_not_recovered(tmp_path):
    resp = graph_response(tables=(HEADER, PERSON), seeds=(HEADER, PERSON), plan_status="partial")
    _, out = run(tmp_path, lambda req, **k: resp)
    row = read(out)[0]
    assert row["bridge_tables_needed"] == ["sales.customer"]
    assert row["bridge_tables_recovered"] == []
    assert row["schema_recall"] == pytest.approx(2 / 3)
    assert row["plan_status"] == "partial"


def test_retrieval_mode_uses_explanation_tables_and_cannot_recover_bridges(tmp_path):
    resp = AskResponse(
        request_id="r", question="q", db_id=DB, mode="retrieval", status="success", sql="SELECT", result=GOLD,
        explanation=Explanation(summary="", tables=[
            ExplainTable(uid=HEADER, display_name="sales.salesorderheader", role="context", reason=""),
            ExplainTable(uid=PERSON, display_name="person.person", role="context", reason="")]),
    )
    _, out = run(tmp_path, lambda req, **k: resp, modes=("retrieval",))
    row = read(out)[0]
    assert row["context_tables"] == ["person.person", "sales.salesorderheader"]
    assert row["bridge_tables_needed"] == ["sales.customer"]   # seeds are the context, so the bridge was never offered
    assert row["bridge_tables_recovered"] == []
    assert row["plan_status"] is None


def test_full_mode_leaves_bridge_metrics_empty(tmp_path):
    resp = AskResponse(request_id="r", question="q", db_id=DB, mode="full", status="success", sql="SELECT", result=GOLD)
    _, out = run(tmp_path, lambda req, **k: resp, modes=("full",))
    row = read(out)[0]
    assert row["context_tables"] == [] and row["bridge_tables_needed"] == [] and row["bridge_tables_recovered"] == []


def test_order_matters_is_passed_to_the_comparison(tmp_path):
    reordered = qr(["firstname", "total"], [["Eugene", "6383.66"], ["Jon", "8248.99"]])
    _, out = run(tmp_path, lambda req, **k: graph_response(result=reordered))
    assert read(out)[0]["correct"] is True
    out.unlink()
    _, out = run(tmp_path, lambda req, **k: graph_response(result=reordered), rows=[question(order_matters=True)])
    assert read(out)[0]["correct"] is False


# ----------------------------------------------------------------- request building


def test_request_shape_and_flags(tmp_path):
    seen = []

    def answer(request, **kwargs):
        seen.append((request, kwargs))
        return graph_response(mode=request.mode)

    run(tmp_path, answer, rows=[question(evidence="firstname is the given name")], modes=("full", "graph"),
        planner_options={"weighted": False})
    (r1, k1), (r2, k2) = seen
    assert (r1.mode, r2.mode) == ("full", "graph")
    assert r1.options.row_limit == ev.GOLD_ROW_LIMIT
    assert r1.question == "total per person\nHint: firstname is the given name"
    assert k1 == {"explain": False, "planner_options": {"weighted": False}, "exclude_example_ids": frozenset({"aw-001"})}


def test_evidence_can_be_left_out_and_planner_options_omitted_when_empty(tmp_path):
    seen = []

    def answer(request, **kwargs):
        seen.append((request, kwargs))
        return graph_response()

    run(tmp_path, answer, rows=[question(evidence="hint")], use_evidence=False)
    assert seen[0][0].question == "total per person" and seen[0][1] == {"explain": False, "exclude_example_ids": frozenset({"aw-001"})}


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        run(tmp_path, lambda req, **k: graph_response(), modes=("bogus",))


# ----------------------------------------------------------------- resume and errors


def test_resume_skips_finished_rows_and_retries_errors(tmp_path):
    rows = [question(id="a"), question(id="b")]
    calls = []

    def flaky(request, **k):
        calls.append(request.question)
        if len(calls) == 2:
            raise RuntimeError("503 from the model")
        return graph_response()

    report, out = run(tmp_path, flaky, rows=rows)
    assert report.written == 1 and [e[:2] for e in report.errors] == [("b", "graph")]
    assert [r["question_id"] for r in read(out)] == ["a"]

    calls.clear()
    report2, out = run(tmp_path, lambda req, **k: graph_response(), rows=rows)
    assert report2.already_done == 1 and report2.written == 1 and not report2.errors
    assert [r["question_id"] for r in read(out)] == ["a", "b"]


def test_other_run_ids_are_not_resumed(tmp_path):
    qfile = write_questions(tmp_path / "q.jsonl", [question()])
    out = tmp_path / "r.jsonl"
    for run_id in ("one", "two"):
        ev.run_eval([qfile], ["graph"], out, run_id, answer=lambda req, **k: graph_response(),
                    run_query=lambda d, s, n: GOLD, now=lambda: NOW)
    assert [r["run_id"] for r in read(out)] == ["one", "two"]


# ----------------------------------------------------------------- summary


def _results(tmp_path):
    rows = [question(id=f"q{i}", tier=1 + i % 2) for i in range(4)]
    wrong = qr(["firstname", "total"], [["x", "1"]])
    flip = iter([GOLD, wrong, GOLD, GOLD])
    _, out = run(tmp_path, lambda req, **k: graph_response(result=next(flip)), rows=rows)
    return out


def test_aggregate_by_mode_and_tier(tmp_path):
    rows = {(r["mode"], r["tier"]): r for r in ev.aggregate(ev.load_results(_results(tmp_path)))}
    assert set(rows) == {("graph", "1"), ("graph", "2"), ("graph", "all")}
    assert rows[("graph", "all")]["n"] == 4 and rows[("graph", "all")]["accuracy"] == 0.75
    assert rows[("graph", "2")]["accuracy"] == 0.5          # q1 is wrong
    assert rows[("graph", "all")]["bridge_recovery"] == 1.0 and rows[("graph", "all")]["bridge_needed"] == 4
    assert rows[("graph", "all")]["input_tokens"] == 1800


def test_summarize_dataframe_and_text(tmp_path):
    pd = pytest.importorskip("pandas")
    out = _results(tmp_path)
    df = ev.summarize(out)
    assert isinstance(df, pd.DataFrame) and list(df["tier"]) == ["1", "2", "all"]
    text = ev.format_summary(ev.aggregate(ev.load_results(out)))
    assert "graph" in text and "75%" in text


def test_summary_of_empty_results_is_empty(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("")
    assert ev.aggregate(ev.load_results(p)) == []


def test_a_matching_result_does_not_rescue_a_failed_status(tmp_path):
    # defensive: only status "success" counts, even if a result happens to be attached
    _, out = run(tmp_path, lambda req, **k: graph_response(status="execution_error", result=GOLD))
    assert read(out)[0]["correct"] is False


# ----------------------------------------------------------------- command line


def test_cli_summarize_and_check_gold(tmp_path, monkeypatch, capsys):
    out = _results(tmp_path)
    assert ev.main(["--summarize", str(out)]) == 0
    assert "graph" in capsys.readouterr().out

    qfile = write_questions(tmp_path / "cli.jsonl", [question(), question(id="aw-002")])
    monkeypatch.setattr(ev, "_default_query", lambda db, sql, n: GOLD)
    assert ev.main(["--questions", str(qfile), "--check-gold"]) == 0
    assert "2 of 2 gold queries usable" in capsys.readouterr().out

    def broken(db, sql, n):
        raise RuntimeError("nope")

    monkeypatch.setattr(ev, "_default_query", broken)
    assert ev.main(["--questions", str(qfile), "--check-gold"]) == 1
    assert "ERROR aw-001" in capsys.readouterr().out


# ----------------------------------------------------------------- sweeps


def test_parse_variant_forms_and_errors():
    assert ev.parse_variant("5") == ev.Variant(5, 0, 1)
    assert ev.parse_variant("8+4") == ev.Variant(8, 4, 1)
    assert ev.parse_variant("8+4:2") == ev.Variant(8, 4, 2)
    assert [ev.parse_variant(s).label for s in ("5", "8+4", "8+4:2")] == ["k5", "k8+4h1", "k8+4h2"]
    assert ev.Variant(5).expand == {"max_added": 0, "hops": 1}
    for bad in ("", "x", "0", "5+", "5+2:0", "5+-1", "5+a"):
        with pytest.raises(ValueError, match="bad variant"):
            ev.parse_variant(bad)


def test_run_eval_passes_top_k_and_expansion_to_the_pipeline_only_when_given(tmp_path):
    seen = []

    def answer(request, explain=False, **kw):
        seen.append((request.options.top_k, kw))
        return graph_response()

    run(tmp_path, answer)
    assert seen[-1] == (None, {"exclude_example_ids": frozenset({"aw-001"})})
    run(tmp_path, answer, top_k=8, expand={"max_added": 4, "hops": 2}, rows=[question(id="aw-002")])
    assert seen[-1] == (8, {"exclude_example_ids": frozenset({"aw-002"}), "expand": {"max_added": 4, "hops": 2}})


def test_sweep_runs_every_variant_under_its_own_run_id_and_resumes(tmp_path):
    calls = []

    def answer(request, explain=False, **kw):
        calls.append((request.options.top_k, kw.get("expand")))
        # a bigger context for the bigger variant, and a wrong answer for the smallest
        tables = (HEADER, CUSTOMER, PERSON) if request.options.top_k > 3 else (HEADER,)
        wrong = qr(["x"], [["0"]])
        return graph_response(tables=tables, seeds=tables[:1], result=GOLD if request.options.top_k > 3 else wrong)

    qfile = write_questions(tmp_path / "q.jsonl", [question()])
    out = tmp_path / "sweep.jsonl"
    variants = [ev.parse_variant(s) for s in ("3", "5", "5+2:2")]
    kw = {"llm_model": "m", "answer": answer, "run_query": lambda db, sql, n: GOLD, "now": lambda: NOW}

    reports = ev.run_sweep(variants, [qfile], ["graph"], out, "sw", **kw)
    assert [r.written for _, r in reports] == [1, 1, 1]
    assert calls == [(3, {"max_added": 0, "hops": 1}), (5, {"max_added": 0, "hops": 1}), (5, {"max_added": 2, "hops": 2})]
    assert {r["run_id"] for r in read(out)} == {"sw-k3", "sw-k5", "sw-k5+2h2"}

    calls.clear()
    again = ev.run_sweep(variants, [qfile], ["graph"], out, "sw", **kw)
    assert calls == [] and [r.already_done for _, r in again] == [1, 1, 1]  # nothing is asked twice

    rows = {r["run_id"]: r for r in ev.aggregate_runs(ev.load_results(out))}
    assert rows["sw-k3"]["accuracy"] == 0.0 and rows["sw-k5"]["accuracy"] == 1.0
    assert rows["sw-k3"]["context_tables"] == 1 and rows["sw-k5"]["context_tables"] == 3
    text = ev.format_runs(list(rows.values()))
    assert "sw-k5+2h2" in text and "accuracy" in text and "100.0%" in text


def test_cli_compare_runs_prints_one_row_per_run(tmp_path, capsys):
    _, out = run(tmp_path, lambda request, explain=False, **kw: graph_response())
    assert ev.main(["--compare-runs", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "run-a" in printed and "graph" in printed


def test_each_question_is_asked_with_its_own_id_excluded_from_the_few_shot_examples(tmp_path):
    """A dev question is also a few-shot example; without this the model would be shown its own answer."""
    excluded = []

    def answer(request, explain=False, **kw):
        excluded.append(kw["exclude_example_ids"])
        return graph_response()

    run(tmp_path, answer, rows=[question(id="aw-dev-01", split="dev"), question(id="aw-dev-02", split="dev")], split="dev")
    assert excluded == [frozenset({"aw-dev-01"}), frozenset({"aw-dev-02"})]


# ----------------------------------------------------------------- embedding quota


def quota_error():
    from schemagraph.graph.vector import RateLimited

    return RateLimited("rate limited (HTTP 429)", retry_after=3600.0)


def test_run_stops_after_three_quota_refusals_in_a_row_and_keeps_finished_rows(tmp_path):
    rows = [question(id=f"q{i}") for i in range(8)]
    calls = []

    def answer(request, **k):
        calls.append(request.question)
        if len(calls) > 2:
            raise quota_error()
        return graph_response()

    report, out = run(tmp_path, answer, rows=rows)
    assert report.written == 2 and len(report.errors) == 3 and "quota" in report.stopped
    assert len(calls) == 5                                      # not 8: the rest were not attempted
    assert [r["question_id"] for r in read(out)] == ["q0", "q1"]

    report2, out = run(tmp_path, lambda req, **k: graph_response(), rows=rows)   # quota back: resumes
    assert report2.stopped is None and report2.already_done == 2 and report2.written == 6


def test_isolated_quota_errors_and_other_errors_do_not_stop_a_run(tmp_path):
    rows = [question(id=f"q{i}") for i in range(6)]
    calls = []

    def answer(request, **k):
        calls.append(1)
        n = len(calls)
        if n in (1, 2, 4):
            raise quota_error()
        if n == 3:
            raise RuntimeError("model error")                    # resets the streak
        return graph_response()

    report, _ = run(tmp_path, answer, rows=rows)
    assert report.stopped is None and len(calls) == 6 and len(report.errors) == 4


def test_sweep_stops_at_the_first_variant_that_hits_the_quota(tmp_path):
    def answer(request, **k):
        raise quota_error()

    qfile = write_questions(tmp_path / "q.jsonl", [question(id=f"q{i}") for i in range(5)])
    variants = [ev.parse_variant(s) for s in ("3", "5")]
    reports = ev.run_sweep(variants, [qfile], ["graph"], tmp_path / "s.jsonl", "sw", llm_model="m", answer=answer,
                           run_query=lambda d, s, n: GOLD, now=lambda: NOW)
    assert len(reports) == 1 and reports[0][1].stopped


# ----------------------------------------------------------------- embedding the questions in advance


class FakeEmbedder:
    def __init__(self):
        self.texts = None

    def prefetch_queries(self, texts):
        self.texts = texts
        return len(texts)


def test_warm_query_cache_embeds_the_text_the_pipeline_will_be_asked(tmp_path):
    qs = ev.load_questions(write_questions(tmp_path / "q.jsonl", [
        question(id="a", question="plain"), question(id="b", question="hinted", evidence="a hint")]), "test")
    emb = FakeEmbedder()
    assert ev.warm_query_cache(qs, True, embedder=emb) == 2
    assert emb.texts == ["plain", "hinted\nHint: a hint"]
    ev.warm_query_cache(qs, False, embedder=emb)
    assert emb.texts == ["plain", "hinted"]
    assert ev.warm_query_cache(qs, True, embedder=object()) == 0       # an embedder without a cache has nothing to do


def cli(tmp_path, monkeypatch, modes, warm):
    from schemagraph.eval import run_eval as module

    qfile = write_questions(tmp_path / "q.jsonl", [question()])
    ran = []
    monkeypatch.setattr(module, "warm_query_cache", warm)
    monkeypatch.setattr(module, "run_eval", lambda *a, **k: ran.append(1) or module.RunReport("r"))
    (tmp_path / "out.jsonl").write_text("")
    code = module.main(["--questions", str(qfile), "--modes", *modes, "--out", str(tmp_path / "out.jsonl")])
    return code, ran


def test_cli_stops_before_running_when_the_questions_cannot_be_embedded(tmp_path, monkeypatch, capsys):
    from schemagraph.graph.vector import RateLimited

    def refuse(*a, **k):
        raise RateLimited("daily quota")

    code, ran = cli(tmp_path, monkeypatch, ["graph"], refuse)
    assert code == 3 and ran == [] and "STOPPED" in capsys.readouterr().err


def test_cli_does_not_embed_for_full_mode_and_survives_other_embedding_errors(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    assert cli(tmp_path, monkeypatch, ["full"], boom) == (0, [1])

    def broken(*a, **k):
        raise RuntimeError("no embedding section")

    code, ran = cli(tmp_path, monkeypatch, ["retrieval"], broken)
    assert (code, ran) == (0, [1]) and "one at a time" in capsys.readouterr().err
