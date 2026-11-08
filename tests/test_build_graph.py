"""graph/build.py with every step replaced by a recording fake: order, skip flags, per-database
failure isolation, and the checks done once before any database is touched."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from schemagraph.config import EmbeddingConfig
from schemagraph.extraction import extract
from schemagraph.graph import build

SAMPLE = Path(__file__).parent.parent / "contracts" / "examples" / "metadata.adventureworks.sample.json"
DB = "adventureworks"


@pytest.fixture(scope="module")
def meta():
    return extract.load_metadata(SAMPLE)


class Recorder:
    """Fake steps. `calls` is the global order; failures are injected by name."""

    def __init__(self, meta, tmp_path, fail=None, problems=None, wrong_db=False):
        self.meta, self.calls, self.fail, self.problems, self.wrong_db = meta, [], fail or {}, problems or [], wrong_db
        self.report = SimpleNamespace(
            tables=7, columns=93, tables_current=7, columns_current=93, tables_total=7, columns_total=93, stopped=None
        )
        self.path = tmp_path / "meta.json"
        self.path.write_text("{}")

    def _step(self, name, db_id=None):
        self.calls.append(name if db_id is None else f"{name}:{db_id}")
        if self.fail.get(name) == db_id or (db_id is None and name in self.fail):
            raise RuntimeError(f"{name} blew up")

    def steps(self):
        def extract_(db_id, **kw):
            self._step("extract", db_id)
            self.extract_kwargs = kw
            return SimpleNamespace(ok=True, error=None)

        def metadata_path(db_id):
            return self.path

        def load(path):
            return self.meta.model_copy(update={"db_id": "other"}) if self.wrong_db else self.meta

        def ingest(meta, rebuild=False):
            self._step("ingest", meta.db_id)
            self.rebuild = rebuild

        def weights(db_id):
            self._step("weights", db_id)
            return 5

        def indexes(dimensions, name):
            self.calls.append(f"indexes:{dimensions}:{name}")

        def embed(db_id, embedder, name, reembed=False):
            self._step("embed", db_id)
            self.reembed = reembed
            return self.report

        def verify(meta):
            self.calls.append("verify")
            return list(self.problems)

        return build.Steps(extract_, load, metadata_path, ingest, weights, indexes, embed, verify)


ENRICH = build.Enrichment(embedder=object(), dimensions=1024, name="voyage")


def run(rec, **opts):
    return build.build_graph(DB, build.Options(**opts), ENRICH, rec.steps())


def test_steps_run_in_order(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    result = run(rec)
    assert rec.calls == [f"extract:{DB}", f"ingest:{DB}", f"weights:{DB}", "indexes:1024:voyage", f"embed:{DB}", "verify"]
    assert result.ok and result.steps_done == ["extract", "ingest", "weights", "embed", "verify"]
    assert (result.tables, result.columns) == (7, 93)
    assert result.declared + result.inferred == 5
    assert (result.embedded_tables, result.embedded_columns, result.edges_weighted) == (7, 93, 5)


def test_skip_enrichment_creates_no_index_and_embeds_nothing(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    result = build.build_graph(DB, build.Options(skip_enrichment=True), None, rec.steps())
    assert result.ok and not any(c.startswith(("indexes", "embed")) for c in rec.calls)


def test_skip_extraction_reuses_the_file_and_needs_it_to_exist(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    result = run(rec, skip_extraction=True)
    assert result.ok and not any(c.startswith("extract") for c in rec.calls)
    rec.path.unlink()
    result = run(rec, skip_extraction=True)
    assert "does not exist" in result.error and result.steps_done == []


def test_options_reach_the_steps(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    run(rec, rebuild=True, skip_inference=True, sample_n=9, min_overlap=0.8, min_confidence=0.7)
    assert rec.rebuild is True
    assert rec.extract_kwargs == {"skip_inference": True, "sample_n": 9, "min_overlap": 0.8, "min_confidence": 0.7}
    rec2 = Recorder(meta, tmp_path)
    run(rec2)
    assert rec2.rebuild is False and "min_overlap" not in rec2.extract_kwargs


def test_a_failing_step_stops_that_database_and_says_how_far_it_got(meta, tmp_path):
    rec = Recorder(meta, tmp_path, fail={"weights": DB})
    result = run(rec)
    assert not result.ok and "weights blew up" in result.error
    assert result.steps_done == ["extract", "ingest"]
    assert "verify" not in rec.calls and f"embed:{DB}" not in rec.calls


def test_extraction_failure_is_reported_not_raised(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    steps = rec.steps()
    steps.extract = lambda db_id, **kw: SimpleNamespace(ok=False, error="OperationalError: no such db")
    result = build.build_graph(DB, build.Options(), ENRICH, steps)
    assert "no such db" in result.error and result.steps_done == []


def test_metadata_for_another_db_id_is_refused_before_ingest(meta, tmp_path):
    rec = Recorder(meta, tmp_path, wrong_db=True)
    result = run(rec)
    assert "not 'adventureworks'" in result.error and not any(c.startswith("ingest") for c in rec.calls)


def test_verify_differences_make_the_run_fail(meta, tmp_path):
    rec = Recorder(meta, tmp_path, problems=["extra table in graph: x"])
    result = run(rec)
    assert result.error is None and not result.ok and result.problems == ["extra table in graph: x"]
    text = build.format_results([result], build.Options())
    assert "extra table in graph" in text and "--rebuild" in text and "0 built, 0 partial, 1 failed" in text
    assert "--rebuild" not in build.format_results([result], build.Options(rebuild=True))


def test_one_database_failing_does_not_stop_the_others(meta, tmp_path):
    rec = Recorder(meta, tmp_path, fail={"ingest": "bad"})
    steps = rec.steps()
    steps.load_metadata = lambda path: meta.model_copy(update={"db_id": steps_current[0]})
    steps_current = [""]
    original_extract = steps.extract

    def extract_(db_id, **kw):
        steps_current[0] = db_id
        return original_extract(db_id, **kw)

    steps.extract = extract_
    results = build.build_many(["good", "bad", "also_good"], build.Options(), ENRICH, steps)
    assert [r.ok for r in results] == [True, False, True]
    assert "1 failed" in build.format_results(results, build.Options())


# ----------------------------------------------------------------- whole-run checks


def config(**kw):
    base = {"provider": "openai", "model": "m", "dimensions": 8, "name": "voyage"}
    return EmbeddingConfig(**(base | kw))


class FakeEmbedder:
    def __init__(self, dims, fail=False):
        self.dims, self.fail = dims, fail
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        if self.fail:
            raise RuntimeError("401 unauthorized")
        return [0.0] * self.dims


def test_prepare_enrichment_checks_the_vector_size_with_one_call():
    embedder = FakeEmbedder(8)
    enrichment = build.prepare_enrichment(config(), lambda cfg: embedder)
    assert (enrichment.dimensions, enrichment.name, enrichment.embedder) == (8, "voyage", embedder)
    assert embedder.queries == [build.PROBE_TEXT]


def test_prepare_enrichment_rejects_a_size_mismatch():
    with pytest.raises(build.SetupError, match="3072-dimensional.*dimensions: 8"):
        build.prepare_enrichment(config(), lambda cfg: FakeEmbedder(3072))


def test_prepare_enrichment_reports_an_unreachable_model():
    with pytest.raises(build.SetupError, match="cannot reach the embedding model.*401"):
        build.prepare_enrichment(config(), lambda cfg: FakeEmbedder(8, fail=True))


def test_prepare_enrichment_without_a_config_section(monkeypatch):
    monkeypatch.setattr("schemagraph.config.get_config", lambda: SimpleNamespace(embedding=None))
    with pytest.raises(build.SetupError, match="--skip-enrichment"):
        build.prepare_enrichment()


# ----------------------------------------------------------------- command line


@pytest.fixture
def cli(monkeypatch, meta, tmp_path):
    from schemagraph.graph import driver

    state = SimpleNamespace(closed=0, connected=0, prepared=0, conn_error=None, prep_error=None)

    def check():
        state.connected += 1
        if state.conn_error:
            raise ConnectionError(state.conn_error)

    def prepare():
        state.prepared += 1
        if state.prep_error:
            raise build.SetupError(state.prep_error)
        return ENRICH

    monkeypatch.setattr(driver, "check_connection", check)
    monkeypatch.setattr(driver, "close_driver", lambda: setattr(state, "closed", state.closed + 1))
    monkeypatch.setattr(build, "prepare_enrichment", prepare)
    state.rec = Recorder(meta, tmp_path)
    return state


def test_cli_success(cli, capsys):
    assert build.main(["--db", DB], steps=cli.rec.steps()) == 0
    out = capsys.readouterr().out
    assert "ok" in out and "embedded 7/7 tables and 93/93 columns" in out and "1 built, 0 partial, 0 failed" in out
    assert cli.closed == 1


def test_cli_setup_failures_exit_2_and_touch_no_database(cli, capsys):
    cli.prep_error = "no key"
    assert build.main(["--db", DB], steps=cli.rec.steps()) == 2
    assert "cannot start: no key" in capsys.readouterr().out and cli.rec.calls == []
    cli.prep_error, cli.conn_error = None, "neo4j down"
    assert build.main(["--db", DB], steps=cli.rec.steps()) == 2
    assert "neo4j down" in capsys.readouterr().out and cli.rec.calls == [] and cli.closed == 2


def test_cli_skip_enrichment_does_not_need_an_embedding_model(cli):
    cli.prep_error = "would fail"
    assert build.main(["--db", DB, "--skip-enrichment"], steps=cli.rec.steps()) == 0
    assert cli.prepared == 0


def test_cli_exit_1_when_a_database_fails_or_differs(cli):
    cli.rec.problems = ["x"]
    assert build.main(["--db", DB], steps=cli.rec.steps()) == 1


def test_cli_argument_errors(cli):
    for argv in ([], ["--all", "--db", DB], ["--db", DB, "--skip-extraction", "--skip-inference"]):
        with pytest.raises(SystemExit):
            build.main(argv, steps=cli.rec.steps())


def test_cli_all_uses_the_databases_the_readonly_role_sees(cli, monkeypatch):
    monkeypatch.setattr("schemagraph.db.list_databases", lambda: ["a", "b"])
    steps = cli.rec.steps()
    steps.load_metadata = lambda path: cli.rec.meta.model_copy(update={"db_id": cli.last})
    cli.last = ""
    inner = steps.extract

    def extract_(db_id, **kw):
        cli.last = db_id
        return inner(db_id, **kw)

    steps.extract = extract_
    assert build.main(["--all"], steps=steps) == 0
    assert [c for c in cli.rec.calls if c.startswith("extract")] == ["extract:a", "extract:b"]


# ----------------------------------------------------------------- partial (rate-limited) embedding


def partial_report(rec, why="rate limited (HTTP 429) and the 120s wait budget (max_wait_s) is used up"):
    rec.report = SimpleNamespace(
        tables=7, columns=40, tables_current=7, columns_current=40, tables_total=7, columns_total=93, stopped=why
    )


def test_a_rate_limited_embedding_is_partial_not_failed(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    partial_report(rec)
    result = run(rec)
    assert result.partial and not result.ok and not result.failed and result.error is None
    assert result.steps_done == ["extract", "ingest", "weights", "embed (partial)", "verify"]
    assert (result.tables_current, result.tables_total, result.columns_current, result.columns_total) == (7, 7, 40, 93)
    assert rec.calls[-1] == "verify"                                       # the graph itself is still checked


def test_partial_output_says_what_is_saved_and_how_to_resume(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    partial_report(rec)
    result = run(rec)
    text = build.format_results([result], build.Options())
    assert "PART" in text and "embedded 7/7 tables and 40/93 columns" in text and "max_wait_s" in text
    assert "rerun the same command" in text and "WITHOUT --rebuild" not in text
    assert "0 built, 1 partial, 0 failed" in text
    assert "WITHOUT --rebuild" in build.format_results([result], build.Options(rebuild=True))


def test_the_report_counts_new_and_total_vectors(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    text = build.format_results([run(rec)], build.Options())
    assert "embedded 7/7 tables and 93/93 columns (100 new)" in text


def test_reembed_reaches_the_embed_step(meta, tmp_path):
    rec = Recorder(meta, tmp_path)
    run(rec, reembed=True)
    assert rec.reembed is True
    rec2 = Recorder(meta, tmp_path)
    run(rec2)
    assert rec2.reembed is False


def test_cli_exit_3_for_partial_and_1_when_anything_failed(cli, capsys):
    partial_report(cli.rec)
    assert build.main(["--db", DB], steps=cli.rec.steps()) == 3
    assert "1 partial" in capsys.readouterr().out
    cli.rec.problems = ["x"]
    assert build.main(["--db", DB], steps=cli.rec.steps()) == 1          # a real difference outranks a partial run


def test_cli_partial_in_one_database_does_not_stop_the_next(cli, meta, monkeypatch):
    partial_report(cli.rec)
    steps = cli.rec.steps()
    current = [""]
    inner = steps.extract

    def extract_(db_id, **kw):
        current[0] = db_id
        return inner(db_id, **kw)

    steps.extract = extract_
    steps.load_metadata = lambda path: meta.model_copy(update={"db_id": current[0]})
    assert build.main(["--db", "a", "--db", "b"], steps=steps) == 3
    assert [c for c in cli.rec.calls if c.startswith("embed")] == ["embed:a", "embed:b"]


def test_cli_reembed_flag(cli):
    assert build.main(["--db", DB, "--reembed"], steps=cli.rec.steps()) == 0 and cli.rec.reembed is True
