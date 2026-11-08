"""The structural explanation (tables given, joins, warnings) needs no model call. The pipeline attaches it to
every answer that had a schema context, so evaluation, which runs with explain=False, can still see which
tables the model was given."""

from types import SimpleNamespace

import pytest

from schemagraph.eval import run_eval
from schemagraph.models import (
    AskRequest,
    ExplainResult,
    GenerationResult,
    JoinPlan,
    PlanAlternative,
    PlannedJoin,
    QueryResult,
    SchemaContext,
    SeedTable,
)
from schemagraph.pipeline import explain, pipeline

DB = "shop"
ORDERS, ITEMS, CUSTOMERS = f"{DB}:public.orders", f"{DB}:public.items", f"{DB}:public.customers"


def join(source="declared", confidence=1.0):
    return PlannedJoin(
        from_column=f"{ITEMS}.order_id", to_column=f"{ORDERS}.id", source=source, confidence=confidence,
        cardinality="N:1", traversal="many_to_one", weight=1.0,
    )


def graph_context(alternatives=None, notes=None):
    plan = JoinPlan(
        db_id=DB, status="full", algorithm="networkx_steiner",
        seed_tables=[SeedTable(uid=ORDERS, score=0.9, matched_on="table"), SeedTable(uid=CUSTOMERS, score=0.5, matched_on="column:name")],
        bridge_tables=[ITEMS], tables=[ORDERS, ITEMS, CUSTOMERS], joins=[join(), join("inferred", 0.8)],
        alternatives=alternatives or [],
    )
    return SchemaContext(db_id=DB, mode="graph", tables=plan.tables, ddl="...", join_plan=plan, notes=notes or [])


def test_explain_context_reports_roles_joins_and_warnings_without_a_model(monkeypatch):
    monkeypatch.setattr(explain, "chat_completion", lambda *a, **k: pytest.fail("no model call expected"))
    ctx = graph_context(notes=["No join plan connected the retrieved tables."])
    seeds = ctx.join_plan.seed_tables
    result = explain.explain_context(ctx, seeds)
    assert result.summary == ""
    assert {t.display_name: t.role for t in result.tables} == {"public.orders": "seed", "public.items": "bridge", "public.customers": "seed"}
    assert [(j.from_, j.source) for j in result.join_path] == [("public.items.order_id", "declared"), ("public.items.order_id", "inferred")]
    assert "confidence 0.80" in result.join_path[1].text
    assert result.warnings == ["No join plan connected the retrieved tables."]
    assert result.interpretation is None


def test_explain_context_mentions_alternative_paths():
    ctx = graph_context(alternatives=[PlanAlternative(tables=[ORDERS, CUSTOMERS], total_cost=3.0)])
    assert "alternative" in explain.explain_context(ctx, []).interpretation


def test_non_graph_modes_get_seed_or_context_roles():
    seeds = [SeedTable(uid=ORDERS, score=0.7, matched_on="table")]
    retrieval = SchemaContext(db_id=DB, mode="retrieval", tables=[ORDERS], ddl="...")
    assert [t.role for t in explain.explain_context(retrieval, seeds).tables] == ["seed"]
    full = SchemaContext(db_id=DB, mode="full", tables=[ORDERS, ITEMS], ddl="...")
    assert [t.role for t in explain.explain_context(full, []).tables] == ["context", "context"]


def test_build_explanation_adds_the_summary_and_reports_usage(monkeypatch):
    monkeypatch.setattr(explain, "chat_completion", lambda messages: ("Counts the orders.", 11, 5))
    usage = {}
    ctx = graph_context()
    result = explain.build_explanation("how many orders", "SELECT 1", ctx, ctx.join_plan.seed_tables, usage=usage)
    assert result.summary == "Counts the orders." and usage == {"input_tokens": 11, "output_tokens": 5}
    assert result == explain.explain_context(ctx, ctx.join_plan.seed_tables).model_copy(update={"summary": "Counts the orders."})


# ---------------------------------------------------------------------------- through the pipeline


@pytest.fixture
def setup(monkeypatch):
    """answer_question with every external piece faked; returns a function to run it."""
    monkeypatch.setattr(pipeline, "get_config", lambda: SimpleNamespace(
        pipeline=SimpleNamespace(top_k=5, max_attempts=1, row_limit=10, examples_k=0)))
    monkeypatch.setattr(pipeline, "retrieve_examples", lambda *a: [])
    monkeypatch.setattr(pipeline, "validate", lambda *a: ExplainResult(ok=True))
    monkeypatch.setattr(pipeline.db, "run_readonly_query", lambda *a: QueryResult(columns=["n"], rows=[["1"]], row_count=1))
    monkeypatch.setattr(explain, "chat_completion", lambda messages: ("A summary.", 3, 2))

    def run(mode, context, seeds, sql="SELECT 1", **kwargs):
        monkeypatch.setattr(pipeline, "build_context", lambda *a, **k: (context, seeds))
        monkeypatch.setattr(pipeline, "generate_sql", lambda *a: GenerationResult(sql=sql, raw_output=sql))
        return pipeline.answer_question(AskRequest(question="q", db_id=DB, mode=mode), **kwargs)

    return run


def test_retrieval_mode_without_explain_still_reports_its_tables_to_the_harness(setup):
    seeds = [SeedTable(uid=ORDERS, score=0.7, matched_on="table"), SeedTable(uid=ITEMS, score=0.6, matched_on="table")]
    ctx = SchemaContext(db_id=DB, mode="retrieval", tables=[ORDERS, ITEMS], ddl="...")
    response = setup("retrieval", ctx, seeds, explain=False)
    assert response.explanation.summary == "" and response.metrics.latency_ms.explanation == 0
    assert run_eval.context_tables(response) == ["public.items", "public.orders"]
    assert run_eval.seed_tables(response) == {"public.orders", "public.items"}


def test_full_mode_without_explain_reports_every_table_it_gave(setup):
    ctx = SchemaContext(db_id=DB, mode="full", tables=[ORDERS, ITEMS, CUSTOMERS], ddl="...")
    response = setup("full", ctx, [], explain=False)
    assert run_eval.context_tables(response) == ["public.customers", "public.items", "public.orders"]
    assert run_eval.seed_tables(response) is None                    # full mode has no retrieval seeds


def test_a_failed_answer_still_reports_the_tables_it_was_given(setup):
    ctx = SchemaContext(db_id=DB, mode="retrieval", tables=[ORDERS], ddl="...")
    response = setup("retrieval", ctx, [SeedTable(uid=ORDERS, score=0.7, matched_on="table")], sql="DELETE FROM orders", explain=True)
    assert response.status == "unsafe_sql"
    assert response.explanation.summary == "" and [t.uid for t in response.explanation.tables] == [ORDERS]
    assert run_eval.context_tables(response) == ["public.orders"]


def test_with_explain_the_summary_is_added_and_tokens_counted(setup):
    ctx = SchemaContext(db_id=DB, mode="retrieval", tables=[ORDERS], ddl="...")
    response = setup("retrieval", ctx, [SeedTable(uid=ORDERS, score=0.7, matched_on="table")], explain=True)
    assert response.explanation.summary == "A summary."
    assert [t.role for t in response.explanation.tables] == ["seed"]
    assert (response.metrics.input_tokens, response.metrics.output_tokens) == (3, 2)


def test_no_schema_means_no_explanation(setup):
    ctx = SchemaContext(db_id=DB, mode="retrieval", tables=[], ddl="")
    response = setup("retrieval", ctx, [], explain=True)
    assert response.status == "no_schema" and response.explanation is None
    assert run_eval.context_tables(response) == []
