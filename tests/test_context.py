"""
The graph functions are replaced with fakes built from the sample files in
contracts/examples/
"""

import json
from pathlib import Path
import pytest
from schemagraph.models import DatabaseMetadata, JoinPlan, SeedTable, TableDetail
from schemagraph.pipeline import context as ctx

EXAMPLES = Path(__file__).parent.parent / "contracts" / "examples"
META = DatabaseMetadata.model_validate_json((EXAMPLES / "metadata.adventureworks.sample.json").read_text())
PLAN = JoinPlan.model_validate_json((EXAMPLES / "join_plan.sample.json").read_text())
DB = "adventureworks"

def details_from_metadata(meta: DatabaseMetadata) -> list[TableDetail]:
    """What queries.get_all_tables would return for the sample metadata, sorted by uid."""
    out = []
    for t in sorted(meta.tables, key=lambda t: t.uid):
        cols = sorted((c for c in meta.columns if c.table_uid == t.uid), key=lambda c: c.ordinal_position)
        col_uids = {c.uid for c in cols}
        refs = [r for r in meta.relationships if r.from_column in col_uids]
        out.append(TableDetail(table=t, columns=cols, references=refs))
    return out

ALL = details_from_metadata(META)
BY_UID = {d.table.uid: d for d in ALL}
SEEDS = PLAN.seed_tables

@pytest.fixture
def fake_graph(monkeypatch):
    """Swap the graph calls for fakes and record how they were called."""
    calls = {}

    def get_all_tables(db_id):
        calls["get_all_tables"] = db_id
        return ALL

    def get_tables(db_id, uids):
        calls["get_tables"] = list(uids)
        return [BY_UID[u] for u in uids if u in BY_UID]

    def retrieve_seed_tables(db_id, question, k):
        calls["retrieve"] = (question, k)
        return calls.get("seeds", SEEDS)

    def plan_joins(db_id, seeds, **options):
        calls["plan_options"] = options
        return calls.get("plan", PLAN)

    monkeypatch.setattr(ctx.queries, "get_all_tables", get_all_tables)
    monkeypatch.setattr(ctx.queries, "get_tables", get_tables)
    monkeypatch.setattr(ctx.retrieve, "retrieve_seed_tables", retrieve_seed_tables)
    monkeypatch.setattr(ctx.planner, "plan_joins", plan_joins)
    return calls

def test_ddl_has_every_table_column_and_key():
    ddl = ctx.render_ddl(ALL)
    assert ddl.count("CREATE TABLE") == len(META.tables)
    assert "CREATE TABLE person.person (" in ddl
    assert "PRIMARY KEY (businessentityid)" in ddl
    for col in META.columns:
        assert f"  {ctx.quote_ident(col.name)} " in ddl

def test_ddl_marks_declared_and_inferred_foreign_keys():
    ddl = ctx.render_ddl(ALL)
    assert "FOREIGN KEY (customerid) REFERENCES sales.customer (customerid)" in ddl
    inferred = [line for line in ddl.splitlines() if "REFERENCES production.product (productid)" in line]
    assert any("inferred" in line for line in inferred)
    declared = [line for line in ddl.splitlines() if "REFERENCES sales.customer (customerid)" in line]
    assert declared and not any("inferred" in line for line in declared)


def test_ddl_statements_are_well_formed():
    """No trailing comma before );and every statement is closed."""
    ddl = ctx.render_ddl(ALL)
    lines = ddl.splitlines()
    for i, line in enumerate(lines):
        if line == ");":
            code = lines[i - 1].split("  --")[0]
            assert not code.rstrip().endswith(",")
    assert ddl.count("CREATE TABLE") == ddl.count(");")

def test_quote_ident():
    assert ctx.quote_ident("customerid") == "customerid"
    assert ctx.quote_ident("CustomerID") == '"CustomerID"'
    assert ctx.quote_ident("first name") == '"first name"'
    assert ctx.quote_ident('a"b') == '"a""b"'

def test_render_joins_matches_spec_example():
    assert ctx.render_joins(PLAN).splitlines() == [
        "sales.salesorderheader.customerid = sales.customer.customerid",
        "sales.customer.personid = person.person.businessentityid",
    ]

def test_render_joins_warns_on_inferred_and_one_to_many():
    j = PLAN.joins[0].model_copy(update={"source": "inferred", "confidence": 0.9, "traversal": "one_to_many"})
    text = ctx.render_joins(PLAN.model_copy(update={"joins": [j]}))
    assert "inferred" in text and "one-to-many" in text

def test_full_mode_uses_every_table_and_no_retrieval(fake_graph):
    context, seeds = ctx.build_context(DB, "anything", "full", 5)
    assert context.tables == [d.table.uid for d in ALL]
    assert seeds == []
    assert context.join_plan is None and context.join_text == ""
    assert "retrieve" not in fake_graph

def test_retrieval_mode_uses_seeds_only(fake_graph):
    context, seeds = ctx.build_context(DB, "orders by customer name", "retrieval", 2)
    assert fake_graph["retrieve"] == ("orders by customer name", 2)
    assert context.tables == [s.uid for s in SEEDS]
    assert "adventureworks:sales.customer" not in context.tables  # the bridge is missing
    assert context.join_plan is None and context.join_text == ""
    assert seeds == SEEDS

def test_graph_mode_adds_bridge_table_and_joins(fake_graph):
    context, _ = ctx.build_context(DB, "q", "graph", 2)
    assert "adventureworks:sales.customer" in context.tables
    assert set(context.tables) == set(PLAN.tables)
    assert context.join_plan == PLAN
    assert "sales.customer.personid = person.person.businessentityid" in context.join_text

def test_graph_mode_falls_back_when_no_plan(fake_graph):
    fake_graph["plan"] = JoinPlan(db_id=DB, status="none", algorithm="none", seed_tables=SEEDS)
    context, _ = ctx.build_context(DB, "q", "graph", 2)
    assert context.tables == [s.uid for s in SEEDS]
    assert context.join_text == ""
    assert context.join_plan.status == "none"
    assert context.notes

def test_graph_mode_keeps_unreachable_seeds(fake_graph):
    lost = "adventureworks:production.product"
    fake_graph["plan"] = PLAN.model_copy(update={"status": "partial", "unreachable_seeds": [lost]})
    context, _ = ctx.build_context(DB, "q", "graph", 3)
    assert lost in context.tables
    assert any("production.product" in n for n in context.notes)

def test_planner_options_are_passed_through(fake_graph):
    ctx.build_context(DB, "q", "graph", 2, planner_options={"include_inferred": False})
    assert fake_graph["plan_options"] == {"include_inferred": False}

def test_no_seeds_gives_empty_context(fake_graph):
    fake_graph["seeds"] = []
    for mode in ("retrieval", "graph"):
        context, seeds = ctx.build_context(DB, "q", mode, 5)
        assert context.tables == [] and seeds == []
