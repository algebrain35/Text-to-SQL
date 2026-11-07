"""pipeline/explain.py: templates need no LLM; the summary uses a fake one."""

from pathlib import Path

from langchain_core.messages import AIMessage

from schemagraph.models import JoinPlan, PlanAlternative, SchemaContext, SeedTable
from schemagraph.pipeline import explain as ex

EXAMPLES = Path(__file__).parent.parent / "contracts" / "examples"
PLAN = JoinPlan.model_validate_json((EXAMPLES / "join_plan.sample.json").read_text())
SEEDS = PLAN.seed_tables
GRAPH = SchemaContext(db_id="adventureworks", mode="graph", tables=PLAN.tables, ddl="", join_plan=PLAN)


class FakeLLM:
    def __init__(self, text="Adds up order totals per person.", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise ConnectionError("LLM down")
        return AIMessage(content=self.text)


def test_graph_mode_roles_and_reasons():
    rows = {t.uid: t for t in ex.explain_tables(GRAPH, SEEDS)}
    bridge = rows["adventureworks:sales.customer"]
    assert bridge.role == "bridge"
    assert "sales.salesorderheader" in bridge.reason and "person.person" in bridge.reason
    seed = rows["adventureworks:sales.salesorderheader"]
    assert seed.role == "seed" and "totaldue" in seed.reason
    assert all(t.display_name == t.uid.split(":", 1)[1] for t in rows.values())


def test_retrieval_mode_tables_are_seeds():
    ctx = SchemaContext(db_id="adventureworks", mode="retrieval", tables=[s.uid for s in SEEDS], ddl="")
    assert {t.role for t in ex.explain_tables(ctx, SEEDS)} == {"seed"}


def test_full_mode_marks_tables_the_sql_uses():
    ctx = SchemaContext(db_id="adventureworks", mode="full", tables=PLAN.tables, ddl="")
    rows = {t.display_name: t for t in ex.explain_tables(ctx, [], "SELECT * FROM person.person")}
    assert {t.role for t in rows.values()} == {"context"}
    assert rows["person.person"].reason == "Used in the query."
    assert rows["sales.customer"].reason != "Used in the query."


def test_join_path_in_words():
    joins = ex.explain_joins(PLAN)
    assert [j.from_ for j in joins] == ["sales.salesorderheader.customerid", "sales.customer.personid"]
    assert joins[0].source == "declared"
    assert "Each sales.salesorderheader row points to one sales.customer row" in joins[0].text
    dumped = joins[0].model_dump(by_alias=True)
    assert "from" in dumped and "from_" not in dumped


def test_inferred_join_is_flagged():
    j = PLAN.joins[0].model_copy(update={"source": "inferred", "confidence": 0.9})
    plan = PLAN.model_copy(update={"joins": [j]})
    assert "inferred" in ex.join_text(j)
    ctx = GRAPH.model_copy(update={"join_plan": plan})
    assert any("inferred" in w for w in ex.warnings(ctx))


def test_interpretation_only_with_alternatives():
    assert ex.interpretation(PLAN) is None
    alt = PlanAlternative(tables=PLAN.tables + ["adventureworks:sales.store"], total_cost=4.0)
    text = ex.interpretation(PLAN.model_copy(update={"alternatives": [alt]}))
    assert "sales.store" in text


def test_summary_uses_llm_and_survives_failure():
    llm = FakeLLM()
    e = ex.build_explanation("q", "SELECT 1", GRAPH, SEEDS, llm=llm)
    assert e.summary == "Adds up order totals per person." and llm.calls == 1
    e = ex.build_explanation("q", "SELECT 1", GRAPH, SEEDS, llm=FakeLLM(fail=True))
    assert e.summary == "" and e.tables  # the rest still comes back


def test_summarize_false_never_calls_llm():
    llm = FakeLLM()
    e = ex.build_explanation("q", "SELECT 1", GRAPH, SEEDS, summarize=False, llm=llm)
    assert llm.calls == 0 and e.summary == "" and len(e.join_path) == 2
