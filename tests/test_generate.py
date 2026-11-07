"""no LLM or API key needed because a fake model stands in"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from schemagraph.config import LLMConfig
from schemagraph.models import Attempt, FewShotExample, SchemaContext
from schemagraph.pipeline import generate as gen

CONTEXT = SchemaContext(
    db_id="adventureworks",
    mode="graph",
    tables=["adventureworks:sales.customer", "adventureworks:person.person"],
    ddl="CREATE TABLE sales.customer (\n  customerid integer NOT NULL\n);",
    join_text="sales.customer.personid = person.person.businessentityid",
    notes=["person.person is big"],
)
EXAMPLE = FewShotExample(
    id="aw-dev-1", db_id="adventureworks", question="How many customers?",
    gold_sql="SELECT count(*) FROM sales.customer;", tier=1, split="dev",
)

class FakeLLM:
    """Stands in for a LangChain chat model: records the prompt, returns a fixed reply."""
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage
        self.seen = None

    def invoke(self, messages):
        self.seen = messages
        return AIMessage(content=self.content, usage_metadata=self.usage)

@pytest.mark.parametrize("reply, expected", [
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("Here you go:\n```sql\nSELECT a\nFROM t\n```\nHope that helps", "SELECT a\nFROM t"),
    ("```\nSELECT 2\n```", "SELECT 2"),
    ("```SQL\nselect 3;;\n```", "select 3"),
    ("SELECT 4;", "SELECT 4"),
    ("  with x as (select 1) select * from x", "with x as (select 1) select * from x"),
    ("```python\nprint(1)\n```\n```sql\nSELECT 5\n```", "SELECT 5"),
    ("I cannot answer that from this schema.", None),
    ("", None),
    ("```sql\n\n```", None),
])
def test_extract_sql(reply, expected):
    assert gen.extract_sql(reply) == expected

def test_prompt_has_rules_schema_joins_notes_examples_and_question():
    msgs = gen.build_prompt("Total orders per person?", CONTEXT, [EXAMPLE], [])
    assert isinstance(msgs[0], SystemMessage) and "SELECT" in msgs[0].content
    assert len(msgs) == 2 and isinstance(msgs[1], HumanMessage)
    body = msgs[1].content
    for piece in (CONTEXT.ddl, CONTEXT.join_text, "person.person is big",
                  "How many customers?", "SELECT count(*) FROM sales.customer;",
                  "Total orders per person?"):
        assert piece in body
    assert body.index("### Schema") < body.index("### Examples") < body.index("### Question")

def test_prompt_leaves_out_empty_sections():
    plain = CONTEXT.model_copy(update={"join_text": "", "notes": [], "mode": "retrieval"})
    body = gen.build_prompt("q", plain, [], [])[1].content
    assert "### Join conditions" not in body
    assert "### Notes" not in body
    assert "### Examples" not in body

def test_same_template_in_every_mode():
    """Only the context differs between modes, never the instructions."""
    a = gen.build_prompt("q", CONTEXT.model_copy(update={"mode": "full"}), [], [])
    b = gen.build_prompt("q", CONTEXT.model_copy(update={"mode": "graph"}), [], [])
    assert [m.content for m in a] == [m.content for m in b]

def test_failed_attempts_are_replayed_with_their_errors():
    tries = [Attempt(sql="SELECT nope FROM t", error='column "nope" does not exist'),
             Attempt(sql="SELECT x FROM t", error="timeout")]
    msgs = gen.build_prompt("q", CONTEXT, [], tries)
    assert len(msgs) == 2 + 2 * len(tries)
    assert isinstance(msgs[2], AIMessage) and "SELECT nope FROM t" in msgs[2].content
    assert isinstance(msgs[3], HumanMessage) and 'column "nope" does not exist' in msgs[3].content
    assert "timeout" in msgs[5].content

def test_generate_sql_returns_sql_and_tokens():
    llm = FakeLLM("```sql\nSELECT 1;\n```", {"input_tokens": 120, "output_tokens": 9, "total_tokens": 129})
    result = gen.generate_sql("q", CONTEXT, [], [], llm=llm)
    assert result.sql == "SELECT 1"
    assert result.raw_output == "```sql\nSELECT 1;\n```"
    assert (result.input_tokens, result.output_tokens) == (120, 9)
    assert llm.seen is not None and isinstance(llm.seen[0], SystemMessage)

def test_generate_sql_without_sql_or_usage():
    result = gen.generate_sql("q", CONTEXT, [], [], llm=FakeLLM("No idea."))
    assert result.sql is None and result.raw_output == "No idea."
    assert (result.input_tokens, result.output_tokens) == (0, 0)

def test_reply_as_content_blocks():
    msg = AIMessage(content=[{"type": "text", "text": "```sql\nSELECT "}, {"type": "text", "text": "7\n```"}])
    assert gen.extract_sql(gen.message_text(msg)) == "SELECT 7"

def test_make_chat_model_uses_config_and_temperature_zero(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = gen.make_chat_model(LLMConfig(provider="anthropic", model="claude-test", max_tokens=300))
    assert model.model == "claude-test"
    assert model.temperature == 0
    assert model.max_tokens == 300


def test_make_chat_model_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = gen.make_chat_model(LLMConfig(provider="openai", model="gpt-test"))
    assert model.model_name == "gpt-test"
    assert model.temperature == 0


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        gen.make_chat_model(LLMConfig(provider="anthropic", model="claude-test"))
