"""pipeline/examples.py: few-shot example selection."""

from schemagraph.models import FewShotExample
from schemagraph.pipeline import examples as exm


def make(id, question, db_id="adventureworks", split="dev"):
    return FewShotExample(id=id, db_id=db_id, question=question, gold_sql="SELECT 1", tier=1, split=split)


POOL = [
    make("a", "How many products are in each subcategory?"),
    make("b", "Total order value for each customer"),
    make("c", "List employees hired after 2010"),
    make("d", "Total order value for each customer", db_id="other_db"),
]


def test_most_similar_first_and_same_db_only():
    got = exm.rank_examples("What is the total order value per customer name?", "adventureworks", POOL, 2)
    assert [e.id for e in got][0] == "b"
    assert all(e.db_id == "adventureworks" for e in got)


def test_identical_question_is_never_its_own_example():
    got = exm.rank_examples("Total order value for each customer", "adventureworks", POOL, 3)
    assert "b" not in [e.id for e in got]


def test_k_zero_and_missing_file(tmp_path):
    assert exm.rank_examples("anything", "adventureworks", POOL, 0) == []
    assert exm.retrieve_examples("adventureworks", "q", 3, path=tmp_path / "nope.jsonl") == []


def test_reads_file_and_skips_test_questions(tmp_path):
    path = tmp_path / "fewshot.jsonl"
    rows = [make("x", "count products"), make("y", "count products in stock", split="test")]
    path.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    got = exm.retrieve_examples("adventureworks", "how many products", 5, path=path)
    assert [e.id for e in got] == ["x"]
