"""Few-shot example retrieval, including keeping a question out of its own prompt."""

import json
from types import SimpleNamespace

import pytest

from schemagraph.pipeline import examples

DB = "adventureworks"


def row(qid, question, db_id=DB, split="dev"):
    return {"id": qid, "db_id": db_id, "question": question, "gold_sql": "SELECT 1", "tier": 1, "split": split,
            "order_matters": False, "evidence": None, "author": "t", "notes": ""}


@pytest.fixture
def fewshot(tmp_path, monkeypatch):
    path = tmp_path / "fewshot.jsonl"
    rows = [row("d1", "How many employees work in each department?"),
            row("d2", "How many vendors supply each product category?"),
            row("d3", "Which three products sold the most units in 2024?"),
            row("b1", "How many employees work in each department?", db_id="other")]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(examples, "get_config", lambda: SimpleNamespace(pipeline=SimpleNamespace(fewshot_path=path)))
    return path


def test_the_closest_example_for_the_same_database_comes_first(fewshot):
    got = examples.retrieve_examples(DB, "How many employees work in each department?", 2)
    assert [e.id for e in got] == ["d1", "d2"]                   # b1 is identical but belongs to another database


def test_a_question_is_never_shown_with_its_own_answer_when_its_id_is_excluded(fewshot):
    question = "How many employees work in each department?"
    assert examples.retrieve_examples(DB, question, 1)[0].id == "d1"            # the leak: it matches itself
    got = examples.retrieve_examples(DB, question, 3, exclude_ids={"d1"})
    assert "d1" not in [e.id for e in got] and len(got) == 2


def test_excluding_nothing_changes_nothing_and_unknown_ids_are_ignored(fewshot):
    base = [e.id for e in examples.retrieve_examples(DB, "employees per department", 3)]
    assert [e.id for e in examples.retrieve_examples(DB, "employees per department", 3, exclude_ids=())] == base
    assert [e.id for e in examples.retrieve_examples(DB, "employees per department", 3, exclude_ids={"zzz"})] == base


def test_test_questions_are_refused_in_the_file(tmp_path, monkeypatch):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row("t1", "q", split="test")) + "\n")
    monkeypatch.setattr(examples, "get_config", lambda: SimpleNamespace(pipeline=SimpleNamespace(fewshot_path=path)))
    with pytest.raises(ValueError, match="never 'test'"):
        examples.retrieve_examples(DB, "q", 1)
