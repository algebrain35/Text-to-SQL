"""bird_questions.py on invented records shaped like BIRD mini-dev (no BIRD data is used)."""

import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from schemagraph.eval import bird_questions as bq
from schemagraph.models import Question

ONE = "SELECT count(*) FROM {a}"
TWO = "SELECT x.name FROM {a} AS x JOIN {b} AS y ON x.id = y.{a}_id WHERE y.v > 3"
THREE = "SELECT x.name FROM {a} x JOIN {b} y ON x.id = y.a_id JOIN {c} z ON y.id = z.b_id"


def record(i, db, sql, **extra):
    base = {"question_id": i, "db_id": db, "question": f"Question number {i} about {db}?", "evidence": f"hint {i}",
            "SQL": sql, "difficulty": "moderate"}
    base.update(extra)
    return base


def sample_items(per_tier=12, dbs=("alpha", "beta", "gamma")):
    items, i = [], 0
    for tier, template in ((1, ONE), (2, TWO), (3, THREE)):
        for n in range(per_tier):
            db = dbs[n % len(dbs)]
            sql = template.format(a=f"{db}_a", b=f"{db}_b", c=f"{db}_c")
            items.append(record(i, db, sql))
            i += 1
    return items


# ----------------------------------------------------------------- one record


def test_tier_counts_distinct_tables():
    assert bq.tier_for("SELECT * FROM t") == 1
    assert bq.tier_for("SELECT * FROM a JOIN b ON a.id = b.id") == 2
    assert bq.tier_for("SELECT * FROM a JOIN b ON 1=1 JOIN c ON 1=1") == 3
    assert bq.tier_for("SELECT * FROM a JOIN b ON 1=1 JOIN c ON 1=1 JOIN d ON 1=1 JOIN e ON 1=1") == 3
    assert bq.tier_for("SELECT * FROM a x JOIN a y ON x.id = y.parent") == 1     # a self-join reads one table
    assert bq.tier_for("WITH t AS (SELECT * FROM a) SELECT * FROM t") == 1       # a CTE name is not a table
    assert bq.tier_for("SELECT 1") is None
    assert bq.tier_for("SELECT * FROM") is None


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT n FROM t ORDER BY n DESC", True),
        ("SELECT n FROM t ORDER BY n DESC LIMIT 5", True),
        ("SELECT n FROM t ORDER BY n DESC LIMIT 1", False),   # one row: order cannot differ
        ("SELECT n FROM t", False),
        ("SELECT n FROM (SELECT n FROM t ORDER BY n) s", False),   # inner ordering is invisible
        ("SELECT n FROM t UNION SELECT n FROM u", False),
        ("SELECT * FROM", False),
        ("SELECT n FROM t WHERE a = 'x ORDER BY n", False),   # unterminated quote
    ],
)
def test_order_matters_heuristic(sql, expected):
    assert bq.order_matters_for(sql) is expected


def test_record_becomes_a_question():
    q = bq.to_question(record(7, "alpha", TWO.format(a="alpha_a", b="alpha_b")), 0)
    assert isinstance(q, Question)
    assert (q.id, q.db_id, q.tier, q.split, q.author) == ("bird-7", "alpha", 2, "test", "bird")
    assert q.evidence == "hint 7" and "difficulty=moderate" in q.notes and "tables=2" in q.notes
    assert q.gold_sql.startswith("SELECT x.name")


def test_blank_evidence_becomes_none_and_id_falls_back_to_position():
    item = record(0, "alpha", ONE.format(a="t"), evidence="  ")
    del item["question_id"]
    q = bq.to_question(item, 4)
    assert q.evidence is None and q.id == "bird-5"


def test_sql_key_variants_and_surrounding_whitespace():
    item = {"db_id": "alpha", "question": "q", "sql": "  SELECT 1 FROM t;  "}
    assert bq.to_question(item, 0).gold_sql == "SELECT 1 FROM t;"


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda r: r.pop("db_id"), "missing db_id"),
        (lambda r: r.update(question=""), "missing question"),
        (lambda r: r.pop("SQL"), "missing SQL"),
        (lambda r: r.update(db_id="Has-Caps"), "db_id"),
        (lambda r: r.update(SQL="SELECT 1"), "reads no tables"),
    ],
)
def test_unusable_records_say_why(mutate, reason):
    item = record(1, "alpha", ONE.format(a="t"))
    mutate(item)
    result = bq.to_question(item, 0)
    assert isinstance(result, str) and reason in result


# ----------------------------------------------------------------- reading


def test_load_items_array_jsonl_and_wrapped(tmp_path):
    items = sample_items(2)
    (tmp_path / "a.json").write_text(json.dumps(items))
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n\n")
    (tmp_path / "c.json").write_text(json.dumps({"data": items}))
    for name in ("a.json", "b.jsonl", "c.json"):
        assert bq.load_items(tmp_path / name) == items


def test_load_items_rejects_other_shapes(tmp_path):
    (tmp_path / "x.json").write_text('"just a string"')
    with pytest.raises(bq.BirdFormatError):
        bq.load_items(tmp_path / "x.json")
    (tmp_path / "y.jsonl").write_text('{"a": 1}\nnot json\n')
    with pytest.raises(bq.BirdFormatError, match=r"y\.jsonl:2"):
        bq.load_items(tmp_path / "y.jsonl")


# ----------------------------------------------------------------- apportion


@pytest.mark.parametrize("total,weights", [(60, [0.2, 0.4, 0.4]), (10, [1, 1, 1]), (7, [5, 1, 1]), (0, [1, 2]), (3, [0, 0, 1])])
def test_apportion_sums_exactly_and_follows_weights(total, weights):
    counts = bq.apportion(total, weights)
    assert sum(counts) == total and all(c >= 0 for c in counts)
    assert all(abs(c - total * w / sum(weights)) < 1 for c, w in zip(counts, weights))


def test_apportion_is_stable_and_rejects_nonsense():
    assert bq.apportion(10, [1, 1, 1]) == [4, 3, 3]          # ties go to the lowest index
    for bad in ([], [0, 0], [-1, 2]):
        with pytest.raises(ValueError):
            bq.apportion(5, bad)


# ----------------------------------------------------------------- choosing


def candidates(per_tier=30, dbs=("alpha", "beta", "gamma")):
    out = []
    for i, item in enumerate(sample_items(per_tier, dbs)):
        out.append(bq.to_question(item, i))
    return out


def test_choose_counts_mix_and_splits():
    picked, warnings = bq.choose(candidates(), n_test=50, n_dev=10, seed=1)
    assert warnings == [] and len(picked) == 60
    assert Counter(q.split for q in picked) == {"test": 50, "dev": 10}
    assert Counter(q.tier for q in picked) == {1: 12, 2: 24, 3: 24}          # 20/40/40
    assert Counter(q.tier for q in picked if q.split == "dev") == {1: 2, 2: 4, 3: 4}   # dev mirrors the mix
    assert len({q.id for q in picked}) == 60


def test_choose_spreads_over_databases():
    picked, _ = bq.choose(candidates(), n_test=30, n_dev=0, seed=3)
    per_db = Counter(q.db_id for q in picked)
    assert max(per_db.values()) - min(per_db.values()) <= 3


def test_choose_is_reproducible_and_seed_dependent():
    a = [q.id for q in bq.choose(candidates(), 20, 5, seed=7)[0]]
    b = [q.id for q in bq.choose(candidates(), 20, 5, seed=7)[0]]
    c = [q.id for q in bq.choose(candidates(), 20, 5, seed=8)[0]]
    assert a == b and a != c
    shuffled = candidates()
    random.Random(0).shuffle(shuffled)
    assert [q.id for q in bq.choose(shuffled, 20, 5, seed=7)[0]] == a      # input order does not matter


def test_choose_sorts_by_numeric_id():
    ids = [q.id for q in bq.choose(candidates(), 20, 5, seed=0)[0]]
    assert ids == sorted(ids, key=lambda s: int(s.split("-")[1]))


def test_short_tier_is_made_up_from_the_others_with_a_warning():
    pool = [q for q in candidates() if q.tier != 1] + [q for q in candidates() if q.tier == 1][:2]
    picked, warnings = bq.choose(pool, n_test=50, n_dev=10, seed=0)
    assert len(picked) == 60 and Counter(q.tier for q in picked)[1] == 2
    assert any("tier 1" in w for w in warnings)


def test_asking_for_more_than_exists_takes_everything():
    pool = candidates(per_tier=3)
    picked, warnings = bq.choose(pool, n_test=50, n_dev=10)
    assert len(picked) == 9 and any("only 9" in w for w in warnings)
    assert Counter(q.split for q in picked)["dev"] <= 9


def test_zero_dev_and_empty_pool():
    picked, _ = bq.choose(candidates(), 10, 0)
    assert all(q.split == "test" for q in picked)
    assert bq.choose([], 5, 2)[0] == []


# ----------------------------------------------------------------- build


def test_build_reports_skips_and_duplicates():
    items = sample_items(10)
    items.append(record(500, "alpha", "SELECT 1"))                       # no tables
    items.append({"db_id": "alpha", "question": "", "SQL": "SELECT 1 FROM t"})
    items.append(dict(items[0]))                                         # same question_id again
    report = bq.build(items, n_test=20, n_dev=4)
    assert report.read == 33 and report.usable == 30
    assert report.skipped.reasons["duplicate question_id"] == 1
    assert sum(report.skipped.reasons.values()) == 3
    assert len(report.questions) == 24


def test_build_with_verify_drops_failing_and_empty_gold():
    items = sample_items(10)

    def fake_verify(questions):
        ids = [q.id for q in questions]
        return SimpleNamespace(results={i: object() for i in ids[3:]}, errors={ids[0]: "gold SQL failed: boom", ids[1]: "slow"},
                               empty=[ids[4]])

    report = bq.build(items, n_test=40, n_dev=0, verify=fake_verify)
    assert report.usable == 30 - 3 - 1
    assert set(report.dropped_by_gold) == {"bird-0", "bird-1", "bird-4"}
    assert "bird-0" not in {q.id for q in report.questions} and "bird-4" not in {q.id for q in report.questions}
    assert report.dropped_by_gold["bird-4"] == "gold SQL returns no rows"


# ----------------------------------------------------------------- command line


def test_cli_writes_a_file_that_loads_as_questions(tmp_path, capsys):
    from schemagraph.eval.run_eval import load_questions

    src, out = tmp_path / "mini.json", tmp_path / "q" / "bird_subset.jsonl"
    src.write_text(json.dumps(sample_items(30)))
    assert bq.main(["--input", str(src), "--out", str(out), "--n-test", "20", "--n-dev", "5", "--seed", "2"]) == 0
    text = capsys.readouterr().out
    assert "fields seen" in text and "wrote 25" in text and "ALL" in text
    assert len(load_questions(out, "test")) == 20 and len(load_questions(out, "dev")) == 5
    first = out.read_bytes()
    assert bq.main(["--input", str(src), "--out", str(out), "--n-test", "20", "--n-dev", "5", "--seed", "2"]) == 0
    assert out.read_bytes() == first                                      # same seed, same file


def test_cli_reports_a_bad_input(tmp_path, capsys):
    assert bq.main(["--input", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o.jsonl")]) == 1
    assert "cannot read" in capsys.readouterr().err
    empty = tmp_path / "e.json"
    empty.write_text("[]")
    assert bq.main(["--input", str(empty), "--out", str(tmp_path / "o.jsonl")]) == 1


def test_cli_rejects_a_bad_tier_mix(tmp_path):
    with pytest.raises(SystemExit):
        bq.main(["--input", "x", "--tier-mix", "1,2"])


# ----------------------------------------------------------------- failure grouping


def test_group_drops_collapses_quoted_names_and_sorts_by_size():
    dropped = {
        "bird-1": 'gold SQL failed: database "card_games" does not exist',
        "bird-2": 'gold SQL failed: database "financial" does not exist',
        "bird-3": 'gold SQL failed: database "card_games" does not exist',
        "bird-4": "gold SQL returns no rows",
    }
    groups = bq.group_drops(dropped)
    assert groups[0] == ('gold SQL failed: database "..." does not exist', ["bird-1", "bird-2", "bird-3"])
    assert groups[1] == ("gold SQL returns no rows", ["bird-4"])


def test_systematic_failure_needs_a_shared_error_covering_half():
    same = {f"bird-{i}": f'database "d{i}" does not exist' for i in range(6)}
    assert bq.systematic_failure(same, 10) == 'database "..." does not exist'
    assert bq.systematic_failure(same, 13) is None                       # under half of all candidates
    mixed = {f"bird-{i}": f"error {chr(97 + i)}" for i in range(6)}
    assert bq.systematic_failure(mixed, 6) is None                       # many different reasons
    empty = {f"bird-{i}": "gold SQL returns no rows" for i in range(6)}
    assert bq.systematic_failure(empty, 6) is None                       # empty results are a data matter
    assert bq.systematic_failure({}, 0) is None


def test_cli_aborts_without_writing_when_every_gold_query_fails(tmp_path, capsys, monkeypatch):
    from schemagraph.eval import run_eval

    def all_fail(questions):
        return SimpleNamespace(results={}, errors={q.id: f'connection failed: database "{q.db_id}" does not exist'
                                                   for q in questions}, empty=[])

    monkeypatch.setattr(run_eval, "check_gold", all_fail)
    src, out = tmp_path / "mini.json", tmp_path / "q.jsonl"
    src.write_text(json.dumps(sample_items(10)))
    assert bq.main(["--input", str(src), "--out", str(out), "--verify"]) == 2
    captured = capsys.readouterr()
    assert "ABORTED" in captured.err and 'database "..." does not exist' in captured.err
    assert captured.out.count("dropped 30") == 1                         # one grouped line, not thirty
    assert not out.exists()
