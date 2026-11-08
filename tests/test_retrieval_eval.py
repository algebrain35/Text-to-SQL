import json

from schemagraph.eval import retrieval_eval as re_
from schemagraph.models import SeedTable


def seed(table, score, on="table"):
    return SeedTable(uid=f"db:{table}", score=score, matched_on=on)


class Q:
    def __init__(self, qid, sql, question="q", evidence=None):
        self.id, self.gold_sql, self.question, self.evidence = qid, sql, question, evidence


SEEDS = [seed("sales.order", 0.87), seed("sales.noise", 0.86), seed("sales.customer", 0.85),
         seed("hr.employee", 0.84), seed("hr.other", 0.83)]


class Edge:
    def __init__(self, from_table, to_table, confidence=1.0):
        self.from_table, self.to_table, self.confidence = f"db:{from_table}", f"db:{to_table}", confidence


EDGES = [Edge("sales.order", "sales.customer"), Edge("sales.customer", "person.person"), Edge("hr.employee", "person.person")]


def test_seeds_for_keeps_top_k_and_adds_referenced_tables():
    plain = re_.seeds_for(SEEDS, EDGES, 3, 0, 1)
    assert [s.uid for s in plain] == ["db:sales.order", "db:sales.noise", "db:sales.customer"]
    one_hop = re_.seeds_for(SEEDS[:1], EDGES, 1, 2, 1)       # order -> customer
    assert [s.uid for s in one_hop] == ["db:sales.order", "db:sales.customer"]
    two_hop = re_.seeds_for(SEEDS[:1], EDGES, 1, 3, 2)       # order -> customer -> person
    assert [s.uid for s in two_hop] == ["db:sales.order", "db:sales.customer", "db:person.person"]


def test_ranks_and_recall_at_k():
    r = re_.QuestionResult("a", "q", ["sales.customer", "sales.order", "x.missing"],
                           seeds=[(s.uid.split(":")[1], s.score, s.matched_on) for s in SEEDS])
    assert r.ranks() == {"sales.customer": 3, "sales.order": 1, "x.missing": None}
    assert r.recall_at(1) == 1 / 3 and r.recall_at(3) == 2 / 3 and r.recall_at(20) == 2 / 3


def run(plan=None, gold="SELECT * FROM sales.order o JOIN sales.customer c ON c.id = o.cid", edges=None):
    calls = []

    def search(db_id, text, k):
        calls.append((text, k))
        return SEEDS

    return re_.evaluate("db", [Q("a", gold, "how many", evidence="hint")], search=search, plan=plan, edges=edges), calls


def test_evaluate_retrieves_once_with_k_max_and_appends_evidence():
    results, calls = run()
    assert calls == [("how many\nHint: hint", 20)]
    assert results[0].error is None and results[0].gold == ["sales.customer", "sales.order"]
    assert results[0].graph == {}


def test_evaluate_scores_each_strategy_through_the_planner_and_reuses_equal_seed_sets():
    planned = []

    def plan(db_id, seeds):
        planned.append(tuple(sorted(s.uid for s in seeds)))
        return [s.uid for s in seeds] + ["db:sales.bridge"]

    results, _ = run(plan, edges=EDGES)
    graph = results[0].graph
    assert list(graph)[:2] == ["top_k=3", "top_k=3 +2 (1-hop)"]
    assert graph["top_k=3"] == (1.0, 4)                      # three seeds + the bridge
    assert len(planned) == len(set(planned)) < len(re_.strategies())  # identical seed sets planned once


def test_expansion_can_recover_a_table_retrieval_missed():
    seeds_only = lambda db, seeds: [s.uid for s in seeds]
    gold = "SELECT * FROM sales.order o JOIN person.person p ON 1=1"
    results, _ = run(seeds_only, gold=gold, edges=[Edge("sales.customer", "person.person")])
    graph = results[0].graph
    assert graph["top_k=3"][0] == 0.5                         # person is not in the top 3
    assert graph["top_k=3 +2 (1-hop)"][0] == 1.0              # customer (a seed) references person


def test_a_failing_question_is_reported_not_raised():
    def search(db_id, text, k):
        raise RuntimeError("neo4j down")

    results = re_.evaluate("db", [Q("a", "SELECT * FROM t")], search=search)
    assert results[0].error == "RuntimeError: neo4j down"
    assert re_.summarize(results)["questions"] == 0
    assert "nothing to score" in re_.format_report(results)


def test_report_lists_misses_and_flags_compressed_scores():
    results, _ = run(plan=lambda db, seeds: [s.uid for s in seeds], gold="SELECT * FROM hr.employee e JOIN x.missing m ON 1=1", edges=EDGES)
    text = re_.format_report(results, "demo")
    assert "(demo)" in text and "Questions missing a gold table at k=5: 1 of 1" in text
    assert "x.missing (rank >5)" in text and "hr.employee (rank 4)" not in text
    assert "scores barely separate" in text
    assert "most questions fully covered" in text


def test_json_round_trip_and_comparison():
    results, _ = run()
    a = re_.to_json(results, "db", "one", "all")
    b = re_.to_json(results, "db", "two", "all")
    text = re_.format_comparison([json.loads(json.dumps(a)), json.loads(json.dumps(b))])
    assert "one" in text and "two" in text and "warning" not in text
    b["summary"]["questions"] = 7
    assert "do not compare" in re_.format_comparison([a, b])


def test_cli_compare_reads_saved_runs(tmp_path, capsys):
    results, _ = run()
    paths = []
    for name in ("one", "two"):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(re_.to_json(results, "db", name, "all")))
        paths.append(str(p))
    assert re_.main(["--compare", *paths]) == 0
    assert "one" in capsys.readouterr().out


def test_report_names_the_errors_grouped_by_message():
    def search(db_id, text, k):
        raise RuntimeError("429 slow down") if text != "q2" else KeyError("x")

    qs = [Q("a", "SELECT * FROM t", "q1"), Q("b", "SELECT * FROM t", "q2"), Q("c", "SELECT * FROM t", "q1")]
    text = re_.format_report(re_.evaluate("db", qs, search=search))
    assert "ERROR RuntimeError: 429 slow down (2: a, c)" in text and "ERROR KeyError: 'x' (1: b)" in text
