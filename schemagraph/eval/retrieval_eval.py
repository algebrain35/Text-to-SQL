"""Retrieval-only evaluation: does the right schema reach the model, and how much extra comes with it?

    python scripts/retrieval_eval.py --db adventureworks --questions-file schemagraph/eval/questions/adventureworks.jsonl
    python scripts/retrieval_eval.py --db adventureworks --questions-file ... --split all --out runs/gemini.json --label gemini-1536
    python scripts/retrieval_eval.py --compare runs/openai.json runs/gemini.json

No LLM and no Postgres are involved, so a run is cheap and repeatable: it embeds each question once and
reads Neo4j. For every question with gold SQL it asks three things.

1. Recall@k. The gold SQL names some tables; how many of them are among the k best-scoring tables
   (k = 1, 3, 5, 8, 10, 20)? This is retrieval alone, with no join planning. It also lists the questions
   that miss at k=5, with the rank of each missing gold table, because those are the ones to read.
2. Score spread. Cosine scores from many embedding models are bunched together (0.81 to 0.87 is common), so a
   fixed top_k admits noise that scores almost the same as the right tables. The table of mean scores at
   ranks 1, 5 and 20 shows how much room there is to tell them apart.
3. Seed strategies. The seeds go to the join planner, which adds bridge tables to connect them. Each strategy
   keeps at most `top_k` seeds and optionally adds the tables the seeds have foreign keys to ("expand N,
   H-hop": up to N tables within H links, see `retrieve.expand_seeds`), the person behind a salesperson or the
   product under a category, which the question rarely names. For each strategy it reports graph recall (the
   planner's tables, as the model would see them) and the average number of tables in the context. A good
   strategy keeps recall high without growing the context. (A cut-off on the score gap was tried and dropped:
   cosine scores are too bunched for it to separate relevant from noise.)

To compare embedding models, build one graph per model (or one named vector set per model, `embedding.name`
in config.yml), run this once per model with `SCHEMAGRAPH_CONFIG=<that config>` and `--out`, then `--compare`.
Results are only comparable between runs on the same question file and split.

Exit code: 0 on success, 1 if any question hit an error, 2 if the run could not start.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemagraph.eval.schema_metrics import schema_recall, tables_in_sql
from schemagraph.graph import retrieve
from schemagraph.models import SeedTable, display_name

KS = (1, 3, 5, 8, 10, 20)
SCORE_RANKS = (1, 5, 20)
TOP_KS = (3, 5, 8, 12)
EXPANSIONS: tuple[tuple[int, int], ...] = ((0, 1), (2, 1), (4, 1), (4, 2))  # (max tables added, links to look)
MISS_K = 5          # the cut-off whose misses are listed
MAX_MISSES_SHOWN = 15
MAX_ERROR_GROUPS = 5

SearchFn = Callable[[str, str, int], list[SeedTable]]   # (db_id, question text, k) -> seeds, best first
PlanFn = Callable[[str, list[SeedTable]], list[str]]    # (db_id, seeds) -> table uids the context would hold
Edges = list[Any]                                       # queries.ReferenceEdge-like: from_table, to_table, confidence


@dataclass
class QuestionResult:
    qid: str
    question: str
    gold: list[str]                                    # 'schema.table' names the gold SQL reads
    seeds: list[tuple[str, float, str]] = field(default_factory=list)   # (display name, score, matched_on), best first
    graph: dict[str, tuple[float, int]] = field(default_factory=dict)   # strategy label -> (graph recall, context size)
    error: str | None = None

    def ranks(self) -> dict[str, int | None]:
        """1-based rank of each gold table among the seeds, None if it was not returned."""
        order = {name: i for i, (name, _, _) in enumerate(self.seeds, start=1)}
        return {t: order.get(t) for t in self.gold}

    def recall_at(self, k: int) -> float:
        return schema_recall(set(self.gold), {name for name, _, _ in self.seeds[:k]})


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def strategy_label(top_k: int, added: int = 0, hops: int = 1) -> str:
    return f"top_k={top_k}" + (f" +{added} ({hops}-hop)" if added else "")


def strategies() -> list[tuple[int, int, int]]:
    """(top_k, tables to add, hops) for every strategy, in report order."""
    return [(k, n, h) for k in TOP_KS for n, h in EXPANSIONS]


def seeds_for(seeds: list[SeedTable], edges: Edges, top_k: int, added: int, hops: int) -> list[SeedTable]:
    """The seeds a strategy hands to the planner: the best `top_k`, plus what they reference."""
    return retrieve.expand_seeds(seeds[:top_k], edges, max_added=added, hops=hops)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def score_at_rank(result: QuestionResult, rank: int) -> float | None:
    return result.seeds[rank - 1][1] if len(result.seeds) >= rank else None


def summarize(results: list[QuestionResult]) -> dict[str, Any]:
    """Everything the report and `--out` carry, from the per-question results. Errored questions are left out."""
    ok = [r for r in results if r.error is None and r.gold]
    out: dict[str, Any] = {
        "questions": len(ok),
        "errors": sum(r.error is not None for r in results),
        "recall_at": {str(k): mean([r.recall_at(k) for r in ok]) for k in KS},
        "all_gold_at": {str(k): mean([float(r.recall_at(k) == 1.0) for r in ok]) for k in KS},
        "mean_score_at_rank": {str(n): mean([s for r in ok if (s := score_at_rank(r, n)) is not None]) for n in SCORE_RANKS},
        "strategies": {},
    }
    labels = list(dict.fromkeys(label for r in ok for label in r.graph))  # report order
    for label in labels:
        have = [r for r in ok if label in r.graph]
        out["strategies"][label] = {
            "graph_recall": mean([r.graph[label][0] for r in have]),
            "all_gold": mean([float(r.graph[label][0] == 1.0) for r in have]),
            "context_tables": mean([float(r.graph[label][1]) for r in have]),
            "questions": len(have),
        }
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def question_text(q: Any, use_evidence: bool = True) -> str:
    """The text the pipeline would be given: BIRD hints are appended, as `run_eval` does."""
    return f"{q.question}\nHint: {q.evidence}" if use_evidence and q.evidence else q.question


def evaluate(
    db_id: str,
    questions: list[Any],
    *,
    search: SearchFn,
    plan: PlanFn | None = None,
    edges: Edges | None = None,
    k_max: int = max(KS),
    use_evidence: bool = True,
) -> list[QuestionResult]:
    """Retrieve once per question (k_max seeds), then score every cut-off and strategy from that one list."""
    results: list[QuestionResult] = []
    for q in questions:
        result = QuestionResult(qid=q.id, question=q.question, gold=sorted(tables_in_sql(q.gold_sql)))
        try:
            seeds = search(db_id, question_text(q, use_evidence), k_max)
            result.seeds = [(display_name(s.uid), s.score, s.matched_on) for s in seeds]
            if plan is not None:
                cache: dict[frozenset[str], set[str]] = {}
                for top_k, added, hops in strategies():
                    kept = seeds_for(seeds, edges or [], top_k, added, hops)
                    key = frozenset(s.uid for s in kept)
                    if key not in cache:
                        cache[key] = {display_name(t) for t in plan(db_id, kept)} if kept else set()
                    context = cache[key]
                    result.graph[strategy_label(top_k, added, hops)] = (schema_recall(set(result.gold), context), len(context))
        except Exception as exc:  # noqa: BLE001 - one question failing must not stop the run
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}"


def format_report(results: list[QuestionResult], label: str = "") -> str:
    s = summarize(results)
    lines = [f"Retrieval evaluation{f' ({label})' if label else ''}: {s['questions']} questions"
             + (f", {s['errors']} errored" if s["errors"] else "")]
    failed: dict[str, list[str]] = {}
    for r in results:
        if r.error is not None:
            failed.setdefault(r.error, []).append(r.qid)
    for message, ids in list(failed.items())[:MAX_ERROR_GROUPS]:
        lines.append(f"  ERROR {message} ({len(ids)}: {', '.join(ids[:5])}{', ...' if len(ids) > 5 else ''})")
    if not s["questions"]:
        return "\n".join(lines + ["nothing to score"])

    lines += ["", "Gold tables found among the k best-scoring tables (percent)",
              "  k            " + "".join(f"{k:>7}" for k in KS),
              "  mean recall  " + "".join(f"{_pct(s['recall_at'][str(k)]):>7}" for k in KS),
              "  all found    " + "".join(f"{_pct(s['all_gold_at'][str(k)]):>7}" for k in KS)]

    scores = s["mean_score_at_rank"]
    lines += ["", "Mean score at rank " + ", ".join(
        f"{n}: {scores[str(n)]:.3f}" if scores[str(n)] is not None else f"{n}: n/a" for n in SCORE_RANKS)]
    top, mid = scores.get("1"), scores.get("5")
    if top is not None and mid is not None and top - mid < 0.05:
        lines.append(f"  rank 1 and rank 5 differ by only {top - mid:.3f}: scores barely separate relevant from noise")

    misses = [r for r in results if r.error is None and r.gold and r.recall_at(MISS_K) < 1.0]
    lines += ["", f"Questions missing a gold table at k={MISS_K}: {len(misses)} of {s['questions']}"]
    for r in misses[:MAX_MISSES_SHOWN]:
        ranks = ", ".join(f"{t} (rank {rank if rank else f'>{len(r.seeds)}'})" for t, rank in r.ranks().items()
                          if rank is None or rank > MISS_K)
        top_names = ", ".join(name for name, _, _ in r.seeds[:MISS_K])
        lines += [f"  {r.qid}: {r.question[:90]}", f"      missing: {ranks}", f"      got:     {top_names}"]
    if len(misses) > MAX_MISSES_SHOWN:
        lines.append(f"  ... and {len(misses) - MAX_MISSES_SHOWN} more")

    if s["strategies"]:
        lines += ["", "Seed strategies, after join planning (what the model would be given)",
                  f"  {'strategy':<20}{'graph recall':>13}{'all gold':>10}{'tables in context':>19}"]
        for name, v in s["strategies"].items():
            lines.append(f"  {name:<20}{_pct(v['graph_recall']):>13}{_pct(v['all_gold']):>10}{v['context_tables']:>19.1f}")
        best = max(s["strategies"].items(), key=lambda kv: (kv[1]["all_gold"] or 0, -(kv[1]["context_tables"] or 0)))
        lines.append(f"  most questions fully covered: {best[0]} ({_pct(best[1]['all_gold']).strip()}% with "
                     f"{best[1]['context_tables']:.1f} tables on average)")
    return "\n".join(lines)


def to_json(results: list[QuestionResult], db_id: str, label: str, split: str) -> dict[str, Any]:
    return {
        "label": label, "db_id": db_id, "split": split, "summary": summarize(results),
        "questions": [
            {"id": r.qid, "gold": r.gold, "seeds": r.seeds, "graph": r.graph, "error": r.error} for r in results
        ],
    }


def format_comparison(runs: list[dict[str, Any]]) -> str:
    """Side-by-side recall of several saved runs."""
    names = [r.get("label") or f"run{i + 1}" for i, r in enumerate(runs)]
    width = max(12, *(len(n) + 2 for n in names))
    lines = ["Gold tables found among the k best-scoring tables (mean recall, percent)",
             "  k   " + "".join(f"{n:>{width}}" for n in names)]
    for k in KS:
        cells = "".join(f"{_pct(r['summary']['recall_at'].get(str(k))):>{width}}" for r in runs)
        lines.append(f"  {k:<4}{cells}")
    lines += ["", "Every gold table found (percent of questions)", "  k   " + "".join(f"{n:>{width}}" for n in names)]
    for k in KS:
        cells = "".join(f"{_pct(r['summary']['all_gold_at'].get(str(k))):>{width}}" for r in runs)
        lines.append(f"  {k:<4}{cells}")
    counts = {(r["db_id"], r["split"], r["summary"]["questions"]) for r in runs}
    if len(counts) > 1:
        lines += ["", "warning: these runs scored different databases, splits or question counts; do not compare them"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score retrieval (recall@k, score spread, seed pruning) against gold SQL.")
    parser.add_argument("--db", help="db_id to evaluate")
    parser.add_argument("--questions-file", type=Path, action="append", default=[], help="JSONL questions with gold SQL (repeatable)")
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"], help="questions to score (default all: nothing is trained here)")
    parser.add_argument("--no-graph", action="store_true", help="skip the join-planning (seed strategies) part; recall@k only")
    parser.add_argument("--no-evidence", action="store_true", help="do not append BIRD hints to the question")
    parser.add_argument("--label", default="", help="name for this run in the report and in --compare")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    parser.add_argument("--compare", type=Path, nargs="+", help="print saved --out files side by side and exit")
    args = parser.parse_args(argv)

    if args.compare:
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare]
        print(format_comparison(runs))
        return 0
    if not args.db or not args.questions_file:
        parser.error("--db and at least one --questions-file are required (or use --compare)")

    from schemagraph.eval import run_eval
    from schemagraph.graph import driver

    try:
        questions = [q for q in run_eval.load_question_files(args.questions_file, args.split) if q.db_id == args.db]
        if not questions:
            print(f"no {args.split} questions for {args.db!r} in {', '.join(map(str, args.questions_file))}")
            return 2
        driver.check_connection()

        from schemagraph.graph import planner, retrieve, vector

        embedder = vector.get_embedder()

        def search(db_id: str, text: str, k: int) -> list[SeedTable]:
            return retrieve.retrieve_seed_tables(db_id, text, k, embedder=embedder)

        from schemagraph.graph import queries

        edges = [] if args.no_graph else queries.get_reference_edges(args.db)

        def plan(db_id: str, seeds: list[SeedTable]) -> list[str]:
            joined = planner.plan_joins(db_id, seeds)
            return joined.tables if joined.status != "none" else [s.uid for s in seeds]

        results = evaluate(args.db, questions, search=search, plan=None if args.no_graph else plan, edges=edges,
                           use_evidence=not args.no_evidence)
    except Exception as exc:  # noqa: BLE001 - Neo4j, config or the question files unusable
        print(f"cannot start: {type(exc).__name__}: {exc}")
        return 2
    finally:
        driver.close_driver()

    print(format_report(results, args.label))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_json(results, args.db, args.label, args.split), indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
