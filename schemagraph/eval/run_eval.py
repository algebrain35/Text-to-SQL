"""Run the evaluation questions through the pipeline and score the answers (SPEC 5.20).

For every question and every mode this asks `answer_question` for SQL, runs the
gold SQL once, and compares the two results (compare.py). It also records which
tables each mode gave the model (schema_metrics.py). One JSONL row per question per
mode is appended to the output file, and the file is written as the run goes, so an
interrupted run can be resumed with the same --run-id: finished rows are skipped.

    python -m schemagraph.eval.run_eval --questions schemagraph/eval/questions/adventureworks.jsonl \\
        --modes full retrieval graph --out results/2026-11-12-a.jsonl --run-id 2026-11-12-a \\
        --llm-model <model>

    python -m schemagraph.eval.run_eval --questions ... --check-gold     # validate gold SQL only
    python -m schemagraph.eval.run_eval --summarize results/2026-11-12-a.jsonl

Gold SQL is checked before anything is sent to the model, so a broken question costs
nothing. The model is never asked to explain (`explain=False`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from schemagraph.eval.compare import results_match
from schemagraph.eval.schema_metrics import bridge_tables_needed, schema_recall, tables_in_sql
from schemagraph.graph.vector import RateLimited
from schemagraph.models import (
    AskOptions,
    AskRequest,
    AskResponse,
    EvalResult,
    Mode,
    QueryResult,
    Question,
    display_name,
)

log = logging.getLogger(__name__)

GOLD_ROW_LIMIT = 100_000  # large enough that neither the gold nor the generated result is truncated
MAX_RATE_LIMITED_IN_A_ROW = 3  # embedding quota refusals before a run gives up (rows already written are kept)
EXIT_QUOTA = 3

AnswerFn = Callable[..., AskResponse]
QueryFn = Callable[[str, str, int], QueryResult]


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def load_questions(path: Path, split: str = "test") -> list[Question]:
    """Read a JSONL question file and keep one split ('test', 'dev', or 'all').

    Blank lines are skipped. A bad line raises ValueError naming the file and line.
    """
    if split not in ("test", "dev", "all"):
        raise ValueError(f"split must be 'test', 'dev' or 'all', got {split!r}")
    questions: list[Question] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = Question.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if split == "all" or question.split == split:
            questions.append(question)
    return questions


def load_question_files(paths: list[Path], split: str = "test") -> list[Question]:
    """Questions from several files, in order. Ids must be unique across all of them."""
    questions = [q for path in paths for q in load_questions(path, split)]
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id!r}")
        seen.add(q.id)
    return questions


# ---------------------------------------------------------------------------
# Gold SQL
# ---------------------------------------------------------------------------


@dataclass
class GoldCheck:
    results: dict[str, QueryResult] = field(default_factory=dict)  # question id -> result
    errors: dict[str, str] = field(default_factory=dict)           # question id -> why it cannot be used
    empty: list[str] = field(default_factory=list)                 # ids whose gold returns no rows (weak tests)


def _default_query(db_id: str, sql: str, row_limit: int) -> QueryResult:
    from schemagraph.db import postgres  # imported here so the module loads without a database

    return postgres.run_readonly_query(db_id, sql, row_limit)


def check_gold(questions: list[Question], run_query: QueryFn | None = None) -> GoldCheck:
    """Run every gold query once. A query that fails, or returns too many rows to compare, is an error."""
    run_query = run_query or _default_query
    check = GoldCheck()
    for q in questions:
        try:
            result = run_query(q.db_id, q.gold_sql, GOLD_ROW_LIMIT)
        except Exception as exc:  # noqa: BLE001 - any failure makes the question unusable
            check.errors[q.id] = f"gold SQL failed: {exc}"
            continue
        if result.truncated:
            check.errors[q.id] = f"gold SQL returns more than {GOLD_ROW_LIMIT} rows"
            continue
        check.results[q.id] = result
        if result.row_count == 0:
            check.empty.append(q.id)
    return check


# ---------------------------------------------------------------------------
# Scoring (pure)
# ---------------------------------------------------------------------------


def context_tables(response: AskResponse) -> list[str]:
    """Display names of the tables the model was given. The join plan wins; otherwise the explanation's list."""
    if response.join_plan is not None and response.join_plan.tables:
        return sorted(display_name(uid) for uid in response.join_plan.tables)
    if response.explanation is not None:
        return sorted(display_name(t.uid) for t in response.explanation.tables)
    return []


def seed_tables(response: AskResponse) -> set[str] | None:
    """Tables retrieval returned, or None when this response does not say.

    Graph mode has them in the join plan. Retrieval mode's context is its seeds.
    Full mode has no seeds, so its bridge metrics are left empty.
    """
    if response.join_plan is not None and response.join_plan.seed_tables:
        return {display_name(s.uid) for s in response.join_plan.seed_tables}
    if response.explanation is not None:
        seeds = {display_name(t.uid) for t in response.explanation.tables if t.role == "seed"}
        if seeds:
            return seeds
    if response.mode == "retrieval":
        return set(context_tables(response))
    return None


def score(
    question: Question,
    response: AskResponse,
    gold: QueryResult,
    *,
    run_id: str,
    llm_model: str,
    timestamp: datetime,
) -> EvalResult:
    """Turn one pipeline response into one result row."""
    gold_tables = tables_in_sql(question.gold_sql)
    context = set(context_tables(response))
    seeds = seed_tables(response)
    needed = bridge_tables_needed(gold_tables, seeds) if seeds is not None else set()
    correct = (
        response.status == "success"
        and response.result is not None
        and results_match(gold, response.result, question.order_matters)
    )
    return EvalResult(
        question_id=question.id,
        db_id=question.db_id,
        mode=response.mode,
        tier=question.tier,
        correct=correct,
        status=response.status,
        generated_sql=response.sql,
        gold_tables=sorted(gold_tables),
        context_tables=sorted(context),
        schema_recall=schema_recall(gold_tables, context),
        bridge_tables_needed=sorted(needed),
        bridge_tables_recovered=sorted(needed & context),
        plan_status=response.join_plan.status if response.join_plan is not None else None,
        input_tokens=response.metrics.input_tokens,
        latency_ms=response.metrics.latency_ms.total,
        attempt_count=response.metrics.attempt_count,
        llm_model=llm_model,
        run_id=run_id,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    run_id: str
    written: int = 0
    already_done: int = 0
    skipped_bad_gold: dict[str, str] = field(default_factory=dict)
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (question id, mode, message)
    stopped: str | None = None  # why the run ended early (embedding quota), else None


class GoldError(RuntimeError):
    """Some gold queries cannot be used and --skip-bad-gold was not given."""


def _default_answer() -> AnswerFn:
    try:
        from schemagraph.pipeline.pipeline import answer_question
    except ImportError as exc:
        raise RuntimeError("schemagraph.pipeline.pipeline.answer_question is not available yet") from exc
    return answer_question


def load_results(path: Path, run_id: str | None = None) -> list[EvalResult]:
    results = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = EvalResult.model_validate_json(line)
            if run_id is None or row.run_id == run_id:
                results.append(row)
    return results


def question_text(q: Question, use_evidence: bool = True) -> str:
    """What the pipeline is asked: the question, with BIRD's hint appended when there is one."""
    return f"{q.question}\nHint: {q.evidence}" if use_evidence and q.evidence else q.question


def warm_query_cache(questions: list[Question], use_evidence: bool = True, embedder: Any = None) -> int:
    """Embed every question in a few batched requests before the run, so it never pays per question.

    Returns how many were embedded (0 if all were cached or no `embedding.query_cache` is set).
    """
    if embedder is None:
        from schemagraph.graph import vector

        embedder = vector.get_embedder()
    prefetch = getattr(embedder, "prefetch_queries", None)
    return prefetch([question_text(q, use_evidence) for q in questions]) if prefetch else 0


def run_eval(
    question_files: list[Path],
    modes: list[Mode],
    out_path: Path,
    run_id: str,
    planner_options: dict[str, Any] | None = None,
    *,
    llm_model: str = "unspecified",
    split: str = "test",
    use_evidence: bool = True,
    skip_bad_gold: bool = False,
    top_k: int | None = None,
    expand: dict[str, int] | None = None,
    answer: AnswerFn | None = None,
    run_query: QueryFn | None = None,
    now: Callable[[], datetime] | None = None,
) -> RunReport:
    """Score every question in every mode and append the rows to `out_path`.

    planner_options (for the ablations, e.g. {"include_inferred": False, "weighted": False})
    is passed to `answer_question` as the keyword `planner_options` when it is not empty;
    the pipeline must accept it. With use_evidence, a BIRD question's hint is appended to
    the question text. `top_k` and `expand` ({"max_added": N, "hops": H}) override the config's retrieval
    settings for the whole run (None keeps the config); `run_sweep` varies them. Raises GoldError, before any model call, if a gold query is
    unusable and skip_bad_gold is False.
    """
    for mode in modes:
        if mode not in get_args(Mode):
            raise ValueError(f"unknown mode {mode!r}; expected one of {get_args(Mode)}")
    now = now or (lambda: datetime.now(timezone.utc))
    questions = load_question_files(question_files, split)
    report = RunReport(run_id=run_id)

    gold = check_gold(questions, run_query)
    if gold.errors:
        if not skip_bad_gold:
            lines = "\n".join(f"  {qid}: {why}" for qid, why in gold.errors.items())
            raise GoldError(f"{len(gold.errors)} gold queries cannot be used:\n{lines}")
        report.skipped_bad_gold = dict(gold.errors)
    questions = [q for q in questions if q.id in gold.results]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r.question_id, r.mode) for r in load_results(out_path, run_id)} if out_path.exists() else set()
    answer = answer or _default_answer()

    limited = 0
    with out_path.open("a", encoding="utf-8") as out:
        for q in questions:
            text = question_text(q, use_evidence)
            for mode in modes:
                if (q.id, mode) in done:
                    report.already_done += 1
                    continue
                request = AskRequest(
                    question=text, db_id=q.db_id, mode=mode,
                    options=AskOptions(row_limit=GOLD_ROW_LIMIT, top_k=top_k),
                )
                try:
                    # A dev question is also a few-shot example: never show the model the one it is being asked.
                    kwargs: dict[str, Any] = {"exclude_example_ids": frozenset({q.id})}
                    if planner_options:
                        kwargs["planner_options"] = planner_options
                    if expand is not None:
                        kwargs["expand"] = expand
                    response = answer(request, explain=False, **kwargs)
                    row = score(q, response, gold.results[q.id], run_id=run_id, llm_model=llm_model, timestamp=now())
                except Exception as exc:  # noqa: BLE001 - one bad call must not end a long run
                    log.error("%s [%s] failed: %s", q.id, mode, exc)
                    report.errors.append((q.id, mode, str(exc)))
                    limited = limited + 1 if isinstance(exc, RateLimited) else 0
                    if limited >= MAX_RATE_LIMITED_IN_A_ROW:
                        report.stopped = (f"the embedding service refused {limited} questions in a row (quota); stopping "
                                          f"instead of failing the rest. Rerun the same command later to resume")
                        return report
                    continue  # not written, so a resumed run tries it again
                limited = 0
                out.write(row.model_dump_json(by_alias=True) + "\n")
                out.flush()
                report.written += 1
    return report


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def aggregate(results: list[EvalResult]) -> list[dict[str, Any]]:
    """One row per mode and tier, plus an 'all' row per mode. Pure, so it needs no pandas."""
    groups: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[(r.mode, str(r.tier))].append(r)
        groups[(r.mode, "all")].append(r)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    rows = []
    for (mode, tier), rs in sorted(groups.items()):
        needed = sum(len(r.bridge_tables_needed) for r in rs)
        recovered = sum(len(r.bridge_tables_recovered) for r in rs)
        rows.append(
            {
                "mode": mode,
                "tier": tier,
                "n": len(rs),
                "accuracy": mean([r.correct for r in rs]),
                "schema_recall": mean([r.schema_recall for r in rs]),
                "bridge_needed": needed,
                "bridge_recovery": recovered / needed if needed else None,
                "attempts": mean([r.attempt_count for r in rs]),
                "input_tokens": mean([r.input_tokens for r in rs]),
                "latency_ms": mean([r.latency_ms for r in rs]),
            }
        )
    return rows


def summarize(results_path: Path, run_id: str | None = None):
    """Accuracy and context metrics by mode and tier, as a pandas DataFrame (needs the `eval` extra)."""
    import pandas as pd

    return pd.DataFrame(aggregate(load_results(results_path, run_id)))


def format_summary(rows: list[dict[str, Any]]) -> str:
    header = f"{'mode':<10}{'tier':<6}{'n':>5}{'accuracy':>10}{'recall':>8}{'bridges':>9}{'recovered':>11}{'tries':>7}{'tokens':>9}{'ms':>8}"
    lines = [header]
    for r in rows:
        recovery = "-" if r["bridge_recovery"] is None else f"{r['bridge_recovery']:.0%}"
        lines.append(
            f"{r['mode']:<10}{r['tier']:<6}{r['n']:>5}{r['accuracy']:>10.0%}{r['schema_recall']:>8.0%}"
            f"{r['bridge_needed']:>9}{recovery:>11}{r['attempts']:>7.1f}{r['input_tokens']:>9.0f}{r['latency_ms']:>8.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sweeps: the same questions under several retrieval settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One retrieval setting: the `top_k` seeds, plus up to `added` tables they reference, `hops` links away."""

    top_k: int
    added: int = 0
    hops: int = 1

    @property
    def label(self) -> str:
        return f"k{self.top_k}" + (f"+{self.added}h{self.hops}" if self.added else "")

    @property
    def expand(self) -> dict[str, int]:
        return {"max_added": self.added, "hops": self.hops}  # added = 0 switches expansion off


def parse_variant(spec: str) -> Variant:
    """'5' -> top_k 5; '8+4' -> top_k 8, add up to 4 tables one link away; '8+4:2' -> two links away."""
    head, plus, rest = spec.partition("+")
    added_text, colon, hops_text = rest.partition(":")
    try:
        if (plus and not added_text) or (colon and not hops_text):
            raise ValueError(spec)
        variant = Variant(int(head), int(added_text) if added_text else 0, int(hops_text) if hops_text else 1)
    except ValueError:
        raise ValueError(f"bad variant {spec!r}; use K, K+N or K+N:H (for example 5, 8+4, 8+4:2)") from None
    if variant.top_k < 1 or variant.added < 0 or variant.hops < 1:
        raise ValueError(f"bad variant {spec!r}: top_k and hops must be at least 1, added at least 0")
    return variant


def run_sweep(
    variants: list[Variant], question_files: list[Path], modes: list[Mode], out_path: Path, run_id: str, **kwargs: Any
) -> list[tuple[Variant, RunReport]]:
    """`run_eval` once per variant, each under the run id '<run_id>-<label>', all into one results file.

    Finished rows are skipped on a rerun, so an interrupted sweep resumes. Gold queries are checked each time
    (cheap); a model call is made only for rows not yet written.
    """
    reports: list[tuple[Variant, RunReport]] = []
    for v in variants:
        report = run_eval(question_files, modes, out_path, f"{run_id}-{v.label}", top_k=v.top_k, expand=v.expand, **kwargs)
        reports.append((v, report))
        if report.stopped:
            break  # the quota is gone for every later variant too
    return reports


def aggregate_runs(results: list[EvalResult]) -> list[dict[str, Any]]:
    """One row per run id and mode: accuracy, recall and how big the context was. Pure."""
    groups: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[(r.run_id, r.mode)].append(r)
    rows = []
    for (run, mode), rs in sorted(groups.items()):
        n = len(rs)
        rows.append({
            "run_id": run, "mode": mode, "n": n,
            "accuracy": sum(r.correct for r in rs) / n,
            "schema_recall": sum(r.schema_recall for r in rs) / n,
            "context_tables": sum(len(r.context_tables) for r in rs) / n,
            "attempts": sum(r.attempt_count for r in rs) / n,
            "input_tokens": sum(r.input_tokens for r in rs) / n,
            "latency_ms": sum(r.latency_ms for r in rs) / n,
        })
    return rows


def format_runs(rows: list[dict[str, Any]]) -> str:
    width = max([8, *(len(r["run_id"]) + 2 for r in rows)])
    lines = [f"{'run':<{width}}{'mode':<10}{'n':>5}{'accuracy':>10}{'recall':>8}{'tables':>8}{'tries':>7}{'tokens':>9}{'ms':>8}"]
    for r in rows:
        lines.append(
            f"{r['run_id']:<{width}}{r['mode']:<10}{r['n']:>5}{r['accuracy']:>10.1%}{r['schema_recall']:>8.0%}"
            f"{r['context_tables']:>8.1f}{r['attempts']:>7.2f}{r['input_tokens']:>9.0f}{r['latency_ms']:>8.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation questions and score them.")
    parser.add_argument("--questions", nargs="+", type=Path, help="question JSONL file(s)")
    parser.add_argument("--modes", nargs="+", default=list(get_args(Mode)), choices=list(get_args(Mode)))
    parser.add_argument("--out", type=Path, help="results JSONL (appended to; resumable with the same --run-id)")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M"))
    parser.add_argument("--llm-model", default="unspecified", help="recorded in every row")
    parser.add_argument("--split", default="test", choices=["test", "dev", "all"])
    parser.add_argument("--skip-bad-gold", action="store_true", help="drop questions whose gold SQL fails")
    parser.add_argument("--no-evidence", action="store_true", help="do not append BIRD hints to the question")
    parser.add_argument("--no-inferred", action="store_true", help="ablation: planner ignores inferred edges")
    parser.add_argument("--unweighted", action="store_true", help="ablation: planner treats every edge as cost 1")
    parser.add_argument("--check-gold", action="store_true", help="only run the gold SQL and report problems")
    parser.add_argument("--summarize", type=Path, help="print the summary of a results file and exit")
    parser.add_argument("--compare-runs", type=Path, metavar="RESULTS", help="print one row per run id and mode of a results file and exit")
    parser.add_argument("--top-k", type=int, help="override pipeline.top_k for this run")
    parser.add_argument("--expand", help="override pipeline.expand_seeds / expand_hops for this run: N or N:H (0 = off)")
    parser.add_argument("--sweep", nargs="+", metavar="VARIANT", help="run the questions once per variant K, K+N or K+N:H "
                        "(top_k, tables to add, links to look), under run ids '<run-id>-k<K>+<N>h<H>', and compare them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.summarize:
        print(format_summary(aggregate(load_results(args.summarize))))
        return 0
    if args.compare_runs:
        print(format_runs(aggregate_runs(load_results(args.compare_runs))))
        return 0
    if not args.questions:
        parser.error("--questions is required")

    if args.check_gold:
        questions = load_question_files(args.questions, args.split)
        check = check_gold(questions)
        print(f"{len(check.results)} of {len(questions)} gold queries usable")
        for qid, why in check.errors.items():
            print(f"  ERROR {qid}: {why}")
        for qid in check.empty:
            print(f"  WARN  {qid}: gold returns no rows, so it barely tests anything")
        return 1 if check.errors else 0

    if not args.out:
        parser.error("--out is required")
    options: dict[str, Any] = {}
    if args.no_inferred:
        options["include_inferred"] = False
    if args.unweighted:
        options["weighted"] = False
    common: dict[str, Any] = {
        "llm_model": args.llm_model, "split": args.split, "use_evidence": not args.no_evidence,
        "skip_bad_gold": args.skip_bad_gold,
    }
    try:
        if set(args.modes) - {"full"}:   # only these embed the question
            try:
                fetched = warm_query_cache(load_question_files(args.questions, args.split), not args.no_evidence)
            except RateLimited as exc:
                print(f"STOPPED: the embedding service is rate limited, nothing was run ({str(exc)[:300]}). "
                      f"Rerun the same command later.", file=sys.stderr)
                return EXIT_QUOTA
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - not fatal: the run reports its own errors
                print(f"note: could not embed the questions in advance ({type(exc).__name__}: {str(exc)[:200]}); "
                      f"embedding them one at a time", file=sys.stderr)
            else:
                if fetched:
                    print(f"embedded {fetched} new questions in advance (cached in embedding.query_cache)")
        if args.sweep:
            variants = [parse_variant(spec) for spec in args.sweep]
            n = len(load_question_files(args.questions, args.split))
            print(f"{n} questions x {len(args.modes)} modes x {len(variants)} variants: up to "
                  f"{n * len(args.modes) * len(variants)} model calls (rows already in {args.out} are skipped)")
            reports = run_sweep(variants, args.questions, args.modes, args.out, args.run_id,
                                planner_options=options or None, **common)
            errors = [e for _, rep in reports for e in rep.errors]
            for v, rep in reports:
                print(f"{v.label}: wrote {rep.written}, already done {rep.already_done}, errors {len(rep.errors)}")
            prefix = f"{args.run_id}-"
            rows = [r for r in aggregate_runs(load_results(args.out)) if r["run_id"].startswith(prefix)]
            print(format_runs(rows))
            stopped = next((rep.stopped for _, rep in reports if rep.stopped), None)
            if stopped:
                print(f"STOPPED: {stopped}", file=sys.stderr)
                return EXIT_QUOTA
            return 1 if errors else 0
        expand = None
        if args.expand is not None:
            added, _, hops = args.expand.partition(":")
            expand = {"max_added": int(added), "hops": int(hops) if hops else 1}
        report = run_eval(
            args.questions, args.modes, args.out, args.run_id, options or None,
            top_k=args.top_k, expand=expand, **common,
        )
    except (GoldError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"run {report.run_id}: wrote {report.written}, already done {report.already_done}, "
          f"bad gold {len(report.skipped_bad_gold)}, errors {len(report.errors)}")
    print(format_summary(aggregate(load_results(args.out, args.run_id))))
    if report.stopped:
        print(f"STOPPED: {report.stopped}", file=sys.stderr)
        return EXIT_QUOTA
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
