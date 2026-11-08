"""Smoke test: push a few questions through the real services and show what comes back.

    python scripts/smoke_test.py --db adventureworks
    python scripts/smoke_test.py --db adventureworks --question "How many employees are there?"
    python scripts/smoke_test.py --db card_games --questions-file schemagraph/eval/questions/bird_subset.jsonl --n 5
    python scripts/smoke_test.py --db adventureworks --no-llm          # retrieval and join planning only
    python scripts/smoke_test.py --db adventureworks --full-sql        # whole SQL, not a clipped line

For each question and mode (full, retrieval, graph) it prints the tables retrieval found with their
scores, the join plan, the tables the model was given, and, when an LLM is configured, the SQL, whether
it ran, and its cost. With a questions file (which carries gold SQL) it also says whether the result
matches the gold result and what share of the gold tables the model was given.

This is not the evaluation: it runs a handful of questions once, prints everything, and stores nothing.
Its job is to show, before a long run, whether the pieces work together: that Gemini retrieval returns
sensible tables, that the planner connects them, and that the model's SQL runs.

Without an `llm:` section in config.yml it runs retrieval and planning only. A stage that fails is
reported on that question and the run goes on. Exit code: 0 if nothing failed, 1 if any question hit an
error, 2 if the run could not start.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemagraph.models import (
    AskRequest,
    AskResponse,
    Mode,
    QueryResult,
    SchemaContext,
    SeedTable,
    display_name,
)

MODES: tuple[Mode, ...] = ("full", "retrieval", "graph")

DEFAULT_QUESTIONS = {
    "adventureworks": [
        "How many employees are there?",
        "What are the names of the 5 most expensive products?",
        "What is the total sales amount per sales territory?",
        "List each customer's id with the number of orders they have placed.",
        "Which vendors supply products in the Bikes category?",
    ]
}


@dataclass
class SmokeQuestion:
    text: str
    qid: str | None = None
    gold_sql: str | None = None
    order_matters: bool = False


@dataclass
class Row:
    """What one question did in one mode."""

    question: str
    mode: Mode
    qid: str | None = None
    seeds: list[tuple[str, float | None, str]] = field(default_factory=list)  # (table, score or None, matched_on)
    tables: list[str] = field(default_factory=list)                     # what the model was given
    plan: str | None = None                                             # graph mode only
    status: str | None = None                                           # pipeline status; None without an LLM
    sql: str | None = None
    rows: int | None = None
    correct: bool | None = None                                         # only with gold SQL and an LLM
    recall: float | None = None                                         # only with gold SQL
    ms: int = 0
    tokens: int = 0
    attempts: int = 0
    error: str | None = None                                            # a stage that raised

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def plan_summary(plan: Any) -> str | None:
    if plan is None:
        return None
    parts = [f"{plan.status} via {plan.algorithm}", f"{len(plan.joins)} joins"]
    if plan.bridge_tables:
        parts.append("bridges: " + ", ".join(sorted(display_name(t) for t in plan.bridge_tables)))
    if plan.unreachable_seeds:
        parts.append("unreachable: " + ", ".join(sorted(display_name(t) for t in plan.unreachable_seeds)))
    if plan.dropped_seeds:
        parts.append("dropped: " + ", ".join(sorted(display_name(t) for t in plan.dropped_seeds)))
    return "; ".join(parts)


def seed_rows(seeds: list[SeedTable]) -> list[tuple[str, float | None, str]]:
    return [(display_name(s.uid), s.score, s.matched_on) for s in seeds]


def pick_questions(
    db_id: str,
    texts: list[str],
    questions: list[Any],
    n: int,
) -> list[SmokeQuestion]:
    """Questions from --question, else from the questions file (first n for this db), else the built-in list."""
    if texts:
        return [SmokeQuestion(t) for t in texts]
    if questions:
        chosen = [q for q in questions if q.db_id == db_id][:n]
        return [
            SmokeQuestion(
                text=f"{q.question}\nHint: {q.evidence}" if q.evidence else q.question,
                qid=q.id,
                gold_sql=q.gold_sql,
                order_matters=q.order_matters,
            )
            for q in chosen
        ]
    return [SmokeQuestion(t) for t in DEFAULT_QUESTIONS.get(db_id, [])[:n]]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

ContextFn = Callable[..., tuple[SchemaContext, list[SeedTable]]]
AnswerFn = Callable[..., AskResponse]
QueryFn = Callable[[str, str, int], QueryResult]


def run_one(
    db_id: str,
    q: SmokeQuestion,
    mode: Mode,
    *,
    use_llm: bool,
    top_k: int,
    build_context: ContextFn,
    answer: AnswerFn,
    gold: QueryResult | None = None,
    expand: dict[str, int] | None = None,
) -> Row:
    """One question in one mode. Never raises: an exception becomes `row.error`."""
    from schemagraph.eval import run_eval
    from schemagraph.eval.compare import results_match
    from schemagraph.eval.schema_metrics import schema_recall, tables_in_sql

    row = Row(question=q.text, mode=mode, qid=q.qid)
    started = time.perf_counter()
    stage = "pipeline" if use_llm else "context"
    try:
        if use_llm:
            response = answer(AskRequest(question=q.text, db_id=db_id, mode=mode), explain=False)
            context = set(run_eval.context_tables(response))
            row.tables = sorted(context)
            if response.join_plan is not None:
                row.seeds = seed_rows(response.join_plan.seed_tables)
            elif response.explanation is not None:  # retrieval mode: the explanation knows the seeds, not their scores
                row.seeds = [(t.display_name, None, "") for t in response.explanation.tables if t.role == "seed"]
            row.plan = plan_summary(response.join_plan)
            row.status, row.sql = response.status, response.sql
            row.rows = response.result.row_count if response.result is not None else None
            row.tokens = response.metrics.input_tokens + response.metrics.output_tokens
            row.attempts = response.metrics.attempt_count
            if gold is not None:
                row.correct = (
                    response.status == "success"
                    and response.result is not None
                    and results_match(gold, response.result, q.order_matters)
                )
        else:
            context_obj, seeds_list = build_context(db_id, q.text, mode, top_k, **({"expand": expand} if expand else {}))
            context = {display_name(t) for t in context_obj.tables}
            row.tables = sorted(context)
            row.seeds = seed_rows(seeds_list)
            row.plan = plan_summary(context_obj.join_plan)
        stage = "scoring"
        if q.gold_sql is not None:
            row.recall = schema_recall(tables_in_sql(q.gold_sql), set(row.tables))
    except Exception as exc:  # noqa: BLE001 - reported on the row; one failure must not stop the run
        row.error = f"{stage}: {type(exc).__name__}: {exc}"
    row.ms = round((time.perf_counter() - started) * 1000)
    return row


def run_smoke(
    db_id: str,
    questions: list[SmokeQuestion],
    modes: list[Mode],
    *,
    use_llm: bool,
    top_k: int,
    build_context: ContextFn,
    answer: AnswerFn,
    run_query: QueryFn,
    expand: dict[str, int] | None = None,
) -> list[Row]:
    from schemagraph.eval.run_eval import GOLD_ROW_LIMIT

    rows: list[Row] = []
    for q in questions:
        gold = None
        if q.gold_sql is not None and use_llm:
            try:
                gold = run_query(db_id, q.gold_sql, GOLD_ROW_LIMIT)
            except Exception as exc:  # noqa: BLE001 - a bad gold query is reported, the question still runs
                rows.append(Row(q.text, "full", q.qid, error=f"gold SQL failed: {exc}"))
        for mode in modes:
            rows.append(run_one(db_id, q, mode, use_llm=use_llm, top_k=top_k, build_context=build_context,
                                answer=answer, gold=gold, expand=expand))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _clip(text: str, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def format_rows(rows: list[Row], use_llm: bool, full_sql: bool = False) -> str:
    lines: list[str] = []
    last = None
    for r in rows:
        key = (r.qid, r.question)
        if key != last:
            lines += ["", f"Q{' ' + r.qid if r.qid else ''}: {_clip(r.question, 160)}"]
            last = key
        head = f"  [{r.mode:<9}]"
        if r.error:
            lines.append(f"{head} ERROR {r.error}")
            continue
        bits = [f"{len(r.tables)} tables"]
        if r.recall is not None:
            bits.append(f"gold-table recall {r.recall:.0%}")
        if use_llm:
            bits.append(f"status {r.status}")
            if r.correct is not None:
                bits.append("MATCHES gold" if r.correct else "differs from gold")
            elif r.rows is not None:
                bits.append(f"{r.rows} rows")
            bits.append(f"{r.attempts} attempt(s), {r.tokens} tokens")
        bits.append(f"{r.ms} ms")
        lines.append(f"{head} " + ", ".join(bits))
        if r.seeds:
            lines.append("      retrieved: " + ", ".join(n if sc is None else f"{n} ({sc:.3f})" for n, sc, _ in r.seeds))
        if r.plan:
            lines.append(f"      plan: {r.plan}")
        if r.mode != "full" and r.tables:
            lines.append("      given to the model: " + ", ".join(r.tables))
        if r.sql:
            if full_sql:
                lines.append("      sql:")
                lines += [f"        {line}" for line in r.sql.strip().splitlines()]
            else:
                lines.append(f"      sql: {_clip(r.sql)}")
    lines += ["", summary(rows, use_llm)]
    return "\n".join(lines)


def summary(rows: list[Row], use_llm: bool) -> str:
    lines = ["Summary"]
    for mode in MODES:
        mine = [r for r in rows if r.mode == mode]
        if not mine:
            continue
        failed = sum(not r.ok for r in mine)
        text = f"  {mode:<9} {len(mine)} runs, {failed} errors"
        recalls = [r.recall for r in mine if r.recall is not None]
        if recalls:
            text += f", mean gold-table recall {sum(recalls) / len(recalls):.0%}"
        if use_llm:
            ran = sum(r.status == "success" for r in mine)
            text += f", SQL ran {ran}/{len(mine)}"
            graded = [r.correct for r in mine if r.correct is not None]
            if graded:
                text += f", matches gold {sum(graded)}/{len(graded)}"
        lines.append(text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a few questions through the real pipeline and print what happens.")
    parser.add_argument("--db", required=True, help="db_id")
    parser.add_argument("--question", action="append", default=[], help="a question to ask (repeatable)")
    parser.add_argument("--questions-file", type=Path, help="JSONL questions (with gold SQL) for --db")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"], help="split to use from --questions-file")
    parser.add_argument("--n", type=int, default=5, help="how many questions from the file or built-in list")
    parser.add_argument("--mode", action="append", choices=MODES, help="mode to run (repeatable; default all three)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="retrieval and join planning only")
    parser.add_argument("--full-sql", action="store_true", help="print the whole generated SQL, not one clipped line")
    args = parser.parse_args(argv)

    from schemagraph.config import get_config
    from schemagraph.eval import run_eval
    from schemagraph.graph import driver

    try:
        loaded = run_eval.load_questions(args.questions_file, args.split) if args.questions_file else []
        questions = pick_questions(args.db, args.question, loaded, args.n)
        if not questions:
            print(f"no questions for {args.db!r}: pass --question, or --questions-file with questions for it")
            return 2
        config = get_config()
        use_llm = not args.no_llm and config.llm is not None
        if not args.no_llm and config.llm is None:
            print("note: no `llm:` section in config.yml, so this run covers retrieval and join planning only\n")
        driver.check_connection()

        from schemagraph import db
        from schemagraph.pipeline import context, pipeline

        rows = run_smoke(
            args.db, questions, list(args.mode or MODES), use_llm=use_llm, top_k=args.top_k,
            build_context=context.build_context, answer=pipeline.answer_question, run_query=db.run_readonly_query,
            expand=context.expand_options(getattr(config, "pipeline", None)),
        )
    except Exception as exc:  # noqa: BLE001 - Neo4j, Postgres or config unusable: nothing to show
        print(f"cannot start: {type(exc).__name__}: {exc}")
        return 2
    finally:
        driver.close_driver()

    print(format_rows(rows, use_llm, full_sql=args.full_sql))
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
