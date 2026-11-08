"""Checks on a set of evaluation questions, shared by every question file.

    python -m schemagraph.eval.question_set schemagraph/eval/questions/adventureworks.jsonl \\
        schemagraph/eval/questions/bird_subset.jsonl \\
        --fewshot schemagraph/eval/questions/fewshot_examples.jsonl

Prints how the questions are spread over databases, tiers and splits, and lists
anything that would corrupt an evaluation: duplicate ids, a gold query that is not a
single read-only SELECT, the same question twice, a bad db_id, and (given the
few-shot file) any question that also appears as a few-shot example, which would let
the model see the answer. Development questions may also be few-shot examples (they are not
reported), so only test questions are checked; every few-shot example must itself be a development
question. With --verify it also runs every gold query against the databases and says which fail or
return nothing.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from schemagraph.config import DB_ID_RE
from schemagraph.models import Question
from schemagraph.sql_validation import check_select


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace, so trivial rewrites still count as the same question."""
    return re.sub(r"\s+", " ", text).strip().lower()


def problems(questions: list[Question], fewshot: list[Question] | None = None) -> list[str]:
    """Everything wrong with the set, one sentence each. Empty means usable."""
    found: list[str] = []
    ids = Counter(q.id for q in questions)
    found += [f"duplicate id {i!r} ({n} times)" for i, n in ids.items() if n > 1]

    seen_text: dict[tuple[str, str], str] = {}
    for q in questions:
        if not q.question.strip():
            found.append(f"{q.id}: empty question")
        if not DB_ID_RE.match(q.db_id):
            found.append(f"{q.id}: db_id {q.db_id!r} is not lowercase letters, digits and _")
        check = check_select(q.gold_sql)
        if not check.ok:
            found.append(f"{q.id}: gold SQL rejected: {check.reason}")
        key = (q.db_id, normalize_text(q.question))
        if key in seen_text and seen_text[key] != q.id:
            found.append(f"{q.id}: same question as {seen_text[key]} on {q.db_id}")
        seen_text.setdefault(key, q.id)

    if fewshot:
        found += [f"{s.id}: few-shot examples must use split 'dev', not {s.split!r}" for s in fewshot if s.split != "dev"]
        shot_ids = {s.id for s in fewshot}
        shot_text = {(s.db_id, normalize_text(s.question)) for s in fewshot}
        for q in (q for q in questions if q.split == "test"):
            if q.id in shot_ids:
                found.append(f"{q.id}: also in the few-shot examples (same id)")
            elif (q.db_id, normalize_text(q.question)) in shot_text:
                found.append(f"{q.id}: also in the few-shot examples (same question text)")
    return found


def describe(questions: list[Question]) -> str:
    """A small table: databases down the side, tiers and splits across."""
    if not questions:
        return "no questions"
    dbs = sorted({q.db_id for q in questions})
    lines = [f"{'database':<28}{'tier1':>6}{'tier2':>6}{'tier3':>6}{'test':>6}{'dev':>5}{'total':>7}"]
    for db in dbs + ["ALL"]:
        qs = [q for q in questions if db == "ALL" or q.db_id == db]
        tier = Counter(q.tier for q in qs)
        split = Counter(q.split for q in qs)
        lines.append(
            f"{db:<28}{tier[1]:>6}{tier[2]:>6}{tier[3]:>6}{split['test']:>6}{split['dev']:>5}{len(qs):>7}"
        )
    return "\n".join(lines)


def verify_gold(questions: list[Question]) -> tuple[list[str], int]:
    """Run every gold query. Returns report lines (failures grouped by message) and how many failed."""
    from schemagraph.eval.bird_questions import group_drops
    from schemagraph.eval.run_eval import check_gold

    check = check_gold(questions)
    lines = []
    for why, ids in group_drops(check.errors):
        shown = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
        lines.append(f"  PROBLEM {len(ids)} gold queries failed: {why} (e.g. {shown})")
    if check.empty:
        lines.append(f"  note: {len(check.empty)} gold queries return no rows (weak tests): {', '.join(check.empty[:5])}")
    lines.append(f"gold SQL ran for {len(check.results)} of {len(questions)} questions")
    return lines, len(check.errors)


def main(argv: list[str] | None = None) -> int:
    from schemagraph.eval.run_eval import load_question_files

    parser = argparse.ArgumentParser(description="Check evaluation question files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--fewshot", type=Path, help="few-shot examples file; no question may also appear there")
    parser.add_argument("--verify", action="store_true", help="also run every gold query against its database")
    args = parser.parse_args(argv)

    questions = load_question_files(args.files, "all")
    fewshot = load_question_files([args.fewshot], "all") if args.fewshot else None
    print(describe(questions))
    issues = problems(questions, fewshot)
    for issue in issues:
        print(f"  PROBLEM {issue}")
    if not issues:
        print("no problems found")
    failed = 0
    if args.verify:
        lines, failed = verify_gold(questions)
        print("\n".join(lines))
    return 1 if issues or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
