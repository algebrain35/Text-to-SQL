"""Build `bird_subset.jsonl` from the BIRD mini-dev PostgreSQL question file.

    python -m schemagraph.eval.bird_questions \\
        --input data/bird/mini_dev_postgresql.json \\
        --out schemagraph/eval/questions/bird_subset.jsonl --verify

mini-dev has 500 questions on 11 databases, each with a question, an expert hint
(`evidence`) and a PostgreSQL gold query. This picks a reproducible subset of them in the
evaluation's own format (`models.Question`), split into `test` (scored) and `dev` (for tuning).

Tiers. The spec's tiers are 1 single table, 2 joins within one schema, 3 cross-schema joins
of 3+ tables. Every BIRD database has one schema, so tier 3 cannot occur as written.
Here the tier is the number of distinct tables in the gold query: 1, 2, or 3 and more.
That keeps the tier meaning "how many joins the planner must get right".

order_matters is set when the gold query has a top-level ORDER BY and no LIMIT 1. BIRD's own
scorer ignores row order, so this is stricter; check the flagged questions if one looks
unfair (they are listed in `notes`).

--verify runs every gold query first (needs the BIRD databases loaded and the config) and
drops the ones that fail, time out, return no rows, or return too many to compare, so the
sample only holds questions that can actually be scored.

Selection is by seed, so the same input and seed always give the same file. The questions not
chosen stay available as a pool for few-shot examples; question_set.py checks the two
files never overlap.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from schemagraph.config import DB_ID_RE
from schemagraph.eval import question_set
from schemagraph.eval.schema_metrics import tables_in_sql
from schemagraph.models import Question

DEFAULT_TIER_MIX = (0.2, 0.4, 0.4)  # share of tier 1, 2, 3: favour the questions that need joins
SQL_KEYS = ("SQL", "sql", "query", "gold_sql")


class BirdFormatError(ValueError):
    """The input file is not what we expected."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_items(path: Path) -> list[dict[str, Any]]:
    """The records of a mini-dev file: a JSON array, or one JSON object per line."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise BirdFormatError(f"{path}:{number}: not JSON: {exc}") from exc
    if isinstance(data, dict):
        for key in ("data", "questions", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise BirdFormatError(f"{path}: expected a list of question records")
    return data


# ---------------------------------------------------------------------------
# One record -> one Question (pure)
# ---------------------------------------------------------------------------


def tier_for(gold_sql: str) -> int | None:
    """1, 2 or 3 by the number of distinct tables the query reads; None if it reads none."""
    count = len(tables_in_sql(gold_sql))
    return None if count == 0 else min(count, 3)


def order_matters_for(gold_sql: str) -> bool:
    """True if the gold query orders its output and is not a single-row 'top 1' query."""
    try:
        tree = sqlglot.parse_one(gold_sql, read="postgres")
    except SqlglotError:
        return False
    if not isinstance(tree, exp.Select) or not tree.args.get("order"):
        return False
    limit = tree.args.get("limit")
    if limit is not None and str(limit.expression.this) == "1":
        return False
    return True


@dataclass
class Skipped:
    reasons: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, reason: str, label: str) -> None:
        self.reasons[reason] += 1
        if len(self.examples[reason]) < 3:
            self.examples[reason].append(label)


def to_question(item: dict[str, Any], index: int) -> Question | str:
    """Convert one record, or return the reason it cannot be used."""
    sql = next((item[k] for k in SQL_KEYS if isinstance(item.get(k), str)), None)
    missing = [k for k, v in (("db_id", item.get("db_id")), ("question", item.get("question")), ("SQL", sql)) if not v]
    if missing:
        return f"missing {', '.join(missing)}"
    db_id = str(item["db_id"])
    if not DB_ID_RE.match(db_id):
        return f"db_id {db_id!r} is not lowercase letters, digits and _"
    sql = sql.strip()
    tier = tier_for(sql)
    if tier is None:
        return "gold SQL reads no tables (or does not parse)"
    qid = item.get("question_id", index + 1)
    evidence = (item.get("evidence") or "").strip() or None
    difficulty = item.get("difficulty")
    order = order_matters_for(sql)
    notes = f"difficulty={difficulty}; " if difficulty else ""
    notes += f"tables={len(tables_in_sql(sql))}" + ("; ordered (check order_matters)" if order else "")
    return Question(
        id=f"bird-{qid}", db_id=db_id, question=str(item["question"]).strip(), gold_sql=sql, tier=tier,
        split="test", order_matters=order, evidence=evidence, author="bird", notes=notes,
    )


# ---------------------------------------------------------------------------
# Choosing the subset (pure)
# ---------------------------------------------------------------------------


def apportion(total: int, weights: list[float]) -> list[int]:
    """Split `total` into whole numbers in proportion to `weights`, summing exactly to `total`."""
    if total < 0 or not weights or any(w < 0 for w in weights) or sum(weights) == 0:
        raise ValueError("need a non-negative total and positive weights")
    shares = [total * w / sum(weights) for w in weights]
    counts = [int(s) for s in shares]
    order = sorted(range(len(weights)), key=lambda i: (-(shares[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return counts


def _spread_over_databases(pool: list[Question], rng: random.Random) -> list[Question]:
    """Shuffle, then deal one question from each database in turn, so no database dominates."""
    by_db: dict[str, list[Question]] = defaultdict(list)
    for q in sorted(pool, key=lambda q: q.id):
        by_db[q.db_id].append(q)
    for qs in by_db.values():
        rng.shuffle(qs)
    dbs = sorted(by_db)
    rng.shuffle(dbs)
    dealt: list[Question] = []
    while any(by_db.values()):
        for db in dbs:
            if by_db[db]:
                dealt.append(by_db[db].pop())
    return dealt


def choose(
    candidates: list[Question], n_test: int, n_dev: int, tier_mix: tuple[float, float, float] = DEFAULT_TIER_MIX,
    seed: int = 0,
) -> tuple[list[Question], list[str]]:
    """Pick n_test + n_dev questions with the tier mix, spread over databases, then label n_dev of them dev.

    Returns the chosen questions (split set, sorted by id) and warnings, for example when a
    tier has too few questions and others had to make up the difference.
    """
    total = n_test + n_dev
    warnings: list[str] = []
    if total > len(candidates):
        warnings.append(f"asked for {total} questions but only {len(candidates)} are usable; taking all of them")
        total = len(candidates)
        n_dev = min(n_dev, total)
    rng = random.Random(seed)
    pools = {t: _spread_over_databases([q for q in candidates if q.tier == t], rng) for t in (1, 2, 3)}

    quotas = dict(zip((1, 2, 3), apportion(total, list(tier_mix))))
    for t in (1, 2, 3):  # move any shortfall to the tiers that still have questions, largest tier first
        short = quotas[t] - len(pools[t])
        if short > 0:
            warnings.append(f"tier {t}: wanted {quotas[t]}, only {len(pools[t])} usable")
            quotas[t] = len(pools[t])
            for other in sorted((o for o in (1, 2, 3) if o != t), key=lambda o: -len(pools[o]) + quotas[o]):
                room = len(pools[other]) - quotas[other]
                give = min(short, room)
                quotas[other] += give
                short -= give
    chosen = {t: pools[t][: quotas[t]] for t in (1, 2, 3)}

    # dev questions are drawn from each tier in proportion, so dev looks like test
    sizes = [len(chosen[t]) for t in (1, 2, 3)]
    dev_counts = apportion(n_dev, [float(s) for s in sizes]) if sum(sizes) else [0, 0, 0]
    picked: list[Question] = []
    for t, n in zip((1, 2, 3), dev_counts):
        for i, q in enumerate(chosen[t]):
            picked.append(q.model_copy(update={"split": "dev" if i < n else "test"}))
    picked.sort(key=lambda q: _id_key(q.id))
    return picked, warnings


def _id_key(qid: str) -> tuple[str, int, str]:
    prefix, _, rest = qid.rpartition("-")
    return (prefix, int(rest), "") if rest.isdigit() else (qid, -1, qid)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    read: int = 0
    usable: int = 0
    skipped: Skipped = field(default_factory=Skipped)
    dropped_by_gold: dict[str, str] = field(default_factory=dict)
    candidates: int = 0
    warnings: list[str] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


def build(
    items: list[dict[str, Any]],
    n_test: int = 50,
    n_dev: int = 10,
    tier_mix: tuple[float, float, float] = DEFAULT_TIER_MIX,
    seed: int = 0,
    verify: Callable[[list[Question]], Any] | None = None,
) -> BuildReport:
    """Convert, de-duplicate, optionally check the gold queries, and choose the subset.

    `verify` receives the candidate questions and returns an object with `.results` and `.errors`
    and `.empty` (run_eval.GoldCheck); questions in errors or empty are dropped.
    """
    report = BuildReport(read=len(items))
    candidates: list[Question] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        result = to_question(item, index)
        if isinstance(result, str):
            report.skipped.add(result, f"record {index + 1}")
        elif result.id in seen_ids:
            report.skipped.add("duplicate question_id", result.id)
        else:
            seen_ids.add(result.id)
            candidates.append(result)

    report.candidates = len(candidates)
    if verify is not None:
        check = verify(candidates)
        report.dropped_by_gold = dict(check.errors) | {qid: "gold SQL returns no rows" for qid in check.empty}
        candidates = [q for q in candidates if q.id in check.results and q.id not in check.empty]
    report.usable = len(candidates)
    report.questions, report.warnings = choose(candidates, n_test, n_dev, tier_mix, seed)
    return report


ABORT_SHARE = 0.5  # stop when this share of all candidates fail for one and the same reason
_QUOTED = re.compile(r'"[^"]*"')


def group_drops(dropped: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Group drop reasons by message (quoted names collapsed), biggest group first.

    Collapsing "card_games" / "financial" into one key makes "database X does not exist" a single
    line instead of one line per question.
    """
    groups: dict[str, list[str]] = {}
    for qid, why in dropped.items():
        groups.setdefault(_QUOTED.sub('"..."', " ".join(why.split())), []).append(qid)
    return sorted(((why, sorted(ids)) for why, ids in groups.items()), key=lambda g: (-len(g[1]), g[0]))


def systematic_failure(dropped: dict[str, str], candidates: int) -> str | None:
    """The reason, if at least ABORT_SHARE of all candidates failed for the same (non-empty-result) reason.

    That pattern means the environment is wrong (databases missing, bad credentials), not that the
    questions are, and choosing a subset from the few survivors would be misleading.
    """
    errors = {q: w for q, w in dropped.items() if w != "gold SQL returns no rows"}
    if not candidates or not errors:
        return None
    why, ids = group_drops(errors)[0]
    return why if len(ids) >= ABORT_SHARE * candidates else None


def write_jsonl(questions: list[Question], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")


def _parse_mix(text: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(p) for p in text.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) == 0:
        raise argparse.ArgumentTypeError("tier mix must be three non-negative numbers such as 20,40,40")
    return parts  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bird_subset.jsonl from the BIRD mini-dev questions.")
    parser.add_argument("--input", type=Path, default=Path("data/bird/mini_dev_postgresql.json"))
    parser.add_argument("--out", type=Path, default=Path("schemagraph/eval/questions/bird_subset.jsonl"))
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-dev", type=int, default=10)
    parser.add_argument("--tier-mix", type=_parse_mix, default=DEFAULT_TIER_MIX, metavar="T1,T2,T3",
                        help="relative share of tier 1, 2 and 3 questions (default 20,40,40)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true",
                        help="run every gold query first and drop those that fail or return no rows (needs the databases)")
    args = parser.parse_args(argv)

    try:
        items = load_items(args.input)
    except (OSError, BirdFormatError) as exc:
        print(f"cannot read {args.input}: {exc}", file=sys.stderr)
        return 1
    print(f"read {len(items)} records; fields seen: {sorted({k for item in items for k in item})}")

    verify = None
    if args.verify:
        from schemagraph.eval.run_eval import check_gold

        verify = check_gold
    report = build(items, args.n_test, args.n_dev, args.tier_mix, args.seed, verify)

    for reason, n in report.skipped.reasons.items():
        print(f"  skipped {n}: {reason} (e.g. {', '.join(report.skipped.examples[reason])})")
    for why, ids in group_drops(report.dropped_by_gold):
        shown = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
        print(f"  dropped {len(ids)}: {why} (e.g. {shown})")
    systematic = systematic_failure(report.dropped_by_gold, report.candidates)
    if systematic:
        print(f"ABORTED: {len(report.dropped_by_gold)} of {report.candidates} gold queries failed, most for one "
              f"reason:\n  {systematic}\nThat points at the setup (databases not loaded, wrong credentials), "
              "not at the questions. Fix it and rerun; nothing was written.", file=sys.stderr)
        return 2
    for warning in report.warnings:
        print(f"  WARNING {warning}")
    if not report.questions:
        print("no questions chosen", file=sys.stderr)
        return 1

    write_jsonl(report.questions, args.out)
    print(f"{report.usable} usable of {report.read}; wrote {len(report.questions)} to {args.out}")
    print(question_set.describe(report.questions))
    issues = question_set.problems(report.questions)
    for issue in issues:
        print(f"  PROBLEM {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
