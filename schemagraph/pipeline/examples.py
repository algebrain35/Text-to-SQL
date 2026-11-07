"""Few-shot examples for the prompt (SPEC 5.12).

Examples come ONLY from fewshot_examples.jsonl, which must never contain test
questions. The ones put in the prompt are the most similar to the question by
shared words, from the same database. A missing file means no examples.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from schemagraph.models import FewShotExample

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions" / "fewshot_examples.jsonl"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "of", "for", "in", "on", "to", "and", "or", "by", "with", "is", "are", "was",
    "what", "which", "who", "how", "many", "much", "list", "show", "give", "me", "all", "each", "per",
    "that", "this", "do", "does", "have", "has", "from", "at", "as", "be", "their", "its",
}


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def similarity(a: str, b: str) -> float:
    """Share of distinct words in common (Jaccard), 0 to 1."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> tuple[FewShotExample, ...]:  # mtime: reload when the file changes
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = FewShotExample.model_validate_json(line)
            if ex.split == "test":
                log.warning("%s line %d: skipping %s, test questions are never used as examples", path, n, ex.id)
                continue
            out.append(ex)
    return tuple(out)


def load_examples(path: Path | None = None) -> list[FewShotExample]:
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return []
    return list(_load(str(path), path.stat().st_mtime))


def rank_examples(question: str, db_id: str, pool: list[FewShotExample], k: int) -> list[FewShotExample]:
    """Pure: the k examples from `db_id` most similar to `question`, best first.

    An example with exactly the same question is skipped, so a question can never
    be answered by copying its own gold SQL.
    """
    if k <= 0:
        return []
    asked = question.strip().lower()
    candidates = [ex for ex in pool if ex.db_id == db_id and ex.question.strip().lower() != asked]
    candidates.sort(key=lambda ex: (-similarity(question, ex.question), ex.id))
    return candidates[:k]


def retrieve_examples(db_id: str, question: str, k: int = 3, *, path: Path | None = None) -> list[FewShotExample]:
    """SPEC 5.12. From fewshot_examples.jsonl only."""
    return rank_examples(question, db_id, load_examples(path), k)
