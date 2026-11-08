#!/usr/bin/env python3
"""Score retrieval against gold SQL: recall@k, score spread, and seed-pruning strategies. No LLM involved.

    python scripts/retrieval_eval.py --db adventureworks --questions-file schemagraph/eval/questions/adventureworks.jsonl

The work is in schemagraph/eval/retrieval_eval.py; this file only lets you run it as a script from a checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemagraph.eval.retrieval_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
