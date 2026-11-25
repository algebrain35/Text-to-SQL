#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
OUT="${OUT:-results/compare.jsonl}"
REPEATS="${REPEATS:-1}"
# Only a label written into each row; the model actually called is llm.model in the active config.
# Defaulting to that config value keeps the label honest when the config changes.
if [ -z "${MODEL:-}" ]; then
    MODEL=$("$PYTHON" -c 'from schemagraph.config import get_config; print(get_config().llm.model)') || exit 1
fi
AW=schemagraph/eval/questions/adventureworks.jsonl
BIRD=schemagraph/eval/questions/bird_subset.jsonl

"$PYTHON" -m schemagraph.eval.run_eval --questions "$AW" "$BIRD" --check-gold || exit 1

run() {
    local id=$1 questions=$2
    shift 2
    "$PYTHON" -m schemagraph.eval.run_eval --questions "$questions" --modes graph --top-k 8 \
        --llm-model "$MODEL" --out "$OUT" --run-id "$id" "$@"
}

for r in $(seq 1 "$REPEATS"); do
    suffix=""
    [ "$r" -gt 1 ] && suffix="-r$r"
    run "aw-k8-prize4$suffix" "$AW" --selection prize --prize-scale 4
    run "aw-k8-all$suffix" "$AW" --selection all
    run "bird-k8-prize4$suffix" "$BIRD" --selection prize --prize-scale 4
    run "bird-k8-all$suffix" "$BIRD" --selection all
    run "bird-k8-prize4-noinf$suffix" "$BIRD" --selection prize --prize-scale 4 --no-inferred
done

"$PYTHON" -m schemagraph.eval.run_eval --compare-runs "$OUT"
