#!/usr/bin/env bash
# Gate G1 re-eval with frequent replan (plan ~150–300 ms → n_action_steps=8).
# Env: dynamicVLA_isaac. Needs runs/p1_a and runs/p1_b checkpoints.
#
# Usage:
#   bash scripts/run_g1_reeval.sh              # full A+B+B-zero, 20 trials × 5 speeds
#   bash scripts/run_g1_reeval.sh smoke        # B only, speed 0, 5 trials + mp4
#   bash scripts/run_g1_reeval.sh mid         # A+B+B-zero, speeds 0+0.2, 5 trials + mp4
#
# Env overrides: N_ACTION_STEPS TRIALS DEVICE SEED OUT_ROOT
#
# Important: put --headless on its own argv token (never "\ --headless").
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${1:-full}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-9000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/p1/reeval_n${N_ACTION_STEPS}}"

DEBUG_ARGS=()
case "$MODE" in
  smoke)
    SPEEDS=(0.0)
    TRIALS="${TRIALS:-5}"
    RUN_A=0
    RUN_B=1
    RUN_BZ=0
    DEBUG_ARGS=(--debug)
    ;;
  mid)
    SPEEDS=(0.0 0.2)
    TRIALS="${TRIALS:-5}"
    RUN_A=1
    RUN_B=1
    RUN_BZ=1
    DEBUG_ARGS=(--debug)
    ;;
  full)
    SPEEDS=(0.0 0.1 0.2 0.3 0.4)
    TRIALS="${TRIALS:-20}"
    RUN_A=1
    RUN_B=1
    RUN_BZ=1
    ;;
  *)
    echo "usage: $0 [smoke|mid|full]" >&2
    exit 2
    ;;
esac

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynamicVLA_isaac

echo "[g1] mode=$MODE n_action_steps=$N_ACTION_STEPS trials=$TRIALS speeds=${SPEEDS[*]}"
echo "[g1] out=$OUT_ROOT device=$DEVICE"

run_eval() {
  local model="$1" # A | B
  local tag="$2"   # a | b | b_zero
  shift 2
  local ckpt="$ROOT/runs/p1_$(echo "$model" | tr '[:upper:]' '[:lower:]')/pretrained_model"
  local out="$OUT_ROOT/eval_${tag}"
  if [[ ! -d "$ckpt" ]]; then
    echo "[g1] missing checkpoint: $ckpt" >&2
    exit 1
  fi
  mkdir -p "$out"
  echo "[g1] >>> model=$model tag=$tag → $out"
  python "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$ckpt" \
    --model "$model" \
    --headless \
    --enable_cameras \
    --device "$DEVICE" \
    --n-action-steps "$N_ACTION_STEPS" \
    --speeds "${SPEEDS[@]}" \
    --trials-per-speed "$TRIALS" \
    --seed "$SEED" \
    -o "$out" \
    "${DEBUG_ARGS[@]}" \
    "$@"
}

[[ "$RUN_A" == 1 ]] && run_eval A a
[[ "$RUN_B" == 1 ]] && run_eval B b
[[ "$RUN_BZ" == 1 ]] && run_eval B b_zero --zero-conditioning

if [[ "$RUN_A" == 1 && "$RUN_B" == 1 && "$RUN_BZ" == 1 ]]; then
  python "$ROOT/eval/gate_g1.py" \
    --report-a "$OUT_ROOT/eval_a/gate_g1_report.json" \
    --report-b "$OUT_ROOT/eval_b/gate_g1_report.json" \
    --report-b-zero "$OUT_ROOT/eval_b_zero/gate_g1_report.json" \
    -o "$OUT_ROOT/gate_g1_report.json"
  echo "[g1] done → $OUT_ROOT/gate_g1_report.json"
else
  echo "[g1] partial run under $OUT_ROOT (no A+B+B-zero merge)"
fi
