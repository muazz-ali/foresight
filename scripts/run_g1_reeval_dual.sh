#!/usr/bin/env bash
# Four-arm G1/P5 re-eval on two GPUs (no MP4).
# Wave 1: A @ cuda:0 || B @ cuda:1
# Wave 2: B_zero @ cuda:0 || B_oracle @ cuda:1
#
# Env: dynamicVLA_isaac. Needs runs/p1_a and runs/p1_b.
#
# Usage:
#   bash scripts/run_g1_reeval_dual.sh smoke   # 1 trial, speed 0.2 only
#   bash scripts/run_g1_reeval_dual.sh full    # TRIALS×5 speeds (default TRIALS=91)
#
# Env overrides: TRIALS SEED N_ACTION_STEPS OUT_ROOT GPU0 GPU1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${1:-full}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
SEED="${SEED:-9000}"
GPU0="${GPU0:-cuda:0}"
GPU1="${GPU1:-cuda:1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/p1/reeval_n${N_ACTION_STEPS}_n91_p5}"

case "$MODE" in
  smoke)
    SPEEDS=(0.2)
    TRIALS="${TRIALS:-1}"
    ;;
  full)
    SPEEDS=(0.0 0.1 0.2 0.3 0.4)
    TRIALS="${TRIALS:-91}"
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynamicVLA_isaac

echo "[g1-dual] mode=$MODE trials=$TRIALS speeds=${SPEEDS[*]} seed=$SEED"
echo "[g1-dual] out=$OUT_ROOT  GPUs: $GPU0 | $GPU1  (no --debug / no mp4)"

run_eval() {
  local model="$1"   # A | B
  local tag="$2"     # a | b | b_zero | b_oracle
  local device="$3"
  shift 3
  local ckpt="$ROOT/runs/p1_$(echo "$model" | tr '[:upper:]' '[:lower:]')/pretrained_model"
  local out="$OUT_ROOT/eval_${tag}"
  if [[ ! -d "$ckpt" ]]; then
    echo "[g1-dual] missing checkpoint: $ckpt" >&2
    exit 1
  fi
  mkdir -p "$out"
  echo "[g1-dual] >>> model=$model tag=$tag device=$device → $out"
  python "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$ckpt" \
    --model "$model" \
    --headless \
    --enable_cameras \
    --device "$device" \
    --n-action-steps "$N_ACTION_STEPS" \
    --speeds "${SPEEDS[@]}" \
    --trials-per-speed "$TRIALS" \
    --seed "$SEED" \
    -o "$out" \
    "$@"
}

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/dual_run.log"
: > "$LOG"

echo "[g1-dual] wave1: A@$GPU0 || B@$GPU1" | tee -a "$LOG"
run_eval A a "$GPU0" >"$OUT_ROOT/eval_a.log" 2>&1 &
PID_A=$!
run_eval B b "$GPU1" >"$OUT_ROOT/eval_b.log" 2>&1 &
PID_B=$!
wait "$PID_A"
EC_A=$?
wait "$PID_B"
EC_B=$?
echo "[g1-dual] wave1 done EC_A=$EC_A EC_B=$EC_B" | tee -a "$LOG"
if [[ "$EC_A" -ne 0 || "$EC_B" -ne 0 ]]; then
  echo "[g1-dual] wave1 failed — see eval_a.log / eval_b.log" >&2
  exit 1
fi

echo "[g1-dual] wave2: B_zero@$GPU0 || B_oracle@$GPU1" | tee -a "$LOG"
run_eval B b_zero "$GPU0" --zero-conditioning >"$OUT_ROOT/eval_b_zero.log" 2>&1 &
PID_BZ=$!
run_eval B b_oracle "$GPU1" --oracle-conditioning >"$OUT_ROOT/eval_b_oracle.log" 2>&1 &
PID_BO=$!
wait "$PID_BZ"
EC_BZ=$?
wait "$PID_BO"
EC_BO=$?
echo "[g1-dual] wave2 done EC_BZ=$EC_BZ EC_BO=$EC_BO" | tee -a "$LOG"
if [[ "$EC_BZ" -ne 0 || "$EC_BO" -ne 0 ]]; then
  echo "[g1-dual] wave2 failed — see eval_b_zero.log / eval_b_oracle.log" >&2
  exit 1
fi

python "$ROOT/eval/gate_g1.py" \
  --report-a "$OUT_ROOT/eval_a/gate_g1_report.json" \
  --report-b "$OUT_ROOT/eval_b/gate_g1_report.json" \
  --report-b-zero "$OUT_ROOT/eval_b_zero/gate_g1_report.json" \
  -o "$OUT_ROOT/gate_g1_report.json"

python "$ROOT/eval/gate_p5.py" \
  --report-a "$OUT_ROOT/eval_a/gate_g1_report.json" \
  --report-b "$OUT_ROOT/eval_b/gate_g1_report.json" \
  --report-b-zero "$OUT_ROOT/eval_b_zero/gate_g1_report.json" \
  --report-b-oracle "$OUT_ROOT/eval_b_oracle/gate_g1_report.json" \
  -o "$OUT_ROOT/gate_p5_report.json"

echo "[g1-dual] done → $OUT_ROOT/gate_g1_report.json + gate_p5_report.json"
