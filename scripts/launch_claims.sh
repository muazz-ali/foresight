#!/usr/bin/env bash
# Claim tests for one checkpoint: does the model use the 4 numbers to grasp
# moving objects? One sim eval per scenario, same seeds, so runs pair up.
# Conda: dynamicVLA_isaac. Score with: python eval/claims_report.py <OUT>...
#
#   oracle  sim-truth numbers           → flat success across speeds?
#   filter  camera + Kalman numbers     → same with the real tracker?
#   zero    numbers zeroed              → success collapses?
#   shift   p̂ moved by SHIFT (m)        → hand follows the numbers (misses by ≈ SHIFT)?
#   oracle_gate / filter_gate            → same + close gate (wrist depth + tracked XY)
#
# Usage:
#   CHECKPOINT=runs/v1_rel_delta/pretrained_model OUT=data/eval/claims/v1 \
#     DEVICE=cuda:0 bash scripts/launch_claims.sh
#   SCENARIOS="zero shift" ... bash scripts/launch_claims.sh
#
# Env: CHECKPOINT, OUT, DEVICE, TRIALS, SEED, SPEEDS, SCENARIOS, SHIFT, DEBUG,
#      FORESIGHT_PYTHON
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUT="${OUT:?set OUT}"
DEVICE="${DEVICE:-cuda:0}"
TRIALS="${TRIALS:-20}"
# 9000 = the seeds of data/eval/p3_throw_n100 and p3_throw_oracle_n20.
SEED="${SEED:-9000}"
# shellcheck disable=SC2206
SPEEDS=(${SPEEDS:-0.0 0.10 0.15 0.20})
# shellcheck disable=SC2206
SCENARIOS=(${SCENARIOS:-oracle filter zero shift})
# shellcheck disable=SC2206
SHIFT=(${SHIFT:-0.03 0.0})
DEBUG_FLAG=()
[[ "${DEBUG:-0}" == "1" ]] && DEBUG_FLAG=(--debug)

# Same GPU rule as the other launchers: hide the other cards.
if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
fi

mkdir -p "$OUT/logs"
FAILS=0
for sc in "${SCENARIOS[@]}"; do
  case "$sc" in
    oracle) EXTRA=(--cond-source oracle) ;;
    filter) EXTRA=(--cond-source filter) ;;
    zero) EXTRA=(--cond-source oracle --zero-conditioning) ;;
    shift) EXTRA=(--cond-source oracle --cond-shift-m "${SHIFT[@]}") ;;
    oracle_gate) EXTRA=(--cond-source oracle --close-gate on) ;;
    filter_gate) EXTRA=(--cond-source filter --close-gate on) ;;
    *) echo "[claims] unknown scenario $sc" >&2; exit 2 ;;
  esac
  echo "[claims] $sc → $OUT/$sc (checkpoint=$CHECKPOINT seed=$SEED trials=$TRIALS)"
  "$PY" "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$CHECKPOINT" \
    --headless --enable_cameras --device cuda:0 \
    --speeds "${SPEEDS[@]}" \
    --trials-per-speed "$TRIALS" \
    --seed "$SEED" \
    --n-action-steps 8 \
    "${EXTRA[@]}" "${DEBUG_FLAG[@]}" \
    -o "$OUT/$sc" >"$OUT/logs/$sc.log" 2>&1
  rc=$?
  echo "[claims] $sc exit=$rc"
  [[ $rc -ne 0 ]] && FAILS=$((FAILS + 1))
done
exit "$FAILS"
