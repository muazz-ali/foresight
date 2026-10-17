#!/usr/bin/env bash
# Multi-worker policy eval with different house/object/bowl per worker.
# Meshes are fixed for one Isaac process (sim/scene.py); diversity = many seeds.
#
# Default: 10 workers × 10 trials/speed on cuda:1, sequential (avoids OOM).
# Does not touch data/eval/p3_throw_n100 (the fixed-asset cuda:0 run).
#
# Usage:
#   bash scripts/launch_eval_diverse.sh
#   DEVICE=cuda:1 WORKERS=10 TRIALS=10 bash scripts/launch_eval_diverse.sh
#
# Env: FORESIGHT_PYTHON, DEVICE, CHECKPOINT, OUT, WORKERS, TRIALS, SEED_BASE,
#      SPEEDS, N_ACTION_STEPS, COND_SOURCE, REFLEX
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

DEVICE="${DEVICE:-cuda:1}"
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/p3_throw/pretrained_model}"
OUT="${OUT:-$ROOT/data/eval/p3_throw_n100_diverse}"
WORKERS="${WORKERS:-10}"
TRIALS="${TRIALS:-5}"
SEED_BASE="${SEED_BASE:-10000}"
SEED_STRIDE="${SEED_STRIDE:-2000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
COND_SOURCE="${COND_SOURCE:-filter}"
REFLEX="${REFLEX:-on}"
# shellcheck disable=SC2206
SPEEDS=(${SPEEDS:-0.0 0.10 0.15 0.20})

# Same GPU rule as launch_p1_collect: hide other GPUs; process sees cuda:0.
DEVICE_ARGS=(--device cuda:0)
if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  echo "[diverse] DEVICE=$DEVICE → CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --device cuda:0"
else
  unset CUDA_VISIBLE_DEVICES || true
  DEVICE_ARGS=(--device "$DEVICE")
fi

mkdir -p "$OUT/logs"
echo "[diverse] out=$OUT workers=$WORKERS trials/speed=$TRIALS seed_base=$SEED_BASE"
echo "[diverse] checkpoint=$CHECKPOINT speeds=${SPEEDS[*]}"

FAILS=0
for ((w = 0; w < WORKERS; w++)); do
  SEED=$((SEED_BASE + w * SEED_STRIDE))
  WID=$(printf "w%02d" "$w")
  WOUT="$OUT/$WID"
  mkdir -p "$WOUT"
  LOG="$OUT/logs/${WID}_seed${SEED}.log"
  echo "[diverse] === $WID seed=$SEED trials=$TRIALS → $WOUT ==="
  set +e
  "$PY" "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$CHECKPOINT" \
    --headless --debug --enable_cameras \
    "${DEVICE_ARGS[@]}" \
    --speeds "${SPEEDS[@]}" \
    --trials-per-speed "$TRIALS" \
    --seed "$SEED" \
    --n-action-steps "$N_ACTION_STEPS" \
    --cond-source "$COND_SOURCE" \
    --reflex "$REFLEX" \
    -o "$WOUT" \
    >"$LOG" 2>&1
  EC=$?
  set -e
  echo "[diverse] $WID exit=$EC log=$LOG"
  if [[ "$EC" -ne 0 ]]; then
    FAILS=$((FAILS + 1))
  fi
done

# Merge metrics + print diversity / rate summary.
ALL="$OUT/metrics_all.jsonl"
: >"$ALL"
for ((w = 0; w < WORKERS; w++)); do
  WID=$(printf "w%02d" "$w")
  M="$OUT/$WID/metrics.jsonl"
  if [[ -f "$M" ]]; then
    cat "$M" >>"$ALL"
  fi
done

"$PY" - <<PY
import json
from pathlib import Path
from collections import Counter, defaultdict

p = Path("$ALL")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
print(f"[diverse] merged trials={len(rows)} → {p}")
by_speed = defaultdict(lambda: [0, 0])
cats, seeds, ctrs = Counter(), Counter(), Counter()
for r in rows:
    sp = float(r.get("speed_m_s", -1))
    by_speed[sp][1] += 1
    if r.get("success"):
        by_speed[sp][0] += 1
    if r.get("category"):
        cats[r["category"]] += 1
    if r.get("asset_seed") is not None:
        seeds[r["asset_seed"]] += 1
    if r.get("container_category"):
        ctrs[r["container_category"]] += 1
print("[diverse] per-speed success:")
for sp in sorted(by_speed):
    ok, n = by_speed[sp]
    print(f"  {sp:.2f} m/s  {ok}/{n}  ({100*ok/n if n else 0:.1f}%)")
print(f"[diverse] unique asset_seeds={len(seeds)} categories={dict(cats)} containers={dict(ctrs)}")
PY

echo "[diverse] done workers=$WORKERS fails=$FAILS"
exit "$FAILS"
