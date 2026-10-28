#!/usr/bin/env bash
# Launch B1 dual-GPU workers. GPU0 = unseen, GPU1 = seen.
# One object category per process (Isaac World rebuild forbidden).
# Usage: bash eval/launch_b1_grid.sh data/eval/B1_grid_v1 [phase]

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:?usage: launch_b1_grid.sh OUT_DIR [phase]}
PHASE=${2:-all}
# Default ON for smoke so you can watch cells; set SAVE_MP4=0 for full grid.
if [[ "${SAVE_MP4:-}" == "0" ]]; then
  MP4_FLAG=()
elif [[ "$PHASE" == "smoke" || "$PHASE" == "c30" || "$PHASE" == "delay-check" || "$PHASE" == "neg-control" || "${SAVE_MP4:-}" == "1" ]]; then
  MP4_FLAG=(--save-mp4)
else
  MP4_FLAG=()
fi
# NO_EARLY_STOP=1: grid cells run to n (a bad cell no longer stops the grid).
if [[ "${NO_EARLY_STOP:-}" == "1" ]]; then
  ES_FLAG=(--no-early-stop)
else
  ES_FLAG=()
fi
mkdir -p "$OUT/logs"

# Freeze first (no Isaac).
$PY -c "
import sys
from pathlib import Path
ROOT=Path('$ROOT'); sys.path.insert(0,str(ROOT))
from scripts.run_b1_grid import freeze_config, DEFAULT_CFG
freeze_config(Path('$OUT'), DEFAULT_CFG)
print('froze', '$OUT')
"

run_worker() {
  local gpu=$1 role=$2 obj=$3 s0=$4 s1=$5 phase=$6
  local logf=$OUT/logs/${role}_${obj}_${phase}.log
  echo "start role=$role obj=$obj gpu=$gpu seeds=$s0..$s1 phase=$phase"
  CUDA_VISIBLE_DEVICES=$gpu $PY $ROOT/scripts/run_b1_grid.py \
    --headless --enable_cameras --device cuda:0 \
    --role "$role" --phase "$phase" --object "$obj" \
    --seed-start "$s0" --seed-end "$s1" \
    "${MP4_FLAG[@]}" "${ES_FLAG[@]}" \
    -o "$OUT" >"$logf" 2>&1 &
  echo $! >"$OUT/logs/${role}_${obj}_${phase}.pid"
}

# Validity phases (c30 / delay / neg) on GPU0 with egg only, then full grid.
if [[ "$PHASE" == "all" || "$PHASE" == "c30" || "$PHASE" == "delay-check" || "$PHASE" == "neg-control" ]]; then
  CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_b1_grid.py \
    --headless --enable_cameras --device cuda:0 \
    --role unseen --phase c30 --object egg --seed-start 0 --seed-end 9 \
    "${MP4_FLAG[@]}" \
    -o "$OUT" 2>&1 | tee "$OUT/logs/c30.log"
  CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_b1_grid.py \
    --headless --enable_cameras --device cuda:0 \
    --role unseen --phase delay-check --object egg --seed-start 0 --seed-end 9 \
    "${MP4_FLAG[@]}" \
    -o "$OUT" 2>&1 | tee "$OUT/logs/delay.log"
  CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_b1_grid.py \
    --headless --enable_cameras --device cuda:0 \
    --role unseen --phase neg-control --object egg --seed-start 0 --seed-end 9 \
    "${MP4_FLAG[@]}" \
    -o "$OUT" 2>&1 | tee "$OUT/logs/neg.log"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "smoke" || "$PHASE" == "full" ]]; then
  P=$PHASE
  [[ "$PHASE" == "all" ]] && P=full
  # Smoke uses n=4 (seeds 0..3) → only egg shard has work. Lemon/onion would
  # boot Isaac for empty seed ranges and waste GPU time.
  if [[ "$P" == "smoke" ]]; then
    OBJ_SHARDS=("egg:0:26")
  else
    OBJ_SHARDS=("egg:0:26" "lemon:27:53" "onion:54:79")
  fi
  for obj_s0_s1 in "${OBJ_SHARDS[@]}"; do
    IFS=: read -r obj s0 s1 <<<"$obj_s0_s1"
    run_worker 0 unseen "$obj" "$s0" "$s1" "$P"
    run_worker 1 seen "$obj" "$s0" "$s1" "$P"
    wait
  done
fi

$PY -c "
import sys
from pathlib import Path
ROOT=Path('$ROOT'); sys.path.insert(0,str(ROOT))
from scripts import run_b1_grid as m
m._write_report(Path('$OUT'))
print('report written')
"
echo B1_LAUNCH_DONE
