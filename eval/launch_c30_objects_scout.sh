#!/usr/bin/env bash
# Same C30 / C31 / C33 scout as the egg 1000–1004 run, for every object:
#   egg   seed indices 0–4   → seeds 1000–1004
#   lemon seed indices 27–31 → seeds 1027–1031
#   onion seed indices 54–58 → seeds 1054–1058
# Cells: C30 (Δ=0), C31 (Δ=100), C33 (Δ=300) @ 30 cm/s.
# Does not touch sim/control cfg — launch + report only.
#
# Usage:
#   bash eval/launch_c30_objects_scout.sh data/eval/C1_c30_c31_c33_objects_scout
#   SAVE_MP4=1 bash eval/launch_c30_objects_scout.sh data/eval/C1_c30_c31_c33_objects_scout

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:?usage: launch_c30_objects_scout.sh OUT_DIR}
CELLS=${CELLS:-C30,C31,C33}
# N must cover the highest seed index (onion ends at 58).
N=${N:-80}

if [[ "${SAVE_MP4:-}" == "1" ]]; then
  MP4_FLAG=(--save-mp4)
else
  MP4_FLAG=()
fi

mkdir -p "$OUT/logs"
rm -f "$OUT/STOP"

echo "OUT=$OUT cells=$CELLS n=$N objects=egg,lemon,onion (5 seeds each)"

$PY -c "
import sys
from pathlib import Path
ROOT=Path('$ROOT'); sys.path.insert(0,str(ROOT))
from scripts.run_b1_grid import freeze_config, DEFAULT_CFG
freeze_config(Path('$OUT'), DEFAULT_CFG)
print('froze', '$OUT')
"

run_worker() {
  local gpu=$1 obj=$2 s0=$3 s1=$4
  local logf=$OUT/logs/gpu${gpu}_${obj}.log
  echo "start gpu=$gpu obj=$obj seeds=$s0..$s1 cells=$CELLS"
  CUDA_VISIBLE_DEVICES=$gpu $PY $ROOT/eval/run_b1_grid.py \
    --headless --enable_cameras --device cuda:0 \
    --role all --phase full --object "$obj" \
    --seed-start "$s0" --seed-end "$s1" --n "$N" \
    --cells "$CELLS" --no-early-stop \
    "${MP4_FLAG[@]}" \
    -o "$OUT" >"$logf" 2>&1 &
  echo $! >"$OUT/logs/gpu${gpu}_${obj}.pid"
}

# Two GPUs in parallel for egg + lemon; onion after (one object per Isaac process).
run_worker 0 egg 0 4
run_worker 1 lemon 27 31
wait
run_worker 0 onion 54 58
wait

$PY "$ROOT/tools/diag/write_c30_paired_report.py" "$OUT" \
  -o "$OUT/PAIRED_REPORT.md"
echo C30_OBJECTS_SCOUT_DONE
