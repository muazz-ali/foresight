#!/usr/bin/env bash
# Recheck only C30 / C31 / C33 (30 cm/s). Delay stress = 100 ms and 300 ms:
#   C30 = 30 cm/s, Δ = 0 ms    (same-speed baseline)
#   C31 = 30 cm/s, Δ = 100 ms
#   C33 = 30 cm/s, Δ = 300 ms
# Skips C32 (200 ms) and the rest of the 4×4.
#
# All three are "unseen", so both GPUs share object shards (not seen/unseen).
# No validity phases (c30 / delay / neg) — this is a focused recheck.
#
# Usage:
#   bash eval/launch_c30_c31_c33.sh data/eval/C1_c30_c31_c33
#   N=20 SAVE_MP4=1 bash eval/launch_c30_c31_c33.sh data/eval/C1_c30_c31_c33
#   CELLS=C31,C33 bash eval/launch_c30_c31_c33.sh data/eval/C1_c31_c33  # delays only

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:?usage: launch_c30_c31_c33.sh OUT_DIR}
# Default: C30 + delay cells at 100 ms / 300 ms. Override with CELLS=.
CELLS=${CELLS:-C30,C31,C33}
N=${N:-80}
PHASE=${PHASE:-full}

if [[ "${SAVE_MP4:-}" == "1" ]]; then
  MP4_FLAG=(--save-mp4)
else
  MP4_FLAG=()
fi

mkdir -p "$OUT/logs"
rm -f "$OUT/STOP"

echo "OUT=$OUT cells=$CELLS n=$N phase=$PHASE"

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
  local gpu=$1 obj=$2 s0=$3 s1=$4
  local logf=$OUT/logs/gpu${gpu}_${obj}.log
  echo "start gpu=$gpu obj=$obj seeds=$s0..$s1 cells=$CELLS"
  CUDA_VISIBLE_DEVICES=$gpu $PY $ROOT/eval/run_b1_grid.py \
    --headless --enable_cameras --device cuda:0 \
    --role all --phase "$PHASE" --object "$obj" \
    --seed-start "$s0" --seed-end "$s1" --n "$N" \
    --cells "$CELLS" --no-early-stop \
    "${MP4_FLAG[@]}" \
    -o "$OUT" >"$logf" 2>&1 &
  echo $! >"$OUT/logs/gpu${gpu}_${obj}.pid"
}

# Smoke: few egg seeds only. Full: egg / lemon / onion on both GPUs.
if [[ "$PHASE" == "smoke" ]]; then
  N=4
  run_worker 0 egg 0 3
  wait
else
  # GPU0 = egg, GPU1 = lemon; then onion on GPU0 (Isaac: one object per process).
  run_worker 0 egg 0 26
  run_worker 1 lemon 27 53
  wait
  run_worker 0 onion 54 79
  run_worker 1 kiwi 16 31
  wait
  run_worker 0 lemon 32 47
  run_worker 1 lime 48 63
  wait

fi

$PY -c "
import sys
from pathlib import Path
ROOT=Path('$ROOT'); sys.path.insert(0,str(ROOT))
from scripts import run_b1_grid as m
m._write_report(Path('$OUT'))
print('report written')
"
echo C30_C31_C33_DONE
