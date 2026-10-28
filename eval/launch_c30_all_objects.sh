#!/usr/bin/env bash
# C30 / C31 / C33 scout for every fruit in phase0_cfg object_categories.
# Uses scripts/run_c30_object_scout.py (catalog = requested category — not the
# egg/lemon/onion hardcode in run_b1_grid). No sim/control/cfg edits.
#
# Usage:
#   bash eval/launch_c30_all_objects.sh data/eval/C1_c30_all_objects
#   N_PER=5 OBJECTS=egg,lemon bash eval/launch_c30_all_objects.sh data/eval/C1_smoke

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:?usage: launch_c30_all_objects.sh OUT_DIR}

CELLS=${CELLS:-C30,C31,C33}
N_PER=${N_PER:-5}
OBJECTS=${OBJECTS:-apple,avocado,egg,kiwi,lemon,lime,onion,orange,peach,tangerine}
# DUMP=mp4 → h5+json+mp4; DUMP=h5 → h5+json only (faster re-runs).
DUMP=${DUMP:-mp4}
if [[ "$DUMP" == "h5" ]]; then
  DUMP_FLAG=(--save-h5)
elif [[ "$DUMP" == "mp4" ]]; then
  DUMP_FLAG=(--save-mp4)
else
  echo "DUMP must be mp4 or h5, got $DUMP" >&2
  exit 2
fi

mkdir -p "$OUT/logs"
rm -f "$OUT/STOP"

IFS=',' read -r -a OBJ_ARR <<< "$OBJECTS"
echo "OUT=$OUT cells=$CELLS n_per=$N_PER dump=$DUMP objects=${OBJ_ARR[*]}"

$PY -c "
import sys
from pathlib import Path
ROOT=Path('$ROOT'); sys.path.insert(0,str(ROOT))
from scripts.run_b1_grid import freeze_config, DEFAULT_CFG
freeze_config(Path('$OUT'), DEFAULT_CFG)
print('froze', '$OUT')
"

# seed_id = 1000 + obj_index * N_PER + k
seeds_for() {
  local idx=$1
  local s0=$((1000 + idx * N_PER))
  local parts=()
  local k
  for ((k=0; k<N_PER; k++)); do
    parts+=("$((s0 + k))")
  done
  local IFS=,
  echo "${parts[*]}"
}

run_one_object() {
  local gpu=$1
  local obj=$2
  local idx=$3
  local seeds
  seeds=$(seeds_for "$idx")
  local obj_out=$OUT/$obj
  local logf=$OUT/logs/gpu${gpu}_${obj}.log
  mkdir -p "$obj_out" "$OUT/logs"
  echo "start gpu=$gpu obj=$obj seeds=$seeds cells=$CELLS"
  CUDA_VISIBLE_DEVICES=$gpu $PY $ROOT/scripts/run_c30_object_scout.py \
    --headless --enable_cameras --device cuda:0 \
    --object "$obj" --seeds "$seeds" --cells "$CELLS" \
    "${DUMP_FLAG[@]}" -o "$obj_out" >"$logf" 2>&1
  echo "done obj=$obj exit=$?"
}

pids=()
for i in "${!OBJ_ARR[@]}"; do
  obj=${OBJ_ARR[$i]}
  gpu=$((i % 2))
  if [[ -n "${pids[$gpu]:-}" ]]; then
    wait "${pids[$gpu]}" || true
  fi
  run_one_object "$gpu" "$obj" "$i" &
  pids[$gpu]=$!
  echo $! >"$OUT/logs/gpu${gpu}_${obj}.pid"
done
for gpu in 0 1; do
  if [[ -n "${pids[$gpu]:-}" ]]; then
    wait "${pids[$gpu]}" || true
  fi
done

$PY "$ROOT/tools/diag/report_c30_all_objects.py" "$OUT" \
  --objects "$OBJECTS" --n-per "$N_PER" \
  -o "$OUT/REPORT.md"
echo "wrote $OUT/REPORT.md"
echo C30_ALL_OBJECTS_DONE
