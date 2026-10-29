#!/usr/bin/env bash
# One B1 grid worker that restarts itself after an Isaac freeze.
# A freeze trips run_b1_grid.py's episode watchdog (--episode-timeout-s):
# Python stacks go to logs/watchdog_<object>.txt and the worker exits 1.
# Each restart uses --skip-done, so finished episodes are reused, not re-run.
# Each try gets its own Kit log (parallel workers used to overwrite one file).
#
# Usage: GPU=0 TAG=lemon bash eval/run_b1_worker.sh /abs/OUT_DIR [run_b1_grid.py args...]
#   e.g. ... --role all --phase full --n 80 --object lemon --seed-start 32 --seed-end 36 --save-mp4
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=${1:?usage: GPU=.. TAG=.. run_b1_worker.sh OUT_DIR [args]}
shift
TAG=${TAG:?set TAG}
TRIES=${TRIES:-3}
mkdir -p "$OUT/logs"
rc=1
for try in $(seq 1 "$TRIES"); do
  # --kit_args=VALUE form: argparse rejects a separate value that starts with "--".
  CUDA_VISIBLE_DEVICES=${GPU:?set GPU} "$PY" "$ROOT/scripts/run_b1_grid.py" \
    --headless --enable_cameras --device cuda:0 --skip-done \
    "--kit_args=--/log/file=$OUT/logs/kit_${TAG}_try${try}.log" \
    -o "$OUT" "$@" >>"$OUT/logs/$TAG.log" 2>&1
  rc=$?
  echo "WORKER $TAG try $try exit $rc $(date '+%F %T')" >>"$OUT/logs/$TAG.log"
  [ "$rc" -eq 0 ] && exit 0
done
exit "$rc"
