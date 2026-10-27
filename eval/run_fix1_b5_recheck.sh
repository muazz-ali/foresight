#!/usr/bin/env bash
# Fix1 B5 static re-check: v=0, 4 eps × egg/lemon/onion = 12.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:-data/eval/fix1_static_offset/b5_recheck}
mkdir -p "$OUT/logs"
seed0=41000
i=0
for obj in egg lemon onion; do
  seed=$((seed0 + i * 100))
  d=$OUT/$obj
  mkdir -p "$d"
  echo "=== B5 recheck $obj seed=$seed ===" | tee -a "$OUT/logs/launch.log"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY "$ROOT/scripts/run_phase1_collect.py" \
    --headless --enable_cameras --device cuda:0 \
    --speed-mode fixed --speed 0.0 \
    --target-successes 4 --max-attempts 6 \
    --seed "$seed" --object-usd "$obj" --no-debug \
    -o "$d" >>"$OUT/logs/${obj}.log" 2>&1
  ec=$?
  echo "=== done $obj exit=$ec ===" | tee -a "$OUT/logs/launch.log"
  i=$((i + 1))
  if [ "$ec" -ne 0 ]; then
    echo FIX1_B5_FAILED | tee -a "$OUT/logs/launch.log"
    exit "$ec"
  fi
done

$PY "$ROOT/eval/b5_hand_trail_bias.py" --root "$OUT" \
  -o "$OUT/b5_report.json" | tee "$OUT/logs/b5_score.log"
echo FIX1_B5_DONE | tee -a "$OUT/logs/launch.log"
