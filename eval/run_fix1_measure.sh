#!/usr/bin/env bash
# Long-running Fix1 measure (egg → lemon → onion). Safe under nohup.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:-data/eval/fix1_static_offset}
mkdir -p "$OUT/logs"
rm -f "$OUT"/.lock_egg "$OUT"/.lock_lemon "$OUT"/.lock_onion
rm -f "$OUT/settles.jsonl"

for obj in egg lemon onion; do
  echo "=== fix1 measure object=$obj $(date -Is) ===" | tee -a "$OUT/logs/launch.log"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY "$ROOT/eval/fix1_static_offset.py" \
    --headless --enable_cameras --device cuda:0 \
    --object "$obj" -o "$OUT" \
    >>"$OUT/logs/${obj}.log" 2>&1
  ec=$?
  echo "=== done $obj exit=$ec $(date -Is) ===" | tee -a "$OUT/logs/launch.log"
  if [ "$ec" -ne 0 ]; then
    echo "FIX1_MEASURE_FAILED object=$obj" | tee -a "$OUT/logs/launch.log"
    exit "$ec"
  fi
done

$PY -c "
import sys
from pathlib import Path
sys.path.insert(0, r'$ROOT')
from scripts.fix1_static_offset import analyze, write_report
out = Path(r'$OUT')
fit = analyze(out)
write_report(out, fit)
print('dominant', fit.get('dominant'), 'rms_mm', fit.get('residual_rms_mm'))
print('pass_pinch', fit.get('pass_pinch'))
" | tee "$OUT/logs/analyze.log"

echo FIX1_MEASURE_DONE | tee -a "$OUT/logs/launch.log"
