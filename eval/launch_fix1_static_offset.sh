#!/usr/bin/env bash
# Fix-1 static offset: egg / lemon / onion workers (one Isaac scene each).
# Usage: bash eval/launch_fix1_static_offset.sh [OUT_DIR]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:-data/eval/fix1_static_offset}
mkdir -p "$OUT/logs"
rm -f "$OUT"/.lock_* "$OUT/settles.jsonl"

for obj in egg lemon onion; do
  echo "=== fix1 measure object=$obj ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY "$ROOT/scripts/fix1_static_offset.py" \
    --headless --enable_cameras --device cuda:0 \
    --object "$obj" -o "$OUT" \
    2>&1 | tee "$OUT/logs/${obj}.log"
done

$PY -c "
import sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from scripts.fix1_static_offset import analyze, write_report
out = Path('$OUT')
fit = analyze(out)
write_report(out, fit)
print('dominant', fit.get('dominant'), 'rms_mm', fit.get('residual_rms_mm'))
"
echo FIX1_MEASURE_DONE
