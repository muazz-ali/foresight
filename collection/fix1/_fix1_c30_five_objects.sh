#!/usr/bin/env bash
# C30 first-attempt on 5 non-egg objects, one new seed each (Δ=0, 30 cm/s).
# One Isaac process per object (World cannot rebuild mid-run).
#
#   CUDA_VISIBLE_DEVICES=0 bash collection/fix1/_fix1_c30_five_objects.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/data/eval/fix1_static_offset/c30_five_objects}"
CFG="${CFG:-$ROOT/sim/phase0_cfg.yaml}"
PY="${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

# object:seed  (seeds away from the egg 1000/1001/1003 trio).
# orange is in yaml categories but every mesh fails the 65 mm width gate.
TRIALS=(
  "apple:2100"
  "lemon:2101"
  "onion:2102"
  "avocado:2103"
  "kiwi:2104"
)

mkdir -p "$OUT/logs" "$OUT/cells/C30"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

echo "OUT=$OUT"
for trial in "${TRIALS[@]}"; do
  obj="${trial%%:*}"
  seed="${trial##*:}"
  echo "=== C30 object=$obj seed=$seed ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PY" "$ROOT/collection/fix1/_fix1_c30_first_attempt.py" \
      --headless --enable_cameras --device "$DEVICE" \
      -o "$OUT" -c "$CFG" \
      --object "$obj" --seeds "$seed" --asset-seed "$seed" \
      >"$OUT/logs/${obj}_s${seed}.log" 2>&1
  echo "=== done $obj seed=$seed (exit $?) ==="
done

# Merge per-object summaries into one table.
"$PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT")
rows = []
for p in sorted(out.glob("c30_first_attempt_*.json")):
    if p.name == "c30_first_attempt.json":
        continue
    d = json.loads(p.read_text())
    rows.extend(d.get("rows") or [])
summary = {
    "note": "C30 first-attempt; 5 objects × 1 new seed; v=30 cm/s; Δ=0.",
    "n_ok": sum(1 for r in rows if r.get("success")),
    "n": len(rows),
    "rows": rows,
}
(out / "c30_five_objects.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
