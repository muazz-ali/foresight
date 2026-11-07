#!/usr/bin/env bash
# Fix1 closeout: C30 seeds 1000/1001/1003, first-attempt success (egg).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:-data/eval/fix1_static_offset/c30_seeds}
mkdir -p "$OUT/logs"
# seed index i → seed 1000+i; want 1000,1001,1003 → indices 0,1,3
# run seed-start/end covers a range; run three one-seed jobs for clarity
rm -f "$OUT/STOP"
$PY "$ROOT/eval/run_b1_grid.py" --phase freeze -o "$OUT" \
  -c "$ROOT/sim/phase0_cfg.yaml" >>"$OUT/logs/freeze.log" 2>&1

for idx in 0 1 3; do
  echo "=== C30 seed_index=$idx seed=$((1000+idx)) ===" | tee -a "$OUT/logs/launch.log"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY "$ROOT/eval/run_b1_grid.py" \
    --headless --enable_cameras --device cuda:0 \
    --phase c30 --object egg --n 1 \
    --seed-start "$idx" --seed-end "$idx" \
    -o "$OUT" >>"$OUT/logs/s$((1000+idx)).log" 2>&1
  echo "=== done idx=$idx exit=$? ===" | tee -a "$OUT/logs/launch.log"
done

$PY - <<PY
import json, glob
from pathlib import Path
out = Path("$OUT")
rows = []
for p in sorted(out.glob("cells/C30/*.json")):
    if "prechek" in p.name: continue
    d = json.loads(p.read_text())
    rows.append({
        "file": p.name,
        "seed": d.get("seed"),
        "success": d.get("success"),
        "failure_mode": d.get("failure_mode"),
        "first_attempt": True,
    })
# also scrape jsonl
for p in sorted(out.glob("**/C30*.jsonl")):
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        d = json.loads(line)
        rows.append({
            "seed": d.get("seed"),
            "success": d.get("success"),
            "failure_mode": d.get("failure_mode"),
            "first_attempt": True,
            "src": str(p),
        })
# dedupe by seed
by = {}
for r in rows:
    if r.get("seed") is None: continue
    by[int(r["seed"])] = r
summary = {"seeds": by, "n_ok": sum(1 for r in by.values() if r.get("success"))}
(out / "c30_first_attempt.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
echo FIX1_C30_DONE
