#!/usr/bin/env bash
# Pre-collect smoke: many houses × objects × speeds (≥20). One Isaac process
# per (seed, speed) so asset_seed picks a fresh house + mesh each time.
#
# Usage:
#   bash scripts/smoke_diverse_precollect.sh
#   N=24 SPEEDS="0.0 0.10 0.15 0.20" GPUS="0 1" bash scripts/smoke_diverse_precollect.sh
#
# Env: OUT, N, SPEEDS, BASE_SEED, GPUS, DEBUG, FORESIGHT_PYTHON
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

OUT="${OUT:-$ROOT/data/smoke_precollect_diverse}"
N="${N:-20}"
# shellcheck disable=SC2206
SPEEDS=(${SPEEDS:-0.0 0.10 0.15 0.20})
BASE_SEED="${BASE_SEED:-50000}"
# shellcheck disable=SC2206
GPUS=(${GPUS:-0 1})
DEBUG_FLAG=(--debug)
if [[ "${DEBUG:-1}" == "0" ]]; then
  DEBUG_FLAG=()
fi

mkdir -p "$OUT"
JOBS_FILE="$OUT/jobs.tsv"
SUMMARY="$OUT/summary.jsonl"
: >"$SUMMARY"

# jobs: index \t seed \t speed
{
  for ((i = 0; i < N; i++)); do
    seed=$((BASE_SEED + i * 97))
    sp="${SPEEDS[$((i % ${#SPEEDS[@]}))]}"
    printf '%d\t%d\t%s\n' "$i" "$seed" "$sp"
  done
} | tee "$JOBS_FILE"

run_one() {
  local gpu="$1" idx="$2" seed="$3" speed="$4"
  local tag
  tag=$(printf 'i%02d_s%d_v%s' "$idx" "$seed" "$speed")
  local ep_out="$OUT/$tag"
  rm -rf "$ep_out"
  mkdir -p "$ep_out"
  echo "[smoke] gpu$gpu start $tag"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/run_phase0.py" \
    --speed "$speed" -n 1 --robot franka --seed "$seed" \
    --headless --enable_cameras "${DEBUG_FLAG[@]}" --device cuda:0 \
    -o "$ep_out" >"$ep_out/run.log" 2>&1
  local rc=$?
  set -e
  if grep -q '^Traceback (most recent call last):' "$ep_out/run.log"; then
    rc=1
  fi
  "$PY" - <<PY
import json, re
from pathlib import Path
out = Path("$ep_out")
log = (out / "run.log").read_text(errors="replace")
house = obj = box = fail = "?"
m = re.search(r"House (\S+)\s+table (\S+)", log)
if m:
    house = f"{m.group(1)}/{m.group(2)}"
m = re.search(r"Object mesh:\s+(\S+)", log)
if m:
    obj = m.group(1)
m = re.search(r"Box mesh:\s+(\S+)", log)
if m:
    box = m.group(1)
ok = False
mp = out / "metrics.jsonl"
rec = {}
if mp.is_file():
    for line in mp.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ok = bool(rec.get("success"))
        fail = rec.get("failure", fail)
row = {
    "tag": "$tag",
    "gpu": int("$gpu"),
    "idx": int("$idx"),
    "seed": int("$seed"),
    "speed_m_s": float("$speed"),
    "ok": ok,
    "rc": int("$rc"),
    "failure": fail,
    "house": house,
    "object": obj,
    "container": box,
    "max_stage": rec.get("max_stage"),
    "z_max": rec.get("z_max"),
    "abort_reason": rec.get("abort_reason"),
}
with open("$SUMMARY", "a") as fp:
    fp.write(json.dumps(row) + "\n")
print(
    f"[{'PASS' if ok else 'FAIL'}] $tag house={house} obj={obj} "
    f"fail={fail} z_max={rec.get('z_max')} abort={rec.get('abort_reason')}"
)
PY
}

# Round-robin jobs across GPUs (one worker per GPU).
n_gpus=${#GPUS[@]}
pids=()
for ((g = 0; g < n_gpus; g++)); do
  (
    while IFS=$'\t' read -r idx seed speed; do
      if (( idx % n_gpus == g )); then
        run_one "${GPUS[$g]}" "$idx" "$seed" "$speed"
      fi
    done <"$JOBS_FILE"
  ) &
  pids+=($!)
done
ec=0
for pid in "${pids[@]}"; do
  wait "$pid" || ec=1
done

"$PY" - <<PY
import json
from collections import Counter
from pathlib import Path
rows = []
for line in Path("$SUMMARY").read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))
rows.sort(key=lambda r: r["idx"])
n = len(rows)
ok = sum(1 for r in rows if r["ok"])
print()
print("======== PRE-COLLECT DIVERSE SMOKE ========")
print(f"pass={ok} fail={n - ok} / {n}")
print("by speed:")
by_sp = {}
for r in rows:
    by_sp.setdefault(r["speed_m_s"], []).append(r)
for sp in sorted(by_sp):
    rs = by_sp[sp]
    p = sum(1 for r in rs if r["ok"])
    print(f"  {sp*100:.0f} cm/s: {p}/{len(rs)}")
print("failures:", Counter(r["failure"] for r in rows if not r["ok"]))
print("objects:", Counter(r["object"] for r in rows))
houses = {r["house"] for r in rows}
print(f"unique houses: {len(houses)}")
for r in rows:
    mark = "PASS" if r["ok"] else "FAIL"
    print(
        f"  [{mark}] i={r['idx']:02d} seed={r['seed']} "
        f"v={r['speed_m_s']:.2f} obj={r['object']} fail={r['failure']}"
    )
need = max(1, int(0.7 * n + 0.999))  # ceil(0.7 n)
report = {
    "n": n,
    "pass": ok,
    "fail": n - ok,
    "rate": (ok / n) if n else 0.0,
    "need_pass": need,
    "by_speed": {
        str(sp): {
            "n": len(rs),
            "pass": sum(1 for r in rs if r["ok"]),
        }
        for sp, rs in sorted(by_sp.items())
    },
    "failure_counts": dict(Counter(r["failure"] for r in rows if not r["ok"])),
    "unique_houses": len(houses),
    "unique_objects": len({r["object"] for r in rows}),
}
Path("$OUT/report.json").write_text(json.dumps(report, indent=2))
print(f"wrote $OUT/report.json")
print(f"smoke bar: need ≥{need}/{n} (70%) → {'OK' if ok >= need else 'FAIL'}")
# Smoke bar: ≥70% overall (same as G0 per-bin line); not a gate claim.
raise SystemExit(0 if ok >= need else 1)
PY
smoke_ec=$?
exit $smoke_ec
