#!/usr/bin/env bash
# House smoke: N random seeds, one episode each. Same --seed drives house +
# object mesh + container mesh (asset_seed == seed in run_phase0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

OUT="${OUT:-$ROOT/data/house_smoke_20cms}"
SPEED="${SPEED:-0.20}"
N_SEEDS="${N_SEEDS:-5}"
DEBUG_FLAG=()
if [[ "${DEBUG:-1}" == "1" ]]; then
  DEBUG_FLAG=(--debug)
fi

mkdir -p "$OUT"
# Draw distinct random seeds once; write them so the run is replayable.
SEEDS_FILE="$OUT/seeds.txt"
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  seeds=($SEEDS)
else
  mapfile -t seeds < <("$PY" - <<PY
import secrets
n = int("$N_SEEDS")
seen = set()
out = []
while len(out) < n:
    s = secrets.randbelow(100_000)
    if s in seen:
        continue
    seen.add(s)
    out.append(s)
print("\n".join(str(x) for x in out))
PY
)
fi
printf '%s\n' "${seeds[@]}" | tee "$SEEDS_FILE"

pass=0
fail=0
declare -a results

for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"
  ep_out="$OUT/seed_${seed}"
  rm -rf "$ep_out"
  mkdir -p "$ep_out"
  echo "======== seed $seed (run $((i + 1))/${#seeds[@]} @ ${SPEED} m/s) ========"
  set +e
  "$PY" "$ROOT/scripts/run_phase0.py" \
    --speed "$SPEED" -n 1 --robot franka --seed "$seed" \
    --headless --enable_cameras "${DEBUG_FLAG[@]}" \
    -o "$ep_out" >"$ep_out/run.log" 2>&1
  rc=$?
  set -e

  if grep -q '^Traceback (most recent call last):' "$ep_out/run.log"; then
    rc=1
  fi

  # Pull house / object / container from the log for the summary.
  meta=$("$PY" - <<PY
import json, glob, re, sys
from pathlib import Path
out = Path("$ep_out")
house = obj = box = "?"
log = (out / "run.log").read_text(errors="replace")
m = re.search(r"House (\S+)\s+table (\S+)", log)
if m:
    house = f"{m.group(1)} / {m.group(2)}"
m = re.search(r"Object mesh:\s+(\S+)", log)
if m:
    obj = m.group(1)
m = re.search(r"Box mesh:\s+(\S+)", log)
if m:
    box = m.group(1)
ok = False
mp = out / "metrics.jsonl"
if mp.is_file():
    for line in mp.read_text().splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("success") is True:
            ok = True
            break
mp4s = sorted(out.glob("**/*.mp4"))
print(json.dumps({"ok": ok, "house": house, "object": obj, "container": box,
                  "mp4": str(mp4s[0]) if mp4s else ""}))
PY
)

  ok=$(echo "$meta" | "$PY" -c 'import json,sys; print(1 if json.load(sys.stdin)["ok"] else 0)')
  echo "$meta" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
print("  house=%s" % d["house"])
print("  object=%s  container=%s" % (d["object"], d["container"]))
print("  mp4=%s" % (d["mp4"] or "(none)"))
'

  tail -n 25 "$ep_out/run.log" || true

  if [[ "$ok" -eq 1 ]]; then
    pass=$((pass + 1))
    results+=("seed $seed: PASS")
    echo "→ PASS"
  else
    fail=$((fail + 1))
    results+=("seed $seed: FAIL (rc=$rc)")
    echo "→ FAIL (rc=$rc)"
  fi
done

need=${#seeds[@]}
echo
echo "======== HOUSE SMOKE SUMMARY @ ${SPEED} m/s ========"
printf '%s\n' "${results[@]}"
echo "pass=$pass fail=$fail / $need"
echo "seeds: ${seeds[*]}  (saved $SEEDS_FILE)"
if [[ "$pass" -ge "$need" ]]; then
  echo "GATE: OK (all $need)"
  exit 0
fi
echo "GATE: FAIL (need all $need)"
exit 1
