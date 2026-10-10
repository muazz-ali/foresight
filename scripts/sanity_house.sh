#!/usr/bin/env bash
# House-mode sanity: 10 different seeds (≈10 houses). Need ≥8 successes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

OUT="${OUT:-$ROOT/data/house_sanity}"
SPEED="${SPEED:-0.10}"
N_SEEDS="${N_SEEDS:-10}"
BASE_SEED="${BASE_SEED:-40}"
mkdir -p "$OUT"

pass=0
fail=0
declare -a results

for i in $(seq 0 $((N_SEEDS - 1))); do
  seed=$((BASE_SEED + i))
  ep_out="$OUT/seed_${seed}"
  rm -rf "$ep_out"
  mkdir -p "$ep_out"
  echo "======== seed $seed (house run $((i + 1))/$N_SEEDS) ========"
  set +e
  "$PY" "$ROOT/scripts/run_phase0.py" \
    --speed "$SPEED" -n 1 --robot franka --seed "$seed" \
    --headless --enable_cameras \
    -o "$ep_out" >"$ep_out/run.log" 2>&1
  rc=$?
  set -e

  # Isaac sometimes exits 0 after a Traceback during shutdown — treat traceback as fail.
  if grep -q '^Traceback (most recent call last):' "$ep_out/run.log"; then
    rc=1
  fi

  ok=0
  if [[ "$rc" -eq 0 && -f "$ep_out/metrics.jsonl" ]]; then
    if "$PY" - <<PY
import json, sys
ok = False
with open("$ep_out/metrics.jsonl") as fp:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("success") is True:
            ok = True
            break
sys.exit(0 if ok else 1)
PY
    then
      ok=1
    fi
  fi

  # Show the last useful lines for triage.
  tail -n 40 "$ep_out/run.log" || true

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

need=8
if [[ "$N_SEEDS" -lt 8 ]]; then
  need=$N_SEEDS
fi

echo
echo "======== HOUSE SANITY SUMMARY ========"
printf '%s\n' "${results[@]}"
echo "pass=$pass fail=$fail / $N_SEEDS"
if [[ "$pass" -ge "$need" ]]; then
  echo "GATE: OK (≥${need}/${N_SEEDS})"
  exit 0
fi
echo "GATE: FAIL (need ≥${need}/${N_SEEDS})"
exit 1
