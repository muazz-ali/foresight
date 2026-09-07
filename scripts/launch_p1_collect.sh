#!/usr/bin/env bash
# Launch Phase-1 collection across object categories (disjoint seeds).
# Conda: dynamicVLA_isaac
#
# Usage:
#   bash scripts/launch_p1_collect.sh 2000
#   bash scripts/launch_p1_collect.sh 20   # smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_TOTAL="${1:-2000}"
OUT="${OUT:-$ROOT/data/p1-retrain}"
OBJECTS=(apple avocado can cup lemon orange peach potato tangerine tomato)
CONTAINERS=(bowl plate placemat tray)
N_OBJ=${#OBJECTS[@]}
PER=$(( (TARGET_TOTAL + N_OBJ - 1) / N_OBJ ))
SEED_BASE="${SEED_BASE:-1000}"

echo "[p1] target_total=$TARGET_TOTAL per_category≈$PER out=$OUT"
mkdir -p "$OUT"

i=0
for obj in "${OBJECTS[@]}"; do
  ctr="${CONTAINERS[$((i % ${#CONTAINERS[@]}))]}"
  seed=$((SEED_BASE + i * 10000))
  echo "[p1] category=$obj container=$ctr seed=$seed n=$PER"
  python "$ROOT/scripts/run_phase1_collect.py" \
    --target-successes "$PER" \
    --speed-mode stratified \
    --seed "$seed" \
    --object-usd "$obj" \
    --container-usd "$ctr" \
    -o "$OUT" \
    --headless --enable_cameras \
    ${DEVICE:+--device "$DEVICE"}
  i=$((i + 1))
done

echo "[p1] collection launcher done"
