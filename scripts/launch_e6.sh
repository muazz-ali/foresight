#!/usr/bin/env bash
# E6: scripted-expert ceiling vs speed, hand yaw fixed vs pads-along-motion.
# Expert only (no policy). n attempts per cell; success = sim/success.py rule.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
OUT="${OUT:-$ROOT/data/eval/E6}"
N="${N:-40}"
GPU="${GPU:-1}"
WORKERS="${WORKERS:-2}"
# shellcheck disable=SC2206
SPEEDS=(${SPEEDS:-0.20 0.30 0.40 0.50})
OBJ=/home/gpuadmin/Desktop/muazzam/objects/tangerine/tangerine06.usd
CTR=/home/gpuadmin/Desktop/muazzam/objects/bowl/bowl16.usd
JOBS=()
for sp in "${SPEEDS[@]}"; do for mode in fixed along; do JOBS+=("$mode|$sp"); done; done
run_job() {
  local mode sp dir; IFS='|' read -r mode sp <<<"$1"
  dir="$OUT/$mode/v$sp"
  if [[ -f "$dir/metrics.jsonl" ]] && [[ $(wc -l <"$dir/metrics.jsonl") -ge $N ]]; then echo "[e6] skip $dir"; return; fi
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[e6] $dir start $(date +%T)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/scripts/run_phase1_collect.py" \
    --target-successes "$N" --max-attempts "$N" --speed-mode fixed --speed "$sp" \
    --seed 9000 --object-usd "$OBJ" --container-usd "$CTR" -c "$OUT/cfg_$mode.yaml" \
    -o "$dir" --headless --enable_cameras --no-debug --no-save >"$dir/run.log" 2>&1
  echo "[e6] $dir exit=$? $(date +%T)"
}
for ((k = 0; k < WORKERS; k++)); do
  ( sleep $((k * 8)); for ((j = k; j < ${#JOBS[@]}; j += WORKERS)); do run_job "${JOBS[$j]}"; done ) &
done
wait
echo "[e6] all done $(date +%T)"
