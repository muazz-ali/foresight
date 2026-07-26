#!/usr/bin/env bash
# Run Gate G0 (Phase 0 Franka expert speed sweep). No DynamicVLA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-${DYNAMICVLA_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ "${1:-}" == "smoke" ]]; then
  mkdir -p "$ROOT/data/g0_smoke"
  exec "$PY" "$ROOT/scripts/run_phase0.py" --speed 0.0 -n 1 --robot franka \
    --headless --enable_cameras --debug -o "$ROOT/data/g0_smoke"
fi

mkdir -p "$ROOT/data/g0"
exec "$PY" "$ROOT/scripts/run_phase0.py" --g0 --robot franka \
  --headless --enable_cameras \
  --trials-per-speed "${TRIALS:-20}" \
  -o "$ROOT/data/g0"
