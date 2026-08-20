#!/usr/bin/env bash
# Gate G0 helpers (Phase 0 Franka expert).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [smoke|debug-mp4|g0]

  smoke       1 static episode + debug MP4  → data/g0_smoke/
  debug-mp4   2 episodes + stitched MP4     → data/g0_debug_mp4/
  g0          full speed sweep (default)    → data/g0/

Env: FORESIGHT_PYTHON, TRIALS (default 20), SEED (default 40)
  Meshes fixed per process; diversify with distinct SEED / workers.
EOF
}

cmd="${1:-g0}"
case "$cmd" in
  -h|--help|help) usage; exit 0 ;;
  smoke)
    mkdir -p "$ROOT/data/g0_smoke"
    exec "$PY" "$ROOT/scripts/run_phase0.py" --speed 0.0 -n 1 --robot franka \
      --seed "${SEED:-40}" \
      --headless --enable_cameras --debug -o "$ROOT/data/g0_smoke"
    ;;
  debug-mp4)
    mkdir -p "$ROOT/data/g0_debug_mp4"
    exec "$PY" "$ROOT/scripts/run_phase0.py" --speed 0.0 -n 2 --robot franka \
      --seed "${SEED:-40}" \
      --headless --enable_cameras --debug -o "$ROOT/data/g0_debug_mp4"
    ;;
  g0|"")
    mkdir -p "$ROOT/data/g0"
    exec "$PY" "$ROOT/scripts/run_phase0.py" --g0 --robot franka \
      --seed "${SEED:-40}" \
      --headless --enable_cameras \
      --trials-per-speed "${TRIALS:-20}" \
      -o "$ROOT/data/g0"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
