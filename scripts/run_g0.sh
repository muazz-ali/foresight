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

Env: FORESIGHT_PYTHON, TRIALS (default 20), SEED (default 40),
  DEVICE (e.g. cuda:1 when GPU 0 is busy)

  DEVICE=cuda:N hides other GPUs (CUDA_VISIBLE_DEVICES) so Isaac/Vulkan
  never probes a full card. Does not touch other jobs on those GPUs.

Examples:
  DEVICE=cuda:1 bash scripts/run_g0.sh smoke
  DEVICE=cuda:1 bash scripts/run_g0.sh g0
EOF
}

# Conda Isaac needs its own numpy; ~/.local often shadows it and breaks startup.
export PYTHONNOUSERSITE=1

# Isaac AppLauncher --device. For cuda:N, pin visibility so Vulkan does not
# try a full GPU 0 first (OOM cascade). Inside the process that GPU is cuda:0.
DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
    DEVICE_ARGS=(--device cuda:0)
    echo "[g0] DEVICE=$DEVICE → CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --device cuda:0"
  else
    DEVICE_ARGS=(--device "$DEVICE")
  fi
fi

cmd="${1:-g0}"
case "$cmd" in
  -h|--help|help) usage; exit 0 ;;
  smoke)
    mkdir -p "$ROOT/data/g0_smoke"
    exec "$PY" "$ROOT/scripts/run_phase0.py" --speed 0.20 -n 20 --robot franka \
      --seed "${SEED:-70}" \
      --headless --enable_cameras --debug -o "$ROOT/data/g0_smoke" \
      "${DEVICE_ARGS[@]}"
    ;;
  debug-mp4)
    mkdir -p "$ROOT/data/g0_debug_mp4"
    exec "$PY" "$ROOT/scripts/run_phase0.py" --speed 0.0 -n 2 --robot franka \
      --seed "${SEED:-40}" \
      --headless --enable_cameras --debug -o "$ROOT/data/g0_debug_mp4" \
      "${DEVICE_ARGS[@]}"
    ;;
  g0|"")
    mkdir -p "$ROOT/data/g0"
    exec "$PY" "$ROOT/scripts/run_phase0.py" --g0 --robot franka \
      --seed "${SEED:-40}" \
      --headless --enable_cameras \
      --trials-per-speed "${TRIALS:-20}" \
      -o "$ROOT/data/g0" \
      "${DEVICE_ARGS[@]}"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
