#!/usr/bin/env bash
# Phase-1 oracle collection (scripted expert → h5/json[/mp4]).
# Conda: dynamicVLA_isaac
#
# Usage:
#   bash scripts/launch_p1_collect.sh smoke
#   bash scripts/launch_p1_collect.sh 4000 --no-debug   # successes only, no mp4
#   bash scripts/launch_p1_collect.sh 4000 --debug      # mp4 + save failures too
#   DEBUG=0 bash scripts/launch_p1_collect.sh 4000      # same as --no-debug
#
# Env: FORESIGHT_PYTHON, OUT, SEED_BASE, DEVICE, CFG, CHUNK,
#      SMOKE_OBJECT, SMOKE_CONTAINER, DEBUG, SAVE_FAILURES
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Conda Isaac needs its own numpy; ~/.local often shadows it and breaks startup.
export PYTHONNOUSERSITE=1

# Same GPU rule as scripts/run_g0.sh: DEVICE=cuda:N hides the other GPUs so
# Isaac/Vulkan never probes a full card. Inside the process that GPU is cuda:0.
DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
    DEVICE_ARGS=(--device cuda:0)
    echo "[p1] DEVICE=$DEVICE → CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --device cuda:0"
  else
    DEVICE_ARGS=(--device "$DEVICE")
  fi
fi

# House, table, light, object mesh and container mesh are fixed per process
# (asset_seed = --seed). Split each category into chunks of CHUNK successes,
# each its own process and seed, so the dataset sees many worlds, not 13.
CHUNK="${CHUNK:-25}"

# Keep in sync with scene.object_categories / container_categories in phase0_cfg.yaml
# (exclude tomato/beer/bottle/cup/potato/plate/tray — see phase0_cfg exclude_usd_*).
OBJECTS=(apple avocado can egg kiwi lemon lime onion orange peach tangerine)
CONTAINERS=(bowl box)
SEED_BASE="${SEED_BASE:-1000}"
CFG="${CFG:-$ROOT/sim/phase0_cfg.yaml}"

# Debug / mp4 flag: CLI --debug|--no-debug wins over env DEBUG=0|1.
# Default: ON for smoke/diag, OFF for large stratified N (set below per mode).
DEBUG_CLI=""
SAVE_FAILURES="${SAVE_FAILURES:-0}"
CMD=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) DEBUG_CLI=1; shift ;;
    --no-debug) DEBUG_CLI=0; shift ;;
    --save-failures) SAVE_FAILURES=1; shift ;;
    --no-save-failures) SAVE_FAILURES=0; shift ;;
    -h|--help|help) CMD=help; shift ;;
    *)
      if [[ -z "$CMD" ]]; then
        CMD="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done
CMD="${CMD:-2000}"

_resolve_debug() {
  # $1 = default when neither CLI nor env set (1 or 0)
  local default="$1"
  if [[ -n "$DEBUG_CLI" ]]; then
    DEBUG="$DEBUG_CLI"
  elif [[ -n "${DEBUG+x}" && -n "${DEBUG}" ]]; then
    case "${DEBUG}" in
      1|true|TRUE|yes|YES|on|ON) DEBUG=1 ;;
      0|false|FALSE|no|NO|off|OFF) DEBUG=0 ;;
      *) echo "[p1] bad DEBUG=${DEBUG} (use 0 or 1)" >&2; exit 1 ;;
    esac
  else
    DEBUG="$default"
  fi
  if [[ "$DEBUG" == "1" ]]; then
    DEBUG_FLAG=(--debug)
  else
    DEBUG_FLAG=(--no-debug)
  fi
  if [[ "$SAVE_FAILURES" == "1" ]]; then
    FAIL_FLAG=(--save-failures)
  else
    FAIL_FLAG=()
  fi
  echo "[p1] debug/mp4=$([[ $DEBUG == 1 ]] && echo ON || echo OFF) save_failures=$([[ $SAVE_FAILURES == 1 ]] && echo ON || echo OFF)"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [smoke|diag-container|N] [--debug|--no-debug] [--save-failures]

  smoke           5 successes (0/10/15/20 + extra 15); debug ON by default
  diag-container  onion+bowl smoke (fit / place check)
  N               stratified collect ≈N CLEAN successes across categories,
                  CHUNK successes per Isaac process (default 25; each process
                  draws its own house / light / meshes). Default OUT=data/p3_throw.
                  debug OFF by default (successes only, no mp4)

Misses that leave the table abort the episode (sim/collect.py off_table stop).
There is no separate stage-timeout diag — that path was a no-op under the mm/ms loader.

Flags (also via env):
  --debug / --no-debug     write .mp4 and (with debug) save failures
  --save-failures          save failed episodes even with --no-debug
  DEBUG=0|1                same as --no-debug / --debug
  SAVE_FAILURES=0|1

Examples:
  bash scripts/launch_p1_collect.sh 4000 --no-debug
  DEBUG=1 bash scripts/launch_p1_collect.sh 4000
  bash scripts/launch_p1_collect.sh smoke --no-debug

Env: OUT, SEED_BASE, DEVICE (e.g. cuda:1), CHUNK, FORESIGHT_PYTHON, CFG,
     SMOKE_OBJECT, SMOKE_CONTAINER
EOF
}

_run_smoke() {
  local label="$1"
  local out="$2"
  local cfg="$3"
  local obj="$4"
  local ctr="$5"
  mkdir -p "$out"
  echo "[p1] $label → $out  obj=$obj ctr=$ctr cfg=$cfg"
  "$PY" "$ROOT/scripts/run_phase1_collect.py" \
    --target-successes 5 \
    --max-attempts 40 \
    --speed-mode list \
    --speed-list 0.0 0.10 0.15 0.20 0.15 \
    --seed "${SEED_BASE}" \
    --object-usd "$obj" \
    --container-usd "$ctr" \
    -c "$cfg" \
    -o "$out" \
    --headless --enable_cameras \
    "${DEBUG_FLAG[@]}" \
    "${FAIL_FLAG[@]}" \
    "${DEVICE_ARGS[@]}"
  echo "[p1] $label collect done → $out"
  if [[ "$DEBUG" == "1" ]] || [[ -n "$(find "$out" -name '*.h5' 2>/dev/null | head -1)" ]]; then
    "$PY" "$ROOT/scripts/check_collect_recordings.py" --root "$out" -c "$cfg" || true
  fi
  if [[ -f "$out/collect_report.json" ]]; then
    "$PY" -c "import json; r=json.load(open('$out/collect_report.json')); print(json.dumps({k:r[k] for k in ('successes','attempts','success_rate','elapsed_s')}, indent=2))"
  fi
}

case "$CMD" in
  help) usage; exit 0 ;;
  smoke)
    _resolve_debug 1
    OUT="${OUT:-$ROOT/data/p1_smoke}"
    # Default apple+bowl: bowls are open-top (XY fit only) in phase0_cfg.yaml.
    _run_smoke "smoke" "$OUT" "$CFG" \
      "${SMOKE_OBJECT:-apple}" "${SMOKE_CONTAINER:-bowl}"
    ;;
  diag-container)
    _resolve_debug 1
    OUT="${OUT:-$ROOT/data/p1_diag_onion_bowl}"
    _run_smoke "diag-container" "$OUT" "$CFG" onion bowl
    ;;
  diag-timeout)
    echo "[p1] diag-timeout removed: writing stage_timeout_s was a no-op (mm/ms loader)." >&2
    echo "[p1] Off-table misses already abort the episode in sim/collect.py." >&2
    echo "[p1] Use: bash scripts/launch_p1_collect.sh diag-container" >&2
    exit 2
    ;;
  *)
    if ! [[ "$CMD" =~ ^[0-9]+$ ]]; then
      echo "Unknown command: $CMD" >&2
      usage >&2
      exit 1
    fi
    # Large collects: no mp4 / successes-only unless user opts in.
    _resolve_debug 0
    TARGET_TOTAL="$CMD"
    # Not data/p1-retrain: that name belongs to the old bounce-box dataset that
    # select_box_episodes.py and the plan's old-data mix still read.
    OUT="${OUT:-$ROOT/data/p3_throw}"
    N_OBJ=${#OBJECTS[@]}
    PER=$(( (TARGET_TOTAL + N_OBJ - 1) / N_OBJ ))
    if ! [[ "$CHUNK" =~ ^[0-9]+$ ]] || [[ "$CHUNK" -lt 1 ]]; then
      echo "[p1] bad CHUNK=$CHUNK (positive integer)" >&2; exit 1
    fi
    N_CHUNKS=$(( (PER + CHUNK - 1) / CHUNK ))
    echo "[p1] target_total=$TARGET_TOTAL per_category≈$PER chunk=$CHUNK chunks/category=$N_CHUNKS out=$OUT"
    echo "[p1] objects=${OBJECTS[*]}"
    echo "[p1] containers=${CONTAINERS[*]}"
    mkdir -p "$OUT"
    i=0
    fails=0
    for obj in "${OBJECTS[@]}"; do
      left=$PER
      for (( c=0; c<N_CHUNKS; c++ )); do
        n=$(( left < CHUNK ? left : CHUNK ))
        left=$(( left - n ))
        # Alternate containers per chunk so every category sees bowl and box.
        ctr="${CONTAINERS[$(( (i + c) % ${#CONTAINERS[@]} ))]}"
        # Category stride 100000, chunk stride 1000: a chunk walks one seed per
        # attempt (≤ max(3n, n+50)), so seeds never overlap for CHUNK ≤ 333.
        seed=$((SEED_BASE + i * 100000 + c * 1000))
        echo "[p1] category=$obj chunk=$((c + 1))/$N_CHUNKS container=$ctr seed=$seed n=$n"
        # Do not stop the whole job if one pairing has no usable meshes.
        set +e
        "$PY" "$ROOT/scripts/run_phase1_collect.py" \
          --target-successes "$n" \
          --speed-mode stratified \
          --seed "$seed" \
          --object-usd "$obj" \
          --container-usd "$ctr" \
          -c "$CFG" \
          -o "$OUT" \
          --headless --enable_cameras \
          "${DEBUG_FLAG[@]}" \
          "${FAIL_FLAG[@]}" \
          "${DEVICE_ARGS[@]}"
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
          echo "[p1] WARN: category=$obj chunk=$((c + 1)) container=$ctr failed (exit $rc); continuing" >&2
          fails=$((fails + 1))
        fi
      done
      i=$((i + 1))
    done
    echo "[p1] collection launcher done (chunk_failures=$fails)"
    if [[ "$fails" -gt 0 ]]; then
      exit 1
    fi
    ;;
esac
