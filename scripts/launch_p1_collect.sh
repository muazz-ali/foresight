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
# Env: FORESIGHT_PYTHON, OUT, SEED_BASE, DEVICE, CFG,
#      SMOKE_OBJECT, SMOKE_CONTAINER, DEBUG, SAVE_FAILURES
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Categories match sim/phase0_cfg.yaml (enabled object / container lists).
OBJECTS=(apple avocado egg kiwi lemon lime onion orange peach potato tangerine tomato)
CONTAINERS=(bowl box plate tray)
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
Usage: $(basename "$0") [smoke|diag-container|diag-timeout|N] [--debug|--no-debug] [--save-failures]

  smoke           5 successes (0/10/15/20 + extra 15); debug ON by default
  diag-container  onion+bowl @ default stage_timeout
  diag-timeout    onion+bowl @ stage_timeout_s=3.0
  N               stratified collect ≈N successes across categories
                  debug OFF by default (successes only, no mp4)

Flags (also via env):
  --debug / --no-debug     write .mp4 and (with debug) save failures
  --save-failures          save failed episodes even with --no-debug
  DEBUG=0|1                same as --no-debug / --debug
  SAVE_FAILURES=0|1

Examples:
  bash scripts/launch_p1_collect.sh 4000 --no-debug
  DEBUG=1 bash scripts/launch_p1_collect.sh 4000
  bash scripts/launch_p1_collect.sh smoke --no-debug

Env: OUT, SEED_BASE, DEVICE, FORESIGHT_PYTHON, CFG,
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
    ${DEVICE:+--device "$DEVICE"}
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
    _run_smoke "smoke" "$OUT" "$CFG" \
      "${SMOKE_OBJECT:-apple}" "${SMOKE_CONTAINER:-bowl}"
    ;;
  diag-container)
    _resolve_debug 1
    OUT="${OUT:-$ROOT/data/p1_diag_onion_bowl}"
    _run_smoke "diag-container" "$OUT" "$CFG" onion bowl
    ;;
  diag-timeout)
    _resolve_debug 1
    OUT="${OUT:-$ROOT/data/p1_diag_onion_bowl_timeout3}"
    mkdir -p "$ROOT/data"
    TMP_CFG="$ROOT/data/phase0_timeout3.yaml"
    "$PY" - "$CFG" "$TMP_CFG" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.load(open(src), Loader=yaml.FullLoader)
cfg["state_machine_params"]["stage_timeout_s"] = 3.0
yaml.dump(cfg, open(dst, "w"), default_flow_style=False, sort_keys=False)
print(f"wrote {dst} stage_timeout_s=3.0")
PY
    _run_smoke "diag-timeout" "$OUT" "$TMP_CFG" onion bowl
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
    OUT="${OUT:-$ROOT/data/p1-retrain}"
    N_OBJ=${#OBJECTS[@]}
    PER=$(( (TARGET_TOTAL + N_OBJ - 1) / N_OBJ ))
    echo "[p1] target_total=$TARGET_TOTAL per_category≈$PER out=$OUT"
    mkdir -p "$OUT"
    i=0
    for obj in "${OBJECTS[@]}"; do
      ctr="${CONTAINERS[$((i % ${#CONTAINERS[@]}))]}"
      seed=$((SEED_BASE + i * 10000))
      echo "[p1] category=$obj container=$ctr seed=$seed n=$PER"
      "$PY" "$ROOT/scripts/run_phase1_collect.py" \
        --target-successes "$PER" \
        --speed-mode stratified \
        --seed "$seed" \
        --object-usd "$obj" \
        --container-usd "$ctr" \
        -c "$CFG" \
        -o "$OUT" \
        --headless --enable_cameras \
        "${DEBUG_FLAG[@]}" \
        "${FAIL_FLAG[@]}" \
        ${DEVICE:+--device "$DEVICE"}
      i=$((i + 1))
    done
    echo "[p1] collection launcher done"
    ;;
esac
