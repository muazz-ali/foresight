#!/usr/bin/env bash
# Generic eval launcher for the research-plan-v2 experiments (E1-E6).
# JOBS file: one job per line:  <out_dir>|<speed>|<extra eval_policy.py args>
# Each job runs scripts/eval_policy.py with the shared flags below plus its extra args.
# Env: JOBS (file), CHECKPOINT, TRIALS, SEED, WORKERS_PER_GPU, GPUS, DEBUG (1 = mp4 + images)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/v2_abs_noise/pretrained_model}"
TRIALS="${TRIALS:-30}"
SEED="${SEED:-9000}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
# shellcheck disable=SC2206
GPUS=(${GPUS:-0})
DEBUG_FLAG=()
[[ "${DEBUG:-0}" == "1" ]] && DEBUG_FLAG=(--debug)
mapfile -t LINES < <(grep -v '^\s*#' "$JOBS" | grep -v '^\s*$')

run_job() {
  local gpu="$1" line="$2" dir sp extra
  IFS='|' read -r dir sp extra <<<"$line"
  if [[ -f "$dir/metrics.jsonl" ]] && [[ $(wc -l <"$dir/metrics.jsonl") -ge $TRIALS ]]; then
    echo "[jobs] skip $dir (done)"; return 0
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[jobs] gpu$gpu $dir v$sp start $(date +%T)"
  local -a xargs_=()
  eval "xargs_=($extra)"  # extra args may contain quoted strings (e.g. --language "...")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$CHECKPOINT" --headless --enable_cameras "${DEBUG_FLAG[@]}" --device cuda:0 \
    --speeds "$sp" --trials-per-speed "$TRIALS" --seed "$SEED" --n-action-steps 8 "${xargs_[@]}" \
    -o "$dir" >"$dir/run.log" 2>&1
  echo "[jobs] gpu$gpu $dir exit=$? $(date +%T)"
}

N_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
for ((k = 0; k < N_WORKERS; k++)); do
  gpu="${GPUS[$((k % ${#GPUS[@]}))]}"
  (
    sleep $(( k * 8 ))  # stagger Isaac start-up (start-up race seen with 6 at once)
    for ((j = k; j < ${#LINES[@]}; j += N_WORKERS)); do run_job "$gpu" "${LINES[$j]}"; done
  ) &
done
wait
echo "[jobs] all done $(date +%T)"
