#!/usr/bin/env bash
# Stage 1: closed-loop A/B of the P_s close rule (control/capture.py) on one world.
# Same checkpoint, same seeds, sim-truth numbers; only the close timing differs.
#   base   --ps-close observe   policy decides when to close (P_s only logged)
#   time   --ps-close time      + close at the P_s peak (P_s >= 0.5, not rising)
#   veto   --ps-close veto      + time, and hold a policy close while P_s < 0.5 (<= 8 frames)
#   dist   --ps-close dist      control: close when distance <= 2.56 cm and not shrinking
# Score: python eval/stage1_report.py data/eval/stage1
#
# Env: CHECKPOINT, OUT, TRIALS, SEED (asset world; 9000 = tangerine06 + bowl16),
#      ARMS, SPEEDS, WORKERS_PER_GPU, GPUS, DEBUG, FORESIGHT_PYTHON
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/v2_abs_noise/pretrained_model}"
OUT="${OUT:-$ROOT/data/eval/stage1}"
TRIALS="${TRIALS:-40}"
SEED="${SEED:-9000}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
# shellcheck disable=SC2206
GPUS=(${GPUS:-0 1})
# shellcheck disable=SC2206
ARMS=(${ARMS:-base time veto})
# shellcheck disable=SC2206
SPEEDS=(${SPEEDS:-0.0 0.10 0.15 0.20})
DEBUG_FLAG=(--debug)
[[ "${DEBUG:-1}" == "0" ]] && DEBUG_FLAG=()

JOBS=()
for sp in "${SPEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do JOBS+=("$arm|$sp"); done
done

run_job() {
  local gpu="$1" job="$2" arm sp mode dir
  IFS='|' read -r arm sp <<<"$job"
  case "$arm" in base) mode=observe ;; time) mode=time ;; veto) mode=veto ;; dist) mode=dist ;;
    *) echo "[stage1] unknown arm $arm" >&2; return 2 ;; esac
  dir="$OUT/$arm/v$sp"
  if [[ -f "$dir/metrics.jsonl" ]] && [[ $(wc -l <"$dir/metrics.jsonl") -ge $TRIALS ]]; then
    echo "[stage1] skip $arm v$sp (done)"; return 0
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[stage1] gpu$gpu $arm v$sp start $(date +%T)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$CHECKPOINT" --headless --enable_cameras "${DEBUG_FLAG[@]}" --device cuda:0 \
    --speeds "$sp" --trials-per-speed "$TRIALS" --seed "$SEED" --n-action-steps 8 \
    --cond-source oracle --ps-close "$mode" --ps-min 0.5 --ps-max-wait 8 \
    -o "$dir" >"$dir/run.log" 2>&1
  echo "[stage1] gpu$gpu $arm v$sp exit=$? $(date +%T)"
}

N_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
for ((k = 0; k < N_WORKERS; k++)); do
  gpu="${GPUS[$((k % ${#GPUS[@]}))]}"
  (
    for ((j = k; j < ${#JOBS[@]}; j += N_WORKERS)); do
      run_job "$gpu" "${JOBS[$j]}"
    done
  ) &
done
wait
echo "[stage1] all done → $OUT"
