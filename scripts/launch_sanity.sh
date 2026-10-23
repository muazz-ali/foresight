#!/usr/bin/env bash
# Input sanity matrix at 0 cm/s: change ONE input the policy reads, keep the
# rest, in several worlds. Every episode writes .h5/.json + .mp4 (--debug).
# Conda: dynamicVLA_isaac. Score: python eval/sanity_report.py data/eval/sanity
#
# Checks (per world):
#   base           nothing changed
#   cam_static / cam_wrist / cam_both     gray image fed for that camera
#   lang_obj / lang_ctr / lang_both       wrong object / container / both named
#   lang_generic   "Pick the object and place it in the container."
#   lang_other     an unrelated task text
#   num_zero       the 4 numbers zeroed
#   num_shift      p̂ moved +3 cm in x
#   num_vel        v̂ = (0, 0.10) m/s while the object is still
#   grip_closed    gripper slot of the state reads "closed"
#
# Usage: bash scripts/launch_sanity.sh            (4 workers: 2 per GPU)
# Env: CHECKPOINT, OUT, TRIALS, WORKERS_PER_GPU, GPUS, FORESIGHT_PYTHON
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FORESIGHT_PYTHON:-/home/gpuadmin/anaconda3/envs/dynamicVLA_isaac/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/v2_abs_noise/pretrained_model}"
OUT="${OUT:-$ROOT/data/eval/sanity}"
TRIALS="${TRIALS:-10}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
# shellcheck disable=SC2206
GPUS=(${GPUS:-0 1})

# World = asset seed (house, light, object mesh, container mesh fixed per process).
# name|seed|wrong object|wrong container
WORLDS=(
  "tangerine_bowl|9000|egg|box"
  "avocado_bowl|12000|lemon|box"
  "onion_box|10000|apple|bowl"
)

JOBS=()
for w in "${WORLDS[@]}"; do
  IFS='|' read -r wname seed wobj wctr <<<"$w"
  obj="${wname%_*}"; ctr="${wname#*_}"
  JOBS+=("$wname|$seed|base|")
  JOBS+=("$wname|$seed|cam_static|--blank-cam static")
  JOBS+=("$wname|$seed|cam_wrist|--blank-cam wrist")
  JOBS+=("$wname|$seed|cam_both|--blank-cam both")
  JOBS+=("$wname|$seed|lang_obj|--language=Pick the $wobj and place it in the $ctr.")
  JOBS+=("$wname|$seed|lang_ctr|--language=Pick the $obj and place it in the $wctr.")
  JOBS+=("$wname|$seed|lang_both|--language=Pick the $wobj and place it in the $wctr.")
  JOBS+=("$wname|$seed|lang_generic|--language=Pick the object and place it in the container.")
  JOBS+=("$wname|$seed|lang_other|--language=Open the drawer and push the button.")
  JOBS+=("$wname|$seed|num_zero|--zero-conditioning")
  JOBS+=("$wname|$seed|num_shift|--cond-shift-m 0.03 0.0")
  JOBS+=("$wname|$seed|num_vel|--cond-vel-override 0.0 0.10")
  JOBS+=("$wname|$seed|grip_closed|--gripper-state-override closed")
done

run_job() {
  local gpu="$1" job="$2"
  IFS='|' read -r wname seed check extra <<<"$job"
  local dir="$OUT/$wname/$check"
  if [[ -f "$dir/metrics.jsonl" ]] && [[ $(wc -l <"$dir/metrics.jsonl") -ge $TRIALS ]]; then
    echo "[sanity] skip $wname/$check (done)"; return 0
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  local args=()
  if [[ "$extra" == --language=* ]]; then
    args=(--language "${extra#--language=}")
  elif [[ -n "$extra" ]]; then
    read -r -a args <<<"$extra"
  fi
  echo "[sanity] gpu$gpu $wname/$check ${args[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_policy.py" \
    --checkpoint "$CHECKPOINT" --headless --enable_cameras --debug --device cuda:0 \
    --speeds 0.0 --trials-per-speed "$TRIALS" --seed "$seed" --n-action-steps 8 \
    --cond-source oracle "${args[@]}" -o "$dir" >"$dir/run.log" 2>&1
  echo "[sanity] gpu$gpu $wname/$check exit=$?"
}

# Round-robin the jobs over GPUS × WORKERS_PER_GPU background workers.
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
echo "[sanity] all done → $OUT"
