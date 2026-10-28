#!/usr/bin/env bash
# Record the A/B gate set: 0/10/20/30 cm/s × seeds 31000/32000/33000 × 8 throws = 96 expert
# episodes with the CURRENT sim/phase0_cfg.yaml. Stamps the camera pose (A5 checks it) and a
# config snapshot. Starts a job only when a GPU has ≥9 GB free (each Isaac process ~10 GB).
# Usage: bash eval/record_ab_set.sh data/eval/AB_recordings_vN
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
OUT=$ROOT/${1:?usage: record_ab_set.sh OUT_DIR}
mkdir -p $OUT
$PY - <<PYE
import json,sys; sys.path.insert(0,"$ROOT")
from interfaces.config import load_yaml
c=load_yaml("$ROOT/sim/phase0_cfg.yaml")["static_camera"]
json.dump({k:c.get(k) for k in ("position","rotation_deg","focal_length","horizontal_aperture","width","height","fps")},
          open("$OUT/camera_pose.json","w"), indent=1, default=float)
print("camera stamp:", open("$OUT/camera_pose.json").read())
PYE
cp $ROOT/sim/phase0_cfg.yaml $OUT/phase0_cfg.snapshot.yaml
pids=()
for sp in 0.00 0.10 0.20 0.30; do
  for seed in 31000 32000 33000; do
    while true; do
      gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>9000{print $1; exit}')
      [ -n "$gpu" ] && break
      sleep 20
    done
    d=$OUT/v$sp/s$seed; mkdir -p $d
    CUDA_VISIBLE_DEVICES=$gpu $PY $ROOT/scripts/run_phase1_collect.py --headless --enable_cameras --device cuda:0 \
      --speed-mode fixed --speed $sp --target-successes 8 --max-attempts 8 --seed $seed --no-debug -o $d > $d/run.log 2>&1 &
    pids+=($!); echo "started v$sp s$seed gpu$gpu"; sleep 40
  done
done
wait "${pids[@]}"
echo RECORD_DONE
