#!/usr/bin/env bash
# Research plan v2 queue (E1-E6). GPU0: E4 -> E2 -> E3(V2) -> V2 extra seeds.
# GPU1: wait for E6 + E5 training, then E5 evals and E3 (no-numbers) arms.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
L=scripts/launch_jobs.sh
E5CK="$ROOT/runs/E5_no_numbers/pretrained_model"
gpu0() {
  # Wait for the E1 launcher (its PID is passed in; pgrep on job names can miss the gap between jobs).
  for p in ${WAIT_GPU0:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
  echo "[queue] gpu0 E4 $(date +%T)"; JOBS=data/eval/E4/jobs.txt TRIALS=40 GPUS=0 WORKERS_PER_GPU=3 $L
  echo "[queue] gpu0 E2 $(date +%T)"; JOBS=data/eval/E2/jobs.txt TRIALS=40 GPUS=0 WORKERS_PER_GPU=3 $L
  echo "[queue] gpu0 E3-v2 $(date +%T)"; JOBS=data/eval/E3/jobs_v2.txt TRIALS=40 GPUS=0 WORKERS_PER_GPU=2 $L
  echo "[queue] gpu0 V2-extra $(date +%T)"; JOBS=data/eval/E5/jobs_v2extra.txt TRIALS=40 GPUS=0 WORKERS_PER_GPU=2 $L
  echo "[queue] gpu0 done $(date +%T)"
}
gpu1() {
  for p in ${WAIT_GPU1:-}; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
  if [[ ! -f "$E5CK/model.safetensors" ]]; then echo "[queue] gpu1 E5 checkpoint missing; skip"; return; fi
  echo "[queue] gpu1 E5 $(date +%T)"; CHECKPOINT="$E5CK" JOBS=data/eval/E5/jobs_e5.txt TRIALS=80 GPUS=1 WORKERS_PER_GPU=2 $L
  echo "[queue] gpu1 E3-e5 $(date +%T)"; CHECKPOINT="$E5CK" JOBS=data/eval/E3/jobs_e5.txt TRIALS=40 GPUS=1 WORKERS_PER_GPU=2 $L
  echo "[queue] gpu1 done $(date +%T)"
}
gpu0 & gpu1 & wait
echo "[queue] ALL DONE $(date +%T)"
