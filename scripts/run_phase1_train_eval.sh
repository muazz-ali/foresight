#!/usr/bin/env bash
# After Phase-1 collection finishes: convert → train A/B → eval G1.
# Envs: convert/train use dynamicVLA_training; eval uses dynamicVLA_isaac.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P1_DATA="${P1_DATA:-$ROOT/data/p1}"
REPO_ID="${REPO_ID:-foresight/p1}"
LR_ROOT="${LR_ROOT:-$ROOT/data/lerobot/p1}"
STEPS="${STEPS:-20000}"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:1}"
DEVICE_EVAL="${DEVICE_EVAL:-cuda:0}"

echo "[p1] convert $P1_DATA → $LR_ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynamicVLA_training
python "$ROOT/scripts/convert_to_lerobot.py" \
  --input-dir "$P1_DATA" \
  --repo-id "$REPO_ID" \
  --root "$LR_ROOT" \
  --overwrite

echo "[p1] train Model A"
python "$ROOT/scripts/train_smolvla.py" \
  --model A --repo-id "$REPO_ID" --dataset-root "$LR_ROOT" \
  --output-dir "$ROOT/runs/p1_a" --steps "$STEPS" --device "$DEVICE_TRAIN" \
  --batch-size 8 --chunk-size 50

echo "[p1] train Model B"
python "$ROOT/scripts/train_smolvla.py" \
  --model B --repo-id "$REPO_ID" --dataset-root "$LR_ROOT" \
  --output-dir "$ROOT/runs/p1_b" --steps "$STEPS" --device "$DEVICE_TRAIN" \
  --batch-size 8 --chunk-size 50

echo "[p1] eval A / B / B-zero (Isaac)"
conda activate dynamicVLA_isaac
python "$ROOT/scripts/eval_policy.py" \
  --checkpoint "$ROOT/runs/p1_a/pretrained_model" --model A \
  --headless --enable_cameras --device "$DEVICE_EVAL" \
  -o "$ROOT/data/p1/eval_a"

python "$ROOT/scripts/eval_policy.py" \
  --checkpoint "$ROOT/runs/p1_b/pretrained_model" --model B \
  --headless --enable_cameras --device "$DEVICE_EVAL" \
  -o "$ROOT/data/p1/eval_b"

python "$ROOT/scripts/eval_policy.py" \
  --checkpoint "$ROOT/runs/p1_b/pretrained_model" --model B --zero-conditioning \
  --headless --enable_cameras --device "$DEVICE_EVAL" \
  -o "$ROOT/data/p1/eval_b_zero"

python "$ROOT/eval/gate_g1.py" \
  --report-a "$ROOT/data/p1/eval_a/gate_g1_report.json" \
  --report-b "$ROOT/data/p1/eval_b/gate_g1_report.json" \
  --report-b-zero "$ROOT/data/p1/eval_b_zero/gate_g1_report.json" \
  -o "$ROOT/data/p1/gate_g1_report.json"

echo "[p1] done → $ROOT/data/p1/gate_g1_report.json"
