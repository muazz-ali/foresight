# `policy/` — train the pick policy

Plain words: see [`../WORDS.md`](../WORDS.md).

| File | What it does |
|---|---|
| `features.py` | Dataset field names + text instruction helper |
| `convert_h5.py` | Turn our demo files (H5) into a LeRobot dataset |
| `transforms.py` | Model A vs B; random look-ahead time + noise on the 12 numbers |
| `train_smolvla.py` | Fine-tune SmolVLA (use conda `dynamicVLA_training`) |
| `configs/smolvla_{a,b}.yaml` | Notes for A (no future numbers) / B (with future numbers) |

## Commands

```bash
# 1) Convert demos → LeRobot (training env)
conda activate dynamicVLA_training
python scripts/convert_to_lerobot.py --input-dir data/p1 --repo-id foresight/p1 \
  --root data/lerobot/p1 --overwrite

# 2) Train A (images + robot pose + text only)
python scripts/train_smolvla.py --model A --repo-id foresight/p1 \
  --dataset-root data/lerobot/p1 --output-dir runs/p1_a --steps 20000 --device cuda:1

# 3) Train B (same + ~12 future numbers)
python scripts/train_smolvla.py --model B --repo-id foresight/p1 \
  --dataset-root data/lerobot/p1 --output-dir runs/p1_b --steps 20000 --device cuda:1

# 4) Test in Isaac (sim env), then merge scores
conda activate dynamicVLA_isaac
python scripts/eval_policy.py --checkpoint runs/p1_a/pretrained_model --model A \
  --headless --enable_cameras -o data/p1/eval_a
python scripts/eval_policy.py --checkpoint runs/p1_b/pretrained_model --model B \
  --headless --enable_cameras -o data/p1/eval_b
python scripts/eval_policy.py --checkpoint runs/p1_b/pretrained_model --model B \
  --zero-conditioning --headless --enable_cameras -o data/p1/eval_b_zero
python eval/gate_g1.py \
  --report-a data/p1/eval_a/gate_g1_report.json \
  --report-b data/p1/eval_b/gate_g1_report.json \
  --report-b-zero data/p1/eval_b_zero/gate_g1_report.json \
  -o data/p1/gate_g1_report.json
```

Or after demos exist: `bash scripts/run_phase1_train_eval.sh`
