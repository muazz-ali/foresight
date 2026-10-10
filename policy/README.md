# `policy/` — train the pick policy

Plain words: see [`../WORDS.md`](../WORDS.md).

| File | What it does |
|---|---|
| `features.py` | Dataset field names + text instruction helper |
| `convert_h5.py` | Turn our demo files (H5) into a LeRobot dataset |
| `transforms.py` | Fixed look-ahead time + noise; pack 12-D state (proprio + 4 future numbers) |
| `train_smolvla.py` | Fine-tune SmolVLA (use conda `dynamicVLA_training`) |
| `configs/smolvla.yaml` | Δ / noise defaults for training |

## Commands

```bash
# 1) Convert demos → LeRobot (training env)
conda activate dynamicVLA_training
python scripts/convert_to_lerobot.py --input-dir data/p1 --repo-id foresight/p1 \
  --root data/lerobot/p1 --overwrite

# 2) Train (images + robot pose + text + 4 live future numbers)
python scripts/train_smolvla.py --repo-id foresight/p1 \
  --dataset-root data/lerobot/p1 --output-dir runs/p1 --steps 20000 --device cuda:1

# 3) Test in Isaac (sim env) — see scripts/eval_policy.py and eval/gate_g1.py
```

Or after demos exist: `bash scripts/run_phase1_train_eval.sh`
