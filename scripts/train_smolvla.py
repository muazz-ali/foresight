#!/usr/bin/env python3
"""CLI entry: train SmolVLA A/B (conda: dynamicVLA_training).

Usage:
python scripts/train_smolvla.py --model A --repo-id foresight/p1 --dataset-root data/lerobot/p1 --output-dir runs/p1_a --steps 20000 --device cuda:1

python scripts/train_smolvla.py --model B --repo-id foresight/p1 --dataset-root data/lerobot/p1 --output-dir runs/p1_b --steps 20000 --device cuda:1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.train_smolvla import main

if __name__ == "__main__":
    main()
