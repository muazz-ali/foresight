#!/usr/bin/env python3
"""CLI entry: train SmolVLA (conda: dynamicVLA_training).

Thin launcher. The real work is in policy/train_smolvla.py.
Run convert_to_lerobot.py first; train reads the dataset it makes, and both
use the same 12-number state layout (policy/features.py).

Usage:
python scripts/train_smolvla.py --repo-id foresight/p1 --dataset-root data/lerobot/p1 \
  --output-dir runs/p1 --steps 20000 --device cuda:1
"""

from __future__ import annotations

import sys
from pathlib import Path

# Let `import policy` work when run as `python scripts/...` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.train_smolvla import main

if __name__ == "__main__":
    main()
