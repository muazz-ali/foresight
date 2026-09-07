#!/usr/bin/env python3
"""CLI entry: convert Foresight H5 → LeRobot (conda: dynamicVLA_training).
Usage:
python scripts/convert_to_lerobot.py --input-dir data/p1  --repo-id foresight/p1 --root data/lerobot/p1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.convert_h5 import main

if __name__ == "__main__":
    main()
