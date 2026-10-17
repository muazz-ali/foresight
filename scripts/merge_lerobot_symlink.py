#!/usr/bin/env python3
"""CLI: merge two LeRobot roots via symlink (main) + remapped tail parquet.

Disk-cheap: no PNG tree copy. Tail parquet is rewritten so episode_index /
global index / task_index match the merged numbering.

  python scripts/merge_lerobot_symlink.py \\
    --main data/lerobot/p3_throw \\
    --tail data/lerobot/p3_throw_tail \\
    --out data/lerobot/p3_throw_all --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.merge_lerobot_symlink import main

if __name__ == "__main__":
    main()
