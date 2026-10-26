#!/usr/bin/env python3
"""Explain how to smoke-train on old demos (4-D conditioning → 5-D).

``data/lerobot/p3_throw`` has ``observation.conditioning`` shape **[4]**
(p̂x, p̂y, vx, vy). Claim needs **[5]** (insert p̂z).

Rewriting 115G of parquet in place is unsafe. Preferred smoke paths:

1) **New claim collect** (true 5-D)::

     python scripts/smoke_claim_grid.py --headless --enable_cameras --device cuda:0 \\
       --per-cell 10 -o data/claim_smoke
     python -m policy.convert_h5 --input-dir data/claim_smoke/... --repo-id foresight/claim_smoke

2) **Old demos**: train a **4-D** checkpoint for pipeline smoke only
   (eval with matching 4-D pack), OR re-convert from any remaining H5 that
   still has oracle Z and rebuild conditioning with ``conditioning_vector``
   + ``p_hat_z`` in a small offline job (not this script).

This file only prints the plan (exit 0) so operators do not corrupt p3_throw.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    info = ROOT / "data" / "lerobot" / "p3_throw" / "meta" / "info.json"
    shape = None
    if info.exists():
        feats = json.loads(info.read_text())["features"]
        shape = feats.get("observation.conditioning", {}).get("shape")
    print("old p3_throw conditioning shape:", shape)
    print("claim needs shape: [5]")
    print("→ use scripts/smoke_claim_grid.py for the primary 20 min train smoke.")
    print("→ old-dataset 20 min smoke: train existing 4-D repo with train_smolvla")
    print("  and run eval_policy (pipeline check only; not a claim gate).")


if __name__ == "__main__":
    main()
