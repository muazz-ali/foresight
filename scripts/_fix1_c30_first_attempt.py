#!/usr/bin/env python3
"""Fix1 closeout: C30 seeds 1000/1001/1003, first-attempt success (egg, Δ=0)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = (1000, 1001, 1003)


def main() -> None:
    from isaaclab.app import AppLauncher
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output_dir", default=str(ROOT / "data/eval/fix1_static_offset/c30_seeds"))
    p.add_argument("-c", "--sim_cfg_file", default=str(ROOT / "sim/phase0_cfg.yaml"))
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    AppLauncher(args)

    from eval.b1_success import b1_success
    from sim.collect import run_episode
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine

    cfg = load_cfg(Path(args.sim_cfg_file))
    cfg.setdefault("claim_timing", {})["obs_delay_buffer"] = True
    cfg["claim_timing"]["enforce_apply_at"] = False
    cfg.setdefault("sim", {})["device"] = str(getattr(args, "device", None) or "cuda:0")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scene = Phase0Scene(cfg, enable_cameras=False, object_usd="egg", asset_seed=1000)
    rows = []
    for seed in SEEDS:
        expert = PickPlaceStateMachine(cfg)
        log, meta = run_episode(
            scene,
            expert,
            seed=int(seed),
            speed=0.30,
            save_images=False,
            obs_delay_s=0.0,
        )
        ok, mode = b1_success(log, meta, z_table=float(getattr(scene, "_table_z", 0.0)))
        row = {
            "seed": int(seed),
            "success": bool(ok),
            "failure_mode": mode,
            "first_attempt": True,
            "category": meta.get("category"),
            "close_xy_err_m": getattr(expert, "close_xy_err_m", None),
            "dv_at_close_m_s": getattr(expert, "dv_at_close_m_s", None),
        }
        rows.append(row)
        print(
            f"C30 seed={seed} success={ok} mode={mode} "
            f"xy_err_mm={(1000*row['close_xy_err_m']) if row['close_xy_err_m'] is not None else None}"
        )

    payload = {
        "note": "First attempt only; egg; v=30 cm/s; Δ=0.",
        "n_ok": int(sum(1 for r in rows if r["success"])),
        "n": len(rows),
        "rows": rows,
    }
    (out / "c30_first_attempt.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
