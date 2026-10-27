#!/usr/bin/env python3
"""Fix1 diagnostic: C30 seeds 1001/1003, Δ=0 — along/cross-track at matched speed.

Does not change gates. Writes ``c30_crosscheck.json`` under the Fix1 out dir.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = (1001, 1003)
SPEED_CMS = 30.0
DELTA_MS = 0.0


def _along_cross(hand_xy, obj_xy, vel_xy):
    """Along-track and cross-track hand−object (metres). +along = hand ahead."""
    h = np.asarray(hand_xy, dtype=np.float64).reshape(2)
    o = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    v = np.asarray(vel_xy, dtype=np.float64).reshape(2)
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        return float("nan"), float("nan")
    u = v / nv
    # left-handed 2D normal
    n = np.array([-u[1], u[0]], dtype=np.float64)
    d = h - o
    return float(np.dot(d, u)), float(np.dot(d, n))


def main() -> None:
    from isaaclab.app import AppLauncher
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output_dir", default=str(ROOT / "data/eval/fix1_static_offset"))
    p.add_argument("-c", "--sim_cfg_file", default=str(ROOT / "sim/phase0_cfg.yaml"))
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    AppLauncher(args)

    from sim.collect import run_episode
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine

    cfg = load_cfg(Path(args.sim_cfg_file))
    # Δ = 0: live GT, no obs-delay buffer
    cfg.setdefault("claim_timing", {})["obs_delay_buffer"] = False
    cfg["claim_timing"]["enforce_apply_at"] = False
    cfg.setdefault("sim", {})["device"] = str(getattr(args, "device", None) or "cuda:0")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scene = Phase0Scene(cfg, enable_cameras=False, object_usd="egg", asset_seed=1000)
    rows = []
    for seed in SEEDS:
        expert = PickPlaceStateMachine(cfg)
        # Match B1 C30: 30 cm/s, Δ=0
        log, meta = run_episode(
            scene,
            expert,
            seed=int(seed),
            speed=SPEED_CMS / 100.0,
            save_images=False,
            obs_delay_s=0.0,
        )
        # Find frame nearest matched speed (|Δv| min while in chase/close)
        stages = np.asarray(log["stage"], dtype=np.int64)
        ee = np.asarray(log["ee_pos"], dtype=np.float64)
        obj = np.asarray(log["object_pos"], dtype=np.float64)
        vel = np.asarray(log["object_vel"], dtype=np.float64)
        # Hand velocity via FD
        dt = float(cfg["sim"]["dt"])
        v_hand = np.zeros_like(ee)
        v_hand[1:] = (ee[1:] - ee[:-1]) / dt
        dv = np.linalg.norm(v_hand[:, :2] - vel[:, :2], axis=1)
        # Prefer stage 1 (intercept) near close
        mask = stages == 1
        if not np.any(mask):
            mask = np.ones(len(stages), dtype=bool)
        idx = int(np.argmin(np.where(mask, dv, np.inf)))
        along, cross = _along_cross(ee[idx, :2], obj[idx, :2], vel[idx, :2])
        row = {
            "seed": int(seed),
            "speed_cm_s": SPEED_CMS,
            "delta_ms": DELTA_MS,
            "idx": idx,
            "stage": int(stages[idx]),
            "dv_cms": float(dv[idx] * 100.0),
            "along_mm": float(along * 1000.0),
            "cross_mm": float(cross * 1000.0),
            "d_xy_mm": (1000.0 * (ee[idx, :2] - obj[idx, :2])).tolist(),
            "success": bool(meta.get("success", False)),
            "fail_reason": meta.get("fail_reason") or meta.get("reject_reason"),
            "category": meta.get("category"),
        }
        rows.append(row)
        print(
            f"C30 seed={seed} along_mm={row['along_mm']:.2f} cross_mm={row['cross_mm']:.2f} "
            f"dv_cms={row['dv_cms']:.2f} stage={row['stage']}"
        )

    payload = {
        "note": "Diagnostic only. Expect cross ≤ 0.5 mm after Fix1; along-track unchanged (Fix2).",
        "rows": rows,
    }
    (out / "c30_crosscheck.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out / "c30_crosscheck.json")
    os._exit(0)


if __name__ == "__main__":
    main()
