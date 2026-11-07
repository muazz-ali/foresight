#!/usr/bin/env python3
"""Verify claim_height collect/lock against a live Isaac scene (GT z_top).

Rules under test
----------------
1. Yaml → SI: g=10 mm, c=5 mm, collect ≥20 cm, lock <10 cm, s_z=5 mm.
2. Hand descends from ~25 cm above top to ~5 cm:
   - gap ≥ 20 cm: samples accumulate; z_grasp tracks median.
   - gap < 10 cm: lock once; value freezes.
3. Locked value == median of collected z_grasp (or live if none).

  conda activate dynamicVLA_isaac
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python tools/diag/verify_z_latch.py --headless --enable_cameras --device cuda:0

Writes tools/diag/out/z_latch_<object>.json. NOT a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORESIGHT_ROOT))
OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=str(FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"))
    p.add_argument("--object", default="orange")
    p.add_argument("--asset-seed", type=int, default=40)
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])
    AppLauncher(args)

    from interfaces.grasp_height import ZLatch, z_grasp_m, z_top_from_gt_center
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine

    cfg = load_cfg(args.cfg)
    if getattr(args, "device", None):
        cfg["sim"]["device"] = args.device

    scene = Phase0Scene(
        cfg, enable_cameras=True, object_usd=args.object, asset_seed=args.asset_seed
    )
    latch_cfg = ZLatch.from_cfg(cfg)
    sm = PickPlaceStateMachine(cfg)
    sm._z_latch.z_table = float(scene._table_z)
    latch_cfg.z_table = float(scene._table_z)

    scene.reset_episode(seed=0, speed=0.0)
    quat = np.asarray(cfg["robot"]["init_pose"][3:], dtype=np.float64)
    obj = scene.get_object_state().position.copy()
    half_mesh = float(scene._object_half_height)
    half = max(half_mesh, float(obj[2]) - float(scene._table_z))
    z_top = z_top_from_gt_center(float(obj[2]), half_height_m=half)
    expected_live = z_grasp_m(
        z_top,
        z_table=float(scene._table_z),
        insert_m=latch_cfg.insert_m,
        clear_m=latch_cfg.clear_m,
    )

    sm.reset()
    sm._obj_half_h = half_mesh
    gaps = np.linspace(0.25, 0.05, 41)
    rows = []
    n_samples_at_hi = 0
    locked_at_gap = None
    z_after_lock = []
    for gap in gaps:
        ee = np.array([obj[0], obj[1], z_top + float(gap)], dtype=np.float64)
        for _ in range(3):
            scene.set_ee_target(ee, quat, 1.0)
            scene.step(kinematic_object=True)

        zg = sm._update_z_grasp(ee, obj, wrist_depth=None, ee_quat=quat)
        latched = sm._z_latch.latched_z_grasp is not None
        n_samp = len(sm._z_latch.samples_z_grasp)
        if float(gap) >= latch_cfg.latch_hi_m:
            n_samples_at_hi = n_samp
        if latched and locked_at_gap is None:
            locked_at_gap = float(gap)
        if latched:
            z_after_lock.append(float(zg))
        rows.append(
            {
                "gap_m": round(float(gap), 4),
                "ee_z": round(float(ee[2]), 4),
                "z_grasp": round(float(zg), 4),
                "n_samples": n_samp,
                "latched": latched,
            }
        )

    lo, hi = latch_cfg.latch_lo_m, latch_cfg.latch_hi_m
    ok_settings = (
        abs(latch_cfg.insert_m - 0.01) < 1e-9
        and abs(latch_cfg.clear_m - 0.005) < 1e-9
        and abs(lo - 0.10) < 1e-9
        and abs(hi - 0.20) < 1e-9
        and abs(latch_cfg.z_slack_m - 0.005) < 1e-9
    )
    ok_collect = n_samples_at_hi >= 1
    ok_lock = locked_at_gap is not None and locked_at_gap < lo
    ok_freeze = bool(z_after_lock) and max(z_after_lock) - min(z_after_lock) < 1e-6
    # GT still object → all samples ≈ expected_live; median should match.
    ok_value = (
        sm._z_latch.latched_z_grasp is not None
        and abs(float(sm._z_latch.latched_z_grasp) - expected_live) < 1e-3
    )

    report = {
        "object": scene._active_usd.stem if scene._active_usd else args.object,
        "object_size_m": scene._object_size.tolist() if scene._object_size is not None else None,
        "table_z": float(scene._table_z),
        "obj_z": float(obj[2]),
        "z_top": float(z_top),
        "expected_z_grasp": float(expected_live),
        "settings": {
            "insert_m": latch_cfg.insert_m,
            "clear_m": latch_cfg.clear_m,
            "latch_lo_m": lo,
            "latch_hi_m": hi,
            "z_slack_m": latch_cfg.z_slack_m,
        },
        "n_samples_while_high": n_samples_at_hi,
        "locked_at_gap_m": locked_at_gap,
        "latched_z_grasp": sm._z_latch.latched_z_grasp,
        "ok_settings": ok_settings,
        "ok_collect_above_hi": ok_collect,
        "ok_lock_below_lo": ok_lock,
        "ok_freeze_after": ok_freeze,
        "ok_z_grasp_value": ok_value,
        "pass": bool(ok_settings and ok_collect and ok_lock and ok_freeze and ok_value),
        "trace": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = report["object"]
    out = OUT_DIR / f"z_latch_{name}.json"
    out.write_text(json.dumps(report, indent=2))
    print(
        f"[z_latch] {name} table_z={report['table_z']:.3f} z_top={z_top:.3f} "
        f"expect={expected_live:.3f} samples@hi={n_samples_at_hi} "
        f"lock@{locked_at_gap} settings={ok_settings} collect={ok_collect} "
        f"lock={ok_lock} freeze={ok_freeze} value={ok_value} PASS={report['pass']}",
        flush=True,
    )
    print(f"[z_latch] wrote {out}", flush=True)
    os._exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
