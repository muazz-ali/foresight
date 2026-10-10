#!/usr/bin/env python3
"""Does the scripted throw coast move exactly speed x time? (house mode, arm parked)

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/diag/check_coast_speed.py --headless \
      --asset-seed 69400 --speed 0.2 --substeps 4

Prints per control step: commanded travel vs measured travel (m) and the object's
height, until the object leaves the table. NOT a gate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORESIGHT_ROOT))


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=str(FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"))
    p.add_argument("--asset-seed", type=int, default=69400)
    p.add_argument("--seed", type=int, default=69600)
    p.add_argument("--speed", type=float, default=0.2)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--steps", type=int, default=40)
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])
    AppLauncher(args)

    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.cfg)
    if args.substeps is not None:
        cfg["sim"]["physics_substeps"] = int(args.substeps)
    scene = Phase0Scene(cfg, enable_cameras=False, asset_seed=args.asset_seed)
    home = np.asarray(cfg["robot"]["init_pose"][:3])
    quat = np.asarray(cfg["robot"]["init_pose"][3:])
    scene.reset_episode(seed=args.seed, speed=args.speed)
    prev = scene.get_object_state().position.copy()
    ratios = []
    for t in range(args.steps):
        scene.set_ee_target(home, quat, 1.0)
        scene.step(kinematic_object=True)
        pos = scene.get_object_state().position.copy()
        moved = float(np.linalg.norm(pos[:2] - prev[:2]))
        want = args.speed * scene.dt
        ratios.append(moved / want if want > 0 else 1.0)
        print(f"t={t:2d} moved {moved:.4f} want {want:.4f} ratio {ratios[-1]:.3f} z={pos[2]:.4f} "
              f"off_table={scene.off_table}", flush=True)
        prev = pos
        if scene.off_table:
            break
    r = np.asarray(ratios)
    print(f"[coast] {1 / scene.physics_dt:.0f} Hz: median ratio {np.median(r):.3f} "
          f"min {r.min():.3f} max {r.max():.3f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
