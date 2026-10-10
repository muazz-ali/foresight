#!/usr/bin/env python3
"""Short Isaac smoke: physics grasp + wrist depth. NOT a gate, NOT a collect.

  PYTHONPATH=. python tools/diag/smoke_physics_grasp.py --speeds 0 0.2 --seeds 40 41 --headless --enable_cameras

Prints one line per episode (label, farthest stage, z_max, off_table) and a short
trace around the close: stage, hand-object gap, finger width, wrist depth.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--speeds", type=float, nargs="+", default=[0.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[40])
    p.add_argument("--asset-seed", type=int, default=40)
    p.add_argument("--cfg", default="sim/phase0_cfg.yaml")
    p.add_argument("--trace", type=int, default=1, help="print the close trace")
    p.add_argument("--miss", action="store_true", help="expert never moves: object must leave the table")
    p.add_argument("--dump", default="", help="dir: save wrist rgb + depth png at the close")
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])
    app = AppLauncher(args)
    logging.getLogger().setLevel(logging.WARNING)

    from sim.collect import run_episode
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import STAGE_CLOSE, PickPlaceStateMachine
    from sim.success import classify_failure, episode_diagnostics, episode_success

    cfg = load_cfg(args.cfg)
    scene = Phase0Scene(
        cfg, enable_cameras=bool(getattr(app, "_enable_cameras", False)), asset_seed=args.asset_seed
    )
    expert = PickPlaceStateMachine(cfg, dt=scene.dt)
    if args.miss:
        expert.settle_s = 1e9
    for speed in args.speeds:
        for seed in args.seeds:
            ep, meta = run_episode(
                scene, expert, seed=seed, speed=speed, save_images=bool(args.dump)
            )
            ok = episode_success(
                ep,
                z_lift_min=expert.lift_height,
                container_xy_tol=expert.place_xy_tol,
                place_stage=expert.stage_schema.done,
            )
            d = episode_diagnostics(ep, z_lift_min=expert.lift_height)
            lab = classify_failure(
                success=ok,
                max_stage=d["max_stage"],
                object_z_min=d["z_min"],
                object_z_max=d["z_max"],
                object_z_end=d["z_end"],
                z_lift_min=expert.lift_height,
                schema=expert.stage_schema,
                off_table=meta["off_table"],
            )
            print(
                f"[smoke] speed={speed:.2f} seed={seed} mode={meta['grasp_mode']} "
                f"label={lab} max_stage={d['max_stage']} z_max={d['z_max']:.3f} "
                f"z_end={d['z_end']:.3f} off_table={meta['off_table']} retries={meta['retries']}",
                flush=True,
            )
            if args.trace:
                st = ep["stage"]
                idx = np.flatnonzero(st == STAGE_CLOSE)
                if len(idx):
                    i0 = max(0, int(idx[0]) - 2)
                    if args.dump:
                        import os

                        import cv2

                        os.makedirs(args.dump, exist_ok=True)
                        k = int(idx[0])
                        dep = ep["wrist_cam_depth"][k].astype(np.float32)
                        dep = np.where(np.isfinite(dep), dep, 0.0)
                        cv2.imwrite(f"{args.dump}/rgb_s{speed}_{seed}.png", ep["wrist_cam_rgb"][k][..., ::-1])
                        cv2.imwrite(
                            f"{args.dump}/depth_s{speed}_{seed}.png",
                            np.clip(dep / 0.5 * 255.0, 0, 255).astype(np.uint8),
                        )
                    for t in range(i0, min(len(st), int(idx[0]) + 14)):
                        gap = ep["object_pos"][t] - ep["ee_pos"][t]
                        fw = float(ep["proprio"][t][-2:].sum())  # both finger joints (m)
                        print(
                            f"   t={t:3d} stage={int(st[t])} gap(cm)=({100*gap[0]:5.1f},{100*gap[1]:5.1f},{100*gap[2]:5.1f}) "
                            f"obj_z={ep['object_pos'][t][2]:.3f} fingers={fw:.3f} "
                            f"depth_m={ep['wrist_depth_m'][t]:.3f}",
                            flush=True,
                        )
    print("[smoke] done", flush=True)
    import os

    os._exit(0)  # skip app.close hang (same as run_phase0)


if __name__ == "__main__":
    main()
