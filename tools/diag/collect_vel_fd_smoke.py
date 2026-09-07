#!/usr/bin/env python3
"""Live smoke: 1 free-motion episode per speed bin → *_cond.h5 for vel_fd diag.

Launch (from repo root, env ``dynamicVLA_isaac``):

  python tools/diag/collect_vel_fd_smoke.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[2]
if str(FORESIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(FORESIGHT_ROOT))

from eval.gate_g1 import G1_SPEEDS  # noqa: E402

N_STEPS = 120
DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
DEFAULT_OUT = FORESIGHT_ROOT / "tools" / "diag" / "out" / "vel_fix"


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)

    from sim.collect import write_episode_h5
    from sim.scene import Phase0Scene, load_cfg

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*_cond.h5"):
        old.unlink()

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    scene = Phase0Scene(cfg, enable_cameras=True, asset_seed=int(args.seed))

    for speed in G1_SPEEDS:
        seed = int(args.seed) + int(round(float(speed) * 1000))
        scene.reset_episode(seed=seed, speed=float(speed))
        pos_log: list[np.ndarray] = []
        vel_log: list[np.ndarray] = []
        for _ in range(int(args.n_steps)):
            scene.step(kinematic_object=True, attach_object=False, freeze_object=False)
            st = scene.get_object_state()
            pos_log.append(st.position.copy())
            vel_log.append(st.velocity.copy())
        path = out_dir / f"smoke_v{speed:.2f}_s{seed}_cond.h5"
        write_episode_h5(
            path,
            {
                "object_pos": np.asarray(pos_log, dtype=np.float64),
                "object_vel": np.asarray(vel_log, dtype=np.float64),
                "holding": np.zeros(len(pos_log), dtype=np.float32),
            },
            extra={
                "speed_m_s": np.asarray([float(speed)], dtype=np.float32),
                "seed": np.asarray([seed], dtype=np.int64),
            },
        )
        v0 = float(np.linalg.norm(vel_log[0]))
        print(f"wrote {path} n={len(pos_log)} ||v0||={v0:.4f} label={speed}", flush=True)

    print(f"done out_dir={out_dir}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
