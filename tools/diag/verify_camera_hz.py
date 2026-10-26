#!/usr/bin/env python3
"""Verify camera capture rate vs sim.dt in Isaac (claim Step 1).

Plain words: step the scene many times, read ``last_capture_t0`` after each
step, and check that the gaps match ``sim.dt`` (≈ 33.3 ms → 30 Hz).

Important: yaml ``static_camera.fps`` / ``wrist_camera.fps`` are NOT passed into
Isaac ``CameraCfg``. They label videos / logs. Real capture rate = control rate
(``cam.update(self.dt)`` once per ``scene.step``), with ``update_period=0.0``.

Usage (sim conda)::

  conda activate dynamicVLA_isaac
  cd /home/gpuadmin/Desktop/foresight
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  python tools/diag/verify_camera_hz.py --headless --enable_cameras --device cuda:0

Exit 0 only if median gap ≈ dt and inferred Hz ≈ 1/dt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Launch Isaac, step the scene, measure capture stamps vs sim.dt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=str, default=str(ROOT / "sim" / "phase0_cfg.yaml"))
    parser.add_argument("--steps", type=int, default=60, help="control steps after warm-up")
    parser.add_argument("--warmup", type=int, default=5, help="steps before measuring")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tol-ms",
        type=float,
        default=0.5,
        help="allowed |median gap − dt| in milliseconds",
    )
    # AppLauncher owns --device / --headless / --enable_cameras.

    print("[verify_camera_hz] starting AppLauncher…", flush=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not getattr(args, "enable_cameras", False):
        print("[verify_camera_hz] forcing --enable_cameras (else frames are black)", flush=True)
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    exit_code = 1

    try:
        from interfaces.config import load_yaml
        from sim.scene import Phase0Scene

        cfg = load_yaml(args.cfg)
        # AppLauncher --device overrides yaml when present.
        if getattr(args, "device", None):
            cfg.setdefault("sim", {})["device"] = str(args.device)
        dt = float(cfg["sim"]["dt"])
        yaml_fps = int(cfg.get("static_camera", {}).get("fps", -1))
        scene = Phase0Scene(
            cfg,
            enable_cameras=True,
            asset_seed=int(args.seed),
        )
        scene.reset_episode(seed=int(args.seed), speed=0.0)

        stamps: list[float] = []
        last_imgs: dict = {}
        for i in range(int(args.warmup) + int(args.steps)):
            # Hold still: we only care about the clock / camera stamp.
            ee, quat = scene.get_ee_pose()
            scene.set_ee_target(ee, quat, 1.0)
            scene.step(kinematic_object=True, attach_object=False, freeze_object=False)
            last_imgs, t0 = scene.get_image_packet()
            if i >= int(args.warmup):
                stamps.append(float(t0))

        stamps_a = np.asarray(stamps, dtype=np.float64)
        gaps = np.diff(stamps_a)
        med = float(np.median(gaps))
        mean = float(np.mean(gaps))
        std = float(np.std(gaps))
        hz = 1.0 / med if med > 0 else float("nan")
        tol_s = float(args.tol_ms) * 1e-3

        print("--- camera / control clock ---", flush=True)
        print(f"  sim.dt            = {dt * 1000:.6f} ms  ({1.0 / dt:.4f} Hz)", flush=True)
        print(f"  yaml static fps   = {yaml_fps}  (label only; not passed to CameraCfg)", flush=True)
        print(f"  CameraCfg         = update_period=0.0 (update every cam.update call)", flush=True)
        print(f"  n gaps            = {len(gaps)}", flush=True)
        print(
            f"  gap median/mean/std = {med * 1000:.4f} / {mean * 1000:.4f} / {std * 1000:.4f} ms",
            flush=True,
        )
        print(f"  inferred Hz       = {hz:.4f}", flush=True)
        print(f"  first/last t0     = {stamps_a[0]:.6f} → {stamps_a[-1]:.6f} s", flush=True)

        ok_dt = abs(med - dt) <= tol_s
        ok_hz = abs(hz - (1.0 / dt)) <= 0.5  # within 0.5 Hz
        ok_imgs = "static_cam" in last_imgs and "wrist_cam" in last_imgs
        print(
            f"  check median≈dt   = {'PASS' if ok_dt else 'FAIL'} "
            f"(|{med - dt}| ≤ {tol_s})",
            flush=True,
        )
        print(f"  check Hz≈1/dt     = {'PASS' if ok_hz else 'FAIL'}", flush=True)
        print(f"  check both cams   = {'PASS' if ok_imgs else 'FAIL'}", flush=True)

        # Soft note if yaml fps disagrees with measured rate.
        if yaml_fps > 0 and abs(yaml_fps - hz) > 1.0:
            print(
                f"  NOTE: yaml fps={yaml_fps} ≠ measured {hz:.2f} Hz — "
                "fix the comment/label; Isaac ignores yaml fps.",
                flush=True,
            )

        exit_code = 0 if (ok_dt and ok_hz and ok_imgs) else 1
    except Exception as e:
        print(f"[verify_camera_hz] FAILED: {e}", flush=True)
        exit_code = 1
    finally:
        # Skip simulation_app.close(): headless Isaac often hangs there.
        # Hard-exit with the measured status (same pattern as run_phase0).
        del simulation_app
        print(f"[verify_camera_hz] exiting code={exit_code}", flush=True)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
