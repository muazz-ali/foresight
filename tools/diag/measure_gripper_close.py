#!/usr/bin/env python3
"""Measure how long the Franka fingers take to close (physics grasp, house mode).

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/diag/measure_gripper_close.py \
      --headless --object orange --substeps 4

Two numbers, both timed from the open -> close command, sampled at every
PHYSICS substep (via ``scene.step(substep_cb=...)``):
  (a) free close: nothing between the fingers, time until the gap is within
      ``--tol-mm`` of fully shut;
  (b) stall: hand around a resting object, time until the gap stops moving
      (within ``--tol-mm`` of where it ends up). Also checks the object is
      still in the hand after a 10 cm lift.

Writes tools/diag/out/gripper_close_<substeps>.json. NOT a gate.
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


def settle_time(t: np.ndarray, gap: np.ndarray, final: float, tol: float) -> float:
    """First time after which the gap stays within ``tol`` of ``final`` (NaN if never)."""
    off = np.abs(gap - final) > tol
    if off[-1]:
        return float("nan")
    bad = np.flatnonzero(off)
    return float(t[bad[-1] + 1]) if bad.size else float(t[0])


def _hold(scene, pos, quat, grip, n, **kw):
    for _ in range(n):
        scene.set_ee_target(pos, quat, grip)
        scene.step(**kw)


def _close_trace(scene, pos, quat, dur_s: float, **kw):
    """Command close at t=0; return (t, gap) sampled before every physics substep + at the end."""
    fi = scene._finger_idx
    t0 = scene._sim_time
    ts, gaps = [], []

    def cb(sc, t_sim):
        ts.append(t_sim - t0)
        gaps.append(float(sc.robot.data.joint_pos[0, fi].sum()))

    for _ in range(int(round(dur_s / scene.dt))):
        scene.set_ee_target(pos, quat, 0.0)
        scene.step(substep_cb=cb, **kw)
    ts.append(scene._sim_time - t0)
    gaps.append(float(scene.robot.data.joint_pos[0, fi].sum()))
    return np.asarray(ts), np.asarray(gaps)


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=str(FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"))
    p.add_argument("--object", default="orange", help="category or USD stem (one per process)")
    p.add_argument("--asset-seed", type=int, default=40)
    p.add_argument("--substeps", type=int, default=None, help="override sim.physics_substeps")
    p.add_argument("--tol-mm", type=float, default=1.0)
    p.add_argument("--dur-s", type=float, default=0.6)
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])
    AppLauncher(args)

    from sim.scene import Phase0Scene, load_cfg

    tol = args.tol_mm / 1000.0
    rows = []
    for obj_name in [args.object]:
        cfg = load_cfg(args.cfg)
        if args.substeps is not None:
            cfg["sim"]["physics_substeps"] = int(args.substeps)
        if getattr(args, "device", None):
            cfg["sim"]["device"] = args.device
        scene = Phase0Scene(cfg, enable_cameras=False, object_usd=obj_name, asset_seed=args.asset_seed)
        quat = np.asarray(cfg["robot"]["init_pose"][3:], dtype=np.float64)
        home = np.asarray(cfg["robot"]["init_pose"][:3], dtype=np.float64)

        # (a) free close in the air at home.
        scene.reset_episode(seed=0, speed=0.0)
        _hold(scene, home, quat, 1.0, 25, kinematic_object=True)
        t, gap = _close_trace(scene, home, quat, args.dur_s, kinematic_object=True)
        t_free = settle_time(t, gap, 0.0, tol)

        # (b) close on the resting object, pads at its middle.
        scene.reset_episode(seed=0, speed=0.0)
        obj = scene.get_object_state().position.copy()
        _hold(scene, obj + [0, 0, 0.10], quat, 1.0, 30, kinematic_object=True)
        _hold(scene, obj, quat, 1.0, 30, kinematic_object=True)
        ee_err = float(np.linalg.norm(scene.get_ee_pose()[0] - obj))
        t2, gap2 = _close_trace(
            scene, obj, quat, args.dur_s, kinematic_object=False, attach_object=True
        )
        tail = t2 >= t2[-1] - 0.1
        final = float(gap2[tail].mean())
        t_stall = settle_time(t2, gap2, final, tol)
        # Lift 10 cm: is it still in the hand?
        _hold(scene, obj + [0, 0, 0.10], quat, 0.0, 25, kinematic_object=False, attach_object=True)
        o2 = scene.get_object_state().position
        held = bool(o2[2] - obj[2] > 0.07)

        size = scene._object_size.tolist() if scene._object_size is not None else None
        row = {
            "object": scene._active_usd.stem if scene._active_usd else obj_name,
            "object_size_m": size,
            "physics_hz": round(1.0 / scene.physics_dt, 1),
            "n_substeps": scene.n_substeps,
            "tol_m": tol,
            "free_close_s": t_free,
            "stall_close_s": t_stall,
            "stall_gap_m": final,
            "ee_err_before_close_m": ee_err,
            "held_after_lift": held,
            "free_trace": [[round(a, 4), round(b, 5)] for a, b in zip(t, gap)],
            "stall_trace": [[round(a, 4), round(b, 5)] for a, b in zip(t2, gap2)],
        }
        rows.append(row)
        print(
            f"[close] {row['object']} size={np.round(size, 3) if size else None} "
            f"{row['physics_hz']:.0f} Hz: free close {t_free * 1000:.0f} ms, "
            f"stall {t_stall * 1000:.0f} ms at gap {final * 100:.1f} cm, "
            f"hand err {ee_err * 100:.1f} cm, held after lift={held}",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"gripper_close_{rows[0]['n_substeps']}_{rows[0]['object']}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[close] wrote {out}", flush=True)
    os._exit(0)  # skip app.close hang (same as run_phase0)


if __name__ == "__main__":
    main()
