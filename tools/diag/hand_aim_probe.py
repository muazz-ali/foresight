#!/usr/bin/env python3
"""Hand–aim probe: commanded aim vs pinch after physics step (no objects).

Commands a palm-down straight coast at v ∈ {0, 10, 30} cm/s. Each tick:
  1. read pinch / TCP before step
  2. set DiffIK aim = current_pinch + v·dt  (or hold for v=0)
  3. scene.step
  4. log post-step pinch vs the aim that was just commanded

If |pinch − aim| grows with v → logging/step-order lag (one-tick chase residual).
If offset is ~constant across v → aim/pinch frame mismatch (TCP ≠ pinch).

Usage (Isaac):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python tools/diag/hand_aim_probe.py --headless --enable_cameras --device cuda:0 \\
    -o data/eval/hand_aim_probe.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

FORESIGHT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORESIGHT))

SPEEDS_CMS = (0.0, 10.0, 30.0)
SETTLE_S = 0.8
HOLD_S = 1.0
START_XY = (0.50, 0.0)
Z_GRASP = 0.036


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=FORESIGHT / "sim" / "phase0_cfg.yaml")
    parser.add_argument("-o", "--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args([a for a in sys.argv[1:] if a != ""])
    AppLauncher(args)

    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.cfg)
    cfg.setdefault("claim_timing", {})["obs_delay_buffer"] = False
    cfg["claim_timing"]["enforce_apply_at"] = False
    print("building Phase0Scene…", flush=True)
    scene = Phase0Scene(cfg, enable_cameras=False, object_usd="egg", asset_seed=1000)
    print("reset_episode…", flush=True)
    scene.reset_episode(seed=0, speed=0.0)
    dt = float(scene.dt)
    quat = np.asarray(cfg["robot"]["init_pose"][3:], dtype=np.float64)
    z_grasp = float(scene._table_z) + float(Z_GRASP)
    start = np.array([START_XY[0], START_XY[1], z_grasp], dtype=np.float64)

    for _ in range(int(SETTLE_S / dt)):
        scene.set_ee_target(start, quat, 1.0, vel_w=np.zeros(3))
        scene.step(kinematic_object=True)

    tcp_minus_pinch = scene.get_tcp_minus_pinch_hand_m()
    rows = []
    tick_logs = {}

    for v_cms in SPEEDS_CMS:
        v = float(v_cms) * 0.01  # m/s along +X
        vel = np.array([v, 0.0, 0.0], dtype=np.float64)
        errs = []
        series = []
        print(f"coast v={v_cms} cm/s…", flush=True)
        for _ in range(int(0.3 / dt)):
            scene.set_ee_target(start, quat, 1.0, vel_w=np.zeros(3))
            scene.step(kinematic_object=True)
        # Coast from a fixed start pose; aim = start + v·t (not chase pinch).
        for k in range(int(HOLD_S / dt)):
            t = (k + 1) * dt
            aim = start.copy()
            aim[0] = float(start[0] + v * t)
            scene.set_ee_target(aim, quat, 1.0, vel_w=vel)
            scene.step(kinematic_object=True)
            pinch_post = scene.get_pinch_point()
            tcp_post, _ = scene.get_ee_pose()
            err = pinch_post - aim
            errs.append(err)
            series.append(
                {
                    "t": float(t),
                    "aim": aim.tolist(),
                    "pinch_post": pinch_post.tolist(),
                    "tcp_post": tcp_post.tolist(),
                    "err_mm": (1000.0 * err).tolist(),
                }
            )
        E = np.asarray(errs, dtype=np.float64)
        E2 = E[5:] if E.shape[0] > 5 else E
        med = np.median(E2, axis=0)
        p95 = float(np.percentile(np.linalg.norm(E2, axis=1), 95))
        rows.append(
            {
                "v_cms": float(v_cms),
                "err_med_mm": (1000.0 * med).tolist(),
                "err_norm_med_mm": float(1000.0 * np.median(np.linalg.norm(E2, axis=1))),
                "err_norm_p95_mm": p95,
                "err_along_x_med_mm": float(1000.0 * med[0]),
            }
        )
        tick_logs[str(v_cms)] = series

    along = [r["err_along_x_med_mm"] for r in rows]
    if abs(along[-1] - along[0]) > 2.0:
        verdict = (
            "GROWS with v → one-tick chase / step-order residual "
            f"(Δalong {along[-1] - along[0]:+.1f} mm from 0→30 cm/s). "
            "Fix logging (log pre-step EE next to aim) or command earlier."
        )
    else:
        verdict = (
            "≈CONSTANT across v → aim/pinch frame mismatch "
            f"(tcp−pinch hand = {(1000*tcp_minus_pinch).round(2)} mm). "
            "Align DiffIK TCP with geometric pinch."
        )

    lines = [
        "# Hand–aim probe (no objects)",
        "",
        f"**Start:** XY={START_XY}, Z={Z_GRASP} m. Coast +X.  ",
        f"**TCP − pinch (hand frame):** {(1000*tcp_minus_pinch).round(2).tolist()} mm  ",
        f"**Verdict:** {verdict}",
        "",
        "| v cm/s | err med XYZ mm | ‖err‖ med | ‖err‖ p95 | along-X med mm |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        e = r["err_med_mm"]
        lines.append(
            f"| {r['v_cms']:.0f} | ({e[0]:+.2f}, {e[1]:+.2f}, {e[2]:+.2f}) | "
            f"{r['err_norm_med_mm']:.2f} | {r['err_norm_p95_mm']:.2f} | "
            f"{r['err_along_x_med_mm']:+.2f} |"
        )
    lines += [
        "",
        "Healthy: constant ~0 (frame aligned + logged same tick).  ",
        "At 30 cm/s one control tick ≈ 10 mm of travel — a pure ZOH lag "
        "shows ~half-tick along-track (~5 mm) that scales with v.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "tcp_minus_pinch_hand_mm": (1000 * tcp_minus_pinch).tolist(),
                "rows": rows,
                "verdict": verdict,
                "ticks": tick_logs,
            },
            indent=2,
        )
    )
    print("\n".join(lines), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
