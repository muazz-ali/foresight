#!/usr/bin/env python3
"""DiffIK tracking probe — no objects, no state machine.

Palm-down hand at grasp height. Command straight-line pinch motion at
10 / 20 / 30 cm/s in 8 headings, from starts across the reachable band
(including near-base and the C30 s1001 / s1003 diagonal zones).

Logs commanded vs actual pinch every tick. After the hand is moving
steadily, reports along-track and sideways tracking error (mm).

  conda activate dynamicVLA_isaac
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python tools/diag/probe_diffik_tracking.py --headless --enable_cameras --device cuda:0

Writes ``tools/diag/out/diffik_tracking.json`` + a short markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORESIGHT_ROOT))

DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
OUT_DIR = Path(__file__).resolve().parent / "out"

# Speeds (m/s) and headings (deg, world XY; 0 = +X).
SPEEDS_M_S = (0.10, 0.20, 0.30)
HEADINGS_DEG = tuple(range(0, 360, 45))

# Starts: mid / near-base / C30 fail diagonals / far / sides.
# s1001 chase ~ (0.36, −0.11); s1003 ~ (0.37, +0.22).
STARTS_XY_M = (
    (0.55, 0.00),   # mid table
    (0.45, 0.00),   # mid-near
    (0.32, 0.00),   # near base
    (0.30, -0.08),  # near base, −Y
    (0.30, 0.08),   # near base, +Y
    (0.36, -0.11),  # C30 s1001 diagonal zone
    (0.37, 0.22),   # C30 s1003 diagonal zone
    (0.65, 0.00),   # far +X
    (0.50, 0.12),   # +Y side
    (0.50, -0.12),  # −Y side
)

GRIP_OPEN = 1.0
# Skip the first this many seconds when scoring (ramp / accel).
STEADY_SKIP_S = 0.25
# How long to command the move after skip (steady window).
STEADY_HOLD_S = 0.50
# Settle at the start pose before each move.
SETTLE_S = 0.80


def _unit(heading_deg: float) -> np.ndarray:
    th = math.radians(float(heading_deg))
    return np.array([math.cos(th), math.sin(th)], dtype=np.float64)


def _along_cross(err_xy: np.ndarray, uh: np.ndarray) -> tuple[float, float]:
    """Along / sideways components of err (m). +along = actual ahead of cmd."""
    nh = np.array([-uh[1], uh[0]], dtype=np.float64)
    return float(err_xy @ uh), float(err_xy @ nh)


def _reach_ok(xy: np.ndarray, *, r_min: float = 0.25, r_max: float = 0.78) -> bool:
    r = float(np.linalg.norm(xy[:2]))
    return r_min <= r <= r_max


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("-o", "--output_dir", default=str(OUT_DIR))
    p.add_argument("--grasp-z-above-table-m", type=float, default=0.025,
                   help="Pinch Z = table_z + this (default egg-ish centre).")
    p.add_argument("--speeds", default="0.10,0.20,0.30")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="3 starts × 4 headings × 2 speeds (quick wiring check).",
    )
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    AppLauncher(args)

    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(Path(args.sim_cfg_file))
    cfg.setdefault("claim_timing", {})["enabled_noise_on_replan"] = False
    cfg["claim_timing"]["apply_delta_delay_to_robot"] = False
    cfg.setdefault("sim", {})["device"] = str(getattr(args, "device", None) or "cuda:0")
    dt = float(cfg["sim"]["dt"])
    speeds = tuple(float(x) for x in str(args.speeds).split(",") if x.strip())
    starts = STARTS_XY_M
    headings = HEADINGS_DEG
    if bool(args.smoke):
        starts = STARTS_XY_M[:3]
        headings = (0.0, 90.0, 180.0, 270.0)
        speeds = speeds[:2] if len(speeds) >= 2 else speeds
        print(f"SMOKE: {len(starts)} starts × {len(headings)} hdgs × {len(speeds)} speeds")

    scene = Phase0Scene(cfg, enable_cameras=False, object_usd="egg", asset_seed=1000)
    scene.reset_episode(seed=0, speed=0.0)
    # Palm-down from init pose; grasp height above the table.
    quat = np.asarray(cfg["robot"]["init_pose"][3:], dtype=np.float64)
    z_grasp = float(scene._table_z) + float(args.grasp_z_above_table_m)
    n_settle = max(1, int(round(SETTLE_S / dt)))
    n_skip = max(1, int(round(STEADY_SKIP_S / dt)))
    n_hold = max(1, int(round(STEADY_HOLD_S / dt)))
    n_total = n_skip + n_hold

    rows: list[dict] = []
    tick_log: list[dict] = []

    for sx, sy in starts:
        start = np.array([sx, sy, z_grasp], dtype=np.float64)
        if not _reach_ok(start):
            print(f"skip start {start[:2]} (out of reach ring)")
            continue
        for hdg in headings:
            uh = _unit(hdg)
            for speed in speeds:
                end_xy = start[:2] + uh * speed * (n_total * dt)
                if not _reach_ok(end_xy):
                    # Still run but mark clipped_path; useful near edges.
                    clipped = True
                else:
                    clipped = False

                # Commanded world velocity of the moving pinch target (XY only).
                vel_cmd = np.array([uh[0] * speed, uh[1] * speed, 0.0], dtype=np.float64)

                # Settle at start (v=0 → absolute pose).
                for _ in range(n_settle):
                    scene.set_ee_target(start, quat, GRIP_OPEN, vel_w=np.zeros(3))
                    scene.step(kinematic_object=True)

                alongs, crosses, speeds_act = [], [], []
                for k in range(n_total):
                    t = (k + 1) * dt
                    cmd = start.copy()
                    cmd[0] = float(start[0] + uh[0] * speed * t)
                    cmd[1] = float(start[1] + uh[1] * speed * t)
                    # DiffIK FF: v_cmd = v_target + K·(p* − p_hand).
                    scene.set_ee_target(cmd, quat, GRIP_OPEN, vel_w=vel_cmd)
                    scene.step(kinematic_object=True)
                    pinch = scene.get_pinch_point()
                    err = pinch[:2] - cmd[:2]
                    along, cross = _along_cross(err, uh)
                    # Actual speed along û (FD).
                    if k == 0:
                        prev = pinch.copy()
                        v_act = 0.0
                    else:
                        v_act = float(np.dot((pinch[:2] - prev[:2]) / dt, uh))
                        prev = pinch.copy()
                    tick_log.append(
                        {
                            "start_xy_m": [sx, sy],
                            "heading_deg": float(hdg),
                            "cmd_speed_m_s": float(speed),
                            "t_s": float(t),
                            "cmd_xy_m": cmd[:2].tolist(),
                            "pinch_xy_m": pinch[:2].tolist(),
                            "pinch_z_m": float(pinch[2]),
                            "along_mm": 1000.0 * along,
                            "cross_mm": 1000.0 * cross,
                            "v_act_along_m_s": v_act,
                            "clipped_path": clipped,
                        }
                    )
                    if k >= n_skip:
                        alongs.append(along)
                        crosses.append(cross)
                        speeds_act.append(v_act)

                row = {
                    "start_xy_m": [sx, sy],
                    "heading_deg": float(hdg),
                    "cmd_speed_m_s": float(speed),
                    "cmd_speed_cm_s": 100.0 * float(speed),
                    "clipped_path": clipped,
                    "n_steady": len(alongs),
                    "along_mean_mm": 1000.0 * float(np.mean(alongs)),
                    "along_std_mm": 1000.0 * float(np.std(alongs)),
                    "cross_mean_mm": 1000.0 * float(np.mean(crosses)),
                    "cross_std_mm": 1000.0 * float(np.std(crosses)),
                    "v_act_mean_m_s": float(np.mean(speeds_act)),
                    "v_act_mean_cm_s": 100.0 * float(np.mean(speeds_act)),
                    "speed_err_cm_s": 100.0 * (float(np.mean(speeds_act)) - float(speed)),
                }
                rows.append(row)
                print(
                    f"start=({sx:.2f},{sy:.2f}) hdg={hdg:3.0f}° v={100*speed:.0f} "
                    f"along={row['along_mean_mm']:+.2f}±{row['along_std_mm']:.2f} mm  "
                    f"cross={row['cross_mean_mm']:+.2f}±{row['cross_std_mm']:.2f} mm  "
                    f"v_act={row['v_act_mean_cm_s']:.1f} cm/s"
                    + ("  [clipped]" if clipped else "")
                )

    # Summaries by speed and by start zone.
    def _zone(xy):
        x, y = xy
        if x <= 0.34:
            return "near_base"
        if abs(x - 0.36) < 0.04 and y < -0.05:
            return "diag_s1001"
        if abs(x - 0.37) < 0.04 and y > 0.15:
            return "diag_s1003"
        if x >= 0.60:
            return "far"
        return "mid"

    by_speed = {}
    for sp in speeds:
        sub = [r for r in rows if abs(r["cmd_speed_m_s"] - sp) < 1e-9 and not r["clipped_path"]]
        if not sub:
            sub = [r for r in rows if abs(r["cmd_speed_m_s"] - sp) < 1e-9]
        by_speed[f"{100*sp:.0f}cm_s"] = {
            "n": len(sub),
            "along_mean_mm": float(np.mean([r["along_mean_mm"] for r in sub])),
            "along_std_mm": float(np.std([r["along_mean_mm"] for r in sub])),
            "cross_mean_mm": float(np.mean([r["cross_mean_mm"] for r in sub])),
            "cross_std_mm": float(np.std([r["cross_mean_mm"] for r in sub])),
            "cross_abs_mean_mm": float(np.mean([abs(r["cross_mean_mm"]) for r in sub])),
        }

    by_zone = {}
    for r in rows:
        z = _zone(r["start_xy_m"])
        by_zone.setdefault(z, []).append(r)
    zone_sum = {}
    for z, sub in by_zone.items():
        zone_sum[z] = {
            "n": len(sub),
            "along_mean_mm": float(np.mean([r["along_mean_mm"] for r in sub])),
            "cross_abs_mean_mm": float(np.mean([abs(r["cross_mean_mm"]) for r in sub])),
            "cross_mean_mm": float(np.mean([r["cross_mean_mm"] for r in sub])),
        }

    # Clean gate: unclipped and |along| < 10 mm (drops reach-edge FF blowups).
    clean = {
        f"{100*sp:.0f}cm_s": None for sp in speeds
    }
    for sp in speeds:
        sub = [
            r for r in rows
            if abs(r["cmd_speed_m_s"] - sp) < 1e-9
            and not r["clipped_path"]
            and abs(r["along_mean_mm"]) < 10.0
        ]
        if not sub:
            continue
        alongs = [r["along_mean_mm"] for r in sub]
        crosses = [r["cross_mean_mm"] for r in sub]
        # Residual tip lag from along trail: τ ≈ −along / speed.
        along_m = float(np.mean(alongs)) / 1000.0
        tau_ms = float("nan") if sp < 1e-9 else -1000.0 * along_m / float(sp)
        clean[f"{100*sp:.0f}cm_s"] = {
            "n": len(sub),
            "along_mean_mm": float(np.mean(alongs)),
            "along_std_mm": float(np.std(alongs)),
            "cross_mean_mm": float(np.mean(crosses)),
            "cross_std_mm": float(np.std(crosses)),
            "cross_wobble_mm": float(np.mean([r["cross_std_mm"] for r in sub])),
            "along_wobble_mm": float(np.mean([r["along_std_mm"] for r in sub])),
            "residual_tip_lag_ms": tau_ms,
        }

    ff_k = float(cfg.get("robot", {}).get("diffik_ff_gain", 0.0) or 0.0)
    tip_pad_ms = float(cfg.get("state_machine_params", {}).get("robot_approach_pad_s", 0.0) or 0.0) * 1000.0
    payload = {
        "metric": (
            "e = pinch_actual − pinch_cmd; along = e·û; cross = e·n̂. "
            "+along = hand ahead of the moving command (lag → negative). "
            "Steady window after {:.2f} s for {:.2f} s."
        ).format(STEADY_SKIP_S, STEADY_HOLD_S),
        "diffik_ff_gain": ff_k,
        "tip_lag_pad_ms": tip_pad_ms,
        "z_grasp_m": z_grasp,
        "table_z_m": float(scene._table_z),
        "tcp_offset_m": list(cfg.get("robot", {}).get("tcp_offset", [])),
        "n_runs": len(rows),
        "by_speed": by_speed,
        "by_speed_clean": clean,
        "by_zone": zone_sum,
        "runs": rows,
    }
    (out / "diffik_tracking.json").write_text(json.dumps(payload, indent=2) + "\n")
    # Tick log can be large — write separately.
    with (out / "diffik_tracking_ticks.jsonl").open("w") as fp:
        for t in tick_log:
            fp.write(json.dumps(t) + "\n")

    lines = [
        "# DiffIK tracking probe (no object / no SM)",
        "",
        payload["metric"],
        "",
        f"DiffIK FF gain K = **{ff_k:.2f}** (1/s); tip_lag_pad = **{tip_pad_ms:.0f}** ms.",
        f"Grasp Z = table + {args.grasp_z_above_table_m:.3f} m = **{z_grasp:.4f} m**. "
        f"Palm down. Pinch = ``get_pinch_point``.",
        "",
        "## Gate (unclipped, |along| < 10 mm)",
        "",
        "| speed | n | along mean±std mm | cross wobble mm | residual tip lag ms |",
        "|---|---|---|---|---|",
    ]
    for k, v in clean.items():
        if not v:
            continue
        lines.append(
            f"| {k} | {v['n']} | {v['along_mean_mm']:+.2f}±{v['along_std_mm']:.2f} | "
            f"{v['cross_wobble_mm']:.2f} | {v['residual_tip_lag_ms']:+.1f} |"
        )
    lines += [
        "",
        "## By commanded speed (all unclipped)",
        "",
        "| speed | n | along mean±std mm | cross mean±std mm | |cross| mean mm |",
        "|---|---|---|---|---|",
    ]
    for k, v in by_speed.items():
        lines.append(
            f"| {k} | {v['n']} | {v['along_mean_mm']:+.2f}±{v['along_std_mm']:.2f} | "
            f"{v['cross_mean_mm']:+.2f}±{v['cross_std_mm']:.2f} | {v['cross_abs_mean_mm']:.2f} |"
        )
    lines += [
        "",
        "## By start zone",
        "",
        "| zone | n | along mean mm | cross mean mm | |cross| mean mm |",
        "|---|---|---|---|---|",
    ]
    for z, v in zone_sum.items():
        lines.append(
            f"| {z} | {v['n']} | {v['along_mean_mm']:+.2f} | "
            f"{v['cross_mean_mm']:+.2f} | {v['cross_abs_mean_mm']:.2f} |"
        )
    lines += [
        "",
        "## Sign",
        "",
        "- **along < 0**: pinch trails the moving command (classic DiffIK lag).",
        "- **cross ≠ 0**: sideways miss — cannot be a pure time lag.",
        "- Gate target: |along mean| ≤ 1 mm, cross wobble ≤ 1 mm → tip_lag_pad ≈ 0.",
        "",
        f"JSON: `{out / 'diffik_tracking.json'}`",
        "",
    ]
    md = out / "diffik_tracking.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {out / 'diffik_tracking.json'} and {md}")
    os._exit(0)


if __name__ == "__main__":
    main()
