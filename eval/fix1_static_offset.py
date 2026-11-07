#!/usr/bin/env python3
"""Fix 1 — measure / fit / report the static hand–object offset.

v=0, Δ=0 settles on a yaw × pose × object grid. Fits:

  e(ψ, pose) = b_world + R(ψ)·b_hand + g(pose) + noise

where e = object_centre − pinch (XY). Does NOT change tip_lag, close gate,
or ride correction. Writes ``data/eval/fix1_static_offset/``.

  conda activate dynamicVLA_isaac
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python eval/fix1_static_offset.py --headless --enable_cameras --device cuda:0

Phases: measure | analyze | report (default: all after AppLauncher measure).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
OUT_DEFAULT = FORESIGHT_ROOT / "data" / "eval" / "fix1_static_offset"

YAWS_DEG = (0.0, 45.0, 90.0, 135.0)
RADII_M = (0.35, 0.50, 0.65)
# Reachable fan in front of the Franka (table along +X). Not a full 360° ring.
AZIMUTHS_DEG = (-30.0, 0.0, 30.0)
OBJECTS = ("egg", "lemon", "onion")

V_SETTLE_M_S = 0.002  # 0.2 cm/s
SETTLE_HOLD_S = 0.30
MAX_SETTLE_S = 4.0
GRIP_OPEN = 1.0


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def grasp_quat_yaw(psi_rad: float) -> np.ndarray:
    """Palm-down grasp with fingers yawed by ``psi`` about world Z (wxyz).

    Base = 180° about X (init_pose style), then yaw about Z.
    """
    # qx(π) = (0, 1, 0, 0)
    q_down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    half = 0.5 * float(psi_rad)
    q_yaw = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
    return _quat_mul(q_yaw, q_down)


def rot_z(psi_rad: float) -> np.ndarray:
    c, s = math.cos(psi_rad), math.sin(psi_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def grid_targets() -> list[dict]:
    """108 (yaw × radius × azimuth × object) settle specs."""
    rows = []
    for obj in OBJECTS:
        for yaw in YAWS_DEG:
            for r in RADII_M:
                for az in AZIMUTHS_DEG:
                    th = math.radians(az)
                    rows.append(
                        {
                            "object": obj,
                            "yaw_deg": float(yaw),
                            "r_m": float(r),
                            "azimuth_deg": float(az),
                            "xy_m": [float(r * math.cos(th)), float(r * math.sin(th))],
                        }
                    )
    assert len(rows) == 108, len(rows)
    return rows


def _fit_bias(rows: list[dict]) -> dict:
    """Least-squares: e_xy = b_world + R_z(ψ)[:,:2] @ b_hand_xy  (4 params)."""
    # Stack [ex, ey] = [1,0, R00, R01; 0,1, R10, R11] @ [bw_x, bw_y, bh_x, bh_y]
    A = []
    b = []
    for r in rows:
        if not r.get("ok"):
            continue
        psi = math.radians(float(r["yaw_deg"]))
        R = rot_z(psi)
        ex, ey = float(r["e_pinch_x_m"]), float(r["e_pinch_y_m"])
        # row for ex
        A.append([1.0, 0.0, R[0, 0], R[0, 1]])
        b.append(ex)
        A.append([0.0, 1.0, R[1, 0], R[1, 1]])
        b.append(ey)
    if len(b) < 8:
        return {"ok": False, "reason": f"too few settles ({len(b)//2})"}
    A_m = np.asarray(A, dtype=np.float64)
    b_m = np.asarray(b, dtype=np.float64)
    x, *_ = np.linalg.lstsq(A_m, b_m, rcond=None)
    bw = x[:2]
    bh = x[2:]
    # Residuals
    res = []
    for r in rows:
        if not r.get("ok"):
            continue
        psi = math.radians(float(r["yaw_deg"]))
        R = rot_z(psi)
        pred = bw + R[:2, :2] @ bh
        e = np.array([r["e_pinch_x_m"], r["e_pinch_y_m"]], dtype=np.float64)
        res.append(e - pred)
    res_a = np.asarray(res, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.sum(res_a**2, axis=1))))
    # Pose trend on residual magnitude vs r
    rs, mags = [], []
    for r, rv in zip([rr for rr in rows if rr.get("ok")], res_a):
        rs.append(float(r["r_m"]))
        mags.append(float(np.linalg.norm(rv)))
    rs_a = np.asarray(rs)
    mag_a = np.asarray(mags)
    if len(rs_a) >= 4 and float(np.std(rs_a)) > 1e-6:
        slope_r, intercept_r = np.polyfit(rs_a, mag_a, 1)
    else:
        slope_r, intercept_r = float("nan"), float("nan")
    return {
        "ok": True,
        "n": int(len(res_a)),
        "b_world_m": bw.tolist(),
        "b_world_mm": (bw * 1000.0).tolist(),
        "b_hand_m": bh.tolist(),
        "b_hand_mm": (bh * 1000.0).tolist(),
        "residual_rms_m": rms,
        "residual_rms_mm": 1000.0 * rms,
        "residual_std_mm": float(np.std(mag_a) * 1000.0),
        "slope_res_vs_r_mm_per_m": 1000.0 * float(slope_r) if np.isfinite(slope_r) else None,
        "intercept_res_vs_r_mm": 1000.0 * float(intercept_r) if np.isfinite(intercept_r) else None,
        "need_g_pose": bool(rms > 0.0005 or (np.isfinite(slope_r) and abs(slope_r) > 0.001)),
    }


def _pass_metrics(rows: list[dict], *, use_tcp: bool = False) -> dict:
    """Pass-bar numbers on settled rows (e = o − h, h = pinch or TCP)."""
    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        return {"n": 0}
    if use_tcp:
        ex = np.array([r["e_tcp_x_m"] for r in ok_rows], dtype=np.float64)
        ey = np.array([r["e_tcp_y_m"] for r in ok_rows], dtype=np.float64)
    else:
        ex = np.array([r["e_pinch_x_m"] for r in ok_rows], dtype=np.float64)
        ey = np.array([r["e_pinch_y_m"] for r in ok_rows], dtype=np.float64)
    e2 = np.stack([ex, ey], axis=1)
    per_obj = {}
    for obj in OBJECTS:
        sub = [r for r in ok_rows if r.get("object") == obj]
        if not sub:
            continue
        if use_tcp:
            ox = np.array([r["e_tcp_x_m"] for r in sub])
            oy = np.array([r["e_tcp_y_m"] for r in sub])
        else:
            ox = np.array([r["e_pinch_x_m"] for r in sub])
            oy = np.array([r["e_pinch_y_m"] for r in sub])
        per_obj[obj] = {
            "mean_e_x_mm": float(np.mean(ox) * 1000.0),
            "mean_e_y_mm": float(np.mean(oy) * 1000.0),
            "n": len(sub),
        }
    return {
        "n": len(ok_rows),
        "mean_e_x_mm": float(np.mean(ex) * 1000.0),
        "mean_e_y_mm": float(np.mean(ey) * 1000.0),
        "mean_abs_e_x_mm": float(np.mean(np.abs(ex)) * 1000.0),
        "mean_abs_e_y_mm": float(np.mean(np.abs(ey)) * 1000.0),
        "rms_e_mm": float(np.sqrt(np.mean(np.sum(e2**2, axis=1))) * 1000.0),
        "max_abs_e_mm": float(np.max(np.abs(e2)) * 1000.0),
        "per_object": per_obj,
        "bar_mean_abs_ok": bool(
            all(
                abs(v["mean_e_x_mm"]) <= 0.3 and abs(v["mean_e_y_mm"]) <= 0.3
                for v in per_obj.values()
            )
        )
        if per_obj
        else False,
        "bar_rms_ok": bool(
            float(np.sqrt(np.mean(np.sum(e2**2, axis=1))) * 1000.0) <= 0.5
        ),
        "bar_max_ok": bool(float(np.max(np.abs(e2)) * 1000.0) <= 1.0),
    }


def analyze(out: Path) -> dict:
    """Fit bias model from ``settles.jsonl``; write ``fit.json``."""
    rows = []
    with (out / "settles.jsonl").open() as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    fit = _fit_bias(rows)
    fit["pass_pinch"] = _pass_metrics(rows, use_tcp=False)
    fit["pass_tcp"] = _pass_metrics(rows, use_tcp=True)
    # Dominance
    if fit.get("ok"):
        bw = np.linalg.norm(fit["b_world_m"])
        bh = np.linalg.norm(fit["b_hand_m"])
        # Constant world bias with tiny residual is often DiffIK steady-state
        # (cmd = object, hand short) — still labeled b_world for the model;
        # the apply step aims with +b_world so the hand lands on the object.
        if fit.get("need_g_pose") and fit.get("residual_rms_mm", 0) > 0.5:
            fit["dominant"] = "g_pose"
        elif bw >= 2.0 * bh and bw >= 0.0003:
            fit["dominant"] = "b_world"
        elif bh >= 2.0 * bw and bh >= 0.0003:
            fit["dominant"] = "b_hand"
        elif fit.get("need_g_pose"):
            fit["dominant"] = "g_pose"
        else:
            fit["dominant"] = "mixed" if max(bw, bh) >= 0.0003 else "negligible"
    # TCP vs pinch from first ok row
    for r in rows:
        if r.get("tcp_minus_pinch_hand_mm") is not None:
            fit["tcp_minus_pinch_hand_mm"] = r["tcp_minus_pinch_hand_mm"]
            break
    (out / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    return fit


def write_report(out: Path, fit: dict, *, before: dict | None = None, after: dict | None = None) -> None:
    lines = [
        "# Fix 1 — static hand–object offset",
        "",
        "All expert demos recorded **before** this fix are invalid as training labels.",
        "",
        "## Hypothesis fit",
        "",
        "```",
        json.dumps(fit, indent=2),
        "```",
        "",
        f"**Dominant term:** `{fit.get('dominant')}`",
        "",
        "## Chosen fix",
        "",
    ]
    dom = fit.get("dominant")
    if dom == "b_hand":
        lines.append(
            "Correct TCP / pinch transform (`robot.tcp_offset_mm` → geometric pinch Z)."
        )
    elif dom == "b_world":
        lines.append(
            "Correct object reference: expert aims at `object_xy + b_world` "
            "(stored as `robot.static_aim_bias_mm` / applied in scene aim)."
        )
    elif dom == "g_pose":
        lines.append("Pose-dependent residual → joint controller integral / gravity (sim).")
    else:
        lines.append(f"See fit (`{dom}`); apply source-level corrections listed in fit.json.")
    lines += ["", "## Pass bar", ""]
    if before or after:
        lines.append("| Metric | Before | After | Bar |")
        lines.append("|---|---|---|---|")
        for key, bar in (
            ("mean_abs_e_mm", "≤ 0.3 per axis/object"),
            ("rms_e_mm", "≤ 0.5"),
            ("max_abs_e_mm", "≤ 1.0"),
        ):
            b = before.get(key) if before else None
            a = after.get(key) if after else None
            lines.append(f"| {key} | {b} | {a} | {bar} |")
    else:
        lines.append("_Re-run measure after applying the fix to fill before/after._")
    lines += [
        "",
        "## TCP vs pinch (hand frame)",
        "",
        f"`tcp − pinch` (mm) = `{fit.get('tcp_minus_pinch_hand_mm')}`",
        "",
        "## C30 cross-check",
        "",
        "See `c30_crosscheck.json` (diagnostic only).",
        "",
        "## Files",
        "",
        "- `sim/pinch_point.py`",
        "- `sim/scene.py` (`get_pinch_point`, `get_hand_pose`)",
        "- `eval/fix1_static_offset.py`",
        "- `tests/test_pinch_point.py`",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output_dir", default=str(OUT_DEFAULT))
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("--phase", default="all", help="measure|analyze|report|all")
    p.add_argument("--object", default=None, help="Restrict to one object category")
    p.add_argument("--max-settles", type=int, default=0, help="Cap for smoke (0=all 108)")
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("fix1")
    exit_code = 0

    if args.phase in ("analyze", "report") and args.phase != "all":
        fit = analyze(out) if (out / "settles.jsonl").exists() else {}
        if args.phase == "report" or True:
            write_report(out, fit)
        return

    AppLauncher(args)
    try:
        from sim.scene import Phase0Scene, load_cfg

        cfg = load_cfg(Path(args.sim_cfg_file))
        cfg.setdefault("claim_timing", {})["enabled_noise_on_replan"] = False
        cfg["claim_timing"]["apply_delta_delay_to_robot"] = False
        device = str(getattr(args, "device", None) or "cuda:0")
        cfg.setdefault("sim", {})["device"] = device

        # One object per process (Isaac World cannot rebuild). Use --object or default egg.
        obj = str(args.object or "egg")
        targets = [t for t in grid_targets() if t["object"] == obj]
        if int(args.max_settles) > 0:
            targets = targets[: int(args.max_settles)]

        jsonl = out / "settles.jsonl"
        # Append across object workers; host truncates once.
        if args.phase in ("measure", "all") and not (out / f".lock_{obj}").exists():
            if obj == "egg" and not any((out / f".lock_{o}").exists() for o in OBJECTS):
                jsonl.write_text("")
            (out / f".lock_{obj}").write_text("1\n")

        log.info("building scene object=%s n=%d", obj, len(targets))
        scene = Phase0Scene(
            cfg,
            enable_cameras=False,
            object_usd=obj,
            asset_seed=1000,
        )
        tcp_pinch = scene.get_tcp_minus_pinch_hand_m()
        log.info("tcp−pinch hand mm = %s", (tcp_pinch * 1000.0).tolist())
        # Keep settles on the play band so DiffIK stays well-conditioned.
        bounds = cfg["scene"]["object_xy_bounds"]
        (xmin, xmax), (ymin, ymax) = bounds
        log.info("play bounds x=[%.3f,%.3f] y=[%.3f,%.3f]", xmin, xmax, ymin, ymax)

        for k, spec in enumerate(targets):
            seed = 7000 + k
            meta = scene.reset_episode(seed=seed, speed=0.0)
            st = scene.get_object_state()
            xy = np.asarray(spec["xy_m"], dtype=np.float64)
            xy[0] = float(np.clip(xy[0], xmin + 0.02, xmax - 0.02))
            xy[1] = float(np.clip(xy[1], ymin + 0.02, ymax - 0.02))
            z_obj = float(st.position[2])
            pos = np.array([xy[0], xy[1], z_obj], dtype=np.float64)
            quat_obj = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            scene._coast_pos = pos.copy()
            scene._coast_quat = quat_obj.copy()
            scene._obj_velocity = np.zeros(3)
            scene._write_object_pose(pos, quat_obj, np.zeros(3))

            psi = math.radians(float(spec["yaw_deg"]))
            q_ee = grasp_quat_yaw(psi)
            target = pos.copy()
            t_slow0 = None
            ok = False
            reason = None
            t0 = scene._sim_time
            n_steps = int(MAX_SETTLE_S / float(cfg["sim"]["dt"])) + 5
            prev_ee = None
            prev_t = None
            for _ in range(n_steps):
                scene.set_ee_target(target, q_ee, GRIP_OPEN)
                scene.step(kinematic_object=True, attach_object=False, freeze_object=False)
                ee, _ = scene.get_ee_pose()
                if prev_ee is None:
                    prev_ee, prev_t = ee.copy(), scene._sim_time
                    continue
                dt = scene._sim_time - prev_t
                if dt <= 1e-9:
                    continue
                v_hand = float(np.linalg.norm((ee[:2] - prev_ee[:2]) / dt))
                prev_ee, prev_t = ee.copy(), scene._sim_time
                if v_hand < V_SETTLE_M_S:
                    if t_slow0 is None:
                        t_slow0 = scene._sim_time
                    elif scene._sim_time - t_slow0 >= SETTLE_HOLD_S:
                        ok = True
                        break
                else:
                    t_slow0 = None
            else:
                reason = "settle_timeout"

            hand_pos, hand_quat = scene.get_hand_pose()
            tcp, _ = scene.get_ee_pose()
            pinch = scene.get_pinch_point()
            obj_pos = scene.get_object_state().position.copy()
            e_p = obj_pos - pinch
            e_t = obj_pos - tcp
            row = {
                **spec,
                "seed": seed,
                "ok": bool(ok),
                "reason": reason,
                "object_xyz_m": obj_pos.tolist(),
                "tcp_xyz_m": tcp.tolist(),
                "pinch_xyz_m": pinch.tolist(),
                "hand_xyz_m": hand_pos.tolist(),
                "hand_quat_wxyz": hand_quat.tolist(),
                "e_pinch_x_m": float(e_p[0]),
                "e_pinch_y_m": float(e_p[1]),
                "e_pinch_z_m": float(e_p[2]),
                "e_tcp_x_m": float(e_t[0]),
                "e_tcp_y_m": float(e_t[1]),
                "e_tcp_z_m": float(e_t[2]),
                "e_pinch_xy_mm": [1000.0 * float(e_p[0]), 1000.0 * float(e_p[1])],
                "e_tcp_xy_mm": [1000.0 * float(e_t[0]), 1000.0 * float(e_t[1])],
                "tcp_minus_pinch_hand_mm": (tcp_pinch * 1000.0).tolist(),
                "wall_s": float(scene._sim_time - t0),
                "category": meta.get("category"),
            }
            with jsonl.open("a") as fp:
                fp.write(json.dumps(row) + "\n")
            log.info(
                "[%s %d/%d] yaw=%.0f r=%.2f az=%.0f ok=%s e_pinch_xy_mm=%s e_tcp_xy_mm=%s",
                obj,
                k + 1,
                len(targets),
                spec["yaw_deg"],
                spec["r_m"],
                spec["azimuth_deg"],
                ok,
                [round(x, 2) for x in row["e_pinch_xy_mm"]],
                [round(x, 2) for x in row["e_tcp_xy_mm"]],
            )

        # Analyze only when all objects present (or single-object run).
        n_lines = sum(1 for _ in jsonl.open()) if jsonl.exists() else 0
        if n_lines >= 100 or args.object is not None:
            fit = analyze(out)
            write_report(out, fit)
            log.info(
                "fit dominant=%s rms_mm=%.3f",
                fit.get("dominant"),
                fit.get("residual_rms_mm"),
            )
        log.info("done → %s", out)
    except Exception:
        logging.exception("fix1 failed")
        exit_code = 1
    finally:
        log.info("exiting code=%d (skip app.close hang)", exit_code)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
