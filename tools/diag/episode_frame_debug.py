#!/usr/bin/env python3
"""Frame-by-frame debug for one B1 episode .h5.

Why: agents need the same numbers for pass/fail (ride, close gate, park,
delay, reach-stop). Scores with the expert's inputs: +1 mm hand-eye
(seed+17, same as sim/collect.py) and object-now = live GT + hand-eye
(Δ_true unwrap of delayed obs under constant-velocity coast). Prefer logged
``ee_vel`` (Isaac EE-link) for the speed gate; fall back to FD if absent.

Sign: e = hand − object; along-track + = hand ahead of the object.

Usage:
  PYTHONPATH=. python tools/diag/episode_frame_debug.py \\
    data/eval/.../cells/C33/C33_v30_d300_s1000_egg.h5 \\
    -o data/eval/.../debug_agents/C33/s1000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# Match expert gates (phase0_cfg / close_guard).
XY_OK_M = 0.002
RIDE_BOX_M = 0.010
T_CONTACT_S = 0.130
T_CLOSE_S = 0.180
DV_LIM_MPS = 0.0025 / T_CLOSE_S  # ≈ 0.0139 m/s = 1.39 cm/s
R_ZONE_M = 0.20
REACH_STOP_M = 0.10
LO4 = -3.0718
DT = 1.0 / 30.0


def hand_eye_1mm(seed: int) -> np.ndarray:
    """Same 1 mm hand-eye as ``sim/collect.py`` (rng seed+17, unit direction)."""
    rng = np.random.default_rng(int(seed) + 17)
    direction = rng.normal(size=3)
    n = float(np.linalg.norm(direction))
    return (direction / max(n, 1e-9)) * 0.001


def path_to_zone(p: np.ndarray, v: np.ndarray, r: float = R_ZONE_M) -> float:
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        return float("inf")
    u = v / s
    b = float(p @ u)
    c = float(p @ p - r * r)
    if c <= 0:
        return 0.0
    disc = b * b - c
    if b >= 0 or disc < 0:
        return float("inf")
    return float(-b - np.sqrt(disc))


def analyze(h5_path: Path, *, hand_eye: np.ndarray | None = None) -> dict:
    """Read every control frame and score ride / close / park / reach.

    Gate geometry uses expert object-now ≈ live GT + 1 mm hand-eye (constant-v
    unwrap of delayed obs with Δ_true). Pass ``hand_eye`` from meta when known;
    else rebuild from seed+17.
    """
    with h5py.File(h5_path, "r") as f:
        st = f["stage"][:]
        ee = f["ee_pos"][:]
        ob = f["object_pos"][:]
        vo = f["object_vel"][:]
        act = f["action_pos"][:]
        q = f["proprio"][:, :7]
        grip = f["gripper"][:]
        hold = f["holding"][:] if "holding" in f else np.zeros(len(st))
        delta_meas = f["delta_meas_s"][:] if "delta_meas_s" in f else None
        applied = f["applied_delta_s"][:] if "applied_delta_s" in f else None
        ee_vel_log = f["ee_vel"][:] if "ee_vel" in f else None
        success = bool(int(np.asarray(f["success"][()]).reshape(-1)[0]))
        seed = int(np.asarray(f["seed"][()]).reshape(-1)[0]) if "seed" in f else None
        speed = float(np.asarray(f["speed_m_s"][()]).reshape(-1)[0]) if "speed_m_s" in f else None
        d_ms = int(np.asarray(f["delta_ms"][()]).reshape(-1)[0]) if "delta_ms" in f else None

    n = int(len(st))
    # Expert saw delayed (live − v·Δ_true) + hand_eye, then forecast(Δ_true)
    # → object_now ≈ live + hand_eye under constant-velocity coast.
    if hand_eye is None:
        he = hand_eye_1mm(seed) if seed is not None else np.zeros(3)
    else:
        he = np.asarray(hand_eye, dtype=np.float64).reshape(3)
    ob_now = ob + he.reshape(1, 3)
    # e = hand − object_now; + along û = hand ahead of the object.
    e_xy = ee[:, :2] - ob_now[:, :2]
    gap_l1 = np.abs(e_xy).max(1)  # worst-axis (matches close_xy per-axis style)
    gap_l2 = np.linalg.norm(e_xy, axis=1)
    stages, counts = np.unique(st, return_counts=True)
    stage_hist = {int(s): int(c) for s, c in zip(stages, counts)}

    # Prefer Isaac EE-link velocity (what the expert gate uses); else FD.
    v_src = "ee_vel_link" if ee_vel_log is not None else "ee_pos_fd"
    if ee_vel_log is not None:
        vh = np.asarray(ee_vel_log[:, :2], dtype=np.float64)
    else:
        vh = np.zeros_like(ee[:, :2])
        vh[1:] = (ee[1:, :2] - ee[:-1, :2]) / DT
    dv = vh - vo[:, :2]
    dv_norm = np.linalg.norm(dv, axis=1)

    # Predictive close (same math as close_guard).
    ep = e_xy + dv * T_CONTACT_S
    ok_xy_now = (np.abs(e_xy[:, 0]) <= XY_OK_M) & (np.abs(e_xy[:, 1]) <= XY_OK_M)
    ok_xy_pred = (np.abs(ep[:, 0]) <= XY_OK_M) & (np.abs(ep[:, 1]) <= XY_OK_M)
    ok_xy = ok_xy_now & ok_xy_pred
    ok_v = dv_norm <= DV_LIM_MPS
    ok_both = ok_xy & ok_v

    intercept = np.where(st == 1)[0]
    ride = np.array([t for t in intercept if gap_l1[t] <= RIDE_BOX_M and t >= 1], dtype=int)
    # "Park": stage-1, near ride edge but outside box (10–14 mm).
    park = []
    for t in intercept:
        if t < 1:
            continue
        g = float(gap_l1[t])
        if RIDE_BOX_M < g <= 0.014:
            park.append(int(t))

    v0 = vo[min(5, n - 1), :2]
    spd0 = float(np.linalg.norm(v0))
    heading = float(np.degrees(np.arctan2(v0[1], v0[0]))) if spd0 > 1e-3 else None
    axis_share = float(max(abs(v0[0]), abs(v0[1])) / spd0) if spd0 > 1e-3 else None

    def ride_stats(idx: np.ndarray) -> dict | None:
        if len(idx) < 3 or spd0 < 1e-3:
            return None
        u = vo[idx, :2] / np.maximum(np.linalg.norm(vo[idx, :2], axis=1, keepdims=True), 1e-9)
        nn = np.stack([-u[:, 1], u[:, 0]], 1)
        e = e_xy[idx]
        dvi = dv[idx]
        # Aim lead vs object-now (+ = aim ahead of object along û).
        aim = act[idx, :2] - ob_now[idx, :2]
        return {
            "n": int(len(idx)),
            "e_along_mm_med": float(1000 * np.median((e * u).sum(1))),
            "e_along_mm_p10": float(1000 * np.percentile((e * u).sum(1), 10)),
            "e_along_mm_p90": float(1000 * np.percentile((e * u).sum(1), 90)),
            "e_cross_mm_med": float(1000 * np.median((e * nn).sum(1))),
            "gap_worst_axis_mm_med": float(1000 * np.median(np.abs(e).max(1))),
            "gap_l2_mm_med": float(1000 * np.median(np.linalg.norm(e, axis=1))),
            "dv_along_cms_med": float(100 * np.median((dvi * u).sum(1))),
            "dv_norm_cms_med": float(100 * np.median(np.linalg.norm(dvi, axis=1))),
            "hand_speed_cms_med": float(100 * np.median(np.linalg.norm(vh[idx], axis=1))),
            "obj_speed_cms_med": float(100 * np.median(np.linalg.norm(vo[idx, :2], axis=1))),
            "aim_along_mm_med": float(1000 * np.median((aim * u).sum(1))),
            "aim_cross_mm_med": float(1000 * np.median((aim * nn).sum(1))),
            "hand_z_err_mm_med": float(1000 * np.median(ee[idx, 2] - ob_now[idx, 2])),
            "frac_xy": float(ok_xy[idx].mean()),
            "frac_v": float(ok_v[idx].mean()),
            "frac_both": float(ok_both[idx].mean()),
            "n_xy_pass": int(ok_xy[idx].sum()),
            "n_v_pass": int(ok_v[idx].sum()),
            "n_both_pass": int(ok_both[idx].sum()),
            "v_hand_source": v_src,
            "hand_eye_mm": float(1000 * np.linalg.norm(he)),
            "sign": "+ = hand/aim ahead of object along v",
        }

    closes = np.where(st == 3)[0]
    home = np.where(st == 8)[0]
    close_info = None
    if len(closes):
        c = int(closes[0])
        close_info = {
            "t_s": float(c * DT),
            "frame": c,
            "obj_r_m": float(np.linalg.norm(ob[c, :2])),
            "xy_err_mm": float(1000 * gap_l2[c]),
            "worst_axis_mm": float(1000 * gap_l1[c]),
            "dz_mm": float(1000 * (ee[c, 2] - ob_now[c, 2])),
            "dv_cms": float(100 * dv_norm[c]),
            "ok_xy": bool(ok_xy[c]),
            "ok_v": bool(ok_v[c]),
            "ok_both": bool(ok_both[c]),
            "grip": float(grip[c]),
            "holding_after_5": bool(hold[min(c + 5, n - 1)] > 0.5),
            "stages_after": sorted({int(x) for x in st[c:]}),
            "obj_z_gain_mm": float(1000 * (ob[c:, 2].max() - ob[c, 2])),
        }

    stop_info = None
    if len(home) and (not len(closes) or int(home[0]) < int(closes[0])):
        h = int(home[0])
        if not len(closes) or h < int(closes[0]):
            stop_info = {
                "t_s": float(h * DT),
                "frame": h,
                "obj_r_m": float(np.linalg.norm(ob[h, :2])),
                "hand_r_m": float(np.linalg.norm(ee[h, :2])),
                "gap_mm": float(1000 * gap_l2[h]),
                "ptz_cm": float(100 * path_to_zone(ob[h, :2], vo[h, :2])),
                "obj_xy": ob[h, :2].round(4).tolist(),
            }

    j4_deg = np.degrees(q[:, 3] - LO4)
    chase = (st >= 1) & (st <= 4)
    j4_info = {
        "j4_min_deg_chase": float(j4_deg[chase].min()) if chase.any() else None,
        "hand_r_min_chase": float(np.linalg.norm(ee[chase, :2], axis=1).min()) if chase.any() else None,
        "obj_r_min": float(np.linalg.norm(ob[:, :2], axis=1).min()),
        "obj_r_at_end": float(np.linalg.norm(ob[-1, :2])),
    }

    timeline = []
    if len(ride):
        for t in [int(ride[0]), int(ride[len(ride) // 2]), int(ride[-1])]:
            timeline.append(
                {
                    "tag": "ride",
                    "frame": t,
                    "t_s": float(t * DT),
                    "stage": int(st[t]),
                    "gap_worst_mm": float(1000 * gap_l1[t]),
                    "e_x_mm": float(1000 * e_xy[t, 0]),
                    "e_y_mm": float(1000 * e_xy[t, 1]),
                    "dv_cms": float(100 * dv_norm[t]),
                    "ok_xy": bool(ok_xy[t]),
                    "ok_v": bool(ok_v[t]),
                    "ok_both": bool(ok_both[t]),
                    "aim_lead_mm": float(1000 * np.linalg.norm(act[t, :2] - ob_now[t, :2])),
                    "delta_meas_ms": None if delta_meas is None else float(1000 * delta_meas[t]),
                }
            )
    if len(intercept):
        cand = intercept[np.argsort(gap_l1[intercept])[:5]]
        for t in cand:
            timeline.append(
                {
                    "tag": "best_gap",
                    "frame": int(t),
                    "t_s": float(t * DT),
                    "stage": int(st[t]),
                    "gap_worst_mm": float(1000 * gap_l1[t]),
                    "e_x_mm": float(1000 * e_xy[t, 0]),
                    "e_y_mm": float(1000 * e_xy[t, 1]),
                    "dv_cms": float(100 * dv_norm[t]),
                    "ok_xy": bool(ok_xy[t]),
                    "ok_v": bool(ok_v[t]),
                    "ok_both": bool(ok_both[t]),
                }
            )

    gate_fail = None
    if len(ride):
        xy_fail = ~ok_xy[ride]
        v_fail = ~ok_v[ride]
        gate_fail = {
            "ride_ticks": int(len(ride)),
            "fail_xy_only": int((xy_fail & ~v_fail).sum()),
            "fail_v_only": int((~xy_fail & v_fail).sum()),
            "fail_both_parts": int((xy_fail & v_fail).sum()),
            "pass_both": int(ok_both[ride].sum()),
            "median_|ex|_when_xy_fail_mm": float(
                1000 * np.median(np.abs(e_xy[ride[xy_fail], 0]))
            )
            if xy_fail.any()
            else None,
            "median_|ey|_when_xy_fail_mm": float(
                1000 * np.median(np.abs(e_xy[ride[xy_fail], 1]))
            )
            if xy_fail.any()
            else None,
            "median_dv_when_v_fail_cms": float(100 * np.median(dv_norm[ride[v_fail]]))
            if v_fail.any()
            else None,
        }

    report = {
        "h5": str(h5_path),
        "seed": seed,
        "success": success,
        "n_frames": n,
        "timeout": bool(n >= 300),
        "speed_cms": None if speed is None else float(100 * speed),
        "delta_ms_meta": d_ms,
        "heading_deg": heading,
        "axis_share": axis_share,
        "stage_hist": stage_hist,
        "min_gap_mm_stage01": float(1000 * gap_l1[st <= 1].min()) if (st <= 1).any() else None,
        "t_first_ride_s": float(ride[0] * DT) if len(ride) else None,
        "ride": ride_stats(ride),
        "park_ticks": int(len(park)),
        "park_frac_of_intercept": float(len(park) / max(len(intercept), 1)),
        "n_intercept": int(len(intercept)),
        "close": close_info,
        "stop_before_close": stop_info,
        "j4": j4_info,
        "gate_fail_on_ride": gate_fail,
        "delta_meas_ms_med": None
        if delta_meas is None
        else float(1000 * np.median(delta_meas)),
        "applied_delta_ms_med": None
        if applied is None
        else float(1000 * np.median(applied)),
        "gates": {
            "xy_ok_mm": 1000 * XY_OK_M,
            "ride_box_mm": 1000 * RIDE_BOX_M,
            "dv_lim_cms": 100 * DV_LIM_MPS,
            "t_contact_ms": 1000 * T_CONTACT_S,
            "t_close_ms": 1000 * T_CLOSE_S,
            "reach_stop_mm": 1000 * REACH_STOP_M,
            "v_hand_source": v_src,
            "hand_eye_applied": True,
            "sign": "+ = hand/aim ahead of object along v",
        },
        "timeline": timeline[:12],
        "why_pass_or_fail": None,
    }

    r = report["ride"]
    if success and close_info:
        report["why_pass_or_fail"] = (
            f"PASS: closed at t={close_info['t_s']:.2f}s with worst-axis "
            f"{close_info['worst_axis_mm']:.2f} mm, |Δv|={close_info['dv_cms']:.2f} cm/s "
            f"(limits {1000*XY_OK_M:.0f} mm / {100*DV_LIM_MPS:.2f} cm/s). "
            f"Ride n={0 if r is None else r['n']}, frac_both="
            f"{0 if r is None else r['frac_both']:.3f}, e_along_med="
            f"{0 if r is None else r['e_along_mm_med']:.2f} mm, park_ticks={report['park_ticks']}."
        )
    elif close_info is None and r is not None:
        report["why_pass_or_fail"] = (
            f"FAIL never_closed-like: rode {r['n']} ticks, frac_both={r['frac_both']:.3f} "
            f"(xy={r['frac_xy']:.3f}, v={r['frac_v']:.3f}), e_along_med={r['e_along_mm_med']:.2f} mm "
            f"(need ≤{1000*XY_OK_M:.0f} mm/axis), park_ticks={report['park_ticks']}, "
            f"j4_min={j4_info['j4_min_deg_chase']}, hand_r_min={j4_info['hand_r_min_chase']}."
        )
    elif stop_info is not None:
        report["why_pass_or_fail"] = (
            f"FAIL unreachable-like: stop/home at t={stop_info['t_s']:.2f}s "
            f"obj_r={stop_info['obj_r_m']:.3f} m gap={stop_info['gap_mm']:.1f} mm, "
            f"ride_n={0 if r is None else r['n']}, park={report['park_ticks']}."
        )
    else:
        report["why_pass_or_fail"] = (
            f"outcome success={success}; close={close_info is not None}; "
            f"ride_n={0 if r is None else r['n']}; park={report['park_ticks']}."
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    # Prefer hand_eye from sidecar JSON meta when present.
    he = None
    side = args.h5.with_suffix(".json")
    if side.is_file():
        try:
            meta = json.loads(side.read_text())
            if "hand_eye_m" in meta:
                he = np.asarray(meta["hand_eye_m"], dtype=np.float64)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            he = None
    rep = analyze(args.h5, hand_eye=he)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "success": rep["success"],
                "why": rep["why_pass_or_fail"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
