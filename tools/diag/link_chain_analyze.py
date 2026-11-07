#!/usr/bin/env python3
"""Link-by-link math check for one B1 episode .h5 (+ sidecar .json).

Reads only what collect logged. Missing physics (mass/friction/contact flags)
is marked unavailable — do not invent it.

Links:
  1 Seeing   p_now = p_delayed + v·Δ_meas vs live GT
  2 Aiming   aim vs p_now (ride ticks)
  3 Tracking hand vs aim; v_hand (Isaac) vs v_obj
  4 Close    e_now / e_pred / Δv gates
  5 Close mechanics  gripper + object motion in stage 2
  6 Lift/hold  object−hand while holding

Usage:
  PYTHONPATH=. python tools/diag/link_chain_analyze.py path/to/ep.h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from control.close_guard import (
    DEFAULT_SLACK_M,
    DEFAULT_T_CONTACT_S,
    DEFAULT_CLOSE_TIME_S,
    max_speed_diff_m_s,
)

XY_OK_M = 0.002
RIDE_BOX_M = 0.010


def _dv_lim_m_s() -> float:
    """Live gate lim from phase0 ``close_speed_slack_m`` (fallback plan 2.5 mm)."""
    try:
        from interfaces.config import load_yaml

        cfg = load_yaml(Path(__file__).resolve().parents[2] / "sim" / "phase0_cfg.yaml")
        sm = cfg.get("state_machine_params") or {}
        slack = float(sm.get("close_speed_slack_m", DEFAULT_SLACK_M))
    except Exception:  # noqa: BLE001 — offline diag must not die on cfg glitches
        slack = float(DEFAULT_SLACK_M)
    return max_speed_diff_m_s(slack_m=slack, close_time_s=DEFAULT_CLOSE_TIME_S)


DV_LIM = _dv_lim_m_s()
DT = 1.0 / 30.0
STAGE_APPROACH, STAGE_CLOSE, STAGE_LIFT = 1, 2, 3


def _med_p95(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "med": None, "p95": None}
    return {
        "n": int(x.size),
        "med": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


def _load_meta(h5: Path) -> dict:
    side = h5.with_suffix(".json")
    if not side.is_file():
        return {}
    try:
        return json.loads(side.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def analyze_episode(h5_path: Path) -> dict:
    """Score links 1–6 for one episode. Returns a JSON-serializable dict."""
    meta = _load_meta(h5_path)
    with h5py.File(h5_path, "r") as f:
        st = f["stage"][:]
        ee = f["ee_pos"][:]
        # Prefer same-tick pre-step EE for link-3 hand−aim (pairs with action_pos).
        # Post-step ee_pos still used for ride/gates (physics after DiffIK).
        ee_pre = f["ee_pre"][:] if "ee_pre" in f else None
        ob = f["object_pos"][:]
        vo = f["object_vel"][:]
        act = f["action_pos"][:]
        grip = f["gripper"][:]
        hold = f["holding"][:] if "holding" in f else np.zeros(len(st))
        d_meas = f["delta_meas_s"][:] if "delta_meas_s" in f else np.zeros(len(st))
        d_app = f["applied_delta_s"][:] if "applied_delta_s" in f else np.full(len(st), np.nan)
        # Expert code-path flag (per tick). Absent in runs before it was logged.
        aim_br = f["aim_branch"][:] if "aim_branch" in f else None
        t_cap = f["t_capture_used"][:] if "t_capture_used" in f else None
        t_exec = f["t_exec"][:] if "t_exec" in f else None
        ee_vel = f["ee_vel"][:] if "ee_vel" in f else None
        success = bool(int(np.asarray(f["success"][()]).reshape(-1)[0]))
        seed = int(np.asarray(f["seed"][()]).reshape(-1)[0]) if "seed" in f else None
        speed = float(np.asarray(f["speed_m_s"][()]).reshape(-1)[0]) if "speed_m_s" in f else None
        d_ms = int(np.asarray(f["delta_ms"][()]).reshape(-1)[0]) if "delta_ms" in f else None

    n = len(st)
    he = np.asarray(meta.get("hand_eye_m") or [0, 0, 0], dtype=np.float64).reshape(3)
    # Live GT is logged. Reconstruct delayed obs + expert p_now:
    #   p_delayed ≈ live − v·Δ_true   (const-v coast; HE applied on delayed)
    #   p_now     = p_delayed + v·Δ_meas + HE
    d_true = np.where(np.isfinite(d_app), d_app, 0.0)
    p_delayed = ob - vo * d_true.reshape(-1, 1) + he.reshape(1, 3)
    p_now = p_delayed + vo * d_meas.reshape(-1, 1)
    # Vs live GT (+ HE, since gate uses delayed+forecast(Δ_true) ≈ live+HE).
    p_true_gate = ob + he.reshape(1, 3)
    err1 = np.linalg.norm((p_now - p_true_gate)[:, :2], axis=1)

    v0 = vo[min(5, n - 1), :2]
    spd0 = float(np.linalg.norm(v0))
    heading = float(np.degrees(np.arctan2(v0[1], v0[0]))) if spd0 > 1e-3 else None
    u = vo[:, :2] / np.maximum(np.linalg.norm(vo[:, :2], axis=1, keepdims=True), 1e-9)

    # Ride ticks: stage-1, |e|_∞ ≤ 10 mm vs object-now (expert HE).
    e_gate = ee[:, :2] - p_true_gate[:, :2]
    gap = np.abs(e_gate).max(1)
    ride = np.array(
        [t for t in np.where(st == STAGE_APPROACH)[0] if gap[t] <= RIDE_BOX_M and t >= 1],
        dtype=int,
    )

    # Link 2: aim − p_now on ride (+ = aim ahead of object-now along v).
    if len(ride):
        aim_err_xy = act[ride, :2] - p_now[ride, :2]
        aim_along = (aim_err_xy * u[ride]).sum(1)
        aim_norm = np.linalg.norm(aim_err_xy, axis=1)
    else:
        aim_along = aim_norm = np.array([])

    # Link 3: hand − aim; v_hand − v_obj
    if ee_vel is not None:
        vh = ee_vel[:, :2]
        v_src = "ee_vel_link"
    else:
        vh = np.zeros((n, 2))
        vh[1:] = (ee[1:, :2] - ee[:-1, :2]) / DT
        v_src = "ee_pos_fd"
    if len(ride):
        ee_for_aim = ee_pre if ee_pre is not None else ee
        hand_aim = np.linalg.norm(ee_for_aim[ride, :2] - act[ride, :2], axis=1)
        dv = np.linalg.norm(vh[ride] - vo[ride, :2], axis=1)
        hand_spd = np.linalg.norm(vh[ride], axis=1)
        obj_spd = np.linalg.norm(vo[ride, :2], axis=1)
    else:
        hand_aim = dv = hand_spd = obj_spd = np.array([])

    # Link 4: close gates on ride
    e_now = e_gate
    e_pred = e_now + (vh - vo[:, :2]) * DEFAULT_T_CONTACT_S
    ok_xy = (
        (np.abs(e_now[:, 0]) <= XY_OK_M)
        & (np.abs(e_now[:, 1]) <= XY_OK_M)
        & (np.abs(e_pred[:, 0]) <= XY_OK_M)
        & (np.abs(e_pred[:, 1]) <= XY_OK_M)
    )
    ok_v = np.linalg.norm(vh - vo[:, :2], axis=1) <= DV_LIM
    ok_both = ok_xy & ok_v
    closes = np.where(st == STAGE_CLOSE)[0]
    first_close = int(closes[0]) if len(closes) else None
    if len(ride):
        e_along = (e_now[ride] * u[ride]).sum(1)
        e_cross = (e_now[ride] * np.stack([-u[ride, 1], u[ride, 0]], 1)).sum(1)
        frac_both = float(ok_both[ride].mean())
        ticks_to_gate = None
        if first_close is not None and len(ride):
            # how many ride ticks before close
            ticks_to_gate = int(np.searchsorted(ride, first_close))
    else:
        e_along = e_cross = np.array([])
        frac_both = float("nan")
        ticks_to_gate = None

    # Link 5: stage-2 mechanics (no contact flags in log).
    # Gripper in this h5 is the commanded open fraction after the step; on
    # close entry it is often already 0, so "time to ≤0.2" is N/A — not 0 s shut.
    if len(closes):
        c0, c1 = int(closes[0]), int(closes[-1])
        seg = slice(c0, c1 + 1)
        grip_seg = grip[seg]
        obj_move = np.linalg.norm(ob[c0 : c1 + 1, :2] - ob[c0, :2], axis=1)
        obj_z_pop = float(ob[c0 : c1 + 1, 2].max() - ob[c0, 2])
        hand_z_rise = float(ee[c0 : c1 + 1, 2].max() - ee[c0, 2])
        grip0 = float(grip[c0])
        if grip0 <= 0.2:
            t_shut_s = None
            t_shut_note = "N/A: grip already ≤0.2 on first close tick (logged cmd)"
        else:
            shut = np.where(grip_seg <= 0.2)[0]
            t_shut_s = float(shut[0] * DT) if len(shut) else None
            t_shut_note = "ok" if t_shut_s is not None else "never_reached_0p2"
        link5 = {
            "n_close_ticks": int(c1 - c0 + 1),
            "t_grip_below_0p2_s": t_shut_s,
            "t_grip_note": t_shut_note,
            "obj_xy_travel_mm": _med_p95(1000 * obj_move),
            "obj_z_pop_mm": 1000.0 * obj_z_pop,
            "hand_z_rise_mm": 1000.0 * hand_z_rise,
            "grip_start": grip0,
            "grip_end": float(grip[c1]),
            "contact_flags": "unavailable",
        }
    else:
        link5 = {"n_close_ticks": 0, "contact_flags": "unavailable"}

    # Link 6: lift / hold. Drop = object slips vs hand, not world-Z span
    # (place successes lift tens of cm — that used to false-flag dropped_like).
    lift = np.where(st >= STAGE_LIFT)[0]
    hold_idx = np.array([t for t in lift if hold[t] > 0.5], dtype=int)
    if len(hold_idx) >= 3:
        rel = ob[hold_idx] - ee[hold_idx]
        rel_std = np.std(rel, axis=0)
        z_lift = float(ob[hold_idx, 2].max() - ob[hold_idx[0], 2])
        rel_z_span = float(rel[:, 2].max() - rel[:, 2].min())
        # Slip: relative Z wanders; world Z span is the lift itself — ignore it.
        slipped = bool(rel_std[2] > 0.02 or rel_z_span > 0.03)
        if success and not slipped:
            dropped = False
            drop_note = "ok (place success; relative hold stable)"
        elif success and slipped:
            # Still place-success: annotate soft, do not call "dropped".
            dropped = False
            drop_note = "N/A on success: relative motion noted but place latched"
        else:
            dropped = slipped
            drop_note = "rel_z_slip" if slipped else "ok"
        link6 = {
            "n_hold": int(len(hold_idx)),
            "rel_std_mm": (1000 * rel_std).tolist(),
            "rel_z_span_mm": 1000.0 * rel_z_span,
            "obj_z_lift_mm": 1000.0 * z_lift,
            "dropped_like": dropped,
            "drop_note": drop_note,
        }
    else:
        link6 = {
            "n_hold": int(len(hold_idx)),
            "dropped_like": None,
            "drop_note": "N/A: too few hold ticks",
        }

    # Delay path bookkeeping
    d_meas_med = float(np.nanmedian(d_meas)) if np.isfinite(d_meas).any() else None
    d_app_med = float(np.nanmedian(d_app)) if np.isfinite(d_app).any() else None
    # Chase-aim branch the expert *logged* it ran (not inferred from Δ).
    # Old runs without the flag → "unlogged"; do not guess from Δ.
    if aim_br is None:
        aim_branch_counts = None
        aim_branch = "unlogged"
    else:
        codes = list(meta.get("aim_branch_codes") or [])
        vals, cnts = np.unique(np.asarray(aim_br, dtype=np.int64), return_counts=True)
        aim_branch_counts = {
            (codes[v] if 0 <= v < len(codes) else f"code{v}"): int(c)
            for v, c in zip(vals.tolist(), cnts.tolist())
        }
        chase = {k: v for k, v in aim_branch_counts.items() if k != "none"}
        aim_branch = "+".join(sorted(chase)) if chase else "none"

    n_close_attempts = int(len(closes) > 0)  # SM doesn't log retries separately here
    first_attempt = True  # no multi-attempt flag in log

    return {
        "h5": str(h5_path),
        "meta": {
            "object": meta.get("category") or meta.get("object"),
            "seed": seed,
            "v_cms": None if speed is None else 100.0 * speed,
            "delta_ms": d_ms,
            "heading_deg": heading,
            "w_pre_m": meta.get("w_pre_m"),
            "w_pre_raw_m": meta.get("w_pre_raw_m"),
            "w_pre_clamped": meta.get("w_pre_clamped"),
            "object_half_height_m": meta.get("object_half_height"),
            "mass": "unavailable",
            "friction": "unavailable",
            "failure_mode": meta.get("failure_mode"),
            "success": success,
            "first_attempt": first_attempt,
            "n_close_attempts": n_close_attempts,
            "obs_delay_s_meta": meta.get("obs_delay_s"),
            "aim_branch": aim_branch,
            "aim_branch_counts": aim_branch_counts,
            "delta_meas_ms_med": None if d_meas_med is None else 1000 * d_meas_med,
            "applied_delta_ms_med": None if d_app_med is None else 1000 * d_app_med,
            "close_xy_err_m": meta.get("close_xy_err_m"),
            "dv_at_close_m_s": meta.get("dv_at_close_m_s"),
            "unavailable": [
                "USD mass",
                "USD friction",
                "left/right contact flags",
                "grasp yaw time series",
                "grip force (N)",
                "object width along finger axis (only w_pre)",
            ],
        },
        "link1_seeing": {
            "err_p_now_vs_liveHE_mm": _med_p95(1000 * err1),
            "healthy": "mean≈0, std≈v·5ms (~0.15 mm @30cm/s) — same every cell",
            "v_hand_source_note": "uses logged live GT + reconstructed delay",
        },
        "link2_aiming": {
            "aim_minus_p_now_mm": _med_p95(1000 * aim_norm),
            "aim_along_mm": _med_p95(1000 * aim_along),
            "ride_n": int(len(ride)),
            "healthy": "aim−p_now≈0 on ride (ZOH settle may leave ~v·dt along)",
        },
        "link3_tracking": {
            "hand_minus_aim_mm": _med_p95(1000 * hand_aim),
            "dv_cms": _med_p95(100 * dv),
            "hand_speed_cms": _med_p95(100 * hand_spd),
            "obj_speed_cms": _med_p95(100 * obj_spd),
            "v_hand_source": v_src,
            # Post-step EE vs pre-step aim → ~5–7 mm floor (lemon oks). Gate on |Δv|/frac_both.
            "healthy": (
                "|hand−aim| ~5–7 mm normal for this h5 (post-step EE vs pre-step aim); "
                "discriminate with |Δv| tip / frac_both, not 1 mm hand−aim"
            ),
        },
        "link4_close": {
            "e_along_mm": _med_p95(1000 * e_along),
            "e_cross_mm": _med_p95(1000 * e_cross),
            "frac_both": frac_both,
            "ticks_ride_before_close": ticks_to_gate,
            "first_close_frame": first_close,
            "healthy": "gate opens within a few ride ticks",
        },
        "link5_close_mech": link5,
        "link6_lift_hold": link6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    rep = analyze_episode(args.h5)
    text = json.dumps(rep, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print("wrote", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
