#!/usr/bin/env python3
"""Fix1 closeout offline numbers from B1 v7 + Fix1 settle traces.

1. Regress e_x~v_x, e_y~v_y on matched-speed frames (B1 v7).
2. Predict cross-track for C30 s1001 / s1003.
3. τ_c from v=0 DiffIK convergence (Fix1 settles or AB v5).

Error metric: e = hand − object, + = hand ahead; mean over the matched-speed
window (|Δv| ≤ 1.39 cm/s while stage==intercept or close).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.close_guard import max_speed_diff_m_s  # noqa: E402

B1 = ROOT / "data" / "eval" / "B1_grid_v7"
OUT = ROOT / "data" / "eval" / "fix1_static_offset"
DT = 1.0 / 30.0
DV_LIM = max_speed_diff_m_s(close_time_s=0.180)  # ≈0.0139 m/s


def _matched_mask(ee, obj, vel, stage) -> np.ndarray:
    vh = np.zeros_like(ee)
    vh[1:] = (ee[1:] - ee[:-1]) / DT
    dv = np.linalg.norm(vh[:, :2] - vel[:, :2], axis=1)
    st = np.asarray(stage, dtype=np.int64)
    return (dv <= DV_LIM) & ((st == 1) | (st == 3))


def _collect_matched(root: Path):
    ex, ey, vx, vy = [], [], [], []
    per = {}
    for p in sorted(root.glob("cells/**/*.h5")):
        with h5py.File(p, "r") as h:
            if not all(k in h for k in ("ee_pos", "object_pos", "object_vel", "stage")):
                continue
            ee = np.asarray(h["ee_pos"][:], dtype=np.float64)
            obj = np.asarray(h["object_pos"][:], dtype=np.float64)
            vel = np.asarray(h["object_vel"][:], dtype=np.float64)
            stage = np.asarray(h["stage"][:], dtype=np.int64)
        m = _matched_mask(ee, obj, vel, stage)
        if not np.any(m):
            continue
        e = ee[m, :2] - obj[m, :2]
        v = vel[m, :2]
        ex.append(e[:, 0])
        ey.append(e[:, 1])
        vx.append(v[:, 0])
        vy.append(v[:, 1])
        # Per-file mean over matched window (standard metric).
        u = v.mean(axis=0)
        nu = float(np.linalg.norm(u))
        mean_e = e.mean(axis=0)
        if nu > 1e-6:
            uh = u / nu
            nh = np.array([-uh[1], uh[0]])
            along = float(mean_e @ uh)
            cross = float(mean_e @ nh)
        else:
            along = float(mean_e[0])
            cross = float(mean_e[1])
        per[p.name] = {
            "along_mm": 1000.0 * along,
            "cross_mm": 1000.0 * cross,
            "mean_e_mm": (1000.0 * mean_e).tolist(),
            "mean_v_mms": (1000.0 * v.mean(axis=0)).tolist(),
            "n": int(m.sum()),
        }
    return (
        np.concatenate(ex) if ex else np.zeros(0),
        np.concatenate(ey) if ey else np.zeros(0),
        np.concatenate(vx) if vx else np.zeros(0),
        np.concatenate(vy) if vy else np.zeros(0),
        per,
    )


def _lin(v, e):
    """e ≈ L · v  (through origin). L in seconds."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    denom = float(np.dot(v, v))
    if denom < 1e-12:
        return float("nan"), float("nan")
    L = float(np.dot(v, e) / denom)
    resid = e - L * v
    rms = float(np.sqrt(np.mean(resid**2)))
    return L, rms


def _tau_c_from_settles(jsonl: Path) -> dict:
    """τ_c from exponential approach on AB v5 static closes (proxy).

    Fit |e|(t) ≈ |e0| exp(−t/τ_c) on pre-close frames of v=0 demos, or
    report from Fix1 settle wall times if present.
    """
    ab = ROOT / "data" / "eval" / "AB_recordings_v5" / "v0.00"
    taus = []
    for p in sorted(ab.rglob("*.h5"))[:24]:
        with h5py.File(p, "r") as h:
            stage = np.asarray(h["stage"][:], dtype=np.int64)
            ee = np.asarray(h["ee_pos"][:], dtype=np.float64)
            obj = np.asarray(h["object_pos"][:], dtype=np.float64)
        hits = np.flatnonzero(stage == 3)
        if hits.size == 0:
            continue
        i_c = int(hits[0])
        # last 12 frames before close
        i0 = max(0, i_c - 12)
        mag = np.linalg.norm(ee[i0:i_c, :2] - obj[i0:i_c, :2], axis=1)
        t = np.arange(len(mag), dtype=np.float64) * DT
        mag = np.clip(mag, 1e-6, None)
        # log|e| = log|e0| − t/τ
        if len(mag) < 4:
            continue
        A = np.vstack([np.ones_like(t), -t]).T
        coef, *_ = np.linalg.lstsq(A, np.log(mag), rcond=None)
        inv_tau = float(coef[1])
        if inv_tau > 1e-3:
            taus.append(1.0 / inv_tau)
    settles = {}
    if jsonl.exists():
        rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        walls = [float(r["wall_s"]) for r in rows if r.get("ok")]
        settles = {
            "n": len(walls),
            "mean_settle_wall_s": float(np.mean(walls)) if walls else None,
        }
    return {
        "tau_c_s_mean": float(np.mean(taus)) if taus else None,
        "tau_c_s_std": float(np.std(taus)) if taus else None,
        "n_traces": len(taus),
        "fix1_settles": settles,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex, ey, vx, vy, per = _collect_matched(B1)
    Lx, rms_x = _lin(vx, ex)
    Ly, rms_y = _lin(vy, ey)
    # C30 seeds
    pred = {}
    for seed in (1001, 1003):
        name = f"C30_v30_d000_s{seed}_egg.h5"
        row = per.get(name)
        if row is None:
            # try path search
            hits = list((B1 / "cells" / "C30").glob(f"*s{seed}*.h5"))
            if hits:
                with h5py.File(hits[0], "r") as h:
                    ee = np.asarray(h["ee_pos"][:], np.float64)
                    obj = np.asarray(h["object_pos"][:], np.float64)
                    vel = np.asarray(h["object_vel"][:], np.float64)
                    stage = np.asarray(h["stage"][:], np.int64)
                m = _matched_mask(ee, obj, vel, stage)
                mean_e = (ee[m, :2] - obj[m, :2]).mean(axis=0) if np.any(m) else np.zeros(2)
                mean_v = vel[m, :2].mean(axis=0) if np.any(m) else np.zeros(2)
            else:
                mean_e = mean_v = np.zeros(2)
                m = np.zeros(0, dtype=bool)
        else:
            mean_e = np.asarray(row["mean_e_mm"]) / 1000.0
            mean_v = np.asarray(row["mean_v_mms"]) / 1000.0
        # Model: e ≈ (L_x v_x, L_y v_y); cross = e · n_hat
        e_hat = np.array(
            [
                Lx * mean_v[0] if np.isfinite(Lx) else 0.0,
                Ly * mean_v[1] if np.isfinite(Ly) else 0.0,
            ]
        )
        nu = float(np.linalg.norm(mean_v))
        if nu > 1e-6:
            uh = mean_v / nu
            nh = np.array([-uh[1], uh[0]])
            cross_pred = float(e_hat @ nh)
            along_pred = float(e_hat @ uh)
            cross_meas = float(mean_e @ nh)
            along_meas = float(mean_e @ uh)
        else:
            cross_pred = along_pred = cross_meas = along_meas = float("nan")
        pred[f"s{seed}"] = {
            "along_meas_mm": 1000.0 * along_meas,
            "cross_meas_mm": 1000.0 * cross_meas,
            "along_pred_mm": 1000.0 * along_pred,
            "cross_pred_mm": 1000.0 * cross_pred,
            "mean_v_mms": (1000.0 * mean_v).tolist(),
        }

    tau = _tau_c_from_settles(OUT / "settles.jsonl")
    payload = {
        "metric": "e = hand − object; + = hand ahead; mean over matched-speed window",
        "dv_limit_cms": 100.0 * DV_LIM,
        "n_matched_frames": int(ex.size),
        "L_x_s": Lx,
        "L_y_s": Ly,
        "rms_ex_mm": 1000.0 * rms_x if np.isfinite(rms_x) else None,
        "rms_ey_mm": 1000.0 * rms_y if np.isfinite(rms_y) else None,
        "c30_predict": pred,
        "tau_c": tau,
    }
    (OUT / "closeout_lag_fit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
