"""A1: is the velocity in the policy's input pack right?  Pass: v̂ / v_true = 1.00 ± 0.02.

Offline, no Isaac. Replays the static-camera tracker + Kalman filter (perception.StaticCamEstimator,
current sim/phase0_cfg.yaml camera) over recorded expert throws, and compares the velocity entries
of the pack the policy receives (interfaces.state.conditioning_vector, last two numbers) with:

  v_pos  — velocity from the object's true positions, central difference over ±HALF_WIN s
           (the reference: what the object really did)
  v_sim  — the simulator's logged body velocity (oracle_state[3:5]); this is also what the
           training packs are built from, so v_sim / v_pos checks the training labels

Per frame, ratio = projection of the estimate on the reference: (v̂ · v_ref) / |v_ref|².
1.00 = right speed along the true direction. Also reports |v̂| / |v_ref| and heading error.

Frames used: free object (before hold / lift / off-table / off-area, as in g2_perception_report),
after filter warm-up, hand > HAND_CLEAR m from the object (no occlusion), |v_ref| >= MIN_SPEED.
At 0 m/s the ratio is undefined; the residual speed |v̂| (cm/s) is reported instead.

A throw slows down (friction). A constant-velocity filter lags, so on a decelerating object it
reads high by about a·lag. The decel and the lag-corrected ratio are reported so a real scale
bug (wrong camera / time base) can be told apart from honest lag.

Usage: python eval/a1_velocity_ratio.py [--root data/eval/A1_velocity] [--workers 16]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from interfaces.config import policy_look_ahead_s, load_yaml  # noqa: E402
from interfaces.state import conditioning_vector  # noqa: E402
from perception.estimator import StaticCamEstimator  # noqa: E402

import g2_perception_report as g2  # noqa: E402  (reuse its episode reader / stop rules)

HALF_WIN = 0.10      # s, central-difference half window for v_pos
WARMUP_S = 0.20      # s after the first fix before frames count
HAND_CLEAR = 0.12    # m, hand-object distance below which the hand may hide the object
MIN_SPEED = 0.03     # m/s, below this the ratio is meaningless
_CFG = None
_DELTA = 0.25


def _init():
    global _CFG, _DELTA
    cv2.setNumThreads(1)
    _CFG = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    _DELTA = policy_look_ahead_s(_CFG)


def one(path: str) -> dict:
    ep = g2._read_episode(path)
    ts, orc, rgb = ep["ts"], ep["oracle"], ep["rgb"]
    with h5py.File(path) as h:
        ee = h["ee_pos"][: len(ts)]
    n = len(ts)
    out = {"path": path, "category": ep["category"], "n_free": n, "rows": []}
    if n < 10:
        return out
    est = StaticCamEstimator.from_cfg(_CFG)
    if ep["xy_bounds"] is not None:
        est.search_xy_bounds = tuple(map(tuple, ep["xy_bounds"]))
    out["start_ok"] = bool(est.start(rgb[0], orc[0, :3]))
    p = orc[:, :2]
    t_first = None
    for t in range(n):
        st = est.step(rgb[t], ts[t])
        if est.last_uv is not None and t_first is None:
            t_first = ts[t]
        if st is None or t_first is None or ts[t] - t_first < WARMUP_S:
            continue
        lo = np.searchsorted(ts, ts[t] - HALF_WIN)
        hi = min(np.searchsorted(ts, ts[t] + HALF_WIN), n - 1)
        if hi <= lo:
            continue
        v_pos = (p[hi] - p[lo]) / (ts[hi] - ts[lo])
        v_sim = orc[t, 3:5]
        v_hat = conditioning_vector(st, _DELTA)[-2:]
        hand = float(np.linalg.norm(ee[t, :2] - p[t]))
        # local deceleration along the motion, from positions (for the lag check)
        lo2 = np.searchsorted(ts, ts[t] - 2 * HALF_WIN)
        hi2 = min(np.searchsorted(ts, ts[t] + 2 * HALF_WIN), n - 1)
        acc = np.nan
        if hi2 > hi and lo2 < lo:
            v_a = (p[lo] - p[lo2]) / (ts[lo] - ts[lo2])
            v_b = (p[hi2] - p[hi]) / (ts[hi2] - ts[hi])
            u = v_pos / max(np.linalg.norm(v_pos), 1e-9)
            acc = float((v_b - v_a) @ u / ((ts[hi2] + ts[hi]) / 2 - (ts[lo] + ts[lo2]) / 2))
        out["rows"].append({
            "t": float(ts[t]), "v_hat": v_hat.tolist(), "v_pos": v_pos.tolist(), "v_sim": v_sim.tolist(),
            "hand": hand, "valid": bool(st.valid), "acc": acc,
        })
    return out


def proj(a, b):
    return float(a @ b / (b @ b))


def summarize(results, speed):
    R = [r for res in results for r in res["rows"] if r["hand"] > HAND_CLEAR and r["valid"]]
    s = {"speed": speed, "episodes": len(results), "frames": len(R),
         "start_ok": sum(bool(res.get("start_ok")) for res in results)}
    if not R:
        return s
    vh = np.array([r["v_hat"] for r in R]); vp = np.array([r["v_pos"] for r in R]); vs = np.array([r["v_sim"] for r in R])
    sp = np.linalg.norm(vp, axis=1)
    s["true_speed_median_cm_s"] = float(np.median(sp) * 100)
    s["vhat_speed_median_cm_s"] = float(np.median(np.linalg.norm(vh, axis=1)) * 100)
    m = sp >= MIN_SPEED
    s["frames_moving"] = int(m.sum())
    if m.sum() < 5:
        return s
    r_pos = np.array([proj(a, b) for a, b in zip(vh[m], vp[m])])
    r_sim = np.array([proj(a, b) for a, b in zip(vs[m], vp[m])])
    mag = np.linalg.norm(vh[m], axis=1) / sp[m]
    ang = np.degrees(np.arccos(np.clip(np.sum(vh[m] * vp[m], 1) / (np.linalg.norm(vh[m], axis=1) * sp[m] + 1e-12), -1, 1)))
    acc = np.array([r["acc"] for r, k in zip(R, m) if k])
    # Episode-level medians (frames within an episode are correlated) for the CI
    ep_r = []
    for res in results:
        rr = [proj(np.array(r["v_hat"]), np.array(r["v_pos"])) for r in res["rows"]
              if r["hand"] > HAND_CLEAR and r["valid"] and np.linalg.norm(r["v_pos"]) >= MIN_SPEED]
        if len(rr) >= 3:
            ep_r.append(np.median(rr))
    ep_r = np.array(ep_r)
    boot = [np.median(np.random.default_rng(i).choice(ep_r, len(ep_r))) for i in range(2000)] if len(ep_r) > 1 else [np.nan]
    s.update({
        "ratio_vhat_vs_pos_median": float(np.median(r_pos)),
        "ratio_vhat_vs_pos_p10_p90": [float(x) for x in np.percentile(r_pos, [10, 90])],
        "ratio_episode_median_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "n_episodes_ratio": int(len(ep_r)),
        "magnitude_ratio_median": float(np.median(mag)),
        "heading_err_deg_median": float(np.median(ang)),
        "ratio_vsim_vs_pos_median": float(np.median(r_sim)),
        "decel_median_m_s2": float(np.nanmedian(acc)),
        "pass_1.00_pm_0.02": bool(abs(np.median(r_pos) - 1) <= 0.02),
    })
    return s


def preflight(files) -> None:
    """Stop (no silent adapting) if the recordings don't match the current setup."""
    from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT

    if tuple(CONDITIONING_LAYOUT[-2:]) != ("v_hat_x", "v_hat_y"):
        sys.exit(f"A1 stop: pack layout {CONDITIONING_LAYOUT} no longer ends with v_hat_x, v_hat_y; "
                 "update this script before trusting its numbers.")
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    cam = cfg["static_camera"]
    want_hw = (int(cam["height"]), int(cam["width"]))
    want_dt = 1.0 / float(cam.get("fps") or 25)
    for f in files:
        meta = json.loads(Path(f).with_suffix(".json").read_text()) if Path(f).with_suffix(".json").exists() else {}
        lay = meta.get("conditioning_layout")
        if lay is not None and (tuple(lay) != tuple(CONDITIONING_LAYOUT) or meta.get("conditioning_dim") != CONDITIONING_DIM):
            sys.exit(f"A1 stop: {f} was recorded with pack {lay} (dim {meta.get('conditioning_dim')}), "
                     f"current code packs {CONDITIONING_LAYOUT}. Re-record with the current setup.")
        with h5py.File(f) as h:
            hw = tuple(h["static_cam_rgb"].shape[1:3])
            ts = h["timestamp"][:]
        if hw != want_hw:
            sys.exit(f"A1 stop: {f} frames are {hw}, current static_camera config is {want_hw}. "
                     "Recordings and config don't match; re-record with the current setup.")
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else want_dt
        if abs(dt - want_dt) > 0.15 * want_dt:
            sys.exit(f"A1 stop: {f} frame interval {dt*1000:.1f} ms, config says {want_dt*1000:.1f} ms "
                     f"({cam.get('fps')} fps). Recordings and config don't match.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "eval" / "A1_velocity"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.root)
    report = []
    with Pool(a.workers, initializer=_init) as pool:
        for vdir in sorted(root.glob("v*")):
            speed = float(vdir.name[1:])
            files = sorted(glob.glob(str(vdir / "**" / "*.h5"), recursive=True))
            ready = []
            for f in files:  # a recorder may still be writing the newest file
                try:
                    with h5py.File(f):
                        ready.append(f)
                except OSError:
                    pass
            if len(ready) < len(files):
                print(f"A1: {vdir.name}: skipped {len(files) - len(ready)} file(s) still being written", flush=True)
            files = ready
            if not files:
                print(f"A1: no recordings under {vdir}, skipped", flush=True)
                continue
            preflight(files)
            res = pool.map(one, files)
            n_ok = sum(bool(r.get("start_ok")) for r in res)
            if n_ok < 0.5 * len(res):
                sys.exit(f"A1 stop at {vdir.name}: the tracker found the object at its projected start in only "
                         f"{n_ok}/{len(res)} episodes. The camera pose in sim/phase0_cfg.yaml likely does not "
                         "match these recordings; no ratio reported.")
            report.append(summarize(res, speed))
            print(json.dumps(report[-1]), flush=True)
    (root / "a1_report.json").write_text(json.dumps(report, indent=1))
    lines = ["# A1: velocity in the policy pack vs truth", "",
             "| speed | episodes (tracked) | moving frames | true speed cm/s | v̂/v_true median [p10, p90] | episode CI95 | |v̂|/|v| | heading err ° | v_sim/v_pos | decel m/s² | pass ±0.02 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in report:
        if "ratio_vhat_vs_pos_median" in s:
            lines.append(f"| {s['speed']:.2f} | {s['episodes']} ({s['start_ok']}) | {s['frames_moving']} | {s['true_speed_median_cm_s']:.1f} | "
                         f"{s['ratio_vhat_vs_pos_median']:.3f} [{s['ratio_vhat_vs_pos_p10_p90'][0]:.2f}, {s['ratio_vhat_vs_pos_p10_p90'][1]:.2f}] | "
                         f"{s['ratio_episode_median_ci95'][0]:.3f}–{s['ratio_episode_median_ci95'][1]:.3f} | {s['magnitude_ratio_median']:.3f} | "
                         f"{s['heading_err_deg_median']:.1f} | {s['ratio_vsim_vs_pos_median']:.3f} | {s['decel_median_m_s2']:.2f} | "
                         f"{'yes' if s['pass_1.00_pm_0.02'] else 'no'} |")
        else:
            lines.append(f"| {s['speed']:.2f} | {s['episodes']} ({s.get('start_ok')}) | {s.get('frames_moving', 0)} | "
                         f"{s.get('true_speed_median_cm_s', float('nan')):.1f} | ratio undefined; residual |v̂| median "
                         f"{s.get('vhat_speed_median_cm_s', float('nan')):.2f} cm/s | | | | | | |")
    (root / "REPORT_auto.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
