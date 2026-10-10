#!/usr/bin/env python3
"""G2a offline report: static-camera tracker + Kalman filter vs sim truth.

No Isaac. Reads recorded demo / eval ``.h5`` files (static RGB + sim-truth object
state), runs ``perception.StaticCamEstimator`` on every free-object frame (before
the first hold), and compares its 4-number pack with the oracle pack built from
the same sim-truth state by the same ``interfaces.state.conditioning_vector``.

Frame alignment: in both demo and eval files, ``static_cam_rgb[t]`` is rendered
after the sim step that ``oracle_state[t]`` describes (checked: blob centre is
~1 px from truth at t, ~3.5 px from t−1). Eval ``conditioning[t]`` is the pack the
policy saw *before* step t, i.e. built from ``oracle_state[t-1]`` — so we rebuild the
oracle pack from ``oracle_state[t]`` rather than reuse that column.

Why it exists: before we trust the camera path, ask "how far off are the 4
numbers from the camera, compared to sim truth?" Small error makes the later
policy swap plausible, but does not prove its success. Pass bars: see G2A_*
below; these are offline checks, not the plan's paired policy-swap gate.
Only frames before the first hold are used (free object).

House scene (``--sets house``): reads every .h5 under ``--house-dirs`` (speed and
object category from each sidecar .json; re-detect area from its
``object_xy_bounds``). Scoring stops at the first hold or when the object has
left the table (z more than 3 cm below its spawn height). Also writes .mp4
overlays: tracked blob centre, sim-truth projection and filter p̂ projection.

Run from the repo root in the sim env:
  python scripts/g2_perception_report.py            # report → data/eval/p2_perception_report/
  python scripts/g2_perception_report.py --tune     # Kalman knobs on held-out speeds (not reported)
  python scripts/g2_perception_report.py --sets house --out data/eval/p3_perception_house
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

from interfaces.config import conditioning_delta_s, load_yaml  # noqa: E402
from interfaces.state import ObjectState, conditioning_vector  # noqa: E402
from perception.estimator import StaticCamEstimator  # noqa: E402
from perception.kalman import CVKalman  # noqa: E402

SIM_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
DEMO = FORESIGHT_ROOT / "data" / "p1-retrain"
EVAL = FORESIGHT_ROOT / "data" / "eval"
OUT = EVAL / "p2_perception_report"

# (set, speed m/s, folder, max episodes or None = all). Eval = G1 model oracle runs.
REPORT_SETS = [
    ("demo", 0.00, DEMO / "speed_0.00", 80),
    ("demo", 0.10, DEMO / "speed_0.10", None),
    ("demo", 0.15, DEMO / "speed_0.15", None),
    ("demo", 0.20, DEMO / "speed_0.20", None),
    ("eval", 0.00, EVAL / "scout_b_v0_100runs", None),
    ("eval", 0.15, EVAL / "scout_b_v15_100runs", None),
    ("eval", 0.20, EVAL / "scout_b_v20_100runs", None),
]
# Knob sweep uses speeds that are NOT in the report.
TUNE_SETS = [("demo", s, DEMO / f"speed_{s:.2f}", 40) for s in (0.12, 0.14, 0.16, 0.18)]
HORIZONS = (0.10, 0.20, 0.25, 0.30)  # "look this far ahead" times (s) for the future-error check
BOUNCE_LOOKBACK_S = 0.20  # a wall hit this recently still counts (filter re-learning speed)
SMOOTH_S = 0.24  # house truth speed zigzags on a 3-frame cycle; average over 6 frames
WARMUP_FRAMES = 2  # first fixes: the filter starts at speed 0 (step 3 warms up before first plan)
OFF_TABLE_DROP_M = 0.03  # object this far below / above its spawn height = fell off / lifted
OFF_AREA_M = 0.10  # house: object this far outside the play area (pushed away, out of view)
HOUSE_DIRS = [FORESIGHT_ROOT / "data" / "p3_perception_frames"]
POS_NOISE, VEL_NOISE = 0.015, 0.03  # training noise (policy/configs/smolvla.yaml)
# The G2a bars: median and RMSE of the position-pack error, at 20 cm/s (metres).
G2A_MEDIAN_M, G2A_RMSE_M = 0.01, 0.02

# Set once per worker process (see _init_worker); read by run_episode.
_CFG: dict | None = None
_DELTA = 0.25


def _init_worker() -> None:
    # Each worker loads the sim config once and uses 1 OpenCV thread,
    # so many workers do not fight over CPU cores.
    global _CFG, _DELTA
    cv2.setNumThreads(1)
    _CFG = load_yaml(SIM_CFG)
    _DELTA = conditioning_delta_s(_CFG)


def _oracle(o: np.ndarray) -> ObjectState:
    # One logged oracle_state row → the shared state message
    # (position, velocity, covariance, timestamp, valid).
    return ObjectState(o[:3], o[3:6], o[6:12], o[12], bool(o[13]))


def _wall_hits(vel_xy: np.ndarray) -> np.ndarray:
    """Frame indices where a velocity component flips sign (moving objects only).

    On the hit frame the logged speed along that axis is nearly 0 (the position
    is clamped at the wall), so compare against the last frame with a clear speed.
    """
    hits = set()
    for axis in (0, 1):
        idx = np.where(np.abs(vel_xy[:, axis]) > 0.02)[0]
        s = np.sign(vel_xy[idx, axis])
        hits.update((idx[:-1][s[1:] != s[:-1]] + 1).tolist())
    return np.array(sorted(hits), dtype=int)


def _read_episode(path: str) -> dict:
    """Load one .h5: time, truth, camera frames — only up to the first hold."""
    # Optional sidecar .json: object category and (house) the play area.
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with h5py.File(path) as h:
        hold = h["holding"][:]
        o = h["oracle_state"][:]
        # g = first frame where the object is held, has fallen off the table, was
        # lifted (physics grasp: no glue flag), or (house) has been pushed far
        # outside the play area. Keep frames before it.
        off_table = o[:, 2] < o[0, 2] - OFF_TABLE_DROP_M
        lifted = o[:, 2] > o[0, 2] + OFF_TABLE_DROP_M
        off_area = np.zeros_like(off_table)
        if meta.get("object_xy_bounds"):
            (x0, x1), (y0, y1) = meta["object_xy_bounds"]
            m = OFF_AREA_M
            off_area = (o[:, 0] < x0 - m) | (o[:, 0] > x1 + m) | (o[:, 1] < y0 - m) | (o[:, 1] > y1 + m)
        stop = (hold > 0.5) | lifted | off_table | off_area
        g = int(np.argmax(stop)) if stop.any() else len(hold)
        why = "end" if not stop.any() else ("hold" if hold[g] > 0.5 else "lifted" if lifted[g]
                                             else "off_table" if off_table[g] else "off_area")
        ep = {
            "ts": h["timestamp"][:g],
            "oracle": h["oracle_state"][:g],
            "rgb": h["static_cam_rgb"][:g],
            "n_total": len(hold),
            "stop": why,
        }
    ep["category"] = meta.get("category")
    ep["xy_bounds"] = meta.get("object_xy_bounds")  # house: this table's play area
    return ep


def run_episode(job: tuple[str, float, str, dict | None]) -> dict:
    """Estimator over one episode's free frames → per-frame errors + episode row."""
    set_name, speed, path, kalman_kwargs = job
    ep = _read_episode(path)
    ts, orc, rgb = ep["ts"], ep["oracle"], ep["rgb"]
    n = len(ts)
    est = StaticCamEstimator.from_cfg(_CFG, kalman_kwargs=kalman_kwargs)
    if ep["xy_bounds"] is not None:  # house: re-detect inside this table's play area
        est.search_xy_bounds = tuple(map(tuple, ep["xy_bounds"]))
    row = {"set": set_name, "speed": speed, "path": path, "category": ep["category"], "n_free": n,
           "n_frames_total": ep["n_total"], "stop": ep["stop"]}
    # frames: per-frame flags and errors (has=filter gave a state, meas=camera
    #   found the object, valid=filter says state is good, dp/dv=pack error vs truth).
    # hor: per look-ahead time, error of the filter's guess (fe) and of the
    #   sim-truth straight-line guess (oe) vs where the object really went.
    frames = {k: [] for k in ("has", "meas", "valid", "dp", "dv", "near_bounce", "warmup", "z",
                              "uv", "phat", "phat_true", "dps")}
    hor = {h: {"fe": [], "oe": [], "bounce": []} for h in HORIZONS}
    if n == 0:
        return {"row": {**row, "start_ok": False}, "frames": frames, "hor": hor}
    # Sim stand-in for a detector: truth spawn pose is used once to find the
    # object in frame 0. After that the tracker sees pixels only.
    row["start_ok"] = est.start(rgb[0], orc[0, :3])
    row["mode"] = est.tracker.mode
    # Times when the object hit a wall (velocity flipped). Used to split errors.
    hit_t = ts[_wall_hits(orc[:, 3:5])]
    row["wall_hits"] = int(len(hit_t))
    # Calm sim-truth p̂: truth position + truth speed averaged over the last
    # SMOOTH_S (a perfect causal filter of the truth could give this). In house
    # runs the logged speed zigzags (e.g. 5 / 17 cm/s on alternate frames), so
    # the raw truth p̂ itself jumps by centimetres from frame to frame.
    k_s = max(1, int(round(SMOOTH_S / max(float(np.median(np.diff(ts))) if n > 1 else 0.04, 1e-3))))
    n_fixes = 0
    for t in range(n):
        st = est.step(rgb[t], ts[t])
        n_fixes += est.last_uv is not None
        o = _oracle(orc[t])
        z = None if est.last_uv is None else est.camera.lift(est.last_uv, est.plane_z)
        frames["z"].append(z)
        frames["uv"].append(est.last_uv)  # for the overlay videos
        frames["meas"].append(est.last_uv is not None)
        frames["has"].append(st is not None)
        frames["valid"].append(st is not None and st.valid)
        # Mark recent wall hits because the filter may need time to relearn speed.
        # This split looks backward from the current frame.
        frames["near_bounce"].append(
            bool(np.any((hit_t > ts[t] - BOUNCE_LOOKBACK_S) & (hit_t <= ts[t])))
        )
        frames["warmup"].append(n_fixes <= WARMUP_FRAMES and est.n_redetects == 0)
        # Pack error: filter pack minus truth pack, same Δ, same helper.
        # dp = position part (m), dv = velocity part (m/s). NaN = no estimate yet.
        if st is None:
            frames["dp"].append([np.nan, np.nan])
            frames["dv"].append([np.nan, np.nan])
            frames["phat"].append(None)
            frames["dps"].append([np.nan, np.nan])
            frames["phat_true"].append(conditioning_vector(o, _DELTA)[:2])
        else:
            mine, truth = conditioning_vector(st, _DELTA), conditioning_vector(o, _DELTA)
            diff = mine - truth
            frames["phat"].append(mine[:2])
            frames["phat_true"].append(truth[:2])
            frames["dp"].append(diff[:2])
            j = max(t - k_s, 0)
            v_s = orc[t, 3:5] if t == j else (orc[t, :2] - orc[j, :2]) / (ts[t] - ts[j])
            frames["dps"].append(mine[:2] - (orc[t, :2] + v_s * _DELTA))
            frames["dv"].append(diff[2:])
        # Future check: guess h seconds ahead, compare with the real position
        # then (read from truth by time). Skip if the episode ends before then.
        for h in HORIZONS:
            if st is None or ts[t] + h > ts[-1] + 1e-9:
                continue
            true = np.array([np.interp(ts[t] + h, ts, orc[:, i]) for i in (0, 1)])
            hor[h]["fe"].append(np.linalg.norm(st.predict(h).position[:2] - true))
            hor[h]["oe"].append(np.linalg.norm(o.predict(h).position[:2] - true))
            hor[h]["bounce"].append(
                bool(np.any((hit_t > ts[t] - BOUNCE_LOOKBACK_S) & (hit_t <= ts[t] + h)))
            )
    # Episode summary row (goes to per_episode.csv).
    #   track_rate = share of frames where the camera found the object.
    #   valid_rate = share of frames where the filter called its state valid.
    dp = np.linalg.norm(np.asarray(frames["dp"]), axis=1)
    row["track_rate"] = float(np.mean(frames["meas"]))
    row["pack_p_median_cm"] = float(np.nanmedian(dp) * 100) if np.isfinite(dp).any() else None
    row["filter_bounces"] = est.kf.n_bounces
    row["redetects"] = est.n_redetects
    row["valid_rate"] = float(np.mean(frames["valid"]))
    # Kept for --tune, which replays the filter on saved camera fixes (z) and truth packs.
    frames["ts"] = ts
    frames["oracle_pack"] = [conditioning_vector(_oracle(o), _DELTA) for o in orc]
    return {"row": row, "frames": frames, "hor": hor}


def _episodes(sets, seed: int = 0) -> list[tuple[str, float, str]]:
    """List .h5 files per set. Shuffle with a fixed seed, then apply the cap."""
    jobs = []
    rng = random.Random(seed)
    for set_name, speed, folder, cap in sets:
        files = sorted(str(p) for p in Path(folder).glob("*.h5"))
        rng.shuffle(files)
        jobs += [(set_name, speed, f) for f in (files if cap is None else files[:cap])]
    return jobs


def _house_jobs(dirs) -> list[tuple[str, float, str]]:
    """Every house .h5 under ``dirs``; the speed comes from its sidecar .json."""
    jobs = []
    for d in dirs:
        for f in sorted(Path(d).rglob("*.h5")):
            meta = json.loads(f.with_suffix(".json").read_text())
            jobs.append(("house", round(float(meta["speed_m_s"]), 2), str(f)))
    return jobs


def _video(r: dict, path: Path, cam) -> None:
    """Overlay .mp4 of one episode's free frames.

    Green dot = sim truth now, red ring = tracker blob centre (missing = no fix),
    green cross = sim-truth p̂ (Δ ahead), blue cross = filter p̂.
    """
    import imageio

    ep = _read_episode(r["row"]["path"])
    fr = r["frames"]
    plane = float(ep["oracle"][0, 2])
    with imageio.get_writer(path, fps=12, macro_block_size=1) as w:
        for t, img in enumerate(ep["rgb"]):
            im = np.ascontiguousarray(img[..., :3]).copy()

            def px(xy, zz=plane):
                return tuple(int(round(c)) for c in cam.project(np.r_[xy, zz]))

            cv2.circle(im, px(ep["oracle"][t, :2]), 2, (0, 255, 0), -1)
            cv2.drawMarker(im, px(fr["phat_true"][t]), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            if fr["phat"][t] is not None:
                cv2.drawMarker(im, px(fr["phat"][t]), (40, 120, 255), cv2.MARKER_CROSS, 12, 2)
            if fr["uv"][t] is not None:
                cv2.circle(im, tuple(int(round(c)) for c in fr["uv"][t]), 7, (255, 0, 0), 1)
            err = np.linalg.norm(fr["dp"][t]) * 100
            txt = f"{r['row']['category']} {r['row']['speed']:.2f} m/s t={t}  p_err {err:.1f} cm"
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            w.append_data(im)


def _stats(err_m: np.ndarray) -> dict:
    """Error numbers in cm: median, RMSE, 95th percentile, share over 2 cm."""
    e = np.asarray(err_m, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {
        "n": int(e.size),
        "median_cm": round(float(np.median(e) * 100), 3),
        "rmse_cm": round(float(np.sqrt(np.mean(e**2)) * 100), 3),
        "p95_cm": round(float(np.percentile(e, 95) * 100), 3),
        "frac_gt_2cm": round(float(np.mean(e > 0.02)), 4),
    }


def _group(results: list[dict]) -> dict:
    """Pool all frames of a group of episodes and summarize them.

    Blocks in the output:
      coverage / track_rate / valid_rate: how often we had a state / a camera fix / a valid state.
      pack_p, pack_v: position and velocity pack error vs truth (the G2a number is pack_p).
      *_after_warmup, *_warmup: split off the first fixes, where the filter still thinks speed = 0.
      *_no_recent_hit, *_recent_hit: split by whether a wall bounce just happened.
      horizons: error of the guess N ms ahead vs the real future, filter vs truth
        straight-line, each split by "bounce in the window" or not.
    """
    dp = np.concatenate([np.asarray(r["frames"]["dp"]).reshape(-1, 2) for r in results])
    dv = np.concatenate([np.asarray(r["frames"]["dv"]).reshape(-1, 2) for r in results])
    near = np.concatenate([np.asarray(r["frames"]["near_bounce"], bool) for r in results])
    warm = np.concatenate([np.asarray(r["frames"]["warmup"], bool) for r in results])
    has = np.concatenate([np.asarray(r["frames"]["has"], bool) for r in results])
    meas = np.concatenate([np.asarray(r["frames"]["meas"], bool) for r in results])
    valid = np.concatenate([np.asarray(r["frames"]["valid"], bool) for r in results])
    ep = np.linalg.norm(dp, axis=1)
    eps = np.linalg.norm(np.concatenate(
        [np.asarray(r["frames"]["dps"], dtype=np.float64).reshape(-1, 2) for r in results]), axis=1)
    out = {
        "episodes": len(results),
        "start_failed": sum(not r["row"]["start_ok"] for r in results),
        # Why scoring stopped: hold / off_table / off_area / end (house only differs).
        "stops": {k: sum(r["row"].get("stop") == k for r in results)
                  for k in ("hold", "lifted", "off_table", "off_area", "end")},
        "frames": int(len(ep)),
        "coverage": round(float(has.mean()), 4) if len(has) else None,
        "track_rate": round(float(meas.mean()), 4) if len(meas) else None,
        "valid_rate": round(float(valid.mean()), 4) if len(valid) else None,
        "redetects": sum(r["row"].get("redetects", 0) for r in results),
        "recent_wall_hit_frames_frac": round(float(near.mean()), 4) if len(near) else None,
        "pack_p": _stats(ep),
        "pack_p_after_warmup": _stats(ep[~warm]),
        "pack_p_vs_calm_truth": _stats(eps),
        "pack_p_vs_calm_truth_after_warmup": _stats(eps[~warm]),
        "pack_p_warmup": _stats(ep[warm]),
        "pack_p_no_recent_hit": _stats(ep[~near & ~warm]),
        "pack_p_recent_hit": _stats(ep[near & ~warm]),
        "pack_v": _stats(np.linalg.norm(dv, axis=1)),
        "pack_v_after_warmup": _stats(np.linalg.norm(dv, axis=1)[~warm]),
        "horizons": {},
    }
    for h in HORIZONS:
        fe = np.concatenate([np.asarray(r["hor"][h]["fe"]) for r in results])
        oe = np.concatenate([np.asarray(r["hor"][h]["oe"]) for r in results])
        b = np.concatenate([np.asarray(r["hor"][h]["bounce"], bool) for r in results])
        out["horizons"][f"{int(h * 1000)}ms"] = {
            "filter_vs_true": _stats(fe),
            "oracle_vs_true": _stats(oe),
            "filter_vs_true_no_bounce": _stats(fe[~b]),
            "oracle_vs_true_no_bounce": _stats(oe[~b]),
            "filter_vs_true_bounce": _stats(fe[b]),
            "oracle_vs_true_bounce": _stats(oe[b]),
        }
    return out


def _noise_check(results: list[dict]) -> dict:
    """Filter error per component vs training Gaussian noise (moving objects).

    The policy was trained with fake noise on the 4 numbers. This asks: is the
    real filter error about that size, or bigger? "gaussian_frac_gt_1_2_3sigma"
    is what pure Gaussian noise would give, for comparison.
    """
    dp = np.concatenate([np.asarray(r["frames"]["dp"]).reshape(-1, 2) for r in results]).ravel()
    dv = np.concatenate([np.asarray(r["frames"]["dv"]).reshape(-1, 2) for r in results]).ravel()
    dp, dv = dp[np.isfinite(dp)], dv[np.isfinite(dv)]

    def one(x, sigma):
        return {
            "train_sigma": sigma,
            "filter_std": round(float(np.std(x)), 5),
            "filter_frac_gt_1sigma": round(float(np.mean(np.abs(x) > sigma)), 4),
            "filter_frac_gt_2sigma": round(float(np.mean(np.abs(x) > 2 * sigma)), 4),
            "filter_frac_gt_3sigma": round(float(np.mean(np.abs(x) > 3 * sigma)), 4),
            "gaussian_frac_gt_1_2_3sigma": [0.3173, 0.0455, 0.0027],
        }

    return {"p_hat_component_m": one(dp, POS_NOISE), "v_hat_component_m_s": one(dv, VEL_NOISE)}


def _plots(summary: dict, results: list[dict], out: Path) -> None:
    """Four PNGs: pack error vs bars, error vs look-ahead, error vs training noise, per fruit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2,
    })
    groups = summary["groups"]

    # 1 — pack error per group vs the G2a bars.
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{g['set']} {int(round(g['speed'] * 100))} cm/s" for g in groups]
    x = np.arange(len(groups))
    med = [g["pack_p"].get("median_cm", np.nan) for g in groups]
    rmse = [g["pack_p"].get("rmse_cm", np.nan) for g in groups]
    ax.plot(x, med, "o", ms=9, color=BLUE, label="median")
    ax.plot(x, rmse, "D", ms=8, color=ORANGE, label="RMSE")
    ax.axhline(G2A_MEDIAN_M * 100, color=BLUE, ls=":", lw=1.2)
    ax.axhline(G2A_RMSE_M * 100, color=ORANGE, ls=":", lw=1.2)
    ax.text(len(x) - 0.5, G2A_MEDIAN_M * 100, "median bar 1 cm", color=INK2, va="bottom", ha="right")
    ax.text(len(x) - 0.5, G2A_RMSE_M * 100, "RMSE bar 2 cm", color=INK2, va="bottom", ha="right")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("filter p̂ − sim-truth p̂ (cm)")
    ax.set_ylim(0, max(2.6, np.nanmax(rmse) * 1.15))
    ax.set_title("G2a: how far the camera's p̂ is from the sim-truth p̂ (Δ = 0.25 s)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "g2a_pack_error.png", dpi=150)
    plt.close(fig)

    # 2 — error vs horizon at 20 cm/s: bounce-free windows | windows with a bounce.
    moving20 = [r for r in results if abs(r["row"]["speed"] - 0.20) < 1e-6]
    g20 = _group(moving20)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    hs = [int(h * 1000) for h in HORIZONS]
    for ax, suffix, title in (
        (axes[0], "_no_bounce", "no wall hit in the window"),
        (axes[1], "_bounce", "wall hit in the window"),
    ):
        f = [g20["horizons"][f"{h}ms"][f"filter_vs_true{suffix}"].get("rmse_cm", np.nan) for h in hs]
        o = [g20["horizons"][f"{h}ms"][f"oracle_vs_true{suffix}"].get("rmse_cm", np.nan) for h in hs]
        ax.plot(hs, f, "-o", ms=8, color=BLUE, label="camera + filter")
        ax.plot(hs, o, "-D", ms=7, color=ORANGE, label="sim truth, straight line")
        ax.set_title(title, loc="left")
        ax.set_xlabel("look-ahead (ms)")
        ax.set_xticks(hs)
    axes[0].set_ylabel("RMSE vs true future (cm)")
    axes[0].legend(loc="upper left")
    fig.suptitle("20 cm/s, demos + evals: prediction error vs the real future", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "horizon_20cms.png", dpi=150)
    plt.close(fig)

    # 3 — filter error spread vs training noise (moving objects).
    moving = [r for r in results if r["row"]["speed"] > 0]
    dp = np.concatenate([np.asarray(r["frames"]["dp"]).reshape(-1, 2) for r in moving]).ravel()
    dv = np.concatenate([np.asarray(r["frames"]["dv"]).reshape(-1, 2) for r in moving]).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    rng = np.random.default_rng(0)
    for ax, x_err, sigma, unit, name in (
        (axes[0], dp, POS_NOISE, "cm", "p̂ (per axis)"),
        (axes[1], dv, VEL_NOISE, "cm/s", "v̂ (per axis)"),
    ):
        a = np.sort(np.abs(x_err[np.isfinite(x_err)])) * 100
        g = np.sort(np.abs(rng.normal(0, sigma, 200_000))) * 100
        ax.plot(a, np.linspace(0, 1, len(a)), color=BLUE, label="camera + filter error")
        ax.plot(g, np.linspace(0, 1, len(g)), color=GREY, ls="--",
                label=f"training noise, σ = {sigma * 100:.1f} {unit}")
        ax.set_xlim(0, 4 * sigma * 100)
        ax.set_xlabel(f"|error| ({unit})")
        ax.set_title(name, loc="left")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("share of frames at or below")
    fig.suptitle("Moving objects: filter error vs training noise", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "noise_vs_training.png", dpi=150)
    plt.close(fig)

    # 4 — per fruit (demos, moving).
    pf = summary.get("per_fruit_demo_moving") or summary["per_category_moving"]
    names = sorted(pf, key=lambda k: pf[k]["pack_p"].get("median_cm", 99))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    y = np.arange(len(names))
    axes[0].barh(y, [pf[k]["track_rate"] * 100 for k in names], color=BLUE, height=0.6)
    axes[0].set_xlabel("frames with a camera fix (%)")
    axes[0].set_xlim(0, 100)
    axes[1].barh(y, [pf[k]["pack_p"].get("median_cm", np.nan) for k in names], color=BLUE, height=0.6)
    axes[1].set_xlabel("median p̂ error (cm)")
    axes[0].set_yticks(y, names)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Per object (moving episodes)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "per_fruit.png", dpi=150)
    plt.close(fig)


def report(workers: int, out: Path, sets: str = "box", house_dirs=HOUSE_DIRS, n_videos: int = 8) -> None:
    """Main run: all episodes → summary.json, per_episode.csv, plots (+ house videos)."""
    OUT = Path(out)
    OUT.mkdir(parents=True, exist_ok=True)
    if sets == "house":
        base = _house_jobs(house_dirs)
        speeds = sorted({v for _, v, _ in base})
        group_keys = [("house", v) for v in speeds]
    else:
        base = _episodes(REPORT_SETS)
        group_keys = [(s, v) for s, v, _, _ in REPORT_SETS]
    jobs = [(s, v, f, None) for s, v, f in base]
    print(f"{len(jobs)} episodes, {workers} workers")
    with Pool(workers, initializer=_init_worker) as pool:
        results = list(pool.imap_unordered(run_episode, jobs, chunksize=2))
    results.sort(key=lambda r: (r["row"]["set"], r["row"]["speed"], r["row"]["path"]))

    groups = []
    for set_name, speed in group_keys:
        sub = [r for r in results if r["row"]["set"] == set_name and r["row"]["speed"] == speed]
        groups.append({"set": set_name, "speed": speed, **_group(sub)})
    # Per-fruit split (demos, moving only): shows which objects the tracker finds hard.
    main = "house" if sets == "house" else "demo"

    def per_category(moving_only: bool) -> dict:
        cats = {}
        for r in results:
            if r["row"]["set"] == main and r["row"]["category"]:
                if r["row"]["speed"] > 0 or not moving_only:
                    cats.setdefault(r["row"]["category"], []).append(r)
        out = {k: {"mode": sorted({str(r["row"].get("mode")) for r in v}), **_group(v)}
               for k, v in sorted(cats.items())}
        for v in out.values():
            v.pop("horizons", None)
        return out

    per_fruit = per_category(moving_only=True)

    # Offline G2a verdict: only the 20 cm/s groups count. Both median and RMSE
    # must be under the bars. The online policy swap is a separate G2 check.
    g2a = {}
    for g in groups:
        if abs(g["speed"] - 0.20) < 1e-6:
            s = g["pack_p"]
            g2a[f"{g['set']}_20cms"] = {
                "median_cm": s["median_cm"],
                "rmse_cm": s["rmse_cm"],
                "pass": s["median_cm"] <= G2A_MEDIAN_M * 100 and s["rmse_cm"] <= G2A_RMSE_M * 100,
            }
    g2a["pass"] = all(v["pass"] for v in g2a.values() if isinstance(v, dict))

    # Write the Kalman defaults into the summary so a result always shows its settings.
    kf = CVKalman()
    summary = {
        "what": "G2a offline: static-cam tracker + CV Kalman vs sim-truth pack, free-object frames",
        "delta_s": conditioning_delta_s(load_yaml(SIM_CFG)),
        "kalman": {k: getattr(kf, k) for k in ("meas_std", "accel_std", "bounce_accel_std",
                                                "init_vel_std", "bounce_nis", "max_coast_s",
                                                "outlier_m", "max_speed")},
        "g2a_bars": {"median_cm": G2A_MEDIAN_M * 100, "rmse_cm": G2A_RMSE_M * 100, "speed": "20 cm/s"},
        "g2a": g2a,
        "groups": groups,
        "per_fruit_demo_moving" if sets == "box" else "per_category_moving": per_fruit,
        **({"per_category_all_speeds": per_category(False)} if sets == "house" else {}),
        "noise_check_moving": _noise_check([r for r in results if r["row"]["speed"] > 0]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["set", "speed", "category", "mode", "start_ok", "n_free", "n_frames_total", "stop", "track_rate", "valid_rate",
            "pack_p_median_cm", "wall_hits", "filter_bounces", "redetects", "path"]
    with open(OUT / "per_episode.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r["row"])
    _plots(summary, results, OUT)
    if sets == "house" and n_videos:
        _init_worker()
        from perception.camera import PinholeCamera

        cam = PinholeCamera.from_cfg(_CFG)
        # Mix of good and hard: per category the worst and the best moving episode.
        ok = [r for r in results if r["row"]["start_ok"] and r["row"]["speed"] > 0]
        bad = [r for r in results if not r["row"]["start_ok"]]
        score = lambda r: r["row"]["pack_p_median_cm"] or 99.0  # noqa: E731
        ranked = sorted(ok, key=score)
        picks = bad[:2] + ranked[-(n_videos // 2):] + ranked[: n_videos // 2]
        seen = set()
        vdir = OUT / "videos"
        vdir.mkdir(exist_ok=True)
        for r in picks:
            if r["row"]["path"] in seen:
                continue
            seen.add(r["row"]["path"])
            name = f"{r['row']['category']}_{r['row']['speed']:.2f}_{Path(r['row']['path']).stem}.mp4"
            _video(r, vdir / name, cam)
        print(f"wrote {len(seen)} videos to {vdir}")
    print(json.dumps({"g2a": g2a}, indent=1))
    print(f"wrote {OUT}")


def tune(workers: int) -> None:
    """Sweep Kalman knobs on held-out speeds, re-using one tracker pass per episode."""
    jobs = [(s, v, f, None) for s, v, f in _episodes(TUNE_SETS, seed=1)]
    # Run the slow camera tracker once per episode and save the fixes.
    with Pool(workers, initializer=_init_worker) as pool:
        base = list(pool.imap_unordered(run_episode, jobs, chunksize=2))
    # Knob values to try. Each combo replays the cheap Kalman step only.
    grid = {
        "meas_std": (0.001, 0.002, 0.004, 0.006, 0.008),
        "accel_std": (0.3, 0.5, 1.0, 2.0, 4.0),
        "bounce_accel_std": (3.0, 10.0),
    }
    delta = conditioning_delta_s(load_yaml(SIM_CFG))
    rows = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(grid, combo))
        dp, dv = [], []
        for r in base:
            fr = r["frames"]
            if not r["row"]["start_ok"] or "ts" not in fr:
                continue
            kf = CVKalman(**kw)
            for t, z, opack in zip(fr["ts"], fr["z"], fr["oracle_pack"]):
                kf.step(t, z)
                if kf.started:
                    mine = conditioning_vector(kf.state(0.0), delta)
                    dp.append(np.linalg.norm(mine[:2] - opack[:2]))
                    dv.append(np.linalg.norm(mine[2:] - opack[2:]))
        dp, dv = np.asarray(dp), np.asarray(dv)
        # Score in units of training noise, so p̂ and v̂ count equally.
        score = float(np.sqrt(np.mean((dp / POS_NOISE) ** 2 + (dv / VEL_NOISE) ** 2)))
        sv = _stats(dv)
        rows.append({**kw, "score": round(score, 3), "p": _stats(dp),
                     "v_median_cm_s": sv["median_cm"], "v_rmse_cm_s": sv["rmse_cm"]})
    rows.sort(key=lambda d: d["score"])
    print(f"{len(base)} held-out episodes (speeds {[s for _, s, _, _ in TUNE_SETS]})")
    for d in rows[:10]:
        print(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tune", action="store_true", help="sweep Kalman knobs on held-out speeds")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sets", choices=("box", "house"), default="box",
                    help="box = G2a demo/eval folders; house = every .h5 under --house-dirs")
    ap.add_argument("--house-dirs", nargs="+", default=[str(d) for d in HOUSE_DIRS])
    ap.add_argument("--out", default=str(OUT), help="output folder")
    ap.add_argument("--videos", type=int, default=8, help="house: overlay .mp4s to write")
    args = ap.parse_args()
    if args.tune:
        tune(args.workers)
    else:
        report(args.workers, Path(args.out), args.sets, args.house_dirs, args.videos)
