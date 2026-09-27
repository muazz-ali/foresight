#!/usr/bin/env python3
"""G2a offline report: static-camera tracker + Kalman filter vs sim truth.

No Isaac. Reads recorded demo / eval ``.h5`` files (static RGB + sim-truth object
state), runs ``perception.StaticCamEstimator`` on every free-object frame (before
the first hold), and compares its 4-number pack with the oracle pack built from
the same sim-truth state by the same ``interfaces.state.conditioning_vector``.

Frame alignment: in both demo and eval files, ``static_cam_rgb[t]`` is rendered
after the sim step that ``oracle_state[t]`` describes (checked: blob centre is
~1 px from truth at t, ~3.5 px from t−1). Eval ``conditioning[t]`` is the pack B
saw *before* step t, i.e. built from ``oracle_state[t-1]`` — so we rebuild the
oracle pack from ``oracle_state[t]`` rather than reuse that column.

Run from the repo root in the sim env:
  python scripts/g2_perception_report.py            # report → data/eval/p2_perception_report/
  python scripts/g2_perception_report.py --tune     # Kalman knobs on held-out speeds (not reported)
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

# (set, speed m/s, folder, max episodes or None = all). Eval = G1 Model B oracle runs.
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
HORIZONS = (0.10, 0.20, 0.25, 0.30)
BOUNCE_LOOKBACK_S = 0.20  # a wall hit this recently still counts (filter re-learning speed)
WARMUP_FRAMES = 2  # first fixes: the filter starts at speed 0 (step 3 warms up before B's first call)
POS_NOISE, VEL_NOISE = 0.015, 0.03  # B's training noise (policy/configs/smolvla_b.yaml)
G2A_MEDIAN_M, G2A_RMSE_M = 0.01, 0.02

_CFG: dict | None = None
_DELTA = 0.25


def _init_worker() -> None:
    global _CFG, _DELTA
    cv2.setNumThreads(1)
    _CFG = load_yaml(SIM_CFG)
    _DELTA = conditioning_delta_s(_CFG)


def _oracle(o: np.ndarray) -> ObjectState:
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
    with h5py.File(path) as h:
        hold = h["holding"][:]
        g = int(np.argmax(hold > 0.5)) if hold.max() > 0.5 else len(hold)
        ep = {
            "ts": h["timestamp"][:g],
            "oracle": h["oracle_state"][:g],
            "rgb": h["static_cam_rgb"][:g],
        }
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ep["category"] = meta.get("category")
    return ep


def run_episode(job: tuple[str, float, str, dict | None]) -> dict:
    """Estimator over one episode's free frames → per-frame errors + episode row."""
    set_name, speed, path, kalman_kwargs = job
    ep = _read_episode(path)
    ts, orc, rgb = ep["ts"], ep["oracle"], ep["rgb"]
    n = len(ts)
    est = StaticCamEstimator.from_cfg(_CFG, kalman_kwargs=kalman_kwargs)
    row = {"set": set_name, "speed": speed, "path": path, "category": ep["category"], "n_free": n}
    frames = {k: [] for k in ("has", "meas", "valid", "dp", "dv", "near_bounce", "warmup", "z")}
    hor = {h: {"fe": [], "oe": [], "bounce": []} for h in HORIZONS}
    if n == 0:
        return {"row": {**row, "start_ok": False}, "frames": frames, "hor": hor}
    row["start_ok"] = est.start(rgb[0], orc[0, :3])
    row["mode"] = est.tracker.mode
    hit_t = ts[_wall_hits(orc[:, 3:5])]
    row["wall_hits"] = int(len(hit_t))
    n_fixes = 0
    for t in range(n):
        st = est.step(rgb[t], ts[t])
        n_fixes += est.last_uv is not None
        o = _oracle(orc[t])
        z = None if est.last_uv is None else est.camera.lift(est.last_uv, est.plane_z)
        frames["z"].append(z)
        frames["meas"].append(est.last_uv is not None)
        frames["has"].append(st is not None)
        frames["valid"].append(st is not None and st.valid)
        # Filter vs sim truth only disagree after a wall hit (both predict straight
        # through hits that have not happened yet), so look back, not ahead.
        frames["near_bounce"].append(
            bool(np.any((hit_t > ts[t] - BOUNCE_LOOKBACK_S) & (hit_t <= ts[t])))
        )
        frames["warmup"].append(n_fixes <= WARMUP_FRAMES and est.n_redetects == 0)
        if st is None:
            frames["dp"].append([np.nan, np.nan])
            frames["dv"].append([np.nan, np.nan])
        else:
            diff = conditioning_vector(st, _DELTA) - conditioning_vector(o, _DELTA)
            frames["dp"].append(diff[:2])
            frames["dv"].append(diff[2:])
        for h in HORIZONS:
            if st is None or ts[t] + h > ts[-1] + 1e-9:
                continue
            true = np.array([np.interp(ts[t] + h, ts, orc[:, i]) for i in (0, 1)])
            hor[h]["fe"].append(np.linalg.norm(st.predict(h).position[:2] - true))
            hor[h]["oe"].append(np.linalg.norm(o.predict(h).position[:2] - true))
            hor[h]["bounce"].append(
                bool(np.any((hit_t > ts[t] - BOUNCE_LOOKBACK_S) & (hit_t <= ts[t] + h)))
            )
    dp = np.linalg.norm(np.asarray(frames["dp"]), axis=1)
    row["track_rate"] = float(np.mean(frames["meas"]))
    row["pack_p_median_cm"] = float(np.nanmedian(dp) * 100) if np.isfinite(dp).any() else None
    row["filter_bounces"] = est.kf.n_bounces
    row["redetects"] = est.n_redetects
    row["valid_rate"] = float(np.mean(frames["valid"]))
    frames["ts"] = ts
    frames["oracle_pack"] = [conditioning_vector(_oracle(o), _DELTA) for o in orc]
    return {"row": row, "frames": frames, "hor": hor}


def _episodes(sets, seed: int = 0) -> list[tuple[str, float, str]]:
    jobs = []
    rng = random.Random(seed)
    for set_name, speed, folder, cap in sets:
        files = sorted(str(p) for p in Path(folder).glob("*.h5"))
        rng.shuffle(files)
        jobs += [(set_name, speed, f) for f in (files if cap is None else files[:cap])]
    return jobs


def _stats(err_m: np.ndarray) -> dict:
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
    dp = np.concatenate([np.asarray(r["frames"]["dp"]).reshape(-1, 2) for r in results])
    dv = np.concatenate([np.asarray(r["frames"]["dv"]).reshape(-1, 2) for r in results])
    near = np.concatenate([np.asarray(r["frames"]["near_bounce"], bool) for r in results])
    warm = np.concatenate([np.asarray(r["frames"]["warmup"], bool) for r in results])
    has = np.concatenate([np.asarray(r["frames"]["has"], bool) for r in results])
    meas = np.concatenate([np.asarray(r["frames"]["meas"], bool) for r in results])
    valid = np.concatenate([np.asarray(r["frames"]["valid"], bool) for r in results])
    ep = np.linalg.norm(dp, axis=1)
    out = {
        "episodes": len(results),
        "start_failed": sum(not r["row"]["start_ok"] for r in results),
        "frames": int(len(ep)),
        "coverage": round(float(has.mean()), 4) if len(has) else None,
        "track_rate": round(float(meas.mean()), 4) if len(meas) else None,
        "valid_rate": round(float(valid.mean()), 4) if len(valid) else None,
        "redetects": sum(r["row"].get("redetects", 0) for r in results),
        "recent_wall_hit_frames_frac": round(float(near.mean()), 4) if len(near) else None,
        "pack_p": _stats(ep),
        "pack_p_after_warmup": _stats(ep[~warm]),
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
    """Filter error per component vs B's Gaussian training noise (moving objects)."""
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

    # 3 — filter error spread vs B's training noise (moving objects).
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
                label=f"B's training noise, σ = {sigma * 100:.1f} {unit}")
        ax.set_xlim(0, 4 * sigma * 100)
        ax.set_xlabel(f"|error| ({unit})")
        ax.set_title(name, loc="left")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("share of frames at or below")
    fig.suptitle("Moving objects: filter error vs the noise B was trained with", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "noise_vs_training.png", dpi=150)
    plt.close(fig)

    # 4 — per fruit (demos, moving).
    pf = summary["per_fruit_demo_moving"]
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
    fig.suptitle("Per fruit (demos, 10–20 cm/s)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "per_fruit.png", dpi=150)
    plt.close(fig)


def report(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, v, f, None) for s, v, f in _episodes(REPORT_SETS)]
    print(f"{len(jobs)} episodes, {workers} workers")
    with Pool(workers, initializer=_init_worker) as pool:
        results = list(pool.imap_unordered(run_episode, jobs, chunksize=2))
    results.sort(key=lambda r: (r["row"]["set"], r["row"]["speed"], r["row"]["path"]))

    groups = []
    for set_name, speed, _, _ in REPORT_SETS:
        sub = [r for r in results if r["row"]["set"] == set_name and r["row"]["speed"] == speed]
        groups.append({"set": set_name, "speed": speed, **_group(sub)})
    fruits = {}
    for r in results:
        if r["row"]["set"] == "demo" and r["row"]["speed"] > 0 and r["row"]["category"]:
            fruits.setdefault(r["row"]["category"], []).append(r)
    per_fruit = {
        k: {"mode": sorted({str(r["row"].get("mode")) for r in v}), **_group(v)}
        for k, v in sorted(fruits.items())
    }
    for v in per_fruit.values():
        v.pop("horizons", None)

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

    kf = CVKalman()
    summary = {
        "what": "G2a offline: static-cam tracker + CV Kalman vs sim-truth pack, free-object frames",
        "delta_s": conditioning_delta_s(load_yaml(SIM_CFG)),
        "kalman": {k: getattr(kf, k) for k in ("meas_std", "accel_std", "bounce_accel_std",
                                                "init_vel_std", "bounce_nis", "max_coast_s")},
        "g2a_bars": {"median_cm": G2A_MEDIAN_M * 100, "rmse_cm": G2A_RMSE_M * 100, "speed": "20 cm/s"},
        "g2a": g2a,
        "groups": groups,
        "per_fruit_demo_moving": per_fruit,
        "noise_check_moving": _noise_check([r for r in results if r["row"]["speed"] > 0]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["set", "speed", "category", "mode", "start_ok", "n_free", "track_rate", "valid_rate",
            "pack_p_median_cm", "wall_hits", "filter_bounces", "redetects", "path"]
    with open(OUT / "per_episode.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r["row"])
    _plots(summary, results, OUT)
    print(json.dumps({"g2a": g2a}, indent=1))
    print(f"wrote {OUT}")


def tune(workers: int) -> None:
    """Sweep Kalman knobs on held-out speeds, re-using one tracker pass per episode."""
    jobs = [(s, v, f, None) for s, v, f in _episodes(TUNE_SETS, seed=1)]
    with Pool(workers, initializer=_init_worker) as pool:
        base = list(pool.imap_unordered(run_episode, jobs, chunksize=2))
    grid = {
        "meas_std": (0.001, 0.002, 0.004),
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
        # Score in units of B's training noise, so p̂ and v̂ count equally.
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
    args = ap.parse_args()
    tune(args.workers) if args.tune else report(args.workers)
