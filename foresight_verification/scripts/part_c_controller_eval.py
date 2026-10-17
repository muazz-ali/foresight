#!/usr/bin/env python3
"""Part C — controller ID (α) from expert demos + oracle eval log offsets.

Offline only (dynamicVLA_training). No Isaac / no training.
Writes:
  foresight_verification/partials/part_c.json
  foresight_verification/partials/part_c.md
  foresight_verification/figures/c3_hand_minus_future_*.png
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VER = ROOT / "foresight_verification"
PARTIALS = VER / "partials"
FIGURES = VER / "figures"
DEMO_GLOB = "data/p3_throw/*/speed_*/*.h5"
SEED = 0
DT = 0.04  # 25 Hz from demo timestamps
AXES = ("x", "y", "z")
STAGE_NAMES = {
    0: "reset",
    1: "approach",
    2: "descend",
    3: "close",
    4: "lift",
    5: "carry",
    6: "lower",
    7: "open",
    8: "home",
    9: "done",
}
# Nominal bins for per-speed tables (folder names).
SPEED_BINS = (0.0, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20)
ORACLE_DIRS = [
    ROOT / "data/eval/p3_throw_oracle_n20",
    ROOT / "data/eval/claims/base/oracle",
]
EVAL_SCAN_DIRS = ORACLE_DIRS + [
    ROOT / "data/eval/p3_throw_n100",
    ROOT / "data/eval/p3_throw_n100_diverse",
]


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _decode(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def speed_from_path(path: Path) -> float:
    m = re.search(r"speed_([0-9.]+)", str(path))
    if m:
        return float(m.group(1))
    return float("nan")


def nearest_bin(speed: float) -> float:
    if not math.isfinite(speed):
        return float("nan")
    return float(min(SPEED_BINS, key=lambda b: abs(b - speed)))


def fit_alpha_ols(err: np.ndarray, dh: np.ndarray) -> dict:
    """OLS through origin: dh = α · err. Returns α, R², n."""
    n = int(err.size)
    if n < 8:
        return {"alpha": float("nan"), "r2": float("nan"), "n": n}
    xx = float(np.dot(err, err))
    if xx < 1e-18:
        return {"alpha": float("nan"), "r2": float("nan"), "n": n}
    alpha = float(np.dot(err, dh) / xx)
    yhat = alpha * err
    ss_res = float(np.sum((dh - yhat) ** 2))
    ss_tot = float(np.sum((dh - dh.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
    return {"alpha": alpha, "r2": r2, "n": n, "rmse_m": float(np.sqrt(ss_res / n))}


def fit_second_order(err: np.ndarray, dh: np.ndarray) -> dict:
    """dh[t] = a·dh[t-1] + b·err[t] + c·err[t-1]  (second-order / ARX-ish)."""
    if err.size < 20:
        return {"a": float("nan"), "b": float("nan"), "c": float("nan"), "r2": float("nan"), "n": int(err.size)}
    # Align t>=1
    y = dh[1:]
    X = np.column_stack([dh[:-1], err[1:], err[:-1]])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"a": float("nan"), "b": float("nan"), "c": float("nan"), "r2": float("nan"), "n": int(y.size)}
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
    return {
        "a_dh_prev": float(coef[0]),
        "b_err_t": float(coef[1]),
        "c_err_tm1": float(coef[2]),
        "r2": r2,
        "n": int(y.size),
        "rmse_m": float(np.sqrt(ss_res / max(y.size, 1))),
    }


def load_demo_series(path: Path) -> dict | None:
    with h5py.File(path, "r") as f:
        if "ee_pos" not in f or "action_pos" not in f:
            return None
        h = _f64(f["ee_pos"])
        c = _f64(f["action_pos"])
        stage = _f64(f["stage"]).astype(np.int64) if "stage" in f else np.zeros(len(h), np.int64)
        success = int(f["success"][0]) if "success" in f else 1
        speed = float(f["speed_m_s"][0]) if "speed_m_s" in f else speed_from_path(path)
        ts = _f64(f["timestamp"]) if "timestamp" in f else np.arange(len(h)) * DT
        grip = _f64(f["gripper"]) if "gripper" in f else np.ones(len(h))
    if h.ndim != 2 or h.shape[1] < 3 or len(h) < 3:
        return None
    return {
        "h": h,
        "c": c,
        "stage": stage,
        "success": success,
        "speed": speed,
        "speed_bin": nearest_bin(speed_from_path(path) if math.isfinite(speed_from_path(path)) else speed),
        "timestamp": ts,
        "gripper": grip,
        "path": str(path),
    }


def accumulate_fit(episodes: list[dict], stage_set: set[int] | None = None) -> dict:
    """Collect err/dh per axis and fit first- (+ second-) order models."""
    err_all = {a: [] for a in AXES}
    dh_all = {a: [] for a in AXES}
    for ep in episodes:
        h, c, st = ep["h"], ep["c"], ep["stage"]
        dh = h[1:] - h[:-1]
        err = c[:-1] - h[:-1]
        mask = np.ones(len(dh), dtype=bool)
        if stage_set is not None:
            mask &= np.isin(st[:-1], list(stage_set))
        # Drop near-zero error (already-at-target) — keeps OLS stable.
        for i, a in enumerate(AXES):
            m = mask & (np.abs(err[:, i]) > 1e-6)
            err_all[a].append(err[m, i])
            dh_all[a].append(dh[m, i])
    out = {}
    for a in AXES:
        if not err_all[a]:
            out[a] = {"first_order": {"alpha": float("nan"), "r2": float("nan"), "n": 0}}
            continue
        e = np.concatenate(err_all[a])
        d = np.concatenate(dh_all[a])
        fo = fit_alpha_ols(e, d)
        entry = {"first_order": fo}
        if not (isinstance(fo["r2"], float) and fo["r2"] >= 0.9):
            entry["second_order"] = fit_second_order(e, d)
            entry["second_order_reason"] = f"R2={fo['r2']:.4f}<0.9"
        out[a] = entry
    return out


def simulate_hand(h0: np.ndarray, c: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """h[t+1] = h[t] + α ⊙ (c[t] − h[t])."""
    T = len(c)
    hs = np.zeros_like(c)
    hs[0] = h0
    a = np.asarray(alpha, dtype=np.float64)
    for t in range(T - 1):
        hs[t + 1] = hs[t] + a * (c[t] - hs[t])
    return hs


def best_lag_frames(h_log: np.ndarray, h_sim: np.ndarray, max_lag: int = 25) -> dict:
    """Shift series; report d minimizing RMSE on xy (and per-axis).

    Positive d means logged h lags the simulation (hand behind the α-model).
    Also search negative d (sim lags log) in [-max_lag, +max_lag].
    """
    T = min(len(h_log), len(h_sim))
    h_log = h_log[:T]
    h_sim = h_sim[:T]
    best = {"d_frames": 0, "rmse_m": float("inf")}
    for d in range(-max_lag, max_lag + 1):
        if d >= 0:
            if T - d < 10:
                continue
            err = h_log[d:] - h_sim[: T - d]
        else:
            dd = -d
            if T - dd < 10:
                continue
            err = h_log[: T - dd] - h_sim[dd:]
        rmse = float(np.sqrt(np.mean(err[:, :2] ** 2)))
        if rmse < best["rmse_m"]:
            best = {"d_frames": int(d), "rmse_m": rmse, "rmse_cm": rmse * 100.0}
    # RMSE with no shift (diagnostic)
    best["rmse_cm_at_0"] = float(np.sqrt(np.mean((h_log[:, :2] - h_sim[:, :2]) ** 2)) * 100.0)
    per_axis = {}
    for i, a in enumerate(AXES):
        best_a = {"d_frames": 0, "rmse_m": float("inf")}
        for d in range(-max_lag, max_lag + 1):
            if d >= 0:
                if T - d < 10:
                    continue
                err = h_log[d:, i] - h_sim[: T - d, i]
            else:
                dd = -d
                if T - dd < 10:
                    continue
                err = h_log[: T - dd, i] - h_sim[dd:, i]
            rmse = float(np.sqrt(np.mean(err**2)))
            if rmse < best_a["rmse_m"]:
                best_a = {"d_frames": int(d), "rmse_m": rmse, "rmse_cm": rmse * 100.0}
        per_axis[a] = best_a
    best["per_axis"] = per_axis
    return best


def measured_tracking_lag_cm(ep: dict, alpha: np.ndarray) -> dict:
    """Mean (c − h) during approach/close before first grip close, in cm."""
    h, c, st, g = ep["h"], ep["c"], ep["stage"], ep["gripper"]
    close_idx = np.where(g < 0.5)[0]
    t_end = int(close_idx[0]) if len(close_idx) else len(h)
    mask = np.isin(st, [1, 2, 3]) & (np.arange(len(h)) < max(t_end, 1))
    if mask.sum() < 3:
        mask = np.arange(len(h)) < max(t_end, 1)
    lag = (c[mask] - h[mask]).mean(axis=0) * 100.0  # cm
    # Predicted lag distance for constant-velocity tracking: d ≈ v·dt·(1-α)/α
    # Use command finite-diff velocity on the same mask.
    if mask.sum() >= 2:
        idx = np.where(mask)[0]
        v = np.diff(c[idx], axis=0) / DT
        v_mean = v.mean(axis=0)
    else:
        v_mean = np.zeros(3)
    d_pred = np.zeros(3)
    for i in range(3):
        a = float(alpha[i])
        if 0 < a < 1:
            d_pred[i] = float(v_mean[i] * DT * (1.0 - a) / a) * 100.0  # cm
        else:
            d_pred[i] = float("nan")
    return {
        "d_measured_cm": {a: float(lag[i]) for i, a in enumerate(AXES)},
        "d_predicted_from_alpha_cm": {a: float(d_pred[i]) for i, a in enumerate(AXES)},
        "v_cmd_mean_m_s": {a: float(v_mean[i]) for i, a in enumerate(AXES)},
        "n_frames": int(mask.sum()),
    }


def list_demo_paths(rng: np.random.Generator, max_n: int | None = None) -> list[Path]:
    paths = sorted(ROOT.glob(DEMO_GLOB))
    # Prefer success (all current demos are success=1); shuffle with seed for subsample.
    if max_n is not None and len(paths) > max_n:
        idx = rng.permutation(len(paths))[:max_n]
        paths = [paths[i] for i in sorted(idx)]
    return paths


def run_c1_c2(rng: np.random.Generator) -> tuple[dict, list[dict], np.ndarray]:
    paths = list_demo_paths(rng, max_n=None)
    episodes: list[dict] = []
    skipped = 0
    for p in paths:
        ep = load_demo_series(p)
        if ep is None:
            skipped += 1
            continue
        if ep["success"] != 1:
            continue
        episodes.append(ep)
    print(f"C1 loaded {len(episodes)} success demos (skipped_bad={skipped})")

    global_fit = accumulate_fit(episodes)
    alpha = np.array(
        [global_fit[a]["first_order"]["alpha"] for a in AXES], dtype=np.float64
    )

    # Per phase
    per_phase = {}
    for sid, name in STAGE_NAMES.items():
        per_phase[f"{sid}_{name}"] = accumulate_fit(episodes, {sid})

    # Per speed bin
    by_speed: dict[float, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_speed[ep["speed_bin"]].append(ep)
    per_speed = {f"{sp:.2f}": accumulate_fit(eps) for sp, eps in sorted(by_speed.items())}

    # C2: simulate on a seeded subset (200 eps) for lag stats
    sim_idx = rng.permutation(len(episodes))[: min(200, len(episodes))]
    lag_xy = []
    lag_meas = []
    for i in sim_idx:
        ep = episodes[i]
        hs = simulate_hand(ep["h"][0], ep["c"], alpha)
        lag_xy.append(best_lag_frames(ep["h"], hs))
        lag_meas.append(measured_tracking_lag_cm(ep, alpha))

    d_frames = np.array([x["d_frames"] for x in lag_xy], float)
    meas = {a: np.array([m["d_measured_cm"][a] for m in lag_meas], float) for a in AXES}
    pred = {a: np.array([m["d_predicted_from_alpha_cm"][a] for m in lag_meas], float) for a in AXES}

    c1 = {
        "n_episodes": len(episodes),
        "dt_s": DT,
        "seed": SEED,
        "model": "h[t+1]-h[t] = alpha * (c[t]-h[t]); c=action_pos, h=ee_pos",
        "conveyor_axis": "y",
        "global": global_fit,
        "per_phase": per_phase,
        "per_speed": per_speed,
        "files_root": str(ROOT / "data/p3_throw"),
    }
    c2 = {
        "n_sim_episodes": int(len(sim_idx)),
        "alpha_used": {a: float(alpha[i]) for i, a in enumerate(AXES)},
        "lag_d_frames_xy": {
            "mean": float(d_frames.mean()),
            "std": float(d_frames.std()),
            "median": float(np.median(d_frames)),
        },
        "lag_d_s_xy": {
            "mean": float(d_frames.mean() * DT),
            "std": float(d_frames.std() * DT),
        },
        "d_measured_cm": {
            a: {"mean": float(np.nanmean(meas[a])), "std": float(np.nanstd(meas[a]))} for a in AXES
        },
        "d_predicted_from_alpha_cm": {
            a: {"mean": float(np.nanmean(pred[a])), "std": float(np.nanstd(pred[a]))} for a in AXES
        },
        "notes": (
            "Measured d = mean(c−h) cm during approach/close before first gripper<0.5. "
            "Predicted d = v_cmd·dt·(1−α)/α (ramp-tracking lag of discrete first-order). "
            "d_frames = best shift aligning simulated h to logged h (xy RMSE)."
        ),
    }
    return {"C1": c1, "C2": c2}, episodes, alpha


def _frame_at_time(ts: np.ndarray, t: float | None) -> int | None:
    if t is None or not math.isfinite(float(t)):
        return None
    return int(np.argmin(np.abs(ts - float(t))))


def analyze_oracle_episode(h5_path: Path, meta: dict) -> dict | None:
    with h5py.File(h5_path, "r") as f:
        cs = _decode(f["cond_source"][()]) if "cond_source" in f else meta.get("cond_source", "")
        if "zero_conditioning" in f:
            zraw = f["zero_conditioning"][()]
            zero = bool(int(np.asarray(zraw).reshape(-1)[0]))
        else:
            zero = bool(meta.get("zero_conditioning"))
        if cs != "oracle" or zero:
            return None
        h = _f64(f["ee_pos"])
        obj = _f64(f["object_pos"])
        # Prefer truth_pack; oracle logs often omit it — conditioning IS the true future p.
        truth_src = "missing"
        if "truth_pack" in f:
            truth = _f64(f["truth_pack"])[:, :2]
            truth_src = "truth_pack"
        elif "conditioning" in f and cs == "oracle":
            truth = _f64(f["conditioning"])[:, :2]
            truth_src = "conditioning_oracle"
        else:
            truth = None
        ts = _f64(f["timestamp"])
        grip = _f64(f["gripper"])
        speed = float(np.asarray(f["speed_m_s"]).reshape(-1)[0]) if "speed_m_s" in f else float(meta.get("speed_m_s", float("nan")))
        seed = int(np.asarray(f["seed"]).reshape(-1)[0]) if "seed" in f else int(meta.get("seed", -1))
        success = bool(int(np.asarray(f["success"]).reshape(-1)[0])) if "success" in f else bool(meta.get("success"))

    timing = meta.get("timing") or {}
    close_t = timing.get("close_t")
    closest_t = timing.get("closest_t")
    # Fallbacks from signals
    close_i = _frame_at_time(ts, close_t)
    if close_i is None:
        hit = np.where(grip < 0.5)[0]
        close_i = int(hit[0]) if len(hit) else None
    xy_dist = np.linalg.norm(h[:, :2] - obj[:, :2], axis=1)
    min_i = _frame_at_time(ts, closest_t)
    if min_i is None:
        min_i = int(np.argmin(xy_dist))

    def offset_at(i: int | None) -> dict | None:
        if i is None or i < 0 or i >= len(h):
            return None
        dxy = (h[i, :2] - obj[i, :2]) * 100.0
        return {"x_cm": float(dxy[0]), "y_cm": float(dxy[1]), "norm_cm": float(np.linalg.norm(dxy)), "frame": int(i)}

    # Last-50 frames *ending at close* (not episode end — post-place dwarfs the signal).
    last50 = None
    if truth is not None and len(truth) >= 1:
        end = close_i if close_i is not None else (len(h) - 1)
        end = int(np.clip(end, 0, min(len(h), len(truth)) - 1))
        start = max(0, end - 49)
        diff = (h[start : end + 1, :2] - truth[start : end + 1, :2]) * 100.0  # cm
        last50 = {
            "n": int(diff.shape[0]),
            "source": truth_src,
            "window": "50_frames_ending_at_close",
            "close_frame": int(end),
            "mean_cm": {"x": float(diff[:, 0].mean()), "y": float(diff[:, 1].mean())},
            "std_cm": {"x": float(diff[:, 0].std()), "y": float(diff[:, 1].std())},
            "end_cm": {"x": float(diff[-1, 0]), "y": float(diff[-1, 1])},
            "series_cm": diff.tolist(),
        }

    return {
        "h5": str(h5_path),
        "speed_m_s": speed,
        "seed": seed,
        "success": success,
        "cond_source": cs,
        "hand_minus_object_at_close": offset_at(close_i),
        "hand_minus_object_at_min_dist": offset_at(min_i),
        "last50_hand_minus_truth": last50,
        "truth_source": truth_src,
        "timing_label": timing.get("timing_label"),
        "failure": meta.get("failure"),
    }


def load_metrics_rows(directory: Path) -> list[dict]:
    rows = []
    for name in ("metrics.jsonl", "metrics_all.jsonl"):
        p = directory / name
        if p.is_file():
            with p.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            break
    # Also walk worker subdirs
    if not rows:
        for p in sorted(directory.glob("**/metrics.jsonl")):
            with p.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def resolve_h5(meta: dict, directory: Path) -> Path | None:
    h5 = meta.get("h5")
    if h5:
        candidates = [Path(h5), ROOT / h5, directory / Path(h5).name]
        for p in candidates:
            if p.is_file():
                return p.resolve()
    # guess from seed/speed
    seed = meta.get("seed")
    speed = meta.get("speed_m_s")
    if seed is None:
        return None
    pats = [
        f"*s{seed}_cond.h5",
        f"*_{seed}_cond.h5",
    ]
    for pat in pats:
        hits = list(directory.glob(pat)) + list(directory.glob(f"**/{pat}"))
        if hits:
            return hits[0].resolve()
    if speed is not None:
        tag = f"v{float(speed):.2f}_s{seed}_cond.h5"
        hits = list(directory.glob(f"*{tag}")) + list(directory.glob(f"**/*{tag}"))
        if hits:
            return hits[0].resolve()
    return None


def run_c3() -> dict:
    inventory = {}
    oracle_eps: list[dict] = []
    files_used: list[str] = []

    for d in EVAL_SCAN_DIRS:
        if not d.is_dir():
            inventory[str(d)] = {"exists": False}
            continue
        rows = load_metrics_rows(d)
        n_oracle = sum(1 for r in rows if r.get("cond_source") == "oracle" and not r.get("zero_conditioning"))
        n_filter = sum(1 for r in rows if r.get("cond_source") == "filter")
        n_h5 = len(list(d.glob("**/*_cond.h5")))
        inventory[str(d.relative_to(ROOT))] = {
            "exists": True,
            "n_metrics_rows": len(rows),
            "n_oracle": n_oracle,
            "n_filter": n_filter,
            "n_cond_h5": n_h5,
            "metrics_file": (
                "metrics.jsonl"
                if (d / "metrics.jsonl").is_file()
                else ("metrics_all.jsonl" if (d / "metrics_all.jsonl").is_file() else "UNKNOWN")
            ),
        }

    # Analyze oracle dirs only for C3 stats
    seen_h5: set[str] = set()
    for d in ORACLE_DIRS:
        if not d.is_dir():
            continue
        rows = load_metrics_rows(d)
        for meta in rows:
            if meta.get("cond_source") != "oracle" or meta.get("zero_conditioning"):
                continue
            h5 = resolve_h5(meta, d)
            if h5 is None or not h5.is_file():
                continue
            key = str(h5.resolve())
            if key in seen_h5:
                continue
            seen_h5.add(key)
            ep = analyze_oracle_episode(h5, meta)
            if ep is None:
                continue
            oracle_eps.append(ep)
            try:
                files_used.append(str(h5.resolve().relative_to(ROOT)))
            except ValueError:
                files_used.append(str(h5))

    print(f"C3 oracle episodes analyzed: {len(oracle_eps)}")

    # Per-speed mean/std of hand−object at close / min-dist
    def agg_offsets(key: str) -> dict:
        by: dict[float, list[np.ndarray]] = defaultdict(list)
        for ep in oracle_eps:
            off = ep.get(key)
            if not off:
                continue
            by[nearest_bin(ep["speed_m_s"])].append(np.array([off["x_cm"], off["y_cm"], off["norm_cm"]]))
        out = {}
        for sp, arrs in sorted(by.items()):
            A = np.stack(arrs, axis=0)
            out[f"{sp:.2f}"] = {
                "n": int(A.shape[0]),
                "mean_cm": {"x": float(A[:, 0].mean()), "y": float(A[:, 1].mean()), "norm": float(A[:, 2].mean())},
                "std_cm": {"x": float(A[:, 0].std()), "y": float(A[:, 1].std()), "norm": float(A[:, 2].std())},
            }
        return out

    # Last-50 settle check
    settle = []
    for ep in oracle_eps:
        L = ep.get("last50_hand_minus_truth")
        if not L:
            continue
        series = np.asarray(L["series_cm"], float)
        # Compare early vs late half mean — if |Δmean| small → settled
        mid = len(series) // 2
        early = series[:mid].mean(axis=0)
        late = series[mid:].mean(axis=0)
        settle.append(
            {
                "speed": ep["speed_m_s"],
                "seed": ep["seed"],
                "early_mean_cm": early.tolist(),
                "late_mean_cm": late.tolist(),
                "delta_mean_cm": (late - early).tolist(),
                "late_std_cm": series[mid:].std(axis=0).tolist(),
                "end_cm": L["end_cm"],
            }
        )

    if settle:
        late = np.array([s["late_mean_cm"] for s in settle], float)
        end = np.array([[s["end_cm"]["x"], s["end_cm"]["y"]] for s in settle], float)
        delta = np.array([s["delta_mean_cm"] for s in settle], float)
        settle_summary = {
            "n": len(settle),
            "late_mean_offset_cm": {"x": float(late[:, 0].mean()), "y": float(late[:, 1].mean())},
            "late_std_across_eps_cm": {"x": float(late[:, 0].std()), "y": float(late[:, 1].std())},
            "end_mean_cm": {"x": float(end[:, 0].mean()), "y": float(end[:, 1].mean())},
            "early_to_late_delta_mean_cm": {"x": float(delta[:, 0].mean()), "y": float(delta[:, 1].mean())},
            "settles_to_constant_offset": bool(
                float(np.mean(np.abs(delta[:, 0]))) < 1.0 and float(np.mean(np.abs(delta[:, 1]))) < 1.0
            ),
            "notes": (
                "Offset = hand_xy − true_future_xy (truth_pack if present else oracle conditioning) "
                "in cm, ≤50 frames ending at close. "
                "settles_to_constant_offset=True if mean |early−late| < 1 cm on both axes."
            ),
        }
    else:
        settle_summary = {"n": 0, "unknown": "no truth_pack series in oracle episodes"}

    # Plot 10 episodes (prefer one per speed, mix success)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    plot_eps = []
    by_sp: dict[float, list[dict]] = defaultdict(list)
    for ep in oracle_eps:
        if ep.get("last50_hand_minus_truth"):
            by_sp[nearest_bin(ep["speed_m_s"])].append(ep)
    for sp in sorted(by_sp):
        pool = by_sp[sp]
        rng.shuffle(pool)
        # prefer success
        pool = sorted(pool, key=lambda e: (not e["success"], e["seed"]))
        plot_eps.extend(pool[: max(1, 10 // max(len(by_sp), 1))])
    if len(plot_eps) < 10:
        rest = [e for e in oracle_eps if e not in plot_eps and e.get("last50_hand_minus_truth")]
        rng.shuffle(rest)
        plot_eps.extend(rest[: 10 - len(plot_eps)])
    plot_eps = plot_eps[:10]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ep in plot_eps:
        series = np.asarray(ep["last50_hand_minus_truth"]["series_cm"], float)
        t = np.arange(-len(series) + 1, 1)
        label = f"v={ep['speed_m_s']:.2f} s={ep['seed']} {'ok' if ep['success'] else 'fail'}"
        axes[0].plot(t, series[:, 0], label=label, alpha=0.85)
        axes[1].plot(t, series[:, 1], alpha=0.85)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("hand_x − p_true_x (cm)")
    axes[1].set_ylabel("hand_y − p_true_y (cm)")
    axes[1].set_xlabel("frame (0 = close)")
    axes[0].set_title("Oracle eval: hand − true future object (50 frames → close)")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = FIGURES / "c3_hand_minus_future_last50.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    # Mean curve across all oracle eps
    if settle:
        # pad/truncate to 50
        mats = []
        for ep in oracle_eps:
            L = ep.get("last50_hand_minus_truth")
            if not L:
                continue
            s = np.asarray(L["series_cm"], float)
            if len(s) < 50:
                pad = np.full((50 - len(s), 2), np.nan)
                s = np.vstack([pad, s])
            else:
                s = s[-50:]
            mats.append(s)
        M = np.stack(mats, axis=0)
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        t = np.arange(-49, 1)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for i, a in enumerate(("x", "y")):
            axes[i].plot(t, mean[:, i], color="#1f4e79", lw=2, label="mean")
            axes[i].fill_between(t, mean[:, i] - std[:, i], mean[:, i] + std[:, i], color="#1f4e79", alpha=0.2, label="±1 std")
            axes[i].axhline(0, color="k", lw=0.6)
            axes[i].set_ylabel(f"hand_{a} − p_true_{a} (cm)")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)
        axes[0].set_title(f"Oracle mean offset (50 frames → close, n={M.shape[0]})")
        axes[1].set_xlabel("frame (0 = close)")
        fig.tight_layout()
        fig_path2 = FIGURES / "c3_hand_minus_future_mean.png"
        fig.savefig(fig_path2, dpi=140)
        plt.close(fig)
    else:
        fig_path2 = None

    return {
        "C3": {
            "inventory": inventory,
            "n_oracle_episodes": len(oracle_eps),
            "files_used_n": len(files_used),
            "files_used_sample": files_used[:12],
            "hand_minus_object_xy_at_close_per_speed_cm": agg_offsets("hand_minus_object_at_close"),
            "hand_minus_object_xy_at_min_dist_per_speed_cm": agg_offsets("hand_minus_object_at_min_dist"),
            "last50_settle": settle_summary,
            "figures": [str(fig_path.relative_to(ROOT))]
            + ([str(fig_path2.relative_to(ROOT))] if fig_path2 else []),
            "plot_episodes": [
                {"seed": e["seed"], "speed": e["speed_m_s"], "success": e["success"], "h5": e["h5"]}
                for e in plot_eps
            ],
            "checkpoint": "runs/p3_throw/pretrained_model",
            "notes": (
                "Oracle-only from p3_throw_oracle_n20 and claims/base/oracle. "
                "p3_throw_n100 and p3_throw_n100_diverse are filter cond_source — scanned but not pooled into C3 offsets."
            ),
        }
    }


def _extract_w_h(part_b: dict) -> tuple[dict | None, dict]:
    """Pull w_h from part_b.json (B8 block or summary_pooled_k1_8)."""
    meta: dict = {"source": None}
    b8 = part_b.get("B8") or part_b.get("b8")
    if isinstance(b8, dict):
        wh = b8.get("w_h") or b8.get("wh")
        if isinstance(wh, dict) and ("x" in wh or "y" in wh):
            meta["source"] = "B8.w_h"
            return {"x": wh.get("x"), "y": wh.get("y")}, {**meta, "B8_excerpt": b8}
        # per-axis nested
        if "x" in b8 and isinstance(b8["x"], dict) and "w_h" in b8["x"]:
            meta["source"] = "B8.{axis}.w_h"
            return {"x": b8["x"].get("w_h"), "y": b8.get("y", {}).get("w_h")}, {**meta, "B8_excerpt": b8}

    pooled = part_b.get("summary_pooled_k1_8") or part_b.get("B8_summary_pooled_k1_8")
    if isinstance(pooled, dict) and "x" in pooled and "y" in pooled:
        wx = pooled["x"].get("w_h") if isinstance(pooled["x"], dict) else None
        wy = pooled["y"].get("w_h") if isinstance(pooled["y"], dict) else None
        if wx is not None and wy is not None:
            meta["source"] = "summary_pooled_k1_8.{axis}.w_h"
            excerpt = {
                a: {k: pooled[a].get(k) for k in ("d_cm", "sigma_u_cm", "sigma_n_cm", "w_h", "w_n", "e_star_cm")}
                for a in ("x", "y")
            }
            return {"x": float(wx), "y": float(wy)}, {**meta, "B8_excerpt": excerpt}
    return None, meta


def run_c4(alpha: np.ndarray) -> dict:
    part_b_path = PARTIALS / "part_b.json"
    w_h = None
    extra: dict = {}
    status = "NEED_B8"
    if part_b_path.is_file():
        b = json.loads(part_b_path.read_text())
        w_h, extra = _extract_w_h(b)
    result = {
        "alpha": {a: float(alpha[i]) for i, a in enumerate(AXES)},
        "w_h": w_h if w_h is not None else {"x": None, "y": None, "placeholder": True},
        "alpha_times_one_minus_w_h": {"x": None, "y": None},
        "stability_0_lt_expr_lt_2": {"x": None, "y": None},
        "part_b_path": str(part_b_path.relative_to(ROOT)),
        "part_b_ready": part_b_path.is_file() and w_h is not None,
        "w_h_source": extra.get("source"),
        "status": status,
        "notes": "Need B8 w_h from part_b.json to evaluate α·(1−w_h) and stability bound.",
    }
    if w_h is not None:
        expr = {}
        stab = {}
        for a in ("x", "y"):
            whv = w_h.get(a)
            al = float(alpha[{"x": 0, "y": 1}[a]])
            if whv is None or not math.isfinite(float(whv)):
                expr[a] = None
                stab[a] = None
            else:
                val = al * (1.0 - float(whv))
                expr[a] = float(val)
                stab[a] = bool(0.0 < val < 2.0)
        result["alpha_times_one_minus_w_h"] = expr
        result["stability_0_lt_expr_lt_2"] = stab
        result["status"] = "OK"
        result["notes"] = (
            "Stability requires 0 < α·(1−w_h) < 2 per axis. "
            f"w_h from {extra.get('source')}."
        )
        if "B8_excerpt" in extra:
            result["B8_excerpt"] = extra["B8_excerpt"]
    else:
        result["unknown"] = "part_b.json B8/summary w_h not ready — placeholder keys only; α reported."
        result["FOR_HUMAN"] = "Re-run Part C after Part B lands, or merge will fill C4 from B8."
    return {"C4": result}


def write_md(out: dict) -> str:
    c1 = out["C1"]
    c2 = out["C2"]
    c3 = out["C3"]
    c4 = out["C4"]
    g = c1["global"]
    lines = [
        "# Part C — Controller ID + oracle eval offsets",
        "",
        f"Seed={SEED}. Demo root=`data/p3_throw/*/speed_*/*.h5`. h=`ee_pos`, c=`action_pos`. dt={DT}s.",
        f"Conveyor axis: **{c1['conveyor_axis']}** (object motion).",
        "",
        "## C1 — First-order α fit",
        "",
        f"Episodes used: {c1['n_episodes']} (success preferred; all loaded demos were success).",
        "",
        "| Axis | α | R² | n | second-order? |",
        "|---|---:|---:|---:|---|",
    ]
    for a in AXES:
        fo = g[a]["first_order"]
        so = g[a].get("second_order")
        so_s = (
            f"yes R²={so['r2']:.3f} (a={so['a_dh_prev']:.3f}, b={so['b_err_t']:.3f}, c={so['c_err_tm1']:.3f})"
            if so
            else "no"
        )
        lines.append(
            f"| {a} | {fo['alpha']:.4f} | {fo['r2']:.4f} | {fo['n']} | {so_s} |"
        )
    lines += [
        "",
        "Per-phase / per-speed tables: see `part_c.json` keys `C1.per_phase`, `C1.per_speed`.",
        "",
        "## C2 — Simulate with α vs logged hand",
        "",
        f"Simulated {c2['n_sim_episodes']} episodes.",
        f"- Best lag d (xy): mean **{c2['lag_d_frames_xy']['mean']:.2f} frames** "
        f"({c2['lag_d_s_xy']['mean']:.3f} s), std {c2['lag_d_frames_xy']['std']:.2f}.",
        "- Measured tracking lag d = mean(c−h) before close (cm):",
    ]
    for a in AXES:
        m = c2["d_measured_cm"][a]
        p = c2["d_predicted_from_alpha_cm"][a]
        lines.append(
            f"  - {a}: measured {m['mean']:.2f}±{m['std']:.2f} cm; "
            f"α-ramp predict {p['mean']:.2f}±{p['std']:.2f} cm"
        )
    lines += ["", f"_{c2['notes']}_", "", "## C3 — Oracle eval hand−object offsets", ""]
    lines.append(f"Oracle episodes: {c3['n_oracle_episodes']} (checkpoint `{c3['checkpoint']}`).")
    lines.append("")
    lines.append("### Inventory")
    for path, info in c3["inventory"].items():
        lines.append(f"- `{path}`: {info}")
    lines.append("")
    lines.append("### Hand − object xy at close (cm), per speed")
    lines.append("")
    lines.append("| speed | n | mean x | std x | mean y | std y | mean ‖·‖ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for sp, v in c3["hand_minus_object_xy_at_close_per_speed_cm"].items():
        m, s = v["mean_cm"], v["std_cm"]
        lines.append(
            f"| {sp} | {v['n']} | {m['x']:.2f} | {s['x']:.2f} | {m['y']:.2f} | {s['y']:.2f} | {m['norm']:.2f} |"
        )
    lines.append("")
    lines.append("### Hand − object xy at min-distance frame (cm), per speed")
    lines.append("")
    lines.append("| speed | n | mean x | std x | mean y | std y | mean ‖·‖ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for sp, v in c3["hand_minus_object_xy_at_min_dist_per_speed_cm"].items():
        m, s = v["mean_cm"], v["std_cm"]
        lines.append(
            f"| {sp} | {v['n']} | {m['x']:.2f} | {s['x']:.2f} | {m['y']:.2f} | {s['y']:.2f} | {m['norm']:.2f} |"
        )
    settle = c3["last50_settle"]
    lines += [
        "",
        "### Last-50 hand − true future object (truth_pack)",
        "",
        f"- n={settle.get('n')}",
        f"- late-window mean offset (cm): {settle.get('late_mean_offset_cm')}",
        f"- end-frame mean (cm): {settle.get('end_mean_cm')}",
        f"- early→late Δmean (cm): {settle.get('early_to_late_delta_mean_cm')}",
        f"- settles to constant offset? **{settle.get('settles_to_constant_offset')}**",
        f"- figures: {c3.get('figures')}",
        "",
        f"_{settle.get('notes', '')}_",
        "",
        "## C4 — α·(1 − w_h)",
        "",
        f"Status: **{c4.get('status')}**",
        f"- α: {c4.get('alpha')}",
        f"- w_h: {c4.get('w_h')}",
        f"- α·(1−w_h): {c4.get('alpha_times_one_minus_w_h')}",
        f"- stable (0<·<2): {c4.get('stability_0_lt_expr_lt_2')}",
        "",
        c4.get("notes", ""),
        "",
    ]
    if c4.get("unknown"):
        lines.append(f"UNKNOWN: {c4['unknown']}")
    if c4.get("FOR_HUMAN"):
        lines.append(f"FOR_HUMAN: {c4['FOR_HUMAN']}")
    lines += [
        "",
        "## Files / method",
        "",
        "- Script: `foresight_verification/scripts/part_c_controller_eval.py`",
        f"- Demo files: `{DEMO_GLOB}` (ee_pos, action_pos, stage, gripper, success)",
        "- Oracle eval: `*_cond.h5` + `metrics.jsonl` under oracle dirs listed above",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    PARTIALS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=== C1/C2 ===")
    c12, _episodes, alpha = run_c1_c2(rng)
    print("=== C3 ===")
    c3 = run_c3()
    print("=== C4 ===")
    c4 = run_c4(alpha)

    out = {
        "meta": {
            "part": "C",
            "seed": SEED,
            "units": {"distance": "cm", "time": "s", "alpha": "1/frame"},
            "script": "foresight_verification/scripts/part_c_controller_eval.py",
        },
        **c12,
        **c3,
        **c4,
    }
    # Convenience top-level summary for merge / parent
    out["summary"] = {
        "alpha": {a: c12["C1"]["global"][a]["first_order"]["alpha"] for a in AXES},
        "r2": {a: c12["C1"]["global"][a]["first_order"]["r2"] for a in AXES},
        "d_measured_cm": c12["C2"]["d_measured_cm"],
        "lag_d_frames_xy_mean": c12["C2"]["lag_d_frames_xy"]["mean"],
        "c3_overshoot_at_close_per_speed_cm": c3["C3"]["hand_minus_object_xy_at_close_per_speed_cm"],
        "c3_settle": c3["C3"]["last50_settle"],
        "c4": c4["C4"],
    }

    json_path = PARTIALS / "part_c.json"
    md_path = PARTIALS / "part_c.md"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    md_path.write_text(write_md(out))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("alpha", out["summary"]["alpha"])
    print("r2", out["summary"]["r2"])
    print("d_measured", out["summary"]["d_measured_cm"])


if __name__ == "__main__":
    main()
