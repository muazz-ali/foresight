#!/usr/bin/env python3
"""Seed path diagnostic: object XY vs straight line, forecast vs true.

Reads a recorded episode ``.h5`` (needs ``object_pos``, ``object_vel``,
``stage``, ``timestamp``, ``conditioning``). Writes PNG plots + a short JSON
with sideways deviation (mm).

  PYTHONPATH=. python tools/diag/plot_seed_path.py \\
    data/eval/B1_grid_v7/cells/C30/C30_v30_d000_s1001_egg.h5

Why: C30 s1001 had ~2 mm cross at close — check whether the object path
bends (friction / slide) or only the aim/hand trails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _ride_mask(stage: np.ndarray) -> np.ndarray:
    """Approach + close ticks (stage 1 and 3). Fallback: whole episode."""
    m = np.isin(stage.astype(int), [1, 3])
    return m if bool(m.any()) else np.ones(len(stage), dtype=bool)


def analyze(h5_path: Path) -> dict:
    """Compute path sideways mm and forecast error over the ride."""
    with h5py.File(h5_path, "r") as f:
        obj = np.asarray(f["object_pos"][:], dtype=np.float64)
        vel = np.asarray(f["object_vel"][:], dtype=np.float64)
        stage = np.asarray(f["stage"][:], dtype=np.int64)
        t = np.asarray(f["timestamp"][:], dtype=np.float64)
        cond = np.asarray(f["conditioning"][:], dtype=np.float64) if "conditioning" in f else None
        ee = np.asarray(f["ee_pos"][:], dtype=np.float64) if "ee_pos" in f else None
        success = int(f["success"][0]) if "success" in f else None

    ride = _ride_mask(stage)
    xy = obj[ride, :2]
    vxy = vel[ride, :2]
    speed = np.linalg.norm(vxy, axis=1)
    moving = speed > 0.05
    if not bool(moving.any()):
        uh = xy[-1] - xy[0]
        uh = uh / max(float(np.linalg.norm(uh)), 1e-9)
    else:
        uh = vxy[moving][0]
        uh = uh / max(float(np.linalg.norm(uh)), 1e-9)
    nh = np.array([-uh[1], uh[0]], dtype=np.float64)
    cross = (xy - xy[0]) @ nh
    along = (xy - xy[0]) @ uh
    # Ideal straight line through start along û.
    line = xy[0] + along[:, None] * uh[None, :]

    # Forecast: conditioning[:2] = p̂_xy(t+Δ). Compare to true object later.
    # With Δ≈0, compare to current true; also report lead error vs p + v·τ_pack.
    forecast_err = None
    if cond is not None:
        phat = cond[ride, :2]
        # True position now (Δ=0 pack should match coast at small lead).
        err = phat - xy
        forecast_err = {
            "mean_err_xy_mm": (1000.0 * err).mean(axis=0).tolist(),
            "rmse_xy_mm": (1000.0 * np.sqrt((err**2).mean(axis=0))).tolist(),
            "max_abs_err_mm": float(1000.0 * np.max(np.abs(err))),
        }

    return {
        "h5": str(h5_path),
        "success": success,
        "n_ride": int(ride.sum()),
        "along_cm": float(100.0 * along[-1]) if len(along) else 0.0,
        "sideways_mean_mm": float(1000.0 * np.mean(cross)) if len(cross) else 0.0,
        "sideways_max_abs_mm": float(1000.0 * np.max(np.abs(cross))) if len(cross) else 0.0,
        "sideways_std_mm": float(1000.0 * np.std(cross)) if len(cross) else 0.0,
        "speed_mean_cm_s": float(100.0 * np.mean(speed[moving])) if bool(moving.any()) else None,
        "uh": uh.tolist(),
        "forecast": forecast_err,
        "_arrays": {
            "xy": xy,
            "line": line,
            "cross_mm": 1000.0 * cross,
            "along_m": along,
            "t_ride": t[ride],
            "phat": cond[ride, :2] if cond is not None else None,
            "ee_xy": ee[ride, :2] if ee is not None else None,
            "obj_xy_full": obj[:, :2],
            "stage": stage,
        },
    }


def plot(result: dict, out_dir: Path, stem: str) -> list[Path]:
    """Write XY path and forecast-vs-true PNGs. Returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = result.pop("_arrays")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1) Object XY vs straight line.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    xy, line = a["xy"], a["line"]
    ax.plot(line[:, 0], line[:, 1], "k--", lw=1.5, label="straight line")
    ax.plot(xy[:, 0], xy[:, 1], "b-", lw=2, label="object XY (ride)")
    ax.scatter([xy[0, 0]], [xy[0, 1]], c="g", s=40, zorder=3, label="start")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], c="r", s=40, zorder=3, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"seed path  sideways max |e| = {result['sideways_max_abs_mm']:.2f} mm"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    p1 = out_dir / f"{stem}_xy_vs_line.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    written.append(p1)

    # 2) Sideways vs time + forecast vs true.
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)
    ax0 = axes[0]
    ax0.plot(a["along_m"] * 100.0, a["cross_mm"], "b-")
    ax0.axhline(0.0, color="k", ls="--", lw=0.8)
    ax0.set_xlabel("along track (cm)")
    ax0.set_ylabel("sideways (mm)")
    ax0.set_title("Object sideways deviation from straight line")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    if a["phat"] is not None:
        t0 = float(a["t_ride"][0])
        tt = a["t_ride"] - t0
        ax1.plot(tt, a["xy"][:, 0], "b-", label="true x")
        ax1.plot(tt, a["phat"][:, 0], "b--", label="forecast x")
        ax1.plot(tt, a["xy"][:, 1], "r-", label="true y")
        ax1.plot(tt, a["phat"][:, 1], "r--", label="forecast y")
        ax1.set_xlabel("time on ride (s)")
        ax1.set_ylabel("position (m)")
        ax1.set_title("Forecast (conditioning p̂) vs true object XY")
        ax1.legend(loc="best", fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "no conditioning in h5", ha="center", transform=ax1.transAxes)
    p2 = out_dir / f"{stem}_forecast_vs_true.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    written.append(p2)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", type=Path)
    p.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Default: next to the h5 as <stem>_pathdiag/",
    )
    args = p.parse_args()
    h5 = args.h5.resolve()
    out = args.output_dir or (h5.parent / f"{h5.stem}_pathdiag")
    result = analyze(h5)
    stem = h5.stem
    paths = plot(result, out, stem)
    # result no longer has _arrays
    (out / f"{stem}_pathdiag.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("wrote:")
    for q in paths:
        print(" ", q)
    print(" ", out / f"{stem}_pathdiag.json")


if __name__ == "__main__":
    main()
