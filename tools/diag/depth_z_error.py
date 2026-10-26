#!/usr/bin/env python3
"""Offline gate: wrist→z_top vs GT must stay within depth_rmse_max (default 1 cm).

  PYTHONPATH=. python tools/diag/depth_z_error.py data/smoke/.../ep.h5
  PYTHONPATH=. python tools/diag/depth_z_error.py data/smoke/.../ep.h5 --max-rmse-cm 1.0

Uses saved ``wrist_cam_z`` when present (live cam pose). Else falls back to
``ee_z + tcp_offset`` (0.107 m). Exits 1 if RMSE or |median| fails the gate.
Only frames with hand–top gap ≥ 7 cm are scored (depth untrusted closer in).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from interfaces.grasp_height import z_top_from_gt_center, z_top_from_wrist_depth

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", type=Path)
    p.add_argument("--cam-above-tcp", type=float, default=0.107)
    p.add_argument("--max-rmse-cm", type=float, default=1.0)
    p.add_argument("--min-gap-cm", type=float, default=7.0, help="only score gap ≥ this")
    args = p.parse_args()

    jpath = args.h5.with_suffix(".json")
    meta = json.loads(jpath.read_text()) if jpath.exists() else {}
    half = float(meta.get("object_half_height") or 0.03)

    with h5py.File(args.h5, "r") as f:
        ee = np.asarray(f["ee_pos"])
        eq = np.asarray(f["ee_quat"])
        obj = np.asarray(f["object_pos"])
        depth = np.asarray(f["wrist_cam_depth"])
        cam_z_log = np.asarray(f["wrist_cam_z"]) if "wrist_cam_z" in f else None
        has_static = "static_cam_depth" in f

    gt_top = np.asarray(
        [z_top_from_gt_center(float(z), half_height_m=half) for z in obj[:, 2]]
    )
    gap = ee[:, 2] - gt_top
    min_gap = float(args.min_gap_cm) / 100.0
    old_err, new_err = [], []
    for t in range(len(ee)):
        if float(gap[t]) < min_gap:
            continue
        cz = float(cam_z_log[t]) if cam_z_log is not None else None
        if cz is not None and not np.isfinite(cz):
            cz = None
        z_old = z_top_from_wrist_depth(
            depth[t], ee[t], eq[t], cam_above_tcp_m=0.0
        )
        z_new = z_top_from_wrist_depth(
            depth[t],
            ee[t],
            eq[t],
            cam_z=cz,
            cam_above_tcp_m=float(args.cam_above_tcp),
        )
        if np.isfinite(z_old):
            old_err.append(z_old - gt_top[t])
        if np.isfinite(z_new):
            new_err.append(z_new - gt_top[t])

    def stats(name: str, err: list[float]) -> dict:
        e = np.asarray(err, dtype=np.float64)
        if e.size == 0:
            print(f"{name}: n=0 (no frames with gap≥{args.min_gap_cm} cm)")
            return {"n": 0, "rmse_cm": float("nan"), "med_cm": float("nan")}
        out = {
            "n": int(e.size),
            "mean_cm": 100.0 * float(e.mean()),
            "med_cm": 100.0 * float(np.median(e)),
            "rmse_cm": 100.0 * float(np.sqrt((e * e).mean())),
        }
        print(
            f"{name}: n={out['n']}  mean={out['mean_cm']:+.1f}cm  "
            f"med={out['med_cm']:+.1f}cm  rmse={out['rmse_cm']:.1f}cm"
        )
        return out

    print(f"file={args.h5}")
    print(
        f"half_h={half:.4f}m  static_in_h5={has_static}  "
        f"wrist_cam_z_in_h5={cam_z_log is not None}"
    )
    print(f"GT centre≈{100 * float(obj[0, 2]):.1f}cm  GT top≈{100 * gt_top[0]:.1f}cm")
    stats("wrist OLD (ee_z - depth)", old_err)
    s = stats("wrist NEW (cam_z - depth)", new_err)
    lim = float(args.max_rmse_cm)
    ok = s["n"] > 0 and s["rmse_cm"] <= lim and abs(s["med_cm"]) <= lim
    print(
        f"GATE depth RMSE≤{lim:.1f}cm and |median|≤{lim:.1f}cm @ gap≥{args.min_gap_cm:.0f}cm: "
        f"{'PASS' if ok else 'FAIL'}"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
