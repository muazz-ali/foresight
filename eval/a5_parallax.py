"""A5: static-camera parallax correction.  Pass: x/y error ≤ 2 mm (per-axis RMS).

There is no separate parallax module. The correction in the deployed path is in
perception/estimator.py: the tracked blob pixel is lifted onto a horizontal plane at the object's
CENTRE height (PinholeCamera.lift(uv, plane_z); plane_z set once in StaticCamEstimator.start),
not onto the table. Because the camera looks down at a slant, a plane-height error dz shifts the
lifted point sideways by about dz · (horizontal range / camera height above the plane).

Arms:
  deployed      corrected lift at centre height (parallax + silhouette/lead) — **gate**
  raw_centre    raw tracker UV lifted at centre height (parallax on, no blob fixes)
  no_correction raw UV lifted at table height (parallax off)
  dz=+e / −e    raw UV at true centre ± e (default from A4 latch: −6.7 mm bias, ±2.0 mm σ)
Also: pixel bias of the blob centre vs the projected true centre (tracker, not geometry).

Error = estimate XY − true object XY. Frames: free object, filter warm, hand more than 12 cm
away (no occlusion). Prefer ``data/eval/A5_recordings`` (camera stamp matches current yaml).

Usage:
  PYTHONPATH=. python eval/a5_parallax.py
  PYTHONPATH=. python eval/a5_parallax.py --root data/eval/A5_recordings
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
for p in (ROOT, ROOT / "scripts", ROOT / "eval"):
    sys.path.insert(0, str(p))

from interfaces.config import load_yaml  # noqa: E402
from perception.estimator import StaticCamEstimator  # noqa: E402

import g2_perception_report as g2  # noqa: E402
from a1_velocity_ratio import preflight  # noqa: E402

PASS_MM = 2.0
HAND_CLEAR = 0.12
WARMUP_FRAMES = 3
_CFG = None
_DZ = (0.0, 0.0020)  # A4 latch bias ~0 after depth_z_bias; σ ~2 mm on A1_velocity


def _init(dz):
    global _CFG, _DZ
    cv2.setNumThreads(1)
    _CFG = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    _DZ = dz


def one(path):
    ep = g2._read_episode(path)
    ts, orc, rgb = ep["ts"], ep["oracle"], ep["rgb"]
    meta_p = Path(path).with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    z_table = float(meta.get("table_z") or 0.0)
    with h5py.File(path) as h:
        ee = h["ee_pos"][: len(ts)]
    out = {"start_ok": False, "rows": [], "category": meta.get("category")}
    if len(ts) < 10:
        return out
    est = StaticCamEstimator.from_cfg(_CFG)
    if ep["xy_bounds"] is not None:
        est.search_xy_bounds = tuple(map(tuple, ep["xy_bounds"]))
    out["start_ok"] = bool(est.start(rgb[0], orc[0, :3]))
    cam = est.camera
    seen = 0
    bias, sig = _DZ
    for t in range(len(ts)):
        st = est.step(rgb[t], ts[t])
        uv = est.last_uv
        if uv is None or st is None:
            continue
        seen += 1
        if seen <= WARMUP_FRAMES or np.linalg.norm(ee[t, :2] - orc[t, :2]) <= HAND_CLEAR:
            continue
        true_xy = orc[t, :2]
        zc = float(orc[t, 2])
        row = {"uv_err_px": (np.asarray(uv) - cam.project(orc[t, :3])).tolist()}
        # Gate arm: corrected lift at centre height (parallax + silhouette/lead).
        # This is the measurement the filter eats — not the smoothed KF state.
        if est.last_lift_xy is None:
            continue
        row["deployed"] = (np.asarray(est.last_lift_xy) - true_xy).tolist()
        for name, pz in (
            ("raw_centre", est.plane_z),
            ("no_correction", z_table),
            ("dz_bias", zc + bias),
            ("dz_plus_sigma", zc + sig),
            ("dz_minus_sigma", zc - sig),
        ):
            try:
                row[name] = (np.asarray(cam.lift(uv, pz)) - true_xy).tolist()
            except ValueError:
                row[name] = [np.nan, np.nan]
        row["range_m"] = float(np.linalg.norm(cam.position[:2] - true_xy))
        row["cam_height_m"] = float(cam.position[2] - zc)
        out["rows"].append(row)
    return out


def stats(R, key):
    e = np.array([r[key] for r in R], dtype=float) * 1000
    e = e[np.all(np.isfinite(e), axis=1)]
    if len(e) < 2:
        return None
    rms = np.sqrt(np.mean(e ** 2, axis=0))
    return {"n": int(len(e)), "bias_mm": e.mean(0).tolist(), "sigma_mm": e.std(0).tolist(), "rms_mm": rms.tolist(),
            "median_abs_mm": float(np.median(np.linalg.norm(e, axis=1))),
            "p95_abs_mm": float(np.percentile(np.linalg.norm(e, axis=1), 95)),
            "pass": bool(np.all(rms <= PASS_MM))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root",
        default=str(ROOT / "data" / "eval" / "A5_recordings"),
        help="Episode root (default: A5_recordings, camera-matched)",
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "A5_parallax"))
    ap.add_argument("--dz-mm", type=float, nargs=2, default=[0.0, 2.0], help="height bias and σ to test (A4 latch)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    # Camera stamp written by the recorder: refuse recordings made with another camera pose.
    stamp_p = Path(a.root) / "camera_pose.json"
    if stamp_p.exists():
        stamp = json.loads(stamp_p.read_text())
        cam = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")["static_camera"]
        for k in ("position", "rotation_deg", "focal_length", "horizontal_aperture", "width", "height"):
            if k in stamp and not np.allclose(np.asarray(stamp[k], float), np.asarray(cam.get(k), float), atol=1e-6):
                sys.exit(f"A5 stop: recordings under {a.root} used static_camera.{k} = {stamp[k]}, "
                         f"current config has {cam.get(k)}. Re-record with the current camera.")
    else:
        print(f"A5 note: no camera_pose.json under {a.root}; relying on the tracker-start check only.", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = []
    allrows = []
    with Pool(a.workers, initializer=_init, initargs=((a.dz_mm[0] / 1000, a.dz_mm[1] / 1000),)) as pool:
        for vdir in sorted(Path(a.root).glob("v*")):
            files = sorted(glob.glob(str(vdir / "**" / "*.h5"), recursive=True))
            if not files:
                continue
            preflight(files)
            res = pool.map(one, files)
            n_ok = sum(r["start_ok"] for r in res)
            if n_ok < 0.5 * len(res):
                sys.exit(f"A5 stop at {vdir.name}: tracker started in only {n_ok}/{len(res)} episodes; "
                         "camera config likely does not match these recordings.")
            R = [row for r in res for row in r["rows"]]
            allrows += R
            keys = (
                "deployed",
                "raw_centre",
                "no_correction",
                "dz_bias",
                "dz_plus_sigma",
                "dz_minus_sigma",
                "uv_err_px",
            )
            report.append(
                {
                    "speed": float(vdir.name[1:]),
                    "episodes": len(res),
                    **{k: stats(R, k) for k in keys},
                }
            )
    if not report:
        sys.exit(f"A5 stop: no recordings under {a.root}.")
    geo = {"range_m_median": float(np.median([r["range_m"] for r in allrows])),
           "cam_height_m_median": float(np.median([r["cam_height_m"] for r in allrows]))}
    keys = (
        "deployed",
        "raw_centre",
        "no_correction",
        "dz_bias",
        "dz_plus_sigma",
        "dz_minus_sigma",
        "uv_err_px",
    )
    pooled = {k: stats(allrows, k) for k in keys}
    (out / "a5_report.json").write_text(json.dumps({"per_speed": report, "pooled": pooled, "geometry": geo}, indent=1))
    L = ["# A5: static-camera parallax correction (pass: per-axis RMS x/y error ≤ 2 mm)", "",
         f"Lift = tracked blob pixel → plane at height plane_z. Camera range to object ~{geo['range_m_median']:.2f} m, "
         f"height above object ~{geo['cam_height_m_median']:.2f} m → 1 mm of height error ≈ "
         f"{geo['range_m_median']/geo['cam_height_m_median']:.2f} mm sideways.", "",
         "| speed | arm | frames | bias x, y mm | σ x, y mm | RMS x, y mm | |e| median / p95 mm | pass |", "|---|---|---|---|---|---|---|---|"]
    names = {
        "deployed": "deployed (corrected lift @ centre)",
        "raw_centre": "raw UV @ centre height",
        "no_correction": "no correction (table plane)",
        "dz_bias": f"height {a.dz_mm[0]:+.1f} mm (A4 bias)",
        "dz_plus_sigma": f"height +{a.dz_mm[1]:.2f} mm",
        "dz_minus_sigma": f"height −{a.dz_mm[1]:.2f} mm",
    }
    for r in report + [{"speed": None, **pooled}]:
        for k, label in names.items():
            s = r[k]
            if not s:
                continue
            sp = "all" if r["speed"] is None else f"{r['speed']:.2f}"
            L.append(f"| {sp} | {label} | {s['n']} | {s['bias_mm'][0]:+.2f}, {s['bias_mm'][1]:+.2f} | "
                     f"{s['sigma_mm'][0]:.2f}, {s['sigma_mm'][1]:.2f} | {s['rms_mm'][0]:.2f}, {s['rms_mm'][1]:.2f} | "
                     f"{s['median_abs_mm']:.2f} / {s['p95_abs_mm']:.2f} | {'yes' if s['pass'] else 'no'} |")
    px = pooled["uv_err_px"]
    if px:
        L += ["", f"Blob centre vs projected true centre (pixels, all speeds): bias u, v {px['bias_mm'][0]/1000:+.2f}, "
              f"{px['bias_mm'][1]/1000:+.2f} px; σ {px['sigma_mm'][0]/1000:.2f}, {px['sigma_mm'][1]/1000:.2f} px."]
    d = pooled["deployed"]
    L += ["", f"**Deployed path, pooled: {'PASS' if d and d['pass'] else 'FAIL'}**"]
    (out / "REPORT_auto.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    raise SystemExit(0 if d and d["pass"] else 1)


if __name__ == "__main__":
    main()
