"""A4: latched z_grasp from wrist depth vs GT. Pass: σ_z ≤ 2 mm.

Offline, no Isaac. Replays each episode through the same ZLatch the expert
uses, feeding ``z_top_from_wrist_cloud`` (whole wrist depth image lifted with the
camera pose from the EE, points near the object XY, high percentile of Z) — the
same call the expert makes. Compares the locked z_grasp to GT
``z_grasp_m(centre + half_height)``.

Gate is on **latched** height (what the policy pack gets as p̂z), not raw
per-frame noise. Per-frame stats are reported for diagnosis.

Usage:
  PYTHONPATH=. python eval/a4_z_grasp_depth.py
  PYTHONPATH=. python eval/a4_z_grasp_depth.py --root data/eval/A1_velocity
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interfaces.config import load_yaml  # noqa: E402
from interfaces.grasp_height import (  # noqa: E402
    ZLatch,
    z_grasp_m,
    z_top_from_gt_center,
    wrist_intrinsics,
    z_top_from_wrist_cloud,
)

SIGMA_MAX_MM = 2.0
_CFG = None
_INSERT_M = 0.03
_CLEAR_M = 0.005
_MIN_GAP_M = 0.07
_Z_BIAS_M = 0.0
_TCP = None
_WRIST_POS = None
_WRIST_QUAT = None
_K = None


def _init():
    """Worker init: load yaml once per process."""
    global _CFG, _INSERT_M, _CLEAR_M, _MIN_GAP_M, _Z_BIAS_M, _TCP, _WRIST_POS, _WRIST_QUAT, _K
    _CFG = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    ch = _CFG.get("claim_height") if isinstance(_CFG.get("claim_height"), dict) else {}
    robot = _CFG.get("robot") if isinstance(_CFG.get("robot"), dict) else {}
    wrist = _CFG.get("wrist_camera") if isinstance(_CFG.get("wrist_camera"), dict) else {}
    _INSERT_M = float(ch.get("finger_insert_m", 0.03))
    _CLEAR_M = float(ch.get("table_clear_m", 0.005))
    _MIN_GAP_M = float(ch.get("latch_lo_m", 0.07))
    _Z_BIAS_M = float(ch.get("depth_z_bias_m", 0.0))
    _TCP = np.asarray(robot.get("tcp_offset", [0.0, 0.0, 0.107]), dtype=np.float64).reshape(3)
    _WRIST_POS = np.asarray(wrist.get("pos", [0.065, 0.0, 0.0]), dtype=np.float64).reshape(3)
    _WRIST_QUAT = np.asarray(
        wrist.get("quat_wxyz", [0.0, 0.7071068, 0.7071068, 0.0]), dtype=np.float64
    ).reshape(4)
    _K = wrist_intrinsics(_CFG)


def _stats_mm(err_m: np.ndarray) -> dict:
    """Bias / σ / RMSE / median in millimetres."""
    e = np.asarray(err_m, dtype=np.float64).reshape(-1)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "n": 0,
            "bias_mm": float("nan"),
            "sigma_mm": float("nan"),
            "rmse_mm": float("nan"),
            "med_mm": float("nan"),
            "pass_sigma": False,
        }
    sigma = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
    return {
        "n": int(e.size),
        "bias_mm": 1000.0 * float(np.mean(e)),
        "sigma_mm": 1000.0 * sigma,
        "rmse_mm": 1000.0 * float(np.sqrt(np.mean(e * e))),
        "med_mm": 1000.0 * float(np.median(e)),
        "pass_sigma": bool(sigma * 1000.0 <= SIGMA_MAX_MM),
    }


def one(path: str) -> dict:
    """One episode: latch wrist z_grasp; return latch error + per-frame errors."""
    p = Path(path)
    meta: dict = {}
    jp = p.with_suffix(".json")
    if jp.exists():
        meta = json.loads(jp.read_text())
    half = float(meta.get("object_half_height") or 0.03)
    cat = str(meta.get("category") or meta.get("object_category") or "unknown")
    z_table = float(meta.get("table_height") or meta.get("table_z") or 0.0)

    out = {
        "path": path,
        "category": cat,
        "half_height_mm": 1000.0 * half,
        "err_latch_m": float("nan"),
        "err_frame_m": [],
        "n_frames": 0,
        "n_frame_scored": 0,
        "ok": False,
    }
    try:
        with h5py.File(p, "r") as f:
            if "wrist_cam_depth" not in f or "ee_pos" not in f:
                return out
            ee = np.asarray(f["ee_pos"])
            eq = np.asarray(f["ee_quat"])
            obj = np.asarray(f["object_pos"])
            depth = np.asarray(f["wrist_cam_depth"])
    except OSError:
        return out

    n = min(len(ee), len(obj), len(depth), len(eq))
    out["n_frames"] = int(n)
    latch = ZLatch.from_cfg(_CFG)
    latch.z_table = z_table
    latch.reset()
    gt_g0 = z_grasp_m(
        z_top_from_gt_center(float(obj[0, 2]), half_height_m=half),
        z_table=z_table,
        insert_m=_INSERT_M,
        clear_m=_CLEAR_M,
    )
    zg = None
    frame_err = []
    for t in range(n):
        gt_top = z_top_from_gt_center(float(obj[t, 2]), half_height_m=half)
        # Object XY = logged position (the expert's state; live: static-cam filter).
        zt = z_top_from_wrist_cloud(
            depth[t],
            ee[t],
            eq[t],
            obj[t, :2],
            f_px=_K[0],
            cx=_K[1],
            cy=_K[2],
            z_table=z_table,
            tcp_offset_m=_TCP,
            wrist_pos_m=_WRIST_POS,
            wrist_quat_wxyz=_WRIST_QUAT,
            z_bias_m=_Z_BIAS_M,
        )
        if np.isfinite(zt) and (float(ee[t, 2]) - float(gt_top)) >= _MIN_GAP_M:
            zg_f = z_grasp_m(
                zt, z_table=z_table, insert_m=_INSERT_M, clear_m=_CLEAR_M
            )
            gt_g = z_grasp_m(
                gt_top, z_table=z_table, insert_m=_INSERT_M, clear_m=_CLEAR_M
            )
            frame_err.append(float(zg_f - gt_g))
        zg = latch.update(
            z_top=float(zt) if np.isfinite(zt) else float("nan"),
            ee_z=float(ee[t, 2]),
        )

    out["err_frame_m"] = frame_err
    out["n_frame_scored"] = len(frame_err)
    if zg is not None and np.isfinite(zg):
        out["err_latch_m"] = float(zg) - float(gt_g0)
        out["ok"] = True
    return out


def summarize(rows: list[dict], *, label: str) -> dict:
    """Pool latch errors (gate) and frame errors (info)."""
    latch = np.asarray(
        [r["err_latch_m"] for r in rows if r["ok"] and np.isfinite(r["err_latch_m"])],
        dtype=np.float64,
    )
    frames = np.concatenate(
        [np.asarray(r["err_frame_m"], dtype=np.float64) for r in rows if r["err_frame_m"]]
        or [np.zeros(0)]
    )
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_h: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not r["ok"] or not np.isfinite(r["err_latch_m"]):
            continue
        e = float(r["err_latch_m"])
        by_cat[r["category"]].append(e)
        bin_mm = int(round(float(r["half_height_mm"]) / 5.0) * 5)
        by_h[f"{bin_mm}mm"].append(e)

    latch_s = _stats_mm(latch)
    return {
        "label": label,
        "episodes": len(rows),
        "episodes_scored": int(latch_s["n"]),
        "z_grasp_latch": latch_s,
        "z_grasp_frame": _stats_mm(frames),
        "by_category": {k: _stats_mm(np.asarray(v)) for k, v in sorted(by_cat.items())},
        "by_half_height": {k: _stats_mm(np.asarray(v)) for k, v in sorted(by_h.items())},
        "pass": bool(latch_s["pass_sigma"]),
        "gate_sigma_max_mm": SIGMA_MAX_MM,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Recording root (repeatable). Default: A1_velocity + smoke_roll_meshes",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "A4_z_grasp" / "a4_report.json"),
    )
    args = ap.parse_args()
    roots = args.root or [
        str(ROOT / "data" / "eval" / "A1_velocity"),
        str(ROOT / "data" / "smoke_roll_meshes"),
    ]

    reports = []
    for root in roots:
        root_p = Path(root)
        files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
        ready = []
        for f in files:
            try:
                with h5py.File(f):
                    ready.append(f)
            except OSError:
                pass
        if not ready:
            print(f"A4: no readable .h5 under {root_p}, skipped", flush=True)
            continue
        with Pool(args.workers, initializer=_init) as pool:
            rows = pool.map(one, ready)
        _init()
        rep = summarize(rows, label=str(root_p))
        reports.append(rep)
        print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(reports) and all(bool(r["pass"]) for r in reports)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gate_sigma_max_mm": SIGMA_MAX_MM, "roots": reports, "pass": all_pass}
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# A4: latched z_grasp from wrist depth vs ground truth",
        "",
        f"Gate: latch σ_z ≤ {SIGMA_MAX_MM:.0f} mm (policy pack p̂z).",
        "",
        "| root | episodes | latch bias/σ/RMSE mm | frame σ mm (info) | pass |",
        "|---|---|---|---|---|",
    ]
    for r in reports:
        g, f = r["z_grasp_latch"], r["z_grasp_frame"]
        lines.append(
            f"| `{Path(r['label']).name}` | {r['episodes_scored']}/{r['episodes']} | "
            f"{g['bias_mm']:+.1f} / {g['sigma_mm']:.1f} / {g['rmse_mm']:.1f} | "
            f"{f['sigma_mm']:.1f} | {'yes' if r['pass'] else 'no'} |"
        )
    lines.append("")
    for r in reports:
        lines.append(f"## {Path(r['label']).name} by category (latch)")
        lines.append("")
        lines.append("| category | n | bias mm | σ mm | pass |")
        lines.append("|---|---|---|---|---|")
        for cat, s in r["by_category"].items():
            lines.append(
                f"| {cat} | {s['n']} | {s['bias_mm']:+.1f} | {s['sigma_mm']:.1f} | "
                f"{'yes' if s['pass_sigma'] else 'no'} |"
            )
        lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    lines.append(
        "Method: whole wrist depth image lifted to world points with the camera pose "
        "from the EE (not frozen `Camera.data.pos_w`); object = points within 4 cm of its "
        "XY, above the table, below the fingertips; top = 99th percentile of Z. "
        "Frame σ is info only (every frame with gap ≥ 7 cm, not the latched value)."
    )
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
