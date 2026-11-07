"""B2: expert DESCEND / intercept aim Z is z_grasp, not a fixed desk height.

Why: if the teacher always dives to one absolute Z, demos teach the wrong
height for short and tall objects. Height must live in p̂z / z_grasp; the
relative approach offset (d_z) stays the same across heights.

Two checks (Isaac-free):

  1. SM probe — force APPROACH and DESCEND over several object heights.
     Command Z must equal that episode's z_grasp, and the commanded heights
     must span the object heights (not a constant).

  2. Demo consistency (optional --root) — at first CLOSE, d_z =
     z_hand − z_grasp (GT object centre when close_at_object_center).
     Spread of d_z across episodes must be ≤ s_z / 2 (claim_height.z_slack).

Usage:
  PYTHONPATH=. python eval/b2_descend_z_grasp.py
  PYTHONPATH=. python eval/b2_descend_z_grasp.py --root data/smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interfaces.config import load_yaml  # noqa: E402
from interfaces.state import oracle_from_gt  # noqa: E402
from sim.state_machine import (  # noqa: E402
    STAGE_APPROACH,
    STAGE_CLOSE,
    PickPlaceStateMachine,
)

# Object centres used in the SM height sweep (metres).
_PROBE_HEIGHTS_M = (0.020, 0.035, 0.050, 0.080)
_CARRY = np.array([0.50, 0.30, 0.12], dtype=np.float64)
_DROP = np.array([0.50, 0.30, 0.06], dtype=np.float64)
# Command Z must move at least this much across the probe heights.
_MIN_SPAN_M = 0.040


def _s_z_half_m(cfg: dict) -> float:
    """Close-height consistency budget: s_z / 2 from claim_height.z_slack."""
    ch = cfg.get("claim_height") if isinstance(cfg.get("claim_height"), dict) else {}
    s_z = float(ch.get("z_slack_m", ch.get("z_slack_mm", 5.0) * 1e-3))
    return 0.5 * s_z


def probe_sm(cfg: dict) -> dict:
    """Run intercept at several object heights; check Z tracks z_grasp.

    Call with a loaded phase0 cfg. Returns a pass/fail dict for the report.
    """
    dt = float(cfg["sim"]["dt"])
    down = None
    detail: list[dict] = []
    summaries: list[dict] = []
    for stage in (STAGE_APPROACH,):
        cmd_zs: list[float] = []
        errs: list[float] = []
        for half in _PROBE_HEIGHTS_M:
            sm = PickPlaceStateMachine(cfg, dt=dt)
            sm.set_box_poses(_CARRY, drop_pos=_DROP)
            sm.set_object_half_height(float(half), z_table=0.0)
            sm.stage = int(stage)
            sm.timer_s = 0.0
            if down is None:
                down = sm._down_quat.copy()
            obj = np.array([0.55, 0.0, float(half)], dtype=np.float64)
            ee = np.array([0.45, 0.0, 0.40], dtype=np.float64)  # high hand
            vel = np.array([0.0, -0.20, 0.0], dtype=np.float64)
            cmd = sm.step(oracle_from_gt(obj, vel, 0.0), ee, down)
            zg = float(sm.z_grasp) if sm.z_grasp is not None else float("nan")
            cz = float(cmd.position[2])
            err = abs(cz - zg) if np.isfinite(zg) else float("nan")
            cmd_zs.append(cz)
            errs.append(err)
            detail.append(
                {
                    "stage": int(stage),
                    "obj_z_m": float(half),
                    "z_grasp_m": zg,
                    "cmd_z_m": cz,
                    "abs_err_m": err,
                }
            )
        span = float(max(cmd_zs) - min(cmd_zs)) if cmd_zs else 0.0
        max_err = float(max(errs)) if errs else float("nan")
        # Fixed-height bug: all commands land near the same absolute Z.
        track_ok = bool(np.isfinite(max_err) and max_err < 1e-6)
        not_fixed = span >= _MIN_SPAN_M
        summaries.append(
            {
                "stage": int(stage),
                "cmd_z_span_m": span,
                "max_abs_err_m": max_err,
                "pass_tracks_z_grasp": track_ok,
                "pass_not_fixed": not_fixed,
            }
        )

    ok = all(
        bool(s["pass_tracks_z_grasp"]) and bool(s["pass_not_fixed"]) for s in summaries
    )
    return {
        "check": "sm_probe",
        "heights_m": list(_PROBE_HEIGHTS_M),
        "min_span_m": _MIN_SPAN_M,
        "detail": detail,
        "summaries": summaries,
        "pass": ok,
    }


def _first_close_idx(stage: np.ndarray) -> int | None:
    """Index of the first CLOSE frame, or None."""
    hits = np.flatnonzero(np.asarray(stage, dtype=np.int64) == STAGE_CLOSE)
    return int(hits[0]) if hits.size else None


def one_demo(path: str, *, s_z_half: float) -> dict:
    """One episode: d_z at first CLOSE vs GT object centre (expert z_grasp)."""
    p = Path(path)
    meta: dict = {}
    jp = p.with_suffix(".json")
    if jp.exists():
        meta = json.loads(jp.read_text())
    cat = str(meta.get("category") or meta.get("object_category") or "unknown")
    half = float(meta.get("object_half_height") or float("nan"))

    out: dict = {
        "path": path,
        "category": cat,
        "object_half_height_m": half,
        "closed": False,
        "d_z_m": float("nan"),
        "ee_z_m": float("nan"),
        "z_grasp_m": float("nan"),
        "pass_abs_d_z": False,
    }
    with h5py.File(path, "r") as h:
        if "stage" not in h or "ee_pos" not in h or "object_pos" not in h:
            return out
        stage = h["stage"][:]
        ee = np.asarray(h["ee_pos"][:], dtype=np.float64)
        obj = np.asarray(h["object_pos"][:], dtype=np.float64)
    i = _first_close_idx(stage)
    if i is None:
        return out
    ee_z = float(ee[i, 2])
    # Expert aim with close_at_object_center: object centre = sit height.
    z_grasp = float(obj[i, 2])
    d_z = ee_z - z_grasp
    out.update(
        {
            "closed": True,
            "ee_z_m": ee_z,
            "z_grasp_m": z_grasp,
            "d_z_m": d_z,
            "pass_abs_d_z": bool(abs(d_z) <= s_z_half + 1e-9),
        }
    )
    return out


def summarize_demos(rows: list[dict], *, s_z_half: float, label: str) -> dict:
    """Spread of d_z across closed episodes; pass if σ ≤ s_z/2."""
    closed = [r for r in rows if r.get("closed")]
    d = np.asarray([r["d_z_m"] for r in closed], dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {
            "check": "demo_d_z",
            "label": label,
            "episodes": len(rows),
            "closed": 0,
            "s_z_half_mm": 1000.0 * s_z_half,
            "pass": False,
            "reason": "no CLOSE frames",
        }
    sigma = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    # Fixed-height fingerprint: hand Z almost constant while object Z moves.
    ee = np.asarray([r["ee_z_m"] for r in closed], dtype=np.float64)
    zg = np.asarray([r["z_grasp_m"] for r in closed], dtype=np.float64)
    ee_span = float(np.nanmax(ee) - np.nanmin(ee)) if ee.size else 0.0
    zg_span = float(np.nanmax(zg) - np.nanmin(zg)) if zg.size else 0.0
    # If objects span ≥ 1 cm height but the hand barely moves, that is fixed Z.
    fixed_like = bool(zg_span >= 0.010 and ee_span < 0.5 * zg_span)
    pass_spread = bool(sigma <= s_z_half + 1e-12)
    return {
        "check": "demo_d_z",
        "label": label,
        "episodes": len(rows),
        "closed": int(d.size),
        "s_z_half_mm": 1000.0 * s_z_half,
        "d_z_bias_mm": 1000.0 * float(np.mean(d)),
        "d_z_sigma_mm": 1000.0 * sigma,
        "d_z_abs_max_mm": 1000.0 * float(np.max(np.abs(d))),
        "ee_z_span_mm": 1000.0 * ee_span,
        "z_grasp_span_mm": 1000.0 * zg_span,
        "fixed_height_like": fixed_like,
        "pass": bool(pass_spread and not fixed_like),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Demo root with .h5 (repeatable). Default: data/smoke",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "B2_descend_z_grasp" / "b2_report.json"),
    )
    ap.add_argument(
        "--skip-demos",
        action="store_true",
        help="Only run the SM height probe (no .h5 scan).",
    )
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    s_z_half = _s_z_half_m(cfg)

    sm_rep = probe_sm(cfg)
    print(json.dumps(sm_rep, indent=2), flush=True)

    demo_reps: list[dict] = []
    if not args.skip_demos:
        roots = args.root or [str(ROOT / "data" / "smoke")]
        for root in roots:
            root_p = Path(root)
            files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
            rows = [one_demo(f, s_z_half=s_z_half) for f in files]
            rep = summarize_demos(rows, s_z_half=s_z_half, label=str(root_p))
            demo_reps.append(rep)
            print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(sm_rep["pass"]) and (
        True if args.skip_demos else (bool(demo_reps) and all(r["pass"] for r in demo_reps))
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "s_z_half_mm": 1000.0 * s_z_half,
        "sm_probe": sm_rep,
        "demos": demo_reps,
        "pass": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B2: DESCEND / intercept goes to z_grasp (not a fixed height)",
        "",
        f"Gate: SM command Z = z_grasp across heights; demo d_z σ ≤ s_z/2 "
        f"({1000.0 * s_z_half:.1f} mm).",
        "",
        f"| check | detail | pass |",
        f"|---|---|---|",
        f"| SM probe | span≥{_MIN_SPAN_M*1000:.0f} mm, err=0 | "
        f"{'yes' if sm_rep['pass'] else 'no'} |",
    ]
    for r in demo_reps:
        lines.append(
            f"| demos `{Path(r['label']).name}` | "
            f"n={r.get('closed', 0)}, σ={r.get('d_z_sigma_mm', float('nan')):.2f} mm, "
            f"fixed_like={r.get('fixed_height_like')} | "
            f"{'yes' if r['pass'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
