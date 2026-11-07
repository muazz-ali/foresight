"""B5: no hand-trailing bias — mean d at close = 0 ± 2 mm.

Plain words: when the fingers shut, the hand should sit on the object, not
systematically ahead or behind. A constant lag (old demos: ~2 cm behind the
aim) teaches the policy ``target ≈ hand + d`` and it stops reading p̂.

Isaac-free. For each demo at first CLOSE:

  d = (hand_xy − object_xy) · û_v     when |v| ≥ vmin
  d_x, d_y = hand_xy − object_xy      always (info + static bins)

``û_v`` is the unit object velocity in XY. Positive d = hand ahead of the
object along the motion; negative = hand trails.

Pass: every (speed × object) bin with ≥ ``min_n`` closes has
|mean(d)| ≤ 2 mm (moving). Static bins use |mean(d_x)| and |mean(d_y)| ≤ 2 mm.

Also reports command-target offset (``action_pos``) so you can tell
"expert aimed off" from "DiffIK left the hand off-centre".

Usage:
  PYTHONPATH=. python eval/b5_hand_trail_bias.py
  PYTHONPATH=. python eval/b5_hand_trail_bias.py --root data/eval/A1_velocity
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.state_machine import STAGE_CLOSE  # noqa: E402

BIAS_MAX_M = 0.002  # ± 2 mm
MIN_V_M_S = 0.02  # below this, treat as static (no û_v)
MIN_N = 3  # bins need this many closes to vote


def along_track_m(hand_xy: np.ndarray, obj_xy: np.ndarray, vel_xy: np.ndarray) -> float:
    """Signed hand−object gap along velocity (metres). + = hand ahead.

    Call with XY slices. Returns nan if speed is below ``MIN_V_M_S``.
    """
    h = np.asarray(hand_xy, dtype=np.float64).reshape(-1)[:2]
    o = np.asarray(obj_xy, dtype=np.float64).reshape(-1)[:2]
    v = np.asarray(vel_xy, dtype=np.float64).reshape(-1)[:2]
    nv = float(np.linalg.norm(v))
    if nv < MIN_V_M_S:
        return float("nan")
    return float(np.dot(h - o, v / nv))


def one(path: str, *, t_contact_s: float = 0.130, dt_s: float = 1.0 / 30.0) -> dict:
    """One episode: hand−object offset at pad contact (CLOSE + t_contact).

    Why contact, not gate-fire: the predictive close gate aims ``e + Δv·t`` at
    the pads, so B5 must score the gap when the pads arrive (~130 ms later).
    ``d = hand − object``; + = hand ahead.
    """
    p = Path(path)
    meta: dict = {}
    jp = p.with_suffix(".json")
    if jp.exists():
        meta = json.loads(jp.read_text())
    cat = str(meta.get("category") or meta.get("object_category") or "unknown")

    out: dict = {
        "path": path,
        "category": cat,
        "speed_m_s": float("nan"),
        "closed": False,
        "moving": False,
        "d_along_m": float("nan"),
        "d_x_m": float("nan"),
        "d_y_m": float("nan"),
        "d_cmd_along_m": float("nan"),
        "pass": False,
    }
    with h5py.File(path, "r") as h:
        if "stage" not in h or "ee_pos" not in h or "object_pos" not in h:
            return out
        stage = np.asarray(h["stage"][:], dtype=np.int64)
        ee = np.asarray(h["ee_pos"][:], dtype=np.float64)
        obj = np.asarray(h["object_pos"][:], dtype=np.float64)
        ovel = np.asarray(h["object_vel"][:], dtype=np.float64) if "object_vel" in h else None
        act = np.asarray(h["action_pos"][:], dtype=np.float64) if "action_pos" in h else None
        if "speed_m_s" in h:
            out["speed_m_s"] = float(np.asarray(h["speed_m_s"]).reshape(-1)[0])

    hits = np.flatnonzero(stage == STAGE_CLOSE)
    if hits.size == 0:
        return out
    i0 = int(hits[0])
    # Score at pad contact (gate fire + t_contact), clamped to the close stage.
    n_c = max(0, int(round(float(t_contact_s) / max(float(dt_s), 1e-9))))
    i = min(i0 + n_c, len(stage) - 1)
    while i > i0 and int(stage[i]) != STAGE_CLOSE:
        i -= 1
    d_xy = ee[i, :2] - obj[i, :2]
    vel_xy = (
        np.asarray(ovel[i, :2], dtype=np.float64)
        if ovel is not None
        else np.zeros(2, dtype=np.float64)
    )
    d_along = along_track_m(ee[i, :2], obj[i, :2], vel_xy)
    moving = bool(np.isfinite(d_along))
    d_cmd = float("nan")
    if act is not None and moving:
        d_cmd = along_track_m(act[i, :2], obj[i, :2], vel_xy)

    # Moving: gate on along-track mean later; per-episode flag uses |d|.
    # Static: both XY components must sit inside ±2 mm.
    if moving:
        ok = bool(abs(d_along) <= BIAS_MAX_M + 1e-12)
    else:
        ok = bool(
            abs(float(d_xy[0])) <= BIAS_MAX_M + 1e-12
            and abs(float(d_xy[1])) <= BIAS_MAX_M + 1e-12
        )
    out.update(
        {
            "closed": True,
            "moving": moving,
            "d_along_m": d_along,
            "d_x_m": float(d_xy[0]),
            "d_y_m": float(d_xy[1]),
            "d_cmd_along_m": d_cmd,
            "pass": ok,
        }
    )
    return out


def summarize(
    rows: list[dict],
    *,
    bias_max_m: float = BIAS_MAX_M,
    label: str,
    min_n: int = MIN_N,
) -> dict:
    """Group by speed × category; pass if every large enough bin's mean |d| ≤ budget."""
    closed = [r for r in rows if r.get("closed")]
    bins: dict[tuple, list[dict]] = defaultdict(list)
    for r in closed:
        spd = r.get("speed_m_s", float("nan"))
        key = (
            round(float(spd), 2) if np.isfinite(spd) else None,
            str(r.get("category") or "unknown"),
        )
        bins[key].append(r)

    bin_rows: list[dict] = []
    for key in sorted(bins, key=lambda k: ((k[0] is None), k[0] or -1.0, k[1])):
        grp = bins[key]
        moving = [g for g in grp if g.get("moving")]
        if moving:
            d = np.asarray([g["d_along_m"] for g in moving], dtype=np.float64)
            d = d[np.isfinite(d)]
            mean_d = float(np.mean(d)) if d.size else float("nan")
            cmd = np.asarray([g["d_cmd_along_m"] for g in moving], dtype=np.float64)
            cmd = cmd[np.isfinite(cmd)]
            mean_cmd = float(np.mean(cmd)) if cmd.size else float("nan")
            scored = bool(d.size >= min_n)
            passed = bool(scored and abs(mean_d) <= bias_max_m + 1e-12)
            bin_rows.append(
                {
                    "speed_m_s": key[0],
                    "category": key[1],
                    "n": int(d.size),
                    "kind": "along_track",
                    "d_mean_mm": 1000.0 * mean_d if np.isfinite(mean_d) else float("nan"),
                    "d_std_mm": 1000.0 * float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
                    "d_cmd_mean_mm": (
                        1000.0 * mean_cmd if np.isfinite(mean_cmd) else float("nan")
                    ),
                    "pass_rate": float(np.mean([1.0 if g["pass"] else 0.0 for g in moving])),
                    "pass": passed,
                    "scored": scored,
                }
            )
        else:
            # Static: mean offset on each XY axis must sit in ± budget.
            dx = np.asarray([g["d_x_m"] for g in grp], dtype=np.float64)
            dy = np.asarray([g["d_y_m"] for g in grp], dtype=np.float64)
            mx, my = float(np.mean(dx)), float(np.mean(dy))
            scored = bool(len(grp) >= min_n)
            passed = bool(
                scored
                and abs(mx) <= bias_max_m + 1e-12
                and abs(my) <= bias_max_m + 1e-12
            )
            bin_rows.append(
                {
                    "speed_m_s": key[0],
                    "category": key[1],
                    "n": len(grp),
                    "kind": "static_xy",
                    "d_mean_mm": 1000.0 * mx,  # report x; y in d_y_mean_mm
                    "d_y_mean_mm": 1000.0 * my,
                    "d_std_mm": 1000.0 * float(np.std(dx, ddof=1)) if dx.size > 1 else 0.0,
                    "d_cmd_mean_mm": float("nan"),
                    "pass_rate": float(np.mean([1.0 if g["pass"] else 0.0 for g in grp])),
                    "pass": passed,
                    "scored": scored,
                }
            )

    scored = [b for b in bin_rows if b["scored"]]
    overall = bool(scored) and all(bool(b["pass"]) for b in scored)

    # Pooled moving mean (headline number).
    all_m = np.asarray(
        [r["d_along_m"] for r in closed if r.get("moving")], dtype=np.float64
    )
    all_m = all_m[np.isfinite(all_m)]
    pooled_mean = float(np.mean(all_m)) if all_m.size else float("nan")

    return {
        "check": "hand_trail_bias",
        "label": label,
        "episodes": len(rows),
        "closed": len(closed),
        "bias_max_mm": 1000.0 * bias_max_m,
        "min_n": min_n,
        "pooled_d_along_mean_mm": (
            1000.0 * pooled_mean if np.isfinite(pooled_mean) else float("nan")
        ),
        "pooled_n_moving": int(all_m.size),
        "bins": bin_rows,
        "pass": overall,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Demo root with .h5 (repeatable). Default: A1_velocity + A5_recordings",
    )
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "B5_hand_trail" / "b5_report.json"),
    )
    args = ap.parse_args()
    bias = float(BIAS_MAX_M)

    roots = args.root or [
        str(ROOT / "data" / "eval" / "A1_velocity"),
        str(ROOT / "data" / "eval" / "A5_recordings"),
    ]

    reports: list[dict] = []
    for root in roots:
        root_p = Path(root)
        files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
        rows = [one(f) for f in files]
        rep = summarize(rows, bias_max_m=bias, label=str(root_p), min_n=int(args.min_n))
        reports.append(rep)
        print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(reports) and all(bool(r["pass"]) for r in reports)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bias_max_mm": 1000.0 * bias,
        "roots": reports,
        "pass": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B5: no hand-trailing bias",
        "",
        f"Gate: mean d at close = 0 ± {1000.0 * bias:.0f} mm "
        "(moving: along-track hand−object; static: each of d_x, d_y).",
        "",
        "Positive along-track d = hand ahead of the object. "
        "Command d (info) = action−object along the same axis.",
        "",
    ]
    for r in reports:
        lines.append(f"## `{Path(r['label']).name}` — {'PASS' if r['pass'] else 'FAIL'}")
        lines.append("")
        pooled = r.get("pooled_d_along_mean_mm", float("nan"))
        lines.append(
            f"Pooled moving mean d = {pooled:+.2f} mm "
            f"(n={r.get('pooled_n_moving', 0)})."
        )
        lines.append("")
        lines.append(
            "| speed cm/s | object | n | kind | mean d mm | σ mm | cmd mean mm | pass |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for b in r["bins"]:
            spd = "—" if b["speed_m_s"] is None else f"{100.0 * b['speed_m_s']:.0f}"
            scored = "" if b["scored"] else " (n<" + str(r["min_n"]) + ")"
            if b["kind"] == "static_xy":
                mean_s = (
                    f"x={b['d_mean_mm']:+.2f}, y={b.get('d_y_mean_mm', float('nan')):+.2f}"
                )
            else:
                mean_s = f"{b['d_mean_mm']:+.2f}"
            cmd = b.get("d_cmd_mean_mm", float("nan"))
            cmd_s = "—" if not np.isfinite(cmd) else f"{cmd:+.2f}"
            lines.append(
                f"| {spd} | {b['category']} | {b['n']} | {b['kind']} | "
                f"{mean_s} | {b['d_std_mm']:.2f} | {cmd_s} | "
                f"{'yes' if b['pass'] else 'no'}{scored} |"
            )
        lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    lines.append(
        "If |mean d| is a few mm with cmd mean near v·τ: DiffIK settles off-centre "
        "inside grasp_xyz_ok_m. Tighten the gate or dwell until |d| is small before close."
    )
    lines.append("")
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
