"""B3: speed match at close — |v_obj − v_hand| × t_close ≤ 2.5 mm.

Plain words: fingers take ~80 ms to shut. In that window the object must not
slide more than 2.5 mm across the pads. Same rule as ``control.close_guard``.

Isaac-free. For each demo at first CLOSE and over the finger-close window:

  slip_m = |v_hand_xy − v_obj_xy| × t_close

``v_hand`` from finite difference on ``ee_pos`` (what actually moved).
``v_obj`` from logged ``object_vel`` (else FD on ``object_pos``).
``t_close`` from ``robot.gripper_close_time_s`` (yaml, default 80 ms).

Also reports command-target FD slip (``action_pos``) so you can tell
"expert aimed wrong" from "DiffIK lagged".

Pass: every (speed × object) bin with ≥ ``min_n`` closes has median slip ≤ 2.5 mm.

Usage:
  PYTHONPATH=. python eval/b3_close_speed_match.py
  PYTHONPATH=. python eval/b3_close_speed_match.py --root data/eval/A1_velocity --root data/eval/A5_recordings
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

from control.close_guard import (  # noqa: E402
    DEFAULT_CLOSE_TIME_S,
    DEFAULT_SLACK_M,
    close_speed_check,
    max_speed_diff_m_s,
)
from interfaces.config import load_yaml  # noqa: E402
from sim.state_machine import STAGE_CLOSE  # noqa: E402

SLACK_M = DEFAULT_SLACK_M  # 2.5 mm
MIN_N = 3  # bins need this many closes to vote


def _t_close_s(cfg: dict) -> float:
    """Finger travel time (not the SM dwell)."""
    robot = cfg.get("robot") if isinstance(cfg.get("robot"), dict) else {}
    if "gripper_close_time_s" in robot:
        return float(robot["gripper_close_time_s"])
    if "gripper_close_time_ms" in robot:
        return float(robot["gripper_close_time_ms"]) * 1e-3
    return float(DEFAULT_CLOSE_TIME_S)


def slip_m(v_hand: np.ndarray, v_obj: np.ndarray, *, t_close: float) -> float:
    """Pad slide distance = |Δv_xy| × t_close (metres)."""
    vh = np.asarray(v_hand, dtype=np.float64).reshape(-1)
    vo = np.asarray(v_obj, dtype=np.float64).reshape(-1)
    n = min(2, vh.size, vo.size)  # XY only — slide across the pads
    return float(np.linalg.norm(vh[:n] - vo[:n])) * float(t_close)


def _fd_vel(pos: np.ndarray, i: int, dt: float) -> np.ndarray:
    """Finite-difference velocity at frame i (needs i ≥ 1)."""
    if i < 1 or dt <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return (np.asarray(pos[i], dtype=np.float64) - np.asarray(pos[i - 1], dtype=np.float64)) / dt


def one(
    path: str,
    *,
    t_close: float,
    slack_m: float = SLACK_M,
) -> dict:
    """One episode: max XY slip over the finger-close window at first CLOSE."""
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
        "slip_ee_mm": float("nan"),
        "slip_cmd_mm": float("nan"),
        "pass": False,
    }
    with h5py.File(path, "r") as h:
        if "stage" not in h or "ee_pos" not in h or "timestamp" not in h:
            return out
        stage = np.asarray(h["stage"][:], dtype=np.int64)
        ee = np.asarray(h["ee_pos"][:], dtype=np.float64)
        ts = np.asarray(h["timestamp"][:], dtype=np.float64)
        obj = np.asarray(h["object_pos"][:], dtype=np.float64) if "object_pos" in h else None
        ovel = np.asarray(h["object_vel"][:], dtype=np.float64) if "object_vel" in h else None
        act = np.asarray(h["action_pos"][:], dtype=np.float64) if "action_pos" in h else None
        if "speed_m_s" in h:
            out["speed_m_s"] = float(np.asarray(h["speed_m_s"]).reshape(-1)[0])

    hits = np.flatnonzero(stage == STAGE_CLOSE)
    if hits.size == 0 or len(ts) < 2:
        return out
    i0 = int(hits[0])
    dt = float(np.median(np.diff(ts)))
    if not np.isfinite(dt) or dt <= 0.0:
        return out
    n_win = max(1, int(round(float(t_close) / dt)))
    # Frames while fingers are shutting (stay in CLOSE).
    win = [i for i in range(i0, min(i0 + n_win, len(stage))) if int(stage[i]) == STAGE_CLOSE]
    if not win:
        return out

    slips_ee: list[float] = []
    slips_cmd: list[float] = []
    for i in win:
        if i < 1:
            continue
        vh = _fd_vel(ee, i, dt)
        if ovel is not None:
            vo = np.asarray(ovel[i], dtype=np.float64).reshape(3)
        elif obj is not None:
            vo = _fd_vel(obj, i, dt)
        else:
            continue
        slips_ee.append(slip_m(vh, vo, t_close=t_close))
        if act is not None:
            # Command track: how the target pose moved (expert intent).
            vc = _fd_vel(act, i, dt)
            slips_cmd.append(slip_m(vc, vo, t_close=t_close))

    if not slips_ee:
        return out
    slip_ee = float(max(slips_ee))
    slip_cmd = float(max(slips_cmd)) if slips_cmd else float("nan")
    # Same check the live guard uses (XY padded into 3-vector for the helper).
    vh0 = _fd_vel(ee, win[0], dt) if win[0] >= 1 else np.zeros(3)
    if ovel is not None:
        vo0 = np.asarray(ovel[win[0]], dtype=np.float64).reshape(3)
    else:
        vo0 = _fd_vel(obj, win[0], dt)  # type: ignore[arg-type]
    chk = close_speed_check(
        vh0[:2],
        vo0[:2],
        slack_m=slack_m,
        close_time_s=t_close,
    )
    out.update(
        {
            "closed": True,
            "slip_ee_mm": 1000.0 * slip_ee,
            "slip_cmd_mm": 1000.0 * slip_cmd if np.isfinite(slip_cmd) else float("nan"),
            "speed_diff_m_s": float(chk.speed_diff_m_s),
            "limit_m_s": float(chk.limit_m_s),
            "n_frames": len(slips_ee),
            "pass": bool(slip_ee <= slack_m + 1e-12),
        }
    )
    return out


def summarize(
    rows: list[dict],
    *,
    slack_m: float,
    t_close: float,
    label: str,
    min_n: int = MIN_N,
) -> dict:
    """Group by speed × category; pass if every large enough bin's median ≤ slack."""
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
        ee = np.asarray([g["slip_ee_mm"] for g in grp], dtype=np.float64)
        cmd = np.asarray([g["slip_cmd_mm"] for g in grp], dtype=np.float64)
        cmd = cmd[np.isfinite(cmd)]
        med = float(np.median(ee))
        bin_rows.append(
            {
                "speed_m_s": key[0],
                "category": key[1],
                "n": len(grp),
                "slip_ee_med_mm": med,
                "slip_ee_p90_mm": float(np.percentile(ee, 90)),
                "slip_ee_max_mm": float(np.max(ee)),
                "slip_cmd_med_mm": float(np.median(cmd)) if cmd.size else float("nan"),
                "pass_rate": float(np.mean([1.0 if g["pass"] else 0.0 for g in grp])),
                "pass": bool(len(grp) >= min_n and med <= 1000.0 * slack_m + 1e-9),
                "scored": bool(len(grp) >= min_n),
            }
        )

    scored = [b for b in bin_rows if b["scored"]]
    overall = bool(scored) and all(bool(b["pass"]) for b in scored)
    return {
        "check": "close_speed_match",
        "label": label,
        "episodes": len(rows),
        "closed": len(closed),
        "slack_mm": 1000.0 * slack_m,
        "t_close_s": float(t_close),
        "limit_cm_s": 100.0 * max_speed_diff_m_s(slack_m=slack_m, close_time_s=t_close),
        "min_n": min_n,
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
        default=str(ROOT / "data" / "eval" / "B3_close_speed" / "b3_report.json"),
    )
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    t_close = _t_close_s(cfg)
    slack = float(SLACK_M)

    roots = args.root or [
        str(ROOT / "data" / "eval" / "A1_velocity"),
        str(ROOT / "data" / "eval" / "A5_recordings"),
    ]

    reports: list[dict] = []
    for root in roots:
        root_p = Path(root)
        files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
        rows = [one(f, t_close=t_close, slack_m=slack) for f in files]
        rep = summarize(
            rows, slack_m=slack, t_close=t_close, label=str(root_p), min_n=int(args.min_n)
        )
        reports.append(rep)
        print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(reports) and all(bool(r["pass"]) for r in reports)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slack_mm": 1000.0 * slack,
        "t_close_s": t_close,
        "limit_cm_s": 100.0 * max_speed_diff_m_s(slack_m=slack, close_time_s=t_close),
        "roots": reports,
        "pass": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B3: speed matching at close",
        "",
        f"Gate: |v_obj − v_hand|_xy × t_close ≤ {1000.0 * slack:.1f} mm "
        f"(t_close={1000.0 * t_close:.0f} ms → |Δv| ≤ "
        f"{100.0 * max_speed_diff_m_s(slack_m=slack, close_time_s=t_close):.1f} cm/s).",
        "",
        "Hand speed = FD of `ee_pos`. Command slip (info) = FD of `action_pos`.",
        "",
    ]
    for r in reports:
        lines.append(f"## `{Path(r['label']).name}` — {'PASS' if r['pass'] else 'FAIL'}")
        lines.append("")
        lines.append(
            "| speed cm/s | object | n | ee med/p90/max mm | cmd med mm | pass |"
        )
        lines.append("|---|---|---|---|---|---|")
        for b in r["bins"]:
            spd = "—" if b["speed_m_s"] is None else f"{100.0 * b['speed_m_s']:.0f}"
            scored = "" if b["scored"] else " (n<" + str(r["min_n"]) + ")"
            lines.append(
                f"| {spd} | {b['category']} | {b['n']} | "
                f"{b['slip_ee_med_mm']:.2f} / {b['slip_ee_p90_mm']:.2f} / "
                f"{b['slip_ee_max_mm']:.2f} | "
                f"{b['slip_cmd_med_mm']:.2f} | "
                f"{'yes' if b['pass'] else 'no'}{scored} |"
            )
        lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    lines.append(
        "If ee slip fails but cmd slip is tiny: expert aims with the object; "
        "DiffIK lag leaves the real hand behind during the 80 ms close."
    )
    lines.append("")
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
