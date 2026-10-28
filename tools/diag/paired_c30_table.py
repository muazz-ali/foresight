#!/usr/bin/env python3
"""Paired C30/C31/C33 table for egg seeds (default 1000–1004).

Scores each episode with ``episode_frame_debug`` (hand-eye + object-now,
Isaac ``ee_vel`` when logged). Prints one markdown table.

Usage:
  PYTHONPATH=. python tools/diag/paired_c30_table.py data/eval/C1_c30_c31_c33_fix6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.diag.episode_frame_debug import analyze


def _load_he(h5: Path) -> np.ndarray | None:
    side = h5.with_suffix(".json")
    if not side.is_file():
        return None
    try:
        meta = json.loads(side.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "hand_eye_m" in meta:
        return np.asarray(meta["hand_eye_m"], dtype=np.float64)
    return None


def cell_means(root: Path, cell: str, seeds: list[int]) -> dict:
    """Mean metrics over egg episodes for one cell."""
    rows = []
    for s in seeds:
        matches = sorted((root / "cells" / cell).glob(f"{cell}_*_s{s}_egg.h5"))
        if not matches:
            continue
        h5 = matches[0]
        rep = analyze(h5, hand_eye=_load_he(h5))
        r = rep.get("ride") or {}
        close = rep.get("close") or {}
        rows.append(
            {
                "success": 1.0 if rep["success"] else 0.0,
                "ride_n": float(r.get("n") or 0),
                "park_ticks": float(rep.get("park_ticks") or 0),
                "frac_both": float(r["frac_both"]) if r else float("nan"),
                "frac_v": float(r["frac_v"]) if r else float("nan"),
                "hand_speed_cms": float(r["hand_speed_cms_med"]) if r else float("nan"),
                "aim_along_mm": float(r["aim_along_mm_med"]) if r else float("nan"),
                "close_t_s": float(close["t_s"]) if close else float("nan"),
            }
        )
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "success_n": int(sum(r["success"] for r in rows))}
    for k in (
        "ride_n",
        "park_ticks",
        "frac_both",
        "frac_v",
        "hand_speed_cms",
        "aim_along_mm",
        "close_t_s",
    ):
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[k] = float(np.nanmean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    args = ap.parse_args()
    cells = ["C30", "C31", "C33"]
    stats = {c: cell_means(args.root, c, list(args.seeds)) for c in cells}
    print(
        "| metric | C30 (Δ=0) | C31 (Δ=100) | C33 (Δ=300) |"
    )
    print("|---|---:|---:|---:|")
    def fmt(c, key, nd=2):
        s = stats[c]
        if s.get("n", 0) == 0:
            return "—"
        if key == "success":
            return f"{s['success_n']}/{s['n']}"
        v = s.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{nd}f}"

    rows = [
        ("success", "success", 0),
        ("ride_n", "ride_n", 1),
        ("park_ticks", "park_ticks", 1),
        ("frac_both", "frac_both", 3),
        ("frac_v", "frac_v", 3),
        ("hand speed (ride) cm/s", "hand_speed_cms", 1),
        ("aim_along mm", "aim_along_mm", 1),
        ("close time s", "close_t_s", 2),
    ]
    for label, key, nd in rows:
        if key == "success":
            print(
                f"| {label} | {fmt('C30', 'success')} | {fmt('C31', 'success')} | {fmt('C33', 'success')} |"
            )
        else:
            print(
                f"| {label} | {fmt('C30', key, nd)} | {fmt('C31', key, nd)} | {fmt('C33', key, nd)} |"
            )


if __name__ == "__main__":
    main()
