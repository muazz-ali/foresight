#!/usr/bin/env python3
"""Paired C30/C31/C33 tables for every object shard under an all-objects OUT.

Reads ``OUT/<object>/cells/C3*/``. Object folder name is the product label
(run_b1_grid may still stamp egg/lemon/onion in the .h5 stem).

Writes one markdown report with the same metric grid as the egg scout:
success, ride_n, park_ticks, frac_both, frac_v, hand speed, aim_along, close time.

Usage:
  PYTHONPATH=. python tools/diag/report_c30_all_objects.py \\
    data/eval/C1_c30_all_objects -o data/eval/C1_c30_all_objects/REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.diag.episode_frame_debug import analyze

CELLS = ("C30", "C31", "C33")
CELL_LABEL = {
    "C30": "C30 (Δ=0)",
    "C31": "C31 (Δ=100)",
    "C33": "C33 (Δ=300)",
}
DEFAULT_OBJECTS = (
    "apple,avocado,egg,kiwi,lemon,lime,onion,orange,peach,tangerine"
)


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


def _seeds_for_block(obj_index: int, n_per: int) -> list[int]:
    """Same mapping as launch_c30_all_objects.sh: seed_id = 1000 + index."""
    s0 = obj_index * n_per
    return [1000 + s0 + k for k in range(n_per)]


def cell_means(obj_root: Path, cell: str, seeds: list[int]) -> dict:
    """Mean metrics over one object's episodes in one cell."""
    rows = []
    cell_dir = obj_root / "cells" / cell
    obj_name = obj_root.name
    for s in seeds:
        # Prefer stem with the real object name; fall back to any seed match.
        matches = sorted(cell_dir.glob(f"{cell}_*_s{s}_{obj_name}.h5"))
        if not matches:
            matches = sorted(cell_dir.glob(f"{cell}_*_s{s}_*.h5"))
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
                "seed": int(s),
                "mode": (
                    "ok"
                    if rep["success"]
                    else (
                        (rep.get("stop_before_close") and "unreachable")
                        or "never_closed"
                    )
                ),
            }
        )
    if not rows:
        return {"n": 0, "rows": []}
    out: dict = {
        "n": len(rows),
        "success_n": int(sum(r["success"] for r in rows)),
        "rows": rows,
    }
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


def _fmt(stats: dict, key: str, nd: int = 2) -> str:
    if stats.get("n", 0) == 0:
        return "—"
    if key == "success":
        return f"{stats['success_n']}/{stats['n']}"
    v = stats.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def table_md(stats_by_cell: dict[str, dict]) -> str:
    """One paired markdown table (same rows as the egg scout)."""
    lines = [
        "| metric | C30 (Δ=0) | C31 (Δ=100) | C33 (Δ=300) |",
        "|---|---:|---:|---:|",
    ]
    specs = [
        ("success", "success", 0),
        ("ride_n", "ride_n", 1),
        ("park_ticks", "park_ticks", 1),
        ("frac_both", "frac_both", 3),
        ("frac_v", "frac_v", 3),
        ("hand speed (ride) cm/s", "hand_speed_cms", 1),
        ("aim_along mm", "aim_along_mm", 1),
        ("close time s", "close_t_s", 2),
    ]
    for label, key, nd in specs:
        lines.append(
            "| "
            + " | ".join(
                [label]
                + [_fmt(stats_by_cell[c], key, nd) for c in CELLS]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(root: Path, objects: list[str], n_per: int) -> str:
    """Markdown report: summary scoreboard + per-object paired grids."""
    blocks: list[str] = [
        "# C30 / C31 / C33 — all objects scout",
        "",
        f"**Root:** `{root}`  ",
        f"**Cells:** C30 (Δ=0), C31 (Δ=100), C33 (Δ=300) @ 30 cm/s  ",
        f"**Seeds per object:** {n_per} (ids 1000+k·{n_per} …)  ",
        f"**Objects:** {', '.join(objects)}",
        "",
        "## Scoreboard (success)",
        "",
        "| object | seeds | C30 | C31 | C33 | note |",
        "|---|---|---:|---:|---:|---|",
    ]
    per_obj: dict[str, dict[str, dict]] = {}
    for oi, obj in enumerate(objects):
        seeds = _seeds_for_block(oi, n_per)
        obj_root = root / obj
        stats = {c: cell_means(obj_root, c, seeds) for c in CELLS}
        per_obj[obj] = stats
        seed_lo, seed_hi = seeds[0], seeds[-1]
        n_any = max((stats[c].get("n", 0) for c in CELLS), default=0)
        note = "ok" if n_any > 0 else "no episodes (load/filter fail — see logs/)"
        blocks.append(
            f"| {obj} | {seed_lo}–{seed_hi} | "
            + " | ".join(_fmt(stats[c], "success") for c in CELLS)
            + f" | {note} |"
        )

    blocks += ["", "## Per-object paired grids", ""]
    for obj, stats in per_obj.items():
        seeds = _seeds_for_block(list(objects).index(obj), n_per)
        blocks += [
            f"### {obj} (seeds {seeds[0]}–{seeds[-1]})",
            "",
            table_md(stats),
            "",
        ]

    # Machine-readable twin for later agents.
    summary = {
        "root": str(root),
        "n_per": n_per,
        "objects": {
            obj: {
                "seeds": _seeds_for_block(i, n_per),
                "cells": {
                    c: {
                        k: v
                        for k, v in stats.items()
                        if k != "rows"
                    }
                    for c, stats in per_obj[obj].items()
                },
            }
            for i, obj in enumerate(objects)
        },
    }
    (root / "report_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return "\n".join(blocks).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--objects", type=str, default=DEFAULT_OBJECTS)
    ap.add_argument("--n-per", type=int, default=5)
    args = ap.parse_args()
    objects = [x.strip() for x in str(args.objects).split(",") if x.strip()]
    text = build_report(args.root, objects, int(args.n_per))
    out = args.out or (args.root / "REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
