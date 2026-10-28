#!/usr/bin/env python3
"""Write the C30 / C31 / C33 paired grid report for all objects in a run.

Same metrics as the egg scout table: success, ride_n, park_ticks, frac_both,
frac_v, hand speed, aim_along, close time. One grid per object + a pooled grid.

Does not change sim/control — read-only over episode ``.h5`` / sidecar JSON.

Usage:
  PYTHONPATH=. python tools/diag/write_c30_paired_report.py \\
    data/eval/C1_c30_c31_c33_objects_scout \\
    -o data/eval/C1_c30_c31_c33_objects_scout/PAIRED_REPORT.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.diag.episode_frame_debug import analyze

CELLS = ("C30", "C31", "C33")
CELL_LABEL = {
    "C30": "C30 (Δ=0)",
    "C31": "C31 (Δ=100)",
    "C33": "C33 (Δ=300)",
}
OBJECTS = ("egg", "lemon", "onion")
METRIC_ROWS = (
    ("success", "success", 0),
    ("ride_n", "ride_n", 1),
    ("park_ticks", "park_ticks", 1),
    ("frac_both", "frac_both", 3),
    ("frac_v", "frac_v", 3),
    ("hand speed (ride) cm/s", "hand_speed_cms", 1),
    ("aim_along mm", "aim_along_mm", 1),
    ("close time s", "close_t_s", 2),
)


def _load_he(h5: Path) -> np.ndarray | None:
    side = h5.with_suffix(".json")
    if not side.is_file():
        return None
    try:
        meta = json.loads(side.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if "hand_eye_m" in meta:
        return np.asarray(meta["hand_eye_m"], dtype=np.float64)
    return None


def _score_h5(h5: Path) -> dict:
    """One episode → flat metrics used in the paired grid."""
    rep = analyze(h5, hand_eye=_load_he(h5))
    r = rep.get("ride") or {}
    close = rep.get("close") or {}
    return {
        "h5": str(h5),
        "seed": rep.get("seed"),
        "success": bool(rep["success"]),
        "ride_n": float(r.get("n") or 0),
        "park_ticks": float(rep.get("park_ticks") or 0),
        "frac_both": float(r["frac_both"]) if r else float("nan"),
        "frac_v": float(r["frac_v"]) if r else float("nan"),
        "hand_speed_cms": float(r["hand_speed_cms_med"]) if r else float("nan"),
        "aim_along_mm": float(r["aim_along_mm_med"]) if r else float("nan"),
        "close_t_s": float(close["t_s"]) if close else float("nan"),
        "mode": (
            "ok"
            if rep["success"]
            else (
                "unreachable"
                if rep.get("stop_before_close")
                else "never_closed"
            )
        ),
    }


def discover_episodes(root: Path) -> list[dict]:
    """Find every C30/C31/C33 egg|lemon|onion ``.h5`` under ``cells/``."""
    rows: list[dict] = []
    cells_dir = root / "cells"
    if not cells_dir.is_dir():
        return rows
    for cell in CELLS:
        d = cells_dir / cell
        if not d.is_dir():
            continue
        for h5 in sorted(d.glob(f"{cell}_*_s*_*.h5")):
            # name: C30_v30_d000_s1000_egg.h5
            stem = h5.stem
            parts = stem.split("_")
            obj = parts[-1] if parts else ""
            if obj not in OBJECTS:
                continue
            try:
                seed = int(next(p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()))
            except StopIteration:
                seed = None
            scored = _score_h5(h5)
            scored.update({"cell": cell, "object": obj, "seed": seed if seed is not None else scored["seed"]})
            rows.append(scored)
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Mean metrics + success count for one slice of episodes."""
    if not rows:
        return {"n": 0, "success_n": 0}
    out: dict = {
        "n": len(rows),
        "success_n": int(sum(1 for r in rows if r["success"])),
        "seeds": sorted({int(r["seed"]) for r in rows if r.get("seed") is not None}),
        "modes": {},
    }
    for r in rows:
        m = str(r.get("mode") or "?")
        out["modes"][m] = int(out["modes"].get(m, 0)) + 1
    for _, key, _ in METRIC_ROWS:
        if key == "success":
            continue
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(np.nanmean(vals)) if vals.size else float("nan")
    return out


def fmt_cell(stats: dict, key: str, nd: int) -> str:
    if stats.get("n", 0) == 0:
        return "—"
    if key == "success":
        return f"{stats['success_n']}/{stats['n']}"
    v = stats.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def md_grid(title: str, by_cell: dict[str, dict]) -> list[str]:
    """One markdown paired table (C30 / C31 / C33)."""
    lines = [
        f"### {title}",
        "",
        "| metric | " + " | ".join(CELL_LABEL[c] for c in CELLS) + " |",
        "|---|" + "|".join(["---:"] * len(CELLS)) + "|",
    ]
    for label, key, nd in METRIC_ROWS:
        cells = " | ".join(fmt_cell(by_cell.get(c, {"n": 0}), key, nd) for c in CELLS)
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    return lines


def write_report(root: Path, out_md: Path) -> dict:
    """Score all episodes, write markdown + JSON sidecar next to it."""
    rows = discover_episodes(root)
    # object → cell → rows
    by_obj_cell: dict[str, dict[str, list]] = {o: {c: [] for c in CELLS} for o in OBJECTS}
    pooled: dict[str, list] = {c: [] for c in CELLS}
    for r in rows:
        obj, cell = r["object"], r["cell"]
        if obj in by_obj_cell and cell in by_obj_cell[obj]:
            by_obj_cell[obj][cell].append(r)
        if cell in pooled:
            pooled[cell].append(r)

    lines = [
        "# C30 / C31 / C33 paired report",
        "",
        f"**Root:** `{root}`",
        f"**Episodes scored:** {len(rows)}",
        "**Cells:** C30 Δ=0 · C31 Δ=100 · C33 Δ=300 @ 30 cm/s",
        "**Sign:** + = hand/aim ahead of the object along v",
        "**Scorer:** expert-matched (1 mm hand-eye, object-now; Isaac `ee_vel` when logged)",
        "",
        "## Pooled (all objects)",
        "",
    ]
    pooled_stats = {c: aggregate(pooled[c]) for c in CELLS}
    lines.extend(md_grid("All objects", pooled_stats))

    lines.append("## Per object")
    lines.append("")
    per_obj_stats: dict[str, dict[str, dict]] = {}
    for obj in OBJECTS:
        stats = {c: aggregate(by_obj_cell[obj][c]) for c in CELLS}
        per_obj_stats[obj] = stats
        n = sum(s.get("n", 0) for s in stats.values())
        seeds = sorted({s for c in CELLS for s in (stats[c].get("seeds") or [])})
        title = f"{obj} (n_eps={n}, seeds={seeds})"
        lines.extend(md_grid(title, stats))

    # Per-seed success strip for quick scan.
    lines.append("## Per-seed success")
    lines.append("")
    lines.append("| object | seed | C30 | C31 | C33 |")
    lines.append("|---|---:|:---:|:---:|:---:|")
    # index by (obj, seed, cell)
    idx: dict[tuple, dict] = {}
    for r in rows:
        idx[(r["object"], int(r["seed"]), r["cell"])] = r
    for obj in OBJECTS:
        seeds = sorted({int(r["seed"]) for r in rows if r["object"] == obj and r.get("seed") is not None})
        for seed in seeds:
            marks = []
            for cell in CELLS:
                ep = idx.get((obj, seed, cell))
                if ep is None:
                    marks.append("—")
                elif ep["success"]:
                    marks.append("ok")
                else:
                    marks.append(str(ep.get("mode") or "fail"))
            lines.append(f"| {obj} | {seed} | " + " | ".join(marks) + " |")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out_md.write_text(text)

    blob = {
        "root": str(root),
        "n_episodes": len(rows),
        "pooled": pooled_stats,
        "per_object": per_obj_stats,
        "episodes": rows,
    }
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.root / "PAIRED_REPORT.md")
    blob = write_report(args.root, out)
    print(f"wrote {out} ({blob['n_episodes']} episodes)")
    # Also print pooled grid to stdout for the terminal.
    for line in md_grid("All objects (stdout)", {c: blob["pooled"][c] for c in CELLS}):
        print(line)


if __name__ == "__main__":
    main()
