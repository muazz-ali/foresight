"""B1: oracle expert ≥ 95% success in every grid cell.

Plain words: before you trust a policy on the speed×delay table, the scripted
expert (GT object state) must already win almost every try in each cell.
Otherwise a policy miss is just a bad teacher.

Isaac-free. Reads logged expert episodes (``.h5`` + sidecar ``.json``) or a
``smoke_collect_report.json`` from ``scripts/smoke_claim_grid.py``.

Grids (``--grid``):
  speed  — 0 / 0.10 / 0.20 / 0.30 m/s (delay ignored) — good for A1-style dumps
  smoke  — smoke_claim_grid: speeds × delays 0 / 100 / 200 ms
  plan   — foresight_plan.md claim cells (0.10/0.20/0.30 × 100/200/300 ms)
  none   — only cells that have data (no missing-cell fail)

Pass: every required cell has n ≥ ``min_n`` and success rate ≥ 95%.
Prints Wilson 95% intervals (n=20 is scout; n≥80 to claim).

Usage:
  PYTHONPATH=. python eval/b1_oracle_grid.py
  PYTHONPATH=. python eval/b1_oracle_grid.py --grid smoke --root data/claim_smoke
  PYTHONPATH=. python eval/b1_oracle_grid.py --grid plan --min-n 20
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RATE_MIN = 0.95
MIN_N = 20  # scout; plan § claims want n≥80
# Folder pattern from scripts/smoke_claim_grid.py
_CELL_DIR_RE = re.compile(r"spd_(\d+)_del_(\d+)$")

SPEED_GRID = (0.0, 0.10, 0.20, 0.30)
SMOKE_SPEEDS = (0.0, 0.10, 0.15, 0.20)
SMOKE_DELAYS_S = (0.0, 0.100, 0.200)
PLAN_CELLS = (
    (0.10, 0.100),
    (0.10, 0.300),
    (0.20, 0.100),
    (0.20, 0.300),
    (0.30, 0.200),
    (0.30, 0.300),
)
PLAN_DELAYS_S = (0.0, 0.100, 0.200, 0.300)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes out of n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def required_cells(grid: str) -> list[tuple[float | None, float | None]]:
    """List of (speed_m_s, delay_s) the gate must cover. delay=None → speed-only."""
    g = str(grid).lower()
    if g == "none":
        return []
    if g == "speed":
        return [(float(s), None) for s in SPEED_GRID]
    if g == "smoke":
        return [(float(s), float(d)) for d in SMOKE_DELAYS_S for s in SMOKE_SPEEDS]
    if g == "plan":
        return [(float(s), float(d)) for s, d in PLAN_CELLS]
    raise ValueError(f"unknown --grid {grid!r} (speed|smoke|plan|none)")


def snap_delay_s(delay_s: float, candidates: tuple[float, ...]) -> float | None:
    """Nearest grid delay if within half the minimum spacing; else None."""
    if not candidates:
        return None
    d = float(delay_s)
    best = min(candidates, key=lambda c: abs(float(c) - d))
    gaps = sorted(set(float(c) for c in candidates))
    if len(gaps) >= 2:
        half = 0.5 * min(gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1))
    else:
        half = 0.050
    if abs(float(best) - d) <= half + 1e-12:
        return float(best)
    return None


def _delay_candidates(grid: str) -> tuple[float, ...]:
    g = str(grid).lower()
    if g == "smoke":
        return SMOKE_DELAYS_S
    if g == "plan":
        return PLAN_DELAYS_S
    return PLAN_DELAYS_S  # snap helpers even for speed/none


def _parse_cell_dir(path: Path) -> tuple[float, float] | None:
    """Parse spd_XX_del_YYY folder name → (speed, delay_s)."""
    m = _CELL_DIR_RE.search(path.name)
    if not m:
        return None
    return (int(m.group(1)) / 100.0, int(m.group(2)) / 1000.0)


def one_episode(path: str, *, grid: str) -> dict | None:
    """Load one expert episode; return cell keys + success, or None if unusable."""
    p = Path(path)
    meta: dict = {}
    jp = p.with_suffix(".json")
    if jp.exists():
        raw = json.loads(jp.read_text())
        if isinstance(raw, dict):
            meta = raw

    speed = meta.get("smoke_speed", meta.get("speed_m_s"))
    delay = meta.get("smoke_delay_s")
    success = meta.get("success")

    parent_cell = _parse_cell_dir(p.parent)
    if parent_cell is not None:
        speed = parent_cell[0] if speed is None else speed
        delay = parent_cell[1] if delay is None else delay

    with h5py.File(path, "r") as h:
        if speed is None and "speed_m_s" in h:
            speed = float(np.asarray(h["speed_m_s"][...]).reshape(-1)[0])
        if success is None and "success" in h:
            success = bool(int(np.asarray(h["success"][...]).reshape(-1)[0]))
        if delay is None and "applied_delta_s" in h:
            ad = np.asarray(h["applied_delta_s"][:], dtype=np.float64)
            ad = ad[np.isfinite(ad)]
            if ad.size:
                delay = float(np.median(ad))
        elif delay is None and "reported_delta_s" in h:
            rd = np.asarray(h["reported_delta_s"][:], dtype=np.float64)
            rd = rd[np.isfinite(rd)]
            if rd.size:
                delay = float(np.median(rd))

    if speed is None or success is None:
        return None
    speed = round(float(speed), 2)
    g = str(grid).lower()
    if g == "speed":
        return {
            "path": path,
            "speed_m_s": speed,
            "delay_s": None,
            "success": bool(success),
        }
    raw_delay = float(delay) if delay is not None and np.isfinite(delay) else 0.0
    snapped = snap_delay_s(raw_delay, _delay_candidates(g if g != "none" else "plan"))
    return {
        "path": path,
        "speed_m_s": speed,
        "delay_s": snapped if snapped is not None else round(raw_delay, 3),
        "delay_raw_s": raw_delay,
        "success": bool(success),
    }


def load_smoke_report(path: Path) -> list[dict] | None:
    """Expand smoke_collect_report.json cells into synthetic episode rows."""
    if not path.exists():
        return None
    rep = json.loads(path.read_text())
    if not isinstance(rep, dict) or "cells" not in rep:
        return None
    rows: list[dict] = []
    for cell in rep["cells"]:
        spd = round(float(cell["speed"]), 2)
        delay = float(cell["delay_s"])
        wins = int(cell["wins"])
        attempts = int(cell["attempts"])
        for i in range(attempts):
            rows.append(
                {
                    "path": f"{path}#cell_{spd}_{delay}_{i}",
                    "speed_m_s": spd,
                    "delay_s": delay,
                    "success": bool(i < wins),
                    "from_report": True,
                }
            )
    return rows


def summarize(
    rows: list[dict],
    *,
    grid: str,
    label: str,
    rate_min: float = RATE_MIN,
    min_n: int = MIN_N,
) -> dict:
    """Group by cell; pass if every required cell meets rate and n."""
    buckets: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        spd = r.get("speed_m_s")
        if spd is None:
            continue
        key_spd = round(float(spd), 2)
        delay = r.get("delay_s")
        key = (key_spd, None if delay is None else round(float(delay), 3))
        buckets[key].append(bool(r.get("success")))

    cells: list[dict] = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1] is None, k[1] or -1.0)):
        ys = buckets[key]
        k = int(sum(ys))
        n = int(len(ys))
        rate = k / n if n else float("nan")
        lo, hi = wilson(k, n)
        cells.append(
            {
                "speed_m_s": key[0],
                "delay_s": key[1],
                "successes": k,
                "n": n,
                "rate": rate,
                "wilson95": [lo, hi],
                "pass": bool(n >= min_n and rate + 1e-15 >= float(rate_min)),
                "scored": bool(n >= min_n),
            }
        )

    req = required_cells(grid)
    by_key = {(c["speed_m_s"], c["delay_s"]): c for c in cells}
    missing: list[dict] = []
    req_rows: list[dict] = []
    for spd, delay in req:
        key = (round(float(spd), 2), None if delay is None else round(float(delay), 3))
        if key not in by_key:
            missing.append({"speed_m_s": key[0], "delay_s": key[1], "n": 0})
            req_rows.append(
                {
                    "speed_m_s": key[0],
                    "delay_s": key[1],
                    "successes": 0,
                    "n": 0,
                    "rate": None,
                    "wilson95": [None, None],
                    "pass": False,
                    "scored": False,
                    "missing": True,
                }
            )
        else:
            req_rows.append({**by_key[key], "missing": False})

    if req:
        gate_cells = req_rows
        all_pass = bool(gate_cells) and all(bool(c["pass"]) for c in gate_cells)
    else:
        gate_cells = cells
        scored = [c for c in cells if c["scored"]]
        all_pass = bool(scored) and all(bool(c["pass"]) for c in scored)

    return {
        "check": "oracle_grid",
        "label": label,
        "grid": grid,
        "rate_min": float(rate_min),
        "min_n": int(min_n),
        "episodes": len(rows),
        "cells": cells,
        "required": gate_cells if req else [],
        "missing": missing,
        "pass": all_pass,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Demo root or smoke_collect_report.json parent (repeatable)",
    )
    ap.add_argument(
        "--grid",
        choices=("speed", "smoke", "plan", "none"),
        default="speed",
        help="Required cell set (default: speed-only)",
    )
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--rate-min", type=float, default=RATE_MIN)
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "B1_oracle_grid" / "b1_report.json"),
    )
    args = ap.parse_args()

    roots = args.root or [str(ROOT / "data" / "eval" / "A1_velocity")]
    rows: list[dict] = []
    used: list[str] = []
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            continue
        used.append(str(root_p))
        smoke_rep = root_p / "smoke_collect_report.json"
        if smoke_rep.exists():
            loaded = load_smoke_report(smoke_rep)
            if loaded:
                rows.extend(loaded)
        if root_p.is_file() and root_p.name == "smoke_collect_report.json":
            loaded = load_smoke_report(root_p)
            if loaded:
                rows.extend(loaded)
            continue
        files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
        for f in files:
            ep = one_episode(f, grid=str(args.grid))
            if ep is not None:
                rows.append(ep)

    label = "+".join(Path(u).name for u in used) if used else "empty"
    rep = summarize(
        rows,
        grid=str(args.grid),
        label=label,
        rate_min=float(args.rate_min),
        min_n=int(args.min_n),
    )
    print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(rep["pass"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": args.grid,
        "rate_min": float(args.rate_min),
        "min_n": int(args.min_n),
        "roots": used,
        "report": rep,
        "pass": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B1: oracle expert ≥ 95% in every grid cell",
        "",
        f"Gate: rate ≥ {100.0 * float(args.rate_min):.0f}% with n ≥ {int(args.min_n)} "
        f"(grid=`{args.grid}`). Wilson 95% shown for scout vs claim.",
        f"Roots: {', '.join(used) if used else '(none)'}",
        "",
        f"## `{label}` — {'PASS' if all_pass else 'FAIL'}",
        "",
        "| speed cm/s | delay ms | n | succ | rate | Wilson 95% | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    show = rep["required"] if rep["required"] else rep["cells"]
    for c in show:
        spd = f"{100.0 * c['speed_m_s']:.0f}"
        delay = "—" if c["delay_s"] is None else f"{1000.0 * c['delay_s']:.0f}"
        if c.get("missing"):
            lines.append(f"| {spd} | {delay} | 0 | — | — | — | missing |")
            continue
        lo, hi = c["wilson95"]
        tag = "" if c["scored"] else " (n small)"
        lines.append(
            f"| {spd} | {delay} | {c['n']} | {c['successes']}/{c['n']} | "
            f"{100.0 * c['rate']:.1f}% | {100.0 * lo:.0f}–{100.0 * hi:.0f}% | "
            f"{'yes' if c['pass'] else 'no'}{tag} |"
        )
    if rep.get("missing"):
        lines.append("")
        lines.append(
            f"Missing cells: {len(rep['missing'])}. "
            "Collect with `scripts/smoke_claim_grid.py` or pass `--grid none`."
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
