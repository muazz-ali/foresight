#!/usr/bin/env python3
"""Flatness check: per-link medians across objects and Δ (not success @ n=5).

Reads LINK_REPORT-style episode rows from an eval root. Reports whether
hand−object along/across, |Δv|, and ride ticks-to-close are equal across
objects and across Δ within a tolerance.

Usage:
  PYTHONPATH=. python tools/diag/flatness_report.py data/eval/C1_c30_all_objects_v2_final \\
    -o data/eval/C1_c30_all_objects_v2_final/FLATNESS.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.diag.report_link_chain import CELLS, DELTA, OBJECTS, _agg, _collect


def build(root: Path, *, tol_along_mm: float = 2.0, tol_dv_cms: float = 1.0) -> str:
    rows = [r for r in _collect(root) if "error" not in r]
    by_obj_d: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_obj_d[(r["object"], r["delta_ms"])].append(r)

    lines = [
        "# Flatness report — link medians across objects × Δ",
        "",
        f"**Root:** `{root}`  ",
        f"**Pass band:** |along| spread ≤ {tol_along_mm} mm, |Δv| spread ≤ {tol_dv_cms} cm/s "
        "across objects at fixed Δ (and across Δ for the same object).",
        "",
        "## Per object × Δ (episode medians → cell median)",
        "",
        "| object | Δ | n | succ | along mm | across mm | |Δv| cm/s | "
        "hand−aim mm | ticks→close | frac_both |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cell = {}
    for obj in OBJECTS:
        for d in (0, 100, 300):
            rs = by_obj_d.get((obj, d), [])
            if not rs:
                continue
            along, _ = _agg([_pull_along(r) for r in rs])
            across, _ = _agg([_pull_across(r) for r in rs])
            dv, _ = _agg([r.get("l3_dv") for r in rs])
            ha, _ = _agg([r.get("l3_hand_aim") for r in rs])
            ticks, _ = _agg([r.get("l4_ticks") for r in rs])
            fb, _ = _agg([r.get("l4_frac_both") for r in rs])
            succ = sum(1 for r in rs if r.get("success"))
            cell[(obj, d)] = {
                "along": along,
                "across": across,
                "dv": dv,
                "ha": ha,
                "ticks": ticks,
                "fb": fb,
                "n": len(rs),
                "succ": succ,
            }
            lines.append(
                f"| {obj} | {d} | {len(rs)} | {succ}/{len(rs)} | "
                f"{_f(along)} | {_f(across)} | {_f(dv)} | {_f(ha)} | "
                f"{_f(ticks, 1)} | {_f(fb, 3)} |"
            )

    lines += ["", "## Across-object spread at each Δ", ""]
    lines += [
        "| Δ | n_obj | along spread mm | Δv spread cm/s | ticks spread | flat? |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for d in (0, 100, 300):
        alongs = [cell[(o, d)]["along"] for o in OBJECTS if (o, d) in cell and cell[(o, d)]["along"] is not None]
        dvs = [cell[(o, d)]["dv"] for o in OBJECTS if (o, d) in cell and cell[(o, d)]["dv"] is not None]
        ticks = [cell[(o, d)]["ticks"] for o in OBJECTS if (o, d) in cell and cell[(o, d)]["ticks"] is not None]
        if not alongs:
            continue
        sa = float(np.nanmax(alongs) - np.nanmin(alongs))
        sd = float(np.nanmax(dvs) - np.nanmin(dvs)) if dvs else float("nan")
        st = float(np.nanmax(ticks) - np.nanmin(ticks)) if ticks else float("nan")
        ok = (sa <= tol_along_mm + 1e-9) and (sd <= tol_dv_cms + 1e-9)
        lines.append(
            f"| {d} | {len(alongs)} | {sa:.2f} | {sd:.2f} | {st:.1f} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )

    lines += ["", "## Across-Δ spread per object", ""]
    lines += [
        "| object | along spread mm | Δv spread cm/s | flat? |",
        "|---|---:|---:|---|",
    ]
    for obj in OBJECTS:
        alongs = [
            cell[(obj, d)]["along"]
            for d in (0, 100, 300)
            if (obj, d) in cell and cell[(obj, d)]["along"] is not None
        ]
        dvs = [
            cell[(obj, d)]["dv"]
            for d in (0, 100, 300)
            if (obj, d) in cell and cell[(obj, d)]["dv"] is not None
        ]
        if len(alongs) < 2:
            continue
        sa = float(np.nanmax(alongs) - np.nanmin(alongs))
        sd = float(np.nanmax(dvs) - np.nanmin(dvs)) if dvs else float("nan")
        ok = (sa <= tol_along_mm + 1e-9) and (sd <= tol_dv_cms + 1e-9)
        lines.append(f"| {obj} | {sa:.2f} | {sd:.2f} | {'PASS' if ok else 'FAIL'} |")

    lines.append("")
    return "\n".join(lines)


def _pull_along(r: dict):
    # Prefer link4 e_along; else link2 along
    v = r.get("l4_e_along")
    if v is None:
        v = r.get("l2_along")
    return v


def _pull_across(r: dict):
    return r.get("l4_e_cross")


def _f(v, nd=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tol-along-mm", type=float, default=2.0)
    ap.add_argument("--tol-dv-cms", type=float, default=1.0)
    args = ap.parse_args()
    text = build(
        args.root.resolve(),
        tol_along_mm=float(args.tol_along_mm),
        tol_dv_cms=float(args.tol_dv_cms),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
