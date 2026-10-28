#!/usr/bin/env python3
"""Compare two C30-all-objects eval roots (old vs rerun) → COMPARE.md.

Pairs episodes by (object, cell, seed). Reports success flips and link median
deltas. Used after a no-code-change h5 re-run to check reproducibility.

Usage:
  PYTHONPATH=. python tools/diag/compare_link_runs.py \\
    data/eval/C1_c30_all_objects_v2 \\
    data/eval/C1_c30_all_objects_v2_rerun \\
    -o data/eval/C1_c30_all_objects_v2_rerun/COMPARE.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.diag.report_link_chain import CELLS, DELTA, OBJECTS, _agg, _collect, _fmt


KEYS = (
    ("l1_med", "L1 |p_now−live| mm"),
    ("l2_med", "L2 |aim−p_now| mm"),
    ("l3_hand_aim", "L3 |hand−aim| mm"),
    ("l3_dv", "L3 |Δv| cm/s"),
    ("l4_frac_both", "L4 frac_both"),
    ("l5_z_pop", "L5 z_pop mm"),
)


def _index(rows: list[dict]) -> dict[tuple, dict]:
    out = {}
    for r in rows:
        if "error" in r:
            continue
        seed = r.get("seed")
        if seed is None:
            continue
        out[(r["object"], r["cell"], int(seed))] = r
    return out


def build(old_root: Path, new_root: Path) -> str:
    old_rows = _collect(old_root)
    new_rows = _collect(new_root)
    old_ix = _index(old_rows)
    new_ix = _index(new_rows)
    keys = sorted(set(old_ix) | set(new_ix))

    lines = [
        "# Compare — v2 vs rerun (same seeds, no code change)",
        "",
        f"**Old:** `{old_root}`  ",
        f"**New:** `{new_root}`  ",
        f"**Paired episodes:** {len(keys)}  "
        f"(old={len(old_ix)}, new={len(new_ix)})",
        "",
        "## Scoreboard (success / n)",
        "",
        "| object | cell | Δ ms | old | new | Δ succ |",
        "|---|---|---:|---:|---:|---:|",
    ]

    flips: list[tuple] = []
    by_cell = defaultdict(lambda: {"old_s": 0, "old_n": 0, "new_s": 0, "new_n": 0})
    for obj in OBJECTS:
        for cell in CELLS:
            o_s = o_n = n_s = n_n = 0
            for seed_k in keys:
                if seed_k[0] != obj or seed_k[1] != cell:
                    continue
                o = old_ix.get(seed_k)
                n = new_ix.get(seed_k)
                if o is not None:
                    o_n += 1
                    o_s += int(bool(o.get("success")))
                if n is not None:
                    n_n += 1
                    n_s += int(bool(n.get("success")))
                if o is not None and n is not None and bool(o.get("success")) != bool(n.get("success")):
                    flips.append(
                        (
                            obj,
                            cell,
                            seed_k[2],
                            bool(o.get("success")),
                            bool(n.get("success")),
                            o.get("failure_mode"),
                            n.get("failure_mode"),
                        )
                    )
            if o_n == 0 and n_n == 0:
                continue
            d = n_s - o_s
            lines.append(
                f"| {obj} | {cell} | {DELTA[cell]} | {o_s}/{o_n} | {n_s}/{n_n} | "
                f"{d:+d} |"
            )
            by_cell[cell]["old_s"] += o_s
            by_cell[cell]["old_n"] += o_n
            by_cell[cell]["new_s"] += n_s
            by_cell[cell]["new_n"] += n_n

    lines += ["", "## Totals by cell", "", "| cell | old | new | Δ |", "|---|---:|---:|---:|"]
    for cell in CELLS:
        b = by_cell[cell]
        lines.append(
            f"| {cell} | {b['old_s']}/{b['old_n']} | {b['new_s']}/{b['new_n']} | "
            f"{b['new_s'] - b['old_s']:+d} |"
        )

    lines += [
        "",
        f"## Success flips ({len(flips)})",
        "",
        "| object | cell | seed | old→new | modes |",
        "|---|---|---:|---|---|",
    ]
    if not flips:
        lines.append("| — | — | — | none | — |")
    else:
        for obj, cell, seed, os_, ns_, om, nm in flips:
            lines.append(
                f"| {obj} | {cell} | {seed} | {int(os_)}→{int(ns_)} | "
                f"{om or ('ok' if os_ else '?')} → {nm or ('ok' if ns_ else '?')} |"
            )

    # Link median shifts per object×Δ (paired medians)
    lines += [
        "",
        "## Link medians (old → new)",
        "",
        "Per object × Δ: median of episode medians. Diff = new − old.",
        "",
    ]
    for key, label in KEYS:
        lines += [
            f"### {label}",
            "",
            "| object | Δ | old med | new med | diff |",
            "|---|---:|---:|---:|---:|",
        ]
        for obj in OBJECTS:
            for d_ms in (0, 100, 300):
                o_vals = [
                    old_ix[k][key]
                    for k in keys
                    if k[0] == obj and DELTA[k[1]] == d_ms and k in old_ix
                ]
                n_vals = [
                    new_ix[k][key]
                    for k in keys
                    if k[0] == obj and DELTA[k[1]] == d_ms and k in new_ix
                ]
                om, _ = _agg(o_vals)
                nm, _ = _agg(n_vals)
                if om is None and nm is None:
                    continue
                diff = None if om is None or nm is None else nm - om
                lines.append(
                    f"| {obj} | {d_ms} | {_fmt(om, None)} | {_fmt(nm, None)} | "
                    f"{'—' if diff is None else f'{diff:+.3f}'} |"
                )
        lines.append("")

    # Verdict
    total_flip = len(flips)
    total_old = sum(by_cell[c]["old_s"] for c in CELLS)
    total_new = sum(by_cell[c]["new_s"] for c in CELLS)
    lines += [
        "## Verdict",
        "",
        f"- Total success: **{total_old}** → **{total_new}** ({total_new - total_old:+d})",
        f"- Episode success flips: **{total_flip}**",
    ]
    if total_flip == 0 and total_old == total_new:
        lines.append("- **Reproducible** within this scout (same scoreboard).")
    elif total_flip <= 3 and abs(total_old - total_new) <= 2:
        lines.append("- **Nearly reproducible** — small seed noise; do not treat as a code change.")
    else:
        lines.append(
            "- **Not stable** — investigate non-determinism (PhysX, GPU order) before code fixes."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old_root", type=Path)
    ap.add_argument("new_root", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    text = build(args.old_root.resolve(), args.new_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
