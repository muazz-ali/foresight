#!/usr/bin/env python3
"""Aggregate link-1..6 stats for C1_c30_all_objects_v2 → LINK_REPORT.md.

Groups by object × Δ. First link whose error moves with object/speed/Δ is
the root-cause pointer (see user checklist).

Usage:
  PYTHONPATH=. python tools/diag/report_link_chain.py \\
    data/eval/C1_c30_all_objects_v2 -o data/eval/C1_c30_all_objects_v2/LINK_REPORT.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.diag.link_chain_analyze import analyze_episode

OBJECTS = (
    "apple",
    "avocado",
    "egg",
    "kiwi",
    "lemon",
    "lime",
    "onion",
    "orange",
    "peach",
    "tangerine",
)
CELLS = ("C30", "C31", "C33")
DELTA = {"C30": 0, "C31": 100, "C33": 300}


def _pull(rep: dict, path: str):
    cur: object = rep
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _collect(root: Path) -> list[dict]:
    rows = []
    for obj in OBJECTS:
        for cell in CELLS:
            d = root / obj / "cells" / cell
            if not d.is_dir():
                continue
            for h5 in sorted(d.glob(f"{cell}_*_s*_{obj}.h5")):
                try:
                    rep = analyze_episode(h5)
                except Exception as ex:  # noqa: BLE001 — one bad file must not kill report
                    rows.append(
                        {
                            "object": obj,
                            "cell": cell,
                            "delta_ms": DELTA[cell],
                            "h5": str(h5),
                            "error": str(ex),
                        }
                    )
                    continue
                m = rep["meta"]
                rows.append(
                    {
                        "object": obj,
                        "cell": cell,
                        "delta_ms": DELTA[cell],
                        "seed": m.get("seed"),
                        "success": m.get("success"),
                        "failure_mode": m.get("failure_mode"),
                        "w_pre_m": m.get("w_pre_m"),
                        "w_pre_clamped": m.get("w_pre_clamped"),
                        "half_h_m": m.get("object_half_height_m"),
                        "aim_branch": m.get("aim_branch"),
                        "d_meas_ms": m.get("delta_meas_ms_med"),
                        "d_app_ms": m.get("applied_delta_ms_med"),
                        "heading_deg": m.get("heading_deg"),
                        "l1_med": _pull(rep, "link1_seeing.err_p_now_vs_liveHE_mm.med"),
                        "l1_p95": _pull(rep, "link1_seeing.err_p_now_vs_liveHE_mm.p95"),
                        "l2_med": _pull(rep, "link2_aiming.aim_minus_p_now_mm.med"),
                        "l2_along": _pull(rep, "link2_aiming.aim_along_mm.med"),
                        "ride_n": _pull(rep, "link2_aiming.ride_n"),
                        "l3_hand_aim": _pull(rep, "link3_tracking.hand_minus_aim_mm.med"),
                        "l3_dv": _pull(rep, "link3_tracking.dv_cms.med"),
                        "l3_hand": _pull(rep, "link3_tracking.hand_speed_cms.med"),
                        "l4_e_along": _pull(rep, "link4_close.e_along_mm.med"),
                        "l4_e_cross": _pull(rep, "link4_close.e_cross_mm.med"),
                        "l4_frac_both": _pull(rep, "link4_close.frac_both"),
                        "l4_ticks": _pull(rep, "link4_close.ticks_ride_before_close"),
                        "l5_z_pop": _pull(rep, "link5_close_mech.obj_z_pop_mm"),
                        "l5_t_shut": _pull(rep, "link5_close_mech.t_grip_below_0p2_s"),
                        "l6_hold": _pull(rep, "link6_lift_hold.n_hold"),
                        "l6_drop": _pull(rep, "link6_lift_hold.dropped_like"),
                        "rep": rep,
                    }
                )
    return rows


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return None, None
    return float(np.median(a)), float(np.percentile(a, 95))


def _fmt(med, p95, nd=2):
    if med is None:
        return "—"
    if p95 is None:
        return f"{med:.{nd}f}"
    return f"{med:.{nd}f} / {p95:.{nd}f}"


def build(root: Path) -> tuple[str, dict]:
    rows = _collect(root)
    # Per object × Δ aggregates
    by = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by[(r["object"], r["delta_ms"])].append(r)

    lines = [
        f"# Link-chain report — `{root.name}`",
        "",
        f"**Root:** `{root}`  ",
        "**Method:** reconstruct p_delayed / p_now from live GT + logged Δ; "
        "ride = stage-1 & |e|∞≤10 mm; v_hand = Isaac `ee_vel` when present.  ",
        "**Unavailable in logs:** USD mass/friction, finger contact flags, grip force (N), "
        "grasp-yaw time series, finger-axis width (only `w_pre`).",
        "",
        "## How to read",
        "",
        "For each link: median / p95 of the link error, grouped by object × Δ. "
        "Walk link 1 → 6 by **absolute** health (lemon success band); Δ-movers are notes, "
        "not automatic first cause. L1 of a few tenths of a mm under delay is normal.",
        "",
        "| link | points to |",
        "|---|---|",
        "| 1 ≥1 mm absolute (not tenths) | delay code-path bug |",
        "| 3 |Δv| / 4 frac_both off vs lemon | feedforward / gate |",
        "| 3 |hand−aim| alone ~5–7 mm | normal for this h5 — not the split |",
        "| 5 or 6 (z_pop / hold) by object | gripper + object physics |",
        "",
        "**Schema notes:** L5 `t_grip→0.2` is often N/A (cmd already shut on close entry). "
        "L6 `dropped_like` uses object−hand relative Z only — place successes are not “dropped”.",
        "",
        "## Episode meta (object-dependent)",
        "",
        "| object | Δ | n | succ | modes | w_pre mm | clamped? | half_h mm | aim_branch (logged) |",
        "|---|---:|---:|---:|---|---:|---|---:|---|",
    ]
    for obj in OBJECTS:
        for d_ms in (0, 100, 300):
            rs = by.get((obj, d_ms), [])
            if not rs:
                lines.append(f"| {obj} | {d_ms} | 0 | — | — | — | — | — | — |")
                continue
            modes = {}
            for r in rs:
                m = r.get("failure_mode") or ("ok" if r.get("success") else "?")
                modes[m] = modes.get(m, 0) + 1
            mode_s = ", ".join(f"{k}×{v}" for k, v in sorted(modes.items(), key=lambda kv: -kv[1]))
            w = _agg([None if r.get("w_pre_m") is None else 1000 * r["w_pre_m"] for r in rs])[0]
            hh = _agg([None if r.get("half_h_m") is None else 1000 * r["half_h_m"] for r in rs])[0]
            clamped = sum(1 for r in rs if r.get("w_pre_clamped"))
            # Union of logged branches across episodes (not just the first).
            brs = sorted({str(r.get("aim_branch")) for r in rs})
            br = ", ".join(brs)
            succ = sum(1 for r in rs if r.get("success"))
            lines.append(
                f"| {obj} | {d_ms} | {len(rs)} | {succ}/{len(rs)} | {mode_s} | "
                f"{w if w is None else f'{w:.1f}'} | {clamped}/{len(rs)} | "
                f"{hh if hh is None else f'{hh:.1f}'} | {br} |"
            )

    # Link summary tables
    link_specs = [
        (
            "1 Seeing — |p_now − (live+HE)| mm",
            "l1_med",
            "l1_p95",
            "healthy ≈0 (HE scale ~1 mm)",
        ),
        (
            "2 Aiming — |aim − p_now| mm (ride)",
            "l2_med",
            None,
            "healthy ≈0; along may show ZOH ~v·dt",
        ),
        (
            "2b Aim along mm (+ = ahead)",
            "l2_along",
            None,
            "",
        ),
        (
            "3 Tracking — |hand − aim| mm",
            "l3_hand_aim",
            None,
            "normal floor ~5–7 mm (lemon success band; post-step EE vs pre-step aim) — "
            "do not treat as fail; use |Δv| / frac_both",
        ),
        (
            "3b |Δv| cm/s (Isaac EE vel)",
            "l3_dv",
            None,
            "lemon success band ~2.2–3.2 med; tip vs ~1.39 lim is the gate discriminator",
        ),
        (
            "3c hand speed cm/s",
            "l3_hand",
            None,
            "target ≈30",
        ),
        (
            "4 Close — e_along mm",
            "l4_e_along",
            None,
            "need |e|≲2 to fire",
        ),
        (
            "4b frac_both on ride",
            "l4_frac_both",
            None,
            "lemon scrapes ~0.03–0.05; 0 = starve (primary object split with |Δv|)",
        ),
        (
            "5 Close mech — obj z pop mm",
            "l5_z_pop",
            None,
            "healthy <2 mm; only if closed",
        ),
        (
            "5b t grip→0.2 s",
            "l5_t_shut",
            None,
            "N/A when grip already ≤0.2 on first close tick (this schema) — not a 0 s shut",
        ),
        (
            "6 Lift/hold — dropped_like rate",
            "l6_drop",
            None,
            "relative obj−hand Z only; place successes forced False (world-Z lift ≠ drop)",
        ),
    ]

    lines += ["", "## Link summary (median by object × Δ)", ""]
    for title, key, key_p95, note in link_specs:
        lines += [
            f"### {title}",
            "",
        ]
        if note:
            lines.append(f"_{note}_")
            lines.append("")
        lines.append("| object | Δ=0 | Δ=100 | Δ=300 |")
        lines.append("|---|---:|---:|---:|")
        for obj in OBJECTS:
            cells = []
            for d_ms in (0, 100, 300):
                rs = by.get((obj, d_ms), [])
                med, p95 = _agg([r.get(key) for r in rs])
                if key_p95:
                    # show med/p95 from episode medians' distribution
                    cells.append(_fmt(med, p95, nd=2 if "frac" not in key else 3))
                else:
                    cells.append(_fmt(med, None, nd=3 if "frac" in key else 2))
            lines.append(f"| {obj} | " + " | ".join(cells) + " |")
        lines.append("")

    # Root-cause: absolute walk first (lemon band); soft Δ-movers; skip tiny L1.
    LEMON_DV_HI = 3.2  # lemon success median band upper (~2.2–3.2)
    L1_ABS_MM = 1.0  # tenths of a mm under delay are normal — not first cause
    lines += [
        "## Root-cause pointer (auto)",
        "",
        "Absolute walk 1→4 vs lemon success band, then useful Δ movers. "
        "Do **not** flag link1 when both Δ=0 and Δ>0 medians are <1 mm. "
        "Prefer `link3_dv` / `frac_both` when those move.",
        "",
    ]
    pointers = []
    for obj in OBJECTS:
        r0 = by.get((obj, 0), [])
        r1 = by.get((obj, 100), [])
        if not r0 and not r1 and not by.get((obj, 300)):
            pointers.append(f"- **{obj}**: no episodes (load fail)")
            continue
        if not r0:
            r0 = by.get((obj, 300), []) or r1

        def med(rs, k):
            m, _ = _agg([r.get(k) for r in rs])
            return m

        bits = []
        # Absolute first break (Δ=0 when present)
        l1_0 = med(r0, "l1_med")
        dv0 = med(r0, "l3_dv")
        fb0 = med(r0, "l4_frac_both")
        ha0 = med(r0, "l3_hand_aim")
        if l1_0 is not None and l1_0 >= L1_ABS_MM:
            bits.append(f"link1 absolute ({l1_0:.2f} mm)")
        elif ha0 is not None and 4.5 <= ha0 <= 8.5:
            # expected floor — do not call broken
            pass
        if fb0 is not None and fb0 < 0.01:
            if dv0 is not None and dv0 > LEMON_DV_HI:
                bits.append(
                    f"link3_dv→link4 starve (Δ0 |Δv|={dv0:.2f} > lemon≲{LEMON_DV_HI}, frac_both=0)"
                )
            else:
                bits.append(f"link4 frac_both≈0 @Δ0 (Δv={dv0:.2f})" if dv0 is not None else "link4 frac_both≈0 @Δ0")
        elif dv0 is not None and dv0 > LEMON_DV_HI + 0.5:
            bits.append(f"link3_dv high @Δ0 ({dv0:.2f}; lemon≲{LEMON_DV_HI})")
        elif fb0 is not None and fb0 >= 0.02 and dv0 is not None and dv0 <= LEMON_DV_HI + 0.3:
            bits.append(f"L3–4 near lemon band (Δv={dv0:.2f}, frac_both={fb0:.3f})")

        # Useful Δ movers (skip L1 noise when both <1 mm)
        if r0 and r1:
            for name, k, thr in [
                ("link3_dv", "l3_dv", 0.5),
                ("frac_both", "l4_frac_both", 0.015),
                ("link4_e", "l4_e_along", 2.0),
                ("link2", "l2_med", 2.0),
            ]:
                a, b = med(r0, k), med(r1, k)
                if a is None or b is None:
                    continue
                if abs(a - b) > thr:
                    bits.append(f"Δ-mover {name} (Δ0={a:.3g}, Δ100={b:.3g})")
            a, b = med(r0, "l1_med"), med(r1, "l1_med")
            if (
                a is not None
                and b is not None
                and not (a < L1_ABS_MM and b < L1_ABS_MM)
                and abs(a - b) > 0.5
            ):
                bits.append(f"Δ-mover link1 (Δ0={a:.2f}, Δ100={b:.2f})")

        # Dedupe while keeping order
        seen = set()
        uniq = []
        for b in bits:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        pointers.append(
            f"- **{obj}**: "
            + (", ".join(uniq) if uniq else "no clear break on links 1–4 (check object physics / modes)")
        )
    lines.extend(pointers)

    # Per-object short verdict from modes
    lines += ["", "## Per-object verdict sketch", ""]
    for obj in OBJECTS:
        all_r = []
        for d in (0, 100, 300):
            all_r.extend(by.get((obj, d), []))
        if not all_r:
            lines.append(f"- **{obj}**: no data")
            continue
        succ = sum(1 for r in all_r if r.get("success"))
        modes = {}
        for r in all_r:
            m = r.get("failure_mode") or ("ok" if r.get("success") else "?")
            modes[m] = modes.get(m, 0) + 1
        w = _agg([None if r.get("w_pre_m") is None else 1000 * r["w_pre_m"] for r in all_r])[0]
        lines.append(
            f"- **{obj}**: {succ}/{len(all_r)} success; modes={modes}; "
            f"w_pre≈{w:.1f} mm" if w is not None else f"- **{obj}**: {succ}/{len(all_r)}"
        )

    payload = {
        "root": str(root),
        "n_episodes": len([r for r in rows if "error" not in r]),
        "n_errors": len([r for r in rows if "error" in r]),
        "rows": [{k: v for k, v in r.items() if k != "rep"} for r in rows],
    }
    return "\n".join(lines).rstrip() + "\n", payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    text, payload = build(args.root)
    out = args.out or (args.root / "LINK_REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (out.parent / "link_chain_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
