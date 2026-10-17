#!/usr/bin/env python3
"""Merge foresight_verification/partials → report.json + report.md (slim; no D rows)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTIALS = ROOT / "partials"
OUT_JSON = ROOT / "report.json"
OUT_MD = ROOT / "report.md"


def load(name: str) -> dict:
    p = PARTIALS / name
    if not p.exists():
        return {"_missing": True, "path": str(p)}
    return json.loads(p.read_text())


def slim_d(d: dict) -> dict:
    """Drop per-frame rows and bulky per-k/speed grids; keep headlines."""
    if d.get("_missing"):
        return d
    summary = dict(d.get("summary") or {})
    # `tables` inside summary is multi-MB; leave it in partials only.
    tables_meta = None
    if "tables" in summary:
        tables = summary.pop("tables")
        tables_meta = {
            "omitted": True,
            "keys": list(tables.keys()) if isinstance(tables, dict) else type(tables).__name__,
            "full_path": "foresight_verification/partials/part_d.json",
        }
    out = {
        "summary": summary,
        "tables_meta": tables_meta,
        "compare_B8": d.get("compare_B8"),
        "D1": d.get("D1"),
        "D2": d.get("D2"),
        "D3": d.get("D3"),
        "D4": d.get("D4"),
        "D5": d.get("D5"),
        "rows_omitted": {
            "reason": "size; full rows+tables in partials/part_d.json",
            "n_rows": len(d.get("rows") or []),
        },
    }
    return out


def slim_b(b: dict) -> dict:
    """Keep summary + notes; keep by_group but it is needed for k-specific d."""
    if b.get("_missing"):
        return b
    # by_group is large but useful; keep pooled + note path for full
    out = {
        "part": b.get("part"),
        "seed": b.get("seed"),
        "data": b.get("data"),
        "notes": b.get("notes"),
        "summary_pooled_k1_8": b.get("summary_pooled_k1_8"),
        "per_speed_k1_8": b.get("per_speed_k1_8"),
        "figures": b.get("figures"),
        "by_group_pooled": (b.get("by_group") or {}).get("pooled"),
        "full_by_group_path": "foresight_verification/partials/part_b.json",
    }
    return out


def collect_flags(obj, prefix=""):
    unknowns, for_human, contradictions = [], [], []

    def walk(o, p=""):
        if isinstance(o, dict):
            if o.get("unknown"):
                unknowns.append(f"{p}: {o['unknown']}")
            fh = o.get("FOR_HUMAN") or o.get("for_human")
            if fh:
                for_human.append(f"{p}: {fh}")
            st = str(o.get("status", ""))
            if st.startswith("SKIPPED") or "NEED_APPROVAL" in st:
                for_human.append(f"{p}: {st} — {o.get('notes', o.get('note', ''))}")
            if o.get("contradicts_model"):
                contradictions.append(f"{p}: {o['contradicts_model']}")
            for k, v in o.items():
                walk(v, f"{p}.{k}" if p else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{p}[{i}]")

    walk(obj, prefix)
    return unknowns, for_human, contradictions


def main() -> None:
    a, b, c, d, e = (load(f"part_{k}.json") for k in "abcde")
    report = {
        "meta": {
            "checkpoint": "runs/p3_throw/pretrained_model",
            "seed": 0,
            "units": {"distance": "cm", "velocity": "cm/s", "time": "s"},
            "hard_rules": "no train; no long sim; scripts only under foresight_verification/",
            "partials": {
                "A": "partials/part_a.json",
                "B": "partials/part_b.json",
                "C": "partials/part_c.json",
                "D": "partials/part_d.json",
                "E": "partials/part_e.json",
            },
            "agents": {
                "A": "c8d3cc59-b732-42ba-bad9-65b7713eb911",
                "B": "47af6406-bb68-4edf-a35f-4d70ea951655",
                "C": "03dd0967-ae24-45b5-b889-5c65c79bc2b8",
                "D": "e69fa12a-85b5-4814-a228-a12389bb8a5e",
                "E": "c1c1430c-db0f-4346-bfae-3ba06d209aa5",
            },
        },
        "A": a,
        "B": slim_b(b),
        "C": c,
        "D": slim_d(d),
        "E": e,
    }

    # Headline numbers for quick access
    b8 = (b.get("summary_pooled_k1_8") or {}) if not b.get("_missing") else {}
    dsum = (d.get("summary") or {}) if not d.get("_missing") else {}
    cmp = (d.get("compare_B8") or {}) if not d.get("_missing") else {}
    c3 = (c.get("C3") or {}) if not c.get("_missing") else {}
    c1 = (c.get("C1") or {}) if not c.get("_missing") else {}
    c4 = (c.get("C4") or {}) if not c.get("_missing") else {}

    # Pull S from compare or summary
    def num(*cands, default=None):
        for x in cands:
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, dict):
                for k in ("mean", "value", "S", "s"):
                    if isinstance(x.get(k), (int, float)):
                        return float(x[k])
        return default

    headline = {
        "B8_pooled_k1_8": b8,
        "B_k8_still_d_x_cm": (
            ((b.get("by_group") or {}).get("speed:0.00") or {})
            .get("k=8", {})
            .get("x", {})
            .get("B1", {})
            .get("d_cm")
        ),
        "C1_alpha": (c4.get("alpha") or (c1.get("global") and {
            ax: c1["global"][ax]["first_order"]["alpha"] for ax in ("x", "y", "z")
            if ax in c1.get("global", {})
        })),
        "C2_tracking_lag_cm": (c.get("C2") or {}).get("measured_lag_cm")
        or (c.get("summary") or {}).get("tracking_lag_cm"),
        "C3_close_norm_cm_by_speed": (c3.get("at_close") or c3.get("hand_minus_object_at_close")),
        "C4_stability": {
            "alpha_times_one_minus_w_h": c4.get("alpha_times_one_minus_w_h"),
            "stable": c4.get("stability_0_lt_expr_lt_2"),
        },
        "D_S_delta_1_5cm_chunk0": dsum,
        "D_vs_B8": cmp,
        "E3_sigma_c": (e.get("E3") or {}).get("sigma_c_cm_for_sigma_delta"),
    }
    report["headline"] = headline

    unknowns, for_human, contradictions = [], [], []
    for label, part in [("A", a), ("B", b), ("C", c), ("D", d), ("E", e)]:
        u, f, ct = collect_flags(part, label)
        unknowns += u
        for_human += f
        contradictions += ct

    # Model contradictions we already know
    contradictions += [
        "d depends on k: pooled k=1..8 gives d≈0.8 cm, but k=8 (still) gives d≈2.16 cm — "
        "near-close frames shrink lag; model’s single d is k-dependent.",
        "Measured S_h≈1.08 ≫ B8 w_h≈0.59 and S_p≈0.045 ≪ w_n≈0.42 at δ=1.5 cm chunk0 — "
        "optimal linear mixture does NOT describe the trained net (near-pure copycat).",
        "C3: hand−future does not settle to a constant offset over the last 50 frames "
        "(early→late still moving); late-window mean ≈ (+1.1, −2.9) cm.",
        "C1: first-order α on x has R²≈0.30 — discrete P-gain model is weak on that axis.",
        "No conveyor belt: motion is throw/slide in XY (Part A conveyor_axis).",
        "Labels are commanded targets (A6), matching the model’s y:=action label assumption.",
        "p̂ noise is independent Gaussian after lookahead (A5), matching η_n assumption; "
        "p̂ is NOT rebuilt from noisy v̂.",
    ]

    report["UNKNOWN"] = unknowns
    report["FOR_HUMAN"] = for_human
    report["contradictions_vs_model"] = contradictions

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))

    # --- markdown ---
    bx, by = b8.get("x", {}), b8.get("y", {})
    alpha = headline.get("C1_alpha") or {}
    shx = (dsum.get("S_h_x_c0") or dsum.get("S_h") or {})
    if not isinstance(shx, dict):
        shx = {"mean": shx}
    # prefer compare_B8 measured fields
    measured = {
        "S_h_x": cmp.get("measured_S_h_x") or dsum.get("S_h_x_c0"),
        "S_p_x": cmp.get("measured_S_p_x") or dsum.get("S_p_x_c0"),
        "S_h_y": dsum.get("S_h_y_c0"),
        "S_p_y": dsum.get("S_p_y_c0"),
    }

    def fmt_s(v):
        if isinstance(v, dict):
            m = v.get("mean", v.get("value"))
            s = v.get("std")
            if m is not None and s is not None:
                return f"{m:.3f}±{s:.3f}"
            if m is not None:
                return f"{m:.3f}"
            return json.dumps(v)[:80]
        if isinstance(v, (int, float)):
            return f"{v:.3f}"
        return str(v)[:80]

    # C3 close norms from md-known structure
    close = c3.get("hand_minus_object_at_close") or c3.get("at_close") or {}
    if isinstance(close, dict) and "by_speed" in close:
        close = close["by_speed"]

    lines = [
        "# Foresight shortcut-model verification",
        "",
        "Evidence only. No training. No long Isaac runs.",
        f"Checkpoint: `runs/p3_throw/pretrained_model`. Seed=0. Units: cm / cm/s.",
        "",
        "## 1. Summary table",
        "",
        "| Quantity | Predicted (model / B8) | Measured |",
        "|---|---|---|",
        f"| d_x, d_y (pooled k=1..8) | — | **{bx.get('d_cm', 'UNK'):.3f}**, **{by.get('d_cm', 'UNK'):.3f}** cm |",
        f"| d_x at k=8, speed 0 | ~2.1 (story) | **{headline.get('B_k8_still_d_x_cm')}** cm |",
        f"| σ_u_x / σ_u_y | — | {bx.get('sigma_u_cm', 'UNK'):.3f} / {by.get('sigma_u_cm', 'UNK'):.3f} |",
        f"| σ_n_x / σ_n_y (=std(p̂−y)) | ~1.5 (train noise) | {bx.get('sigma_n_cm', 'UNK'):.3f} / {by.get('sigma_n_cm', 'UNK'):.3f} |",
        f"| w_h_x / w_n_x | {bx.get('w_h', float('nan')):.3f} / {bx.get('w_n', float('nan')):.3f} | S_h / S_p below |",
        f"| e*_x / e*_y | {bx.get('e_star_cm', float('nan')):.3f} / {by.get('e_star_cm', float('nan')):.3f} cm | see C3 close miss |",
        f"| α_x / α_y | — | {alpha.get('x', float('nan')):.3f} / {alpha.get('y', float('nan')):.3f} |",
        f"| α(1−w_h) x/y | in (0,2) | {c4.get('alpha_times_one_minus_w_h')} |",
        f"| S_h_x @δ=1.5cm c0 | w_h≈{bx.get('w_h', float('nan')):.3f} | **{fmt_s(measured['S_h_x'])}** |",
        f"| S_p_x @δ=1.5cm c0 | w_n≈{bx.get('w_n', float('nan')):.3f} | **{fmt_s(measured['S_p_x'])}** |",
        f"| S_h_y / S_p_y | {by.get('w_h', float('nan')):.3f} / {by.get('w_n', float('nan')):.3f} | {fmt_s(measured['S_h_y'])} / {fmt_s(measured['S_p_y'])} |",
        "",
        "### C3 oracle close miss ‖hand−object‖ (cm)",
        "",
        "| speed | mean ‖·‖ |",
        "|---:|---:|",
    ]
    # pull from C3 structure flexibly
    speeds_table = None
    for key in ("per_speed_at_close", "at_close_per_speed", "hand_minus_object_xy_at_close"):
        if key in c3:
            speeds_table = c3[key]
            break
    if speeds_table is None:
        # from summary in part_c
        speeds_table = (c.get("summary") or {}).get("close_norm_by_speed")
    if isinstance(speeds_table, dict):
        for sp, row in sorted(speeds_table.items(), key=lambda kv: float(kv[0]) if str(kv[0]).replace(".", "", 1).isdigit() else 0):
            if isinstance(row, dict):
                nrm = row.get("mean_norm") or row.get("norm_mean") or row.get("mean_abs")
                if nrm is None and "mean_x" in row:
                    import math
                    nrm = math.hypot(row.get("mean_x", 0), row.get("mean_y", 0))
                lines.append(f"| {sp} | {nrm} |")
            else:
                lines.append(f"| {sp} | {row} |")
    else:
        lines.append("| (see part_c.md) | 1.84 / 3.26 / 4.17 / 7.58 at 0/10/15/20 cm/s |")

    lines += [
        "",
        "**Verdict:** The net is a near-pure hand copycat (S_h≈1, S_p≈0). "
        "B8’s optimal mixture (w_h≈0.59) under-states the shortcut. "
        "Architecture-free data change (DART / hand noise) is still the right fix class.",
        "",
        "## 2. UNKNOWN and FOR_HUMAN",
        "",
        "### UNKNOWN",
    ]
    if unknowns:
        lines.extend(f"- {u}" for u in unknowns)
    else:
        lines.append("- (none flagged beyond A10/A11 in part_a)")
    # Always surface A10/A11 from part_a if present
    for aid in ("A10", "A11"):
        u = (a.get(aid) or {}).get("unknown") if isinstance(a.get(aid), dict) else None
        if u and not any(aid in x for x in unknowns):
            lines.append(f"- {aid}: {u}")

    lines += ["", "### FOR_HUMAN"]
    e4 = e.get("E4") or {}
    if for_human:
        lines.extend(f"- {h}" for h in for_human)
    if isinstance(e4, dict):
        qs = e4.get("questions") or e4.get("items") or e4
        if isinstance(qs, list):
            for q in qs:
                lines.append(f"- E4: {q}")
        elif isinstance(qs, dict):
            for k, v in qs.items():
                if k not in ("status",):
                    lines.append(f"- E4.{k}: {v}")
    lines.append("- E1 empirical τ sweep (20 seeds × 6 offsets, Isaac): **needs your approval**.")
    lines.append("")

    lines += [
        "## 3. Contradictions / caveats vs the mathematical model",
        "",
    ]
    lines.extend(f"- {x}" for x in contradictions)
    lines += [
        "",
        "## Partials & scripts",
        "",
        "| Part | JSON | Script |",
        "|---|---|---|",
        "| A | `partials/part_a.json` | (read-only cites) |",
        "| B | `partials/part_b.json` | `scripts/part_b_dataset_stats.py` |",
        "| C | `partials/part_c.json` | `scripts/part_c_controller_eval.py` |",
        "| D | `partials/part_d.json` | `scripts/part_d_sensitivity.py` |",
        "| E | `partials/part_e.json` | `scripts/part_e_dart_notes.md` |",
        "",
        "Full D per-frame rows live only in `partials/part_d.json` (omitted from `report.json` for size).",
        "",
        f"Generated by `{Path(__file__).name}`.",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size//1024} KB)")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
