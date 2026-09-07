#!/usr/bin/env python3
"""Merge Model A / B / B-zero eval reports into Gate G1 verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# MID_M_S = 15 cm/s bin and 15-point bars (flat drop / zero-out).
# HIGH_M_S = 20 cm/s bin and 20-point B-beats-A margin.
MID_M_S = 0.15
HIGH_M_S = 0.20
G1_SPEEDS = (0.0, 0.10, MID_M_S, HIGH_M_S)


def _rate_at(report: dict, speed: float) -> float | None:
    """Success rate at one speed bin, or None if that bin is missing."""
    for s in report.get("summaries", []):
        if abs(s["speed_m_s"] - speed) < 1e-9:
            return float(s["success_rate"])
    return None


def _mid_rate(report: dict) -> float | None:
    """Prefer 15 cm/s; fall back to 20 cm/s if the mid bin was not run."""
    return _rate_at(report, MID_M_S) if _rate_at(report, MID_M_S) is not None else _rate_at(report, HIGH_M_S)


def main() -> None:
    """Merge A / B / B-zero reports → G1 pass/fail JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--report-a", required=True)
    p.add_argument("--report-b", required=True)
    p.add_argument("--report-b-zero", required=True, help="B with --zero-conditioning")
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "p1" / "gate_g1_report.json"),
    )
    args = p.parse_args()

    a = json.loads(Path(args.report_a).read_text())
    b = json.loads(Path(args.report_b).read_text())
    bz = json.loads(Path(args.report_b_zero).read_text())

    a_mid = _mid_rate(a)
    b_mid = _mid_rate(b)
    b0 = _rate_at(b, 0.0)
    b_hi = _rate_at(b, HIGH_M_S)
    bz_mid = _mid_rate(bz)

    margin = (b_mid - a_mid) if (a_mid is not None and b_mid is not None) else None
    flat_drop = (b0 - b_hi) if (b0 is not None and b_hi is not None) else None
    margin_pass = margin is not None and margin >= HIGH_M_S
    flat_pass = flat_drop is not None and flat_drop < MID_M_S
    # Zero-out must hurt: drop at least 15 pts vs full B at mid speed.
    zero_collapse = (
        bz_mid is not None and b_mid is not None and (b_mid - bz_mid) >= MID_M_S
    )

    gate_pass = bool(margin_pass and flat_pass and zero_collapse)
    report = {
        "gate": "G1",
        "gate_g1_pass": gate_pass,
        "margin_pass": margin_pass,
        "flat_pass": flat_pass,
        "zero_out_collapse": zero_collapse,
        "metrics": {
            "A_mid": a_mid,
            "B_mid": b_mid,
            "B_zero_mid": bz_mid,
            "margin_B_minus_A": margin,
            "B_rate_0": b0,
            "B_rate_20": b_hi,
            "flat_drop_0_to_20": flat_drop,
        },
        "criterion": {
            "B_beats_A_mid_ge": HIGH_M_S,
            "B_flat_drop_0_to_20_lt": MID_M_S,
            "zero_out_drop_ge": MID_M_S,
        },
        "reports": {
            "A": args.report_a,
            "B": args.report_b,
            "B_zero": args.report_b_zero,
        },
        "summaries_B": b.get("summaries"),
        "summaries_A": a.get("summaries"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Gate G1: {'PASS' if gate_pass else 'FAIL'} → {out}")


if __name__ == "__main__":
    main()
