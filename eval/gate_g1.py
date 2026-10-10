#!/usr/bin/env python3
"""Merge model / model-zero eval reports into Gate G1 verdict.

Two eval runs of the SAME checkpoint go in:
  1. normal run     (policy sees the 4 numbers)
  2. zero-out run   (the 4 numbers are set to 0)
Two checks must both pass:
  - flat:     success at 0 cm/s vs 20 cm/s drops by less than 15 points.
  - zero-out: at the mid speed, zeroing the numbers costs at least 15 points.
            (If zeroing changes nothing, the model is not using the numbers.)
This file does not count trials or print Wilson intervals. It only reads the
success rates already in the two reports. Its numerical verdict is only a
scout result unless each speed has n >= 80; see foresight_plan.md §8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# MID_M_S = 15 cm/s bin and 15-point bars (flat drop / zero-out).
# HIGH_M_S = 20 cm/s bin for flat-drop check.
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
    """Merge model + model-zero reports → G1 pass/fail JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="Full conditioning eval report JSON")
    p.add_argument("--report-zero", required=True, help="Same checkpoint with --zero-conditioning")
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "p1" / "gate_g1_report.json"),
    )
    args = p.parse_args()

    model = json.loads(Path(args.report).read_text())
    model_zero = json.loads(Path(args.report_zero).read_text())

    # Pull the rates we need. None means "that speed was not run".
    model_mid = _mid_rate(model)
    m0 = _rate_at(model, 0.0)
    m_hi = _rate_at(model, HIGH_M_S)
    zero_mid = _mid_rate(model_zero)

    # Flat check: still-object rate minus fast-object rate. Small drop = good.
    # (Missing bin → flat_pass is False; no guessing.)
    flat_drop = (m0 - m_hi) if (m0 is not None and m_hi is not None) else None
    flat_pass = flat_drop is not None and flat_drop < MID_M_S
    # Zero-out must hurt: drop at least 15 pts vs full model at mid speed.
    zero_collapse = (
        zero_mid is not None and model_mid is not None and (model_mid - zero_mid) >= MID_M_S
    )

    # The two numerical checks pass only if BOTH bars pass.
    # This script does not enforce the separate n>=80 claim rule.
    gate_pass = bool(flat_pass and zero_collapse)
    # Output JSON: verdict, the numbers behind it, the bars used, and source files.
    report = {
        "gate": "G1",
        "gate_g1_pass": gate_pass,
        "flat_pass": flat_pass,
        "zero_out_collapse": zero_collapse,
        "metrics": {
            "model_mid": model_mid,
            "model_zero_mid": zero_mid,
            "model_rate_0": m0,
            "model_rate_20": m_hi,
            "flat_drop_0_to_20": flat_drop,
            "zero_out_drop_mid": (
                (model_mid - zero_mid)
                if model_mid is not None and zero_mid is not None
                else None
            ),
        },
        "criterion": {
            "model_flat_drop_0_to_20_lt": MID_M_S,
            "zero_out_drop_ge": MID_M_S,
        },
        "reports": {
            "model": args.report,
            "model_zero": args.report_zero,
        },
        # Per-speed tables copied through, so one file has the full evidence.
        "summaries_model": model.get("summaries"),
        "summaries_model_zero": model_zero.get("summaries"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Gate G1: {'PASS' if gate_pass else 'FAIL'} → {out}")


if __name__ == "__main__":
    main()
