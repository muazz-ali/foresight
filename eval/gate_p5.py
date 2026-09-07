#!/usr/bin/env python3
"""Merge A / B / B_zero / B_oracle into a P5 four-arm table (no G1 pass claim alone)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.gate_g1 import HIGH_M_S, MID_M_S  # noqa: E402


def _rate_at(report: dict, speed: float) -> float | None:
    """Success rate at one speed bin."""
    for s in report.get("summaries", []):
        if abs(s["speed_m_s"] - speed) < 1e-9:
            return float(s["success_rate"])
    return None


def _n_at(report: dict, speed: float) -> int | None:
    for s in report.get("summaries", []):
        if abs(s["speed_m_s"] - speed) < 1e-9:
            return int(s["attempts"])
    return None


def _succ_at(report: dict, speed: float) -> int | None:
    for s in report.get("summaries", []):
        if abs(s["speed_m_s"] - speed) < 1e-9:
            return int(s["successes"])
    return None


def _mid_speed(report: dict) -> float:
    """15 cm/s if present, else 20 cm/s."""
    if _rate_at(report, MID_M_S) is not None:
        return MID_M_S
    return HIGH_M_S


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def _read_pattern(b: float, bz: float, bo: float, a: float) -> str:
    """Rough P5 readout from debug.md (point estimates only)."""
    # Gaps under 5 pts treated as ≈.
    def near(x: float, y: float, tol: float = 0.05) -> bool:
        return abs(x - y) <= tol

    if near(b, bz) and near(b, bo) and (b - a) < 0.05:
        return "A ≈ B_zero ≈ B ≈ B_oracle → policy ignores conditioning (or n too small)"
    if bz < b - 0.05 and b < bo - 0.05:
        return "B_zero < B < B_oracle → conditioning used; predictor noisy"
    if bz < b - 0.05 and near(b, bo):
        return "B_zero < B ≈ B_oracle → conditioning at ceiling; bottleneck elsewhere"
    if near(b, bz) and near(b, bo):
        return "B ≈ B_zero ≈ B_oracle → policy ignores the conditioning slot"
    return "no arm separates cleanly → n still small, or conditioning weak here"


def main() -> None:
    p = argparse.ArgumentParser(description="P5 four-arm merge")
    p.add_argument("--report-a", required=True)
    p.add_argument("--report-b", required=True)
    p.add_argument("--report-b-zero", required=True)
    p.add_argument("--report-b-oracle", required=True)
    p.add_argument("-o", "--output", default=str(ROOT / "data" / "p1" / "gate_p5_report.json"))
    args = p.parse_args()

    arms = {
        "A": json.loads(Path(args.report_a).read_text()),
        "B": json.loads(Path(args.report_b).read_text()),
        "B_zero": json.loads(Path(args.report_b_zero).read_text()),
        "B_oracle": json.loads(Path(args.report_b_oracle).read_text()),
    }
    mid = _mid_speed(arms["B"])
    table = {}
    for name, rep in arms.items():
        n = _n_at(rep, mid)
        s = _succ_at(rep, mid)
        rate = _rate_at(rep, mid)
        if n is None or s is None or rate is None:
            table[name] = {"mid_speed_m_s": mid, "n": n, "successes": s, "rate": rate}
            continue
        _, lo, hi = wilson(s, n)
        table[name] = {
            "mid_speed_m_s": mid,
            "n": n,
            "successes": s,
            "rate": rate,
            "wilson95": [lo, hi],
            "summaries": rep.get("summaries"),
            "zero_conditioning": rep.get("zero_conditioning"),
            "oracle_conditioning": rep.get("oracle_conditioning"),
        }

    a_m = table["A"].get("rate")
    b_m = table["B"].get("rate")
    bz_m = table["B_zero"].get("rate")
    bo_m = table["B_oracle"].get("rate")
    pattern = "UNKNOWN"
    if None not in (a_m, b_m, bz_m, bo_m):
        pattern = _read_pattern(float(b_m), float(bz_m), float(bo_m), float(a_m))

    out = {
        "gate": "P5",
        "mid_speed_m_s": mid,
        "arms_mid": table,
        "deltas_mid": {
            "B_minus_A": (b_m - a_m) if (a_m is not None and b_m is not None) else None,
            "B_minus_B_zero": (b_m - bz_m) if (b_m is not None and bz_m is not None) else None,
            "B_oracle_minus_B": (bo_m - b_m) if (bo_m is not None and b_m is not None) else None,
        },
        "pattern_hint": pattern,
        "reports": {
            "A": args.report_a,
            "B": args.report_b,
            "B_zero": args.report_b_zero,
            "B_oracle": args.report_b_oracle,
        },
        "note": "Do not claim G1 pass from this file alone; see foresight_plan.md §8 + gate_g1_report.json.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"P5 merge → {path}")


if __name__ == "__main__":
    main()
