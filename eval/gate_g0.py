#!/usr/bin/env python3
"""Summarize Gate G0 report from data/g0/gate_g0_report.json.

Why it exists: G0 asks "does our scripted expert grasp reliably, and fast
enough to collect lots of demos?" This file only PRINTS the report. It does
not decide pass/fail. The collection run wrote PASS/FAIL (and the per-speed
Y/N) into the JSON. Thresholds: see foresight_plan.md §8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--report",
        default=str(ROOT / "data" / "g0" / "gate_g0_report.json"),
    )
    args = p.parse_args()
    report = json.loads(Path(args.report).read_text())
    # Top line: the verdict stored in the file (we do not recompute it).
    print(f"Gate G0: {'PASS' if report.get('gate_g0_pass') else 'FAIL'}")
    print(f"Overall success: {100*report['overall_success_rate']:.1f}%")
    # Throughput = episodes per hour (how fast we can collect demos).
    print(f"Throughput: {report['overall_throughput_eps_per_hour']:.0f} eps/h")
    print()
    # One row per object speed. "succ" = wins/tries. "pass" = that speed's own Y/N.
    print(f"{'speed':>8} {'succ':>8} {'rate':>8} {'eps/h':>8} {'pass':>6}")
    for s in report["summaries"]:
        print(
            f"{s['speed_m_s']:8.2f} "
            f"{s['successes']:3d}/{s['attempts']:<3d} "
            f"{100*s['success_rate']:7.1f}% "
            f"{s['episodes_per_hour']:8.0f} "
            f"{'Y' if s['gate_g0_bin_pass'] else 'N':>6}"
        )


if __name__ == "__main__":
    main()
