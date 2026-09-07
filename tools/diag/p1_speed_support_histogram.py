#!/usr/bin/env python3
"""WORKSTREAM: policy — histogram P1 collection by G1 speed bins + expert SR.

Read-only over data/p1/metrics.jsonl and data/p1/speed_*/*.json.
Does not touch sim/, policy/, eval/, scripts/, or interfaces/.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.gate_g1 import G1_SPEEDS, HIGH_M_S, MID_M_S  # noqa: E402

G1 = list(G1_SPEEDS)


def speed_bin(speed: float) -> float:
    """Map continuous/jittered speed to G1 speed_bin center."""
    if speed <= 0.01:
        return 0.0
    if speed < 0.125:
        return 0.10
    if speed < 0.175:
        return MID_M_S
    return HIGH_M_S


def main() -> None:
    metrics_path = ROOT / "data" / "p1" / "metrics.jsonl"
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    by: dict[float, dict] = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in rows:
        b = speed_bin(float(r["speed_m_s"]))
        by[b]["n"] += 1
        if r.get("success"):
            by[b]["ok"] += 1

    jby: dict[float, int] = defaultdict(int)
    for jp in (ROOT / "data" / "p1").glob("speed_*/*.json"):
        meta = json.loads(jp.read_text())
        jby[speed_bin(float(meta["speed_m_s"]))] += 1

    N = sum(by[b]["n"] for b in G1)
    Nj = sum(jby.values())
    print(f"metrics attempts={len(rows)} successes={sum(1 for r in rows if r.get('success'))}")
    print("speed | n_attempts | %attempts | expert_sr | n_saved_demos | %demos | in-support?")
    flags = []
    for b in G1:
        n, ok = by[b]["n"], by[b]["ok"]
        pct = n / N if N else 0.0
        sr = ok / n if n else float("nan")
        nd = jby[b]
        print(
            f"{b:4.2f} | {n:10d} | {100*pct:8.2f}% | {sr:9.4f} | {nd:13d} | "
            f"{100*nd/Nj if Nj else 0:5.2f}% | yes"
        )
        if pct < 0.10 or sr < 0.7:
            flags.append(b)

    n0, ok0 = by[0.0]["n"], by[0.0]["ok"]
    n20, ok20 = by[HIGH_M_S]["n"], by[HIGH_M_S]["ok"]
    expert_drop = (ok0 / n0 - ok20 / n20) if n0 and n20 else float("nan")
    print(f"expert_drop_0_to_20={expert_drop:.4f}")
    print("verdict=", "DATA-EXPLAINS-DROP" if flags else "DATA-ADEQUATE")
    if flags:
        print("flagged_bins=", flags)


if __name__ == "__main__":
    main()
