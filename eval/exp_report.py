"""Success table for any set of eval_policy.py output dirs (E2, E3, E4).

Each dir has metrics.jsonl (one line per episode). Prints n, successes, Wilson 95%
interval and failure labels. With --ref, every arm is also compared with the
reference arm on the seeds both ran (exact two-sided McNemar on discordant pairs).

Usage:
  python eval/exp_report.py DIR [DIR ...] [--ref DIR] [-o out.md]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the b + c discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(d: Path) -> dict[int, dict]:
    out = {}
    mf = d / "metrics.jsonl"
    if mf.exists():
        for line in mf.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["seed"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()
    ref = load(a.ref) if a.ref else None
    rows = ["| Arm | n | Success | 95% CI | Failures | vs ref: wins/losses, p |", "|---|---|---|---|---|---|"]
    js = {}
    for d in a.dirs:
        ep = load(d)
        n = len(ep)
        k = sum(bool(r["success"]) for r in ep.values())
        lo, hi = wilson(k, n)
        fails = collections.Counter(r.get("failure", "?") for r in ep.values() if not r["success"])
        cmp_txt = ""
        cmp_js = None
        if ref is not None and d != a.ref:
            common = sorted(set(ep) & set(ref))
            b = sum(bool(ep[s]["success"]) and not bool(ref[s]["success"]) for s in common)
            c = sum(not bool(ep[s]["success"]) and bool(ref[s]["success"]) for s in common)
            p = mcnemar_exact(b, c)
            cmp_txt = f"{b}/{c}, p = {p:.3f} (n = {len(common)})"
            cmp_js = {"wins": b, "losses": c, "p": p, "n_common": len(common)}
        name = str(d)
        rows.append(
            f"| {name} | {n} | {k}/{n} ({(k / n if n else math.nan):.0%}) | {lo:.0%}–{hi:.0%} | "
            f"{', '.join(f'{f} {v}' for f, v in fails.most_common()) or '–'} | {cmp_txt} |"
        )
        js[name] = {"n": n, "success": k, "ci": [lo, hi], "failures": dict(fails), "vs_ref": cmp_js}
    text = "\n".join(rows) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text)
        a.out.with_suffix(".json").write_text(json.dumps(js, indent=2))


if __name__ == "__main__":
    main()
