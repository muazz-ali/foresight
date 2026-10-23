"""E6: scripted-expert ceiling vs object speed, hand yaw fixed vs pads along the motion.

Reads data/eval/E6/{fixed,along}/v<speed>/metrics.jsonl (scripts/launch_e6.sh).
Same episode seeds in every cell, so fixed vs along pairs up (exact McNemar).

Throw classes (same geometric rule as Stage 1, fixed in advance):
  realistic = moving toward the robot (v_x < 0) and the straight 3 s path never comes
              within 0.11 m (bowl radius 7.5 cm + object 3.5 cm) of the bowl centre.
  other     = everything else (over the bowl, or away from the robot).

Usage: python eval/e6_report.py [--root data/eval/E6]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.exp_report import mcnemar_exact, wilson  # noqa: E402


def realistic(r: dict) -> bool | None:
    p, v, c = r.get("init_pos"), r.get("init_vel"), r.get("container_pos")
    if p is None or v is None or c is None:
        return None
    p, v, c = (np.asarray(x, dtype=np.float64)[:2] for x in (p, v, c))
    if np.linalg.norm(v) < 1e-6:
        return True
    if v[0] >= 0:
        return False
    t = np.linspace(0.0, 3.0, 301)
    path = p[None, :] + t[:, None] * v[None, :]
    return bool(np.min(np.linalg.norm(path - c[None, :], axis=1)) >= 0.11)


def load(d: Path) -> dict[int, dict]:
    mf = d / "metrics.jsonl"
    if not mf.exists():
        return {}
    return {int(r["seed"]): r for r in (json.loads(x) for x in mf.read_text().splitlines() if x.strip())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/eval/E6"))
    a = ap.parse_args()
    speeds = sorted({float(p.name[1:]) for p in a.root.glob("*/v*")})
    out, lines = {}, [
        "| Speed | Throws | Fixed yaw | Pads along motion | Along vs fixed (wins/losses, p) |",
        "|---|---|---|---|---|",
    ]
    for sp in speeds:
        fx, al = load(a.root / "fixed" / f"v{sp:.2f}"), load(a.root / "along" / f"v{sp:.2f}")
        for cls in ("all", "realistic", "other"):
            def keep(r):
                if cls == "all":
                    return True
                rr = realistic(r)
                return rr is True if cls == "realistic" else rr is False
            sf = {s: r for s, r in fx.items() if keep(r)}
            sa = {s: r for s, r in al.items() if keep(r)}
            kf, ka = sum(r["success"] for r in sf.values()), sum(r["success"] for r in sa.values())
            common = sorted(set(sf) & set(sa))
            b = sum(sa[s]["success"] and not sf[s]["success"] for s in common)
            c = sum(sf[s]["success"] and not sa[s]["success"] for s in common)
            p = mcnemar_exact(b, c)
            lf, hf = wilson(kf, len(sf))
            la, ha = wilson(ka, len(sa))
            out[f"{sp:.2f}_{cls}"] = {"fixed": [kf, len(sf), lf, hf], "along": [ka, len(sa), la, ha],
                                     "along_wins": b, "along_losses": c, "p": p}
            fmt = lambda k, n, lo, hi: f"{k}/{n} ({k / n:.0%}, {lo:.0%}–{hi:.0%})" if n else "–"
            lines.append(f"| {sp:.2f} | {cls} | {fmt(kf, len(sf), lf, hf)} | {fmt(ka, len(sa), la, ha)} | "
                         f"{b}/{c}, p = {p:.3f} |")
    (a.root / "e6.json").write_text(json.dumps(out, indent=2))
    (a.root / "REPORT_auto.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
