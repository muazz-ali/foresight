"""Score the input sanity matrix (scripts/launch_sanity.sh) at 0 cm/s.

Layout: <root>/<world>/<check>/metrics.jsonl + per-episode .h5 (+ .mp4).
For every world × check: success, closed / lifted counts, close time, where the
hand closed relative to the object, and the change against that world's `base`
run on the same seeds (the hand should move ≈ +3 cm x for num_shift if the model
reads p̂; ≈ v̂·Δ = +2.5 cm y for num_vel if it reads v̂).

Usage: python eval/sanity_report.py data/eval/sanity [-o data/eval/sanity/REPORT.md]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

CHECK_ORDER = [
    "base", "cam_static", "cam_wrist", "cam_both",
    "lang_obj", "lang_ctr", "lang_both", "lang_generic", "lang_other",
    "num_zero", "num_shift", "num_vel", "grip_closed",
]


def _episode(r: dict) -> dict:
    """Close time, hand−object XY at close, lift, from one episode's .h5."""
    out = {"closed": False, "close_t": None, "off_xy": None, "lifted": False}
    try:
        with h5py.File(r["h5"], "r") as f:
            g, ee, op, t = f["gripper"][:], f["ee_pos"][:], f["object_pos"][:], f["timestamp"][:]
    except (OSError, KeyError):
        return out
    out["lifted"] = bool(op[:, 2].max() - op[0, 2] > 0.05)
    if (g < 0.5).any():
        i = int(np.argmax(g < 0.5))
        out.update(closed=True, close_t=float(t[i] - t[0]), off_xy=(ee[i, :2] - op[i, :2]).tolist())
    return out


def score(root: Path) -> dict:
    res: dict = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for cdir in sorted(p for p in wdir.iterdir() if p.is_dir()):
            m = cdir / "metrics.jsonl"
            if not m.exists():
                continue
            rows = [json.loads(x) for x in m.read_text().splitlines() if x.strip()]
            eps = {int(r["seed"]): (r, _episode(r)) for r in rows}
            res.setdefault(wdir.name, {})[cdir.name] = eps
    return res


def summarize(res: dict) -> tuple[list[dict], dict]:
    table = []
    agg = defaultdict(lambda: Counter())
    for world, checks in res.items():
        base = checks.get("base", {})
        for check in CHECK_ORDER + sorted(set(checks) - set(CHECK_ORDER)):
            if check not in checks:
                continue
            eps = checks[check]
            n = len(eps)
            succ = sum(bool(r["success"]) for r, _ in eps.values())
            closed = sum(e["closed"] for _, e in eps.values())
            lifted = sum(e["lifted"] for _, e in eps.values())
            ct = [e["close_t"] for _, e in eps.values() if e["close_t"] is not None]
            # Change in where the hand closed vs base, same seed.
            d = []
            for s, (_, e) in eps.items():
                b = base.get(s)
                if b and e["off_xy"] is not None and b[1]["off_xy"] is not None:
                    d.append(np.subtract(e["off_xy"], b[1]["off_xy"]))
            d = np.array(d) * 100 if d else None
            r0 = next(iter(eps.values()))[0]
            row = {
                "world": world,
                "check": check,
                "n": n,
                "success": succ,
                "closed": closed,
                "lifted": lifted,
                "close_t_med_s": float(np.median(ct)) if ct else None,
                "dxy_vs_base_cm": None if d is None else np.median(d, axis=0).round(1).tolist(),
                "failures": dict(Counter(r["failure"] for r, _ in eps.values())),
                "language_fed": r0.get("language"),
                "language_scene": r0.get("language_scene"),
            }
            table.append(row)
            a = agg[check]
            a["n"] += n
            a["success"] += succ
            a["closed"] += closed
            a["lifted"] += lifted
    return table, agg


def to_markdown(table: list[dict], agg: dict) -> str:
    lines = ["# Input sanity matrix — V2, 0 cm/s, oracle numbers", ""]
    lines += ["## All worlds pooled", "", "| check | success | closed | lifted |", "|---|---:|---:|---:|"]
    for c in CHECK_ORDER + sorted(set(agg) - set(CHECK_ORDER)):
        if c in agg:
            a = agg[c]
            lines.append(f"| {c} | {a['success']}/{a['n']} | {a['closed']}/{a['n']} | {a['lifted']}/{a['n']} |")
    lines += ["", "## Per world", "",
              "| world | check | success | closed | lifted | close t (s) | hand Δxy vs base (cm) | failures | task text fed |",
              "|---|---|---:|---:|---:|---:|---|---|---|"]
    for r in table:
        ct = "–" if r["close_t_med_s"] is None else f"{r['close_t_med_s']:.2f}"
        dxy = "–" if r["dxy_vs_base_cm"] is None else f"{r['dxy_vs_base_cm'][0]:+.1f}, {r['dxy_vs_base_cm'][1]:+.1f}"
        fails = ", ".join(f"{k} {v}" for k, v in r["failures"].items() if k != "success") or "–"
        lines.append(
            f"| {r['world']} | {r['check']} | {r['success']}/{r['n']} | {r['closed']}/{r['n']} | {r['lifted']}/{r['n']} "
            f"| {ct} | {dxy} | {fails} | {r['language_fed']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    a = p.parse_args()
    table, agg = summarize(score(a.root))
    md = to_markdown(table, agg)
    print(md)
    out = a.out or (a.root / "REPORT.md")
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
