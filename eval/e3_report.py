"""E3: does the instruction choose the object? Two static objects (target + distractor).

For each episode: at the first close, which object is the hand over?
  target / distractor = within --radius-cm of that object (XY), the nearer one wins;
  neither = closed but far from both; no-close = never closed.

Arms are dirs written by eval_policy.py with --distractor-usd. The instruction
(meta "language") names either the target or the distractor.

Usage: python eval/e3_report.py DIR [DIR ...] [--radius-cm 6]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np


def classify(h5: Path, js: dict, radius: float) -> str:
    dpos = js.get("distractor_pos") or (js.get("meta") or {}).get("distractor_pos")
    with h5py.File(h5, "r") as f:
        g = np.asarray(f["gripper"][:])
        idx = np.flatnonzero(g < 0.5)
        if not len(idx):
            return "no-close"
        i = int(idx[0])
        ee = np.asarray(f["ee_pos"][i, :2], dtype=np.float64)
        obj = np.asarray(f["object_pos"][i, :2], dtype=np.float64)
    dt = float(np.linalg.norm(ee - obj))
    dd = float(np.linalg.norm(ee - np.asarray(dpos[:2], dtype=np.float64))) if dpos else float("inf")
    if min(dt, dd) > radius:
        return "neither"
    return "target" if dt <= dd else "distractor"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--radius-cm", type=float, default=6.0)
    a = ap.parse_args()
    out = {}
    lines = ["| Arm | Instruction names | n | Hand at target | at distractor | neither | no close | Task success |",
             "|---|---|---|---|---|---|---|---|"]
    for d in a.dirs:
        counts = collections.Counter()
        named = collections.Counter()
        succ = 0
        n = 0
        recs = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
        for js in recs:
            h5 = d / f"eval_model_v{float(js['speed_m_s']):.2f}_s{int(js['seed'])}_cond.h5"
            if not h5.exists():
                continue
            n += 1
            counts[classify(h5, js, a.radius_cm / 100.0)] += 1
            lang = str(js.get("language") or "")
            dcat = str(js.get("distractor_category") or "")
            named["distractor" if dcat and dcat in lang else "target"] += 1
            succ += bool(js.get("success"))
        out[str(d)] = {"n": n, **counts, "named": dict(named), "success": succ}
        lines.append(
            f"| {d} | {', '.join(f'{k} {v}' for k, v in named.items())} | {n} | {counts['target']} | "
            f"{counts['distractor']} | {counts['neither']} | {counts['no-close']} | {succ}/{n} |"
        )
    print("\n".join(lines))
    Path(a.dirs[0]).parent.joinpath("e3.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
