"""E1: closed-loop, in-distribution counterfactual tests of what the policy aims with.

Arms (same seeds as the Stage 1 base arm, which is the reference):
  swap  — the 4 numbers come from a different real episode at the same speed.
  shift — p̂ moved +3 cm in x (v̂ unchanged).
  hand  — the model is told its hand is +3 cm in x (real hand unchanged).

Per episode, at the first close (gripper command < 0.5):
  follow = projection of (hand - object) on (fed p̂ - object), divided by |fed p̂ - object|².
           1 = the hand went where the numbers said; 0 = it went to the real object.
  For the hand arm: follow = (hand - object at close, minus the same for the base
           episode with the same seed) projected on the +x shift / 3 cm.
           1 = copies its hand; 0 = ignores the hand.

Usage: python eval/e1_report.py [--root data/eval/E1] [--base data/eval/stage1/base]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def first_close(f) -> int | None:
    g = np.asarray(f["gripper"][:])
    idx = np.flatnonzero(g < 0.5)
    return int(idx[0]) if len(idx) else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def episodes(d: Path) -> dict[int, dict]:
    out = {}
    mf = d / "metrics.jsonl"
    if not mf.exists():
        return out
    for line in mf.read_text().splitlines():
        r = json.loads(line)
        out[int(r["seed"])] = r
    return out


def h5_of(d: Path, speed: float, seed: int) -> Path:
    return d / f"eval_model_v{speed:.2f}_s{seed}_cond.h5"


def close_geom(h5: Path) -> dict | None:
    if not h5.exists():
        return None
    with h5py.File(h5, "r") as f:
        i = first_close(f)
        if i is None:
            return {"closed": False}
        ee = np.asarray(f["ee_pos"][i, :2], dtype=np.float64)
        obj = np.asarray(f["object_pos"][i, :2], dtype=np.float64)
        fed = np.asarray(f["conditioning"][i, :2], dtype=np.float64)
        return {"closed": True, "ee": ee, "obj": obj, "fed": fed, "frame": i}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/eval/E1"))
    ap.add_argument("--base", type=Path, default=Path("data/eval/stage1/base"))
    a = ap.parse_args()
    report = {}
    lines = ["# E1: in-distribution counterfactual tests (closed loop)", "",
             "| Arm | Speed | n | Success (arm) | Success (base, same seeds) | Closed | Follow median [IQR] | Contrast (cm) |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ("swap", "shift", "hand"):
        for sp in (0.0, 0.20):
            d = a.root / arm / f"v{sp if sp else 0.0}"
            if sp == 0.20:
                d = a.root / arm / "v0.20"
            bd = a.base / ("v0.0" if sp == 0 else "v0.20")
            ep, bep = episodes(d), episodes(bd)
            seeds = sorted(ep)
            if not seeds:
                continue
            k = sum(bool(ep[s]["success"]) for s in seeds)
            kb = sum(bool(bep[s]["success"]) for s in seeds if s in bep)
            nb = sum(1 for s in seeds if s in bep)
            follows, dists, closed = [], [], 0
            for s in seeds:
                g = close_geom(h5_of(d, sp, s))
                if not g or not g["closed"]:
                    continue
                closed += 1
                if arm in ("hand", "shift"):
                    # Paired with the base episode of the same seed: how much of the +3 cm x
                    # input change shows up in where the hand closes (relative to the object).
                    gb = close_geom(h5_of(bd, sp, s))
                    if not gb or not gb["closed"]:
                        continue
                    rel = (g["ee"] - g["obj"]) - (gb["ee"] - gb["obj"])
                    follows.append(float(rel[0] / 0.03))
                    dists.append(3.0)
                elif arm == "swap":
                    # Where the donor's object really was at this frame (its p̂ is a 0.25 s
                    # forecast, so p̂ itself sits ahead of any object at speed).
                    ds = (ep[s].get("sanity") or {}).get("cond_donor_seed")
                    dh5 = h5_of(bd, sp, int(ds)) if ds is not None else None
                    if dh5 is None or not dh5.exists():
                        continue
                    with h5py.File(dh5, "r") as fd:
                        kf = min(g["frame"], len(fd["object_pos"]) - 1)
                        dobj = np.asarray(fd["object_pos"][kf, :2], dtype=np.float64)
                    v = dobj - g["obj"]
                    d2 = float(v @ v)
                    if d2 < 0.01 ** 2:
                        continue
                    follows.append(float((g["ee"] - g["obj"]) @ v / d2))
                    dists.append(float(math.sqrt(d2) * 100))
                else:
                    v = g["fed"] - g["obj"]
                    d2 = float(v @ v)
                    if d2 < 0.01 ** 2:  # fed point within 1 cm of the object: no contrast
                        continue
                    follows.append(float((g["ee"] - g["obj"]) @ v / d2))
                    dists.append(float(math.sqrt(d2) * 100))
            fq = np.percentile(follows, [25, 50, 75]) if follows else [math.nan] * 3
            lo, hi = wilson(k, len(seeds))
            report[f"{arm}_v{sp:.2f}"] = {
                "n": len(seeds), "success": k, "success_ci": [lo, hi], "base_success": kb, "base_n": nb,
                "closed": closed, "follow_median": float(fq[1]), "follow_iqr": [float(fq[0]), float(fq[2])],
                "n_follow": len(follows), "fed_minus_obj_cm_median": float(np.median(dists)) if dists else math.nan,
            }
            lines.append(
                f"| {arm} | {sp:.2f} | {len(seeds)} | {k}/{len(seeds)} ({lo:.0%}–{hi:.0%}) | {kb}/{nb} | {closed} | "
                f"{fq[1]:.2f} [{fq[0]:.2f}, {fq[2]:.2f}] (n={len(follows)}) | "
                f"{(np.median(dists) if dists else math.nan):.1f} cm |"
            )
    (a.root / "e1.json").write_text(json.dumps(report, indent=2))
    (a.root / "REPORT_auto.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
