"""Claim scoreboard: one table per model from scripts/launch_claims.sh outputs.

Each model dir holds scenario subdirs (oracle / filter / zero / shift), each a
normal eval_policy.py output (metrics.jsonl + per-episode .h5). Reports:

  success per speed (Wilson 95%), drop still → 20 cm/s     — claim 1: flat
  zero-out success vs oracle                                — claim 2: numbers matter
  shift follow = (hand−object offset at close, shift run − oracle run, same
    seed) along the shift ÷ shift size. ≈1: the hand goes where the numbers
    say. ≈0: the numbers are ignored.                       — claim 2, causal form
  |hand − object| at the close (median cm)                  — why grasps fail

Usage:
  python eval/claims_report.py data/eval/claims/base data/eval/claims/v1 ... [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

SCENARIOS = ("oracle", "filter", "zero", "shift", "oracle_gate", "filter_gate")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rows(scen_dir: Path) -> list[dict]:
    m = scen_dir / "metrics.jsonl"
    return [json.loads(x) for x in m.read_text().splitlines() if x.strip()] if m.exists() else []


def _close_offset(h5: str) -> np.ndarray | None:
    """Hand minus object XY (m) at the first close, or None if never closed."""
    try:
        with h5py.File(h5, "r") as f:
            g = f["gripper"][:]
            if not (g < 0.5).any():
                return None
            i = int(np.argmax(g < 0.5))
            return np.asarray(f["ee_pos"][i, :2] - f["object_pos"][i, :2], dtype=np.float64)
    except (OSError, KeyError):
        return None


def score_model(model_dir: Path) -> dict:
    out: dict = {"model": str(model_dir), "scenarios": {}}
    by_seed: dict[str, dict] = {}
    for sc in SCENARIOS:
        rows = _rows(model_dir / sc)
        if not rows:
            continue
        per = defaultdict(lambda: [0, 0, []])
        for r in rows:
            s = per[round(float(r["speed_m_s"]), 2)]
            s[0] += int(bool(r["success"]))
            s[1] += 1
            off = _close_offset(r["h5"]) if r.get("h5") else None
            if off is not None:
                s[2].append(float(np.linalg.norm(off)) * 100)
        table = {}
        for sp in sorted(per):
            k, n, d = per[sp]
            lo, hi = wilson(k, n)
            table[f"{sp:.2f}"] = {
                "success": k,
                "n": n,
                "rate": k / n if n else float("nan"),
                "wilson95": [lo, hi],
                "close_dist_cm_median": float(np.median(d)) if d else None,
            }
        rates = [v["rate"] for v in table.values()]
        sc_out = {"per_speed": table, "checkpoint": rows[0].get("checkpoint"), "frames": rows[0].get("frames")}
        if "0.00" in table and "0.20" in table:
            sc_out["drop_0_to_20"] = table["0.00"]["rate"] - table["0.20"]["rate"]
        sc_out["mean_rate"] = float(np.mean(rates)) if rates else None
        out["scenarios"][sc] = sc_out
        by_seed[sc] = {(round(float(r["speed_m_s"]), 2), int(r["seed"])): r for r in rows}

    # Shift follow: same seed, shift vs oracle, offset change along the shift.
    if "shift" in by_seed and "oracle" in by_seed:
        shift = None
        follows = defaultdict(list)
        for key, r in by_seed["shift"].items():
            shift = np.asarray(r.get("cond_shift_m") or [0, 0], dtype=np.float64)
            base = by_seed["oracle"].get(key)
            if base is None or np.linalg.norm(shift) < 1e-9:
                continue
            a, b = _close_offset(r["h5"]), _close_offset(base["h5"])
            if a is None or b is None:
                continue
            u = shift / np.linalg.norm(shift)
            follows[f"{key[0]:.2f}"].append(float((a - b) @ u / np.linalg.norm(shift)))
        allf = [x for v in follows.values() for x in v]
        out["shift_follow"] = {
            "shift_m": None if shift is None else shift.tolist(),
            "median_all": float(np.median(allf)) if allf else None,
            "n_pairs": len(allf),
            "per_speed": {k: float(np.median(v)) for k, v in sorted(follows.items())},
        }
    if "zero" in out["scenarios"] and "oracle" in out["scenarios"]:
        z, o = out["scenarios"]["zero"]["mean_rate"], out["scenarios"]["oracle"]["mean_rate"]
        out["zero_collapse"] = {"oracle_mean": o, "zero_mean": z, "drop": None if o is None else o - z}
    return out


def _print(rep: dict) -> None:
    print(f"\n=== {rep['model']}")
    for sc, s in rep["scenarios"].items():
        cells = []
        for sp, v in s["per_speed"].items():
            cd = v["close_dist_cm_median"]
            cells.append(
                f"{sp}: {v['success']:>2}/{v['n']:<3} [{v['wilson95'][0]*100:3.0f}-{v['wilson95'][1]*100:3.0f}%] "
                f"d={cd:.1f}cm" if cd is not None else f"{sp}: {v['success']}/{v['n']}"
            )
        drop = s.get("drop_0_to_20")
        print(f"  {sc:6s} " + " | ".join(cells) + (f" | drop0→20={drop*100:+.0f}pt" if drop is not None else ""))
    if "zero_collapse" in rep:
        zc = rep["zero_collapse"]
        print(f"  zero-out: mean success {zc['oracle_mean']*100:.0f}% → {zc['zero_mean']*100:.0f}%")
    if "shift_follow" in rep:
        sf = rep["shift_follow"]
        print(f"  shift {sf['shift_m']} m: hand follows {sf['median_all']} (1 = uses numbers, 0 = ignores) "
              f"n={sf['n_pairs']} per speed {sf['per_speed']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    a = p.parse_args()
    reps = [score_model(m) for m in a.models]
    for r in reps:
        _print(r)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(reps, indent=2))


if __name__ == "__main__":
    main()
