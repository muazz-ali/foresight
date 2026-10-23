"""Score Stage 1 (scripts/launch_stage1.sh): does closing on P_s beat the policy's own timing?

Endpoints (fixed before the run):
  primary    task success, arm vs base, paired by seed, pooled over speeds (exact McNemar)
  secondary  first-close grasp (object reaches 12 cm before the first re-open), per speed;
             flatness = max - min success over speeds; veto arm vs base

Layout: <root>/<arm>/v<speed>/metrics.jsonl + .h5 (arm = base | time | veto).
Usage: python eval/stage1_report.py data/eval/stage1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import binomtest

LIFT_Z = 0.12


def first_close(h5: str) -> dict:
    with h5py.File(h5, "r") as f:
        g, op, ee = f["gripper"][:], f["object_pos"][:], f["ee_pos"][:]
        ps = f["p_s"][:] if "p_s" in f else np.full(len(g), np.nan)
    c = g < 0.5
    if not c.any():
        return {"closed": False, "grasp": False, "attempt": False, "ps_peak": float(np.nanmax(ps)) if np.isfinite(ps).any() else None}
    i = int(np.argmax(c))
    j = i + int(np.argmax(~c[i:])) if (~c[i:]).any() else len(g)
    pre = ps[:i + 1]
    return {
        "closed": True,
        "grasp": bool(op[i:j, 2].max() >= LIFT_Z),
        "attempt": bool(ee[i:j, 2].max() - ee[max(i - 1, 0), 2] >= 0.05),
        "ps_at_close": float(ps[i]) if np.isfinite(ps[i]) else None,
        "ps_peak": float(np.nanmax(pre)) if np.isfinite(pre).any() else None,
        "close_t": i,
    }


def load(root: Path) -> dict:
    out: dict = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sp_dir in sorted(arm_dir.glob("v*")):
            m = sp_dir / "metrics.jsonl"
            if not m.exists():
                continue
            for line in m.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r.update(first_close(r["h5"]))
                kinds = [e["kind"] for e in (r.get("ps_log") or [])]
                r["close_kind"] = kinds[0] if kinds else "none"
                out.setdefault(arm_dir.name, {}).setdefault(round(float(r["speed_m_s"]), 2), {})[int(r["seed"])] = r
    return out


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar on paired outcomes: b better on n01, worse on n10."""
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"n": len(a), "base": int(sum(a)), "arm": int(sum(b)), "arm_only": n01, "base_only": n10, "p": round(float(p), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    data = load(a.root)
    res: dict = {"arms": {}, "paired": {}}
    lines = ["# Stage 1 — closing on P_s vs the policy's own timing", "",
             "Tangerine06 + bowl16 (seed 9000 world), V2 checkpoint, sim-truth numbers, same seeds in every arm.", ""]
    speeds = sorted({s for arm in data.values() for s in arm})
    # per arm x speed table
    lines += ["## Per arm and speed", "",
              "| arm | speed | n | success | first close lifts | lift tried | closed by rule | P_s at close (median) | failures |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm in ("base", "time", "veto", "dist"):
        if arm not in data:
            continue
        res["arms"][arm] = {}
        for sp in speeds:
            eps = list(data[arm].get(sp, {}).values())
            if not eps:
                continue
            n = len(eps)
            row = {
                "n": n,
                "success": sum(bool(e["success"]) for e in eps),
                "grasp": sum(e["grasp"] for e in eps),
                "attempt": sum(e["attempt"] for e in eps),
                "rule": sum(e["close_kind"] == "rule" for e in eps),
                "ps_close_med": float(np.median([e["ps_at_close"] for e in eps if e.get("ps_at_close") is not None] or [np.nan])),
                "failures": dict(Counter(e["failure"] for e in eps if e["failure"] != "success")),
            }
            res["arms"][arm][sp] = row
            fails = ", ".join(f"{k} {v}" for k, v in row["failures"].items()) or "–"
            lines.append(f"| {arm} | {sp:.2f} | {n} | {row['success']}/{n} ({row['success']/n:.0%}) | {row['grasp']}/{n} "
                         f"| {row['attempt']}/{n} | {row['rule']}/{n} | {row['ps_close_med']:.2f} | {fails} |")
        succ = [res["arms"][arm][s]["success"] / res["arms"][arm][s]["n"] for s in speeds if s in res["arms"][arm]]
        res["arms"][arm]["flatness_max_minus_min"] = round(max(succ) - min(succ), 3) if succ else None
    # paired tests
    lines += ["", "## Paired by seed vs base (exact McNemar)", "",
              "| arm | outcome | speed | n | base | arm | arm-only wins | base-only wins | p |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("time", "veto", "dist"):
        if arm not in data or "base" not in data:
            continue
        for outcome in ("success", "grasp"):
            pooled_a, pooled_b = [], []
            for sp in speeds + ["pooled"]:
                if sp == "pooled":
                    xa, xb = pooled_a, pooled_b
                else:
                    common = sorted(set(data["base"].get(sp, {})) & set(data[arm].get(sp, {})))
                    xa = [bool(data["base"][sp][s][outcome]) for s in common]
                    xb = [bool(data[arm][sp][s][outcome]) for s in common]
                    pooled_a += xa
                    pooled_b += xb
                if not xa:
                    continue
                t = mcnemar(xa, xb)
                res["paired"][f"{arm}|{outcome}|{sp}"] = t
                spl = sp if sp == "pooled" else f"{sp:.2f}"
                lines.append(f"| {arm} | {outcome} | {spl} | {t['n']} | {t['base']} | {t['arm']} | {t['arm_only']} | {t['base_only']} | {t['p']} |")
    lines += ["", "## Flatness (max − min success over speeds)", ""]
    for arm, v in res["arms"].items():
        lines.append(f"- {arm}: {v['flatness_max_minus_min']}")
    md = "\n".join(lines) + "\n"
    print(md)
    (a.root / "REPORT_auto.md").write_text(md)
    (a.root / "stage1.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
