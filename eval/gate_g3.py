#!/usr/bin/env python3
"""Gate G3 scorer: same model, same seeds, reflex off (baseline) vs on.

Inputs are eval run folders (each has metrics.jsonl from scripts/eval_policy.py):
  --baseline   reflex off (the G3 baseline)
  --treatment  reflex on (+ buffer / latency as chosen)
  --zero       optional: treatment settings with --zero-conditioning
We compare RATES, not per-seed flips (policy outcomes still vary run to run).

Checks (foresight_plan.md §5 Gate G3):
  timing:  late+early+short-close failure rate drops by at least half (pooled).
  smooth:  mean pre-grasp jerk goes down (path length is reported beside it).
  fast:    success at 20 cm/s goes up vs the baseline.
  zero:    zeroing the 4 numbers costs >= 15 points at the mid speed.
Rates carry Wilson 95% intervals. n >= 80 per bin before claiming a result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.timing_labels import TIMING_FAILURES  # noqa: E402

MID_M_S = 0.15
HIGH_M_S = 0.20
ZERO_DROP_MIN = 0.15  # same bar as G1
TIMING_KEEP_MAX = 0.5  # treatment timing-failure rate must be <= half the baseline
CLAIM_N = 80


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def load_run(path) -> list[dict]:
    """metrics.jsonl rows from a run folder (or the file itself)."""
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _label(r: dict) -> str | None:
    t = r.get("timing") or {}
    return t.get("timing_label")


def _mean(xs: list) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def summarize(rows: list[dict]) -> dict:
    """Per-speed numbers for one run: success, timing failures, labels, smoothness."""
    out: dict[float, dict] = {}
    for speed in sorted({round(float(r["speed_m_s"]), 4) for r in rows}):
        rs = [r for r in rows if abs(float(r["speed_m_s"]) - speed) < 1e-6]
        labels: dict[str, int] = {}
        for r in rs:
            lab = _label(r) or "unlabelled"
            labels[lab] = labels.get(lab, 0) + 1
        pre = [(r.get("timing") or {}).get("pre_grasp") or {} for r in rs]
        out[speed] = {
            "success": _rate(sum(bool(r["success"]) for r in rs), len(rs)),
            "timing_fail": _rate(sum(_label(r) in TIMING_FAILURES for r in rs), len(rs)),
            "labels": labels,
            "path_len_m": _mean([p.get("path_len_m") for p in pre]),
            "jerk_rms": _mean([p.get("jerk_rms") for p in pre]),
            "seeds": sorted(int(r["seed"]) for r in rs),
        }
    return out


def _pooled(summary: dict, key: str) -> tuple[int, int]:
    return (sum(s[key]["k"] for s in summary.values()), sum(s[key]["n"] for s in summary.values()))


def score(base_rows: list[dict], treat_rows: list[dict], zero_rows: list[dict] | None = None) -> dict:
    base, treat = summarize(base_rows), summarize(treat_rows)
    zero = summarize(zero_rows) if zero_rows else None
    speeds = sorted(set(base) | set(treat))
    per_speed = []
    for sp in speeds:
        b, t = base.get(sp), treat.get(sp)
        row = {"speed_m_s": sp, "baseline": b, "treatment": t}
        if b and t:
            row["seeds_match"] = b["seeds"] == t["seeds"]
            row["timing_fail_drop"] = t["timing_fail"]["rate"] - b["timing_fail"]["rate"]
            row["success_diff"] = t["success"]["rate"] - b["success"]["rate"]
        if zero and sp in zero:
            row["zero"] = zero[sp]
        per_speed.append(row)

    # Timing: pooled over all speeds (per-bin counts are small).
    bk, bn = _pooled(base, "timing_fail")
    tk, tn = _pooled(treat, "timing_fail")
    b_rate = bk / bn if bn else None
    t_rate = tk / tn if tn else None
    timing_pass = None
    if b_rate is not None and t_rate is not None:
        timing_pass = bool(b_rate > 0 and t_rate <= TIMING_KEEP_MAX * b_rate)

    # Smoothness: mean jerk over all episodes that logged it.
    b_jerk = _mean([s["jerk_rms"] for s in base.values()])
    t_jerk = _mean([s["jerk_rms"] for s in treat.values()])
    smooth_pass = None if b_jerk is None or t_jerk is None else bool(t_jerk < b_jerk)

    b20, t20 = base.get(HIGH_M_S), treat.get(HIGH_M_S)
    fast_pass = None
    if b20 and t20:
        fast_pass = bool(t20["success"]["rate"] > b20["success"]["rate"])

    zero_pass = None
    zero_drop = None
    mid = MID_M_S if MID_M_S in treat else HIGH_M_S
    if zero and mid in zero and mid in treat:
        zero_drop = treat[mid]["success"]["rate"] - zero[mid]["success"]["rate"]
        zero_pass = bool(zero_drop >= ZERO_DROP_MIN)

    checks = {"timing_halved": timing_pass, "smoother": smooth_pass,
              "success_20_up": fast_pass, "zero_out_collapse": zero_pass}
    min_n = min([s["success"]["n"] for s in list(base.values()) + list(treat.values())] or [0])
    return {
        "gate": "G3",
        "gate_g3_pass": all(v is True for v in checks.values()),
        "checks": checks,
        "claim_ready": min_n >= CLAIM_N,
        "min_n_per_bin": min_n,
        "timing_fail_pooled": {
            "baseline": _rate(bk, bn), "treatment": _rate(tk, tn),
            "keep_max_fraction": TIMING_KEEP_MAX,
            "fraction_kept": (t_rate / b_rate) if b_rate else None,
        },
        "jerk_rms_mean": {"baseline": b_jerk, "treatment": t_jerk},
        "path_len_m_mean": {
            "baseline": _mean([s["path_len_m"] for s in base.values()]),
            "treatment": _mean([s["path_len_m"] for s in treat.values()]),
        },
        "zero_out_drop_mid": zero_drop,
        "timing_failures": list(TIMING_FAILURES),
        "per_speed": per_speed,
    }


def _pct(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "—"
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} ({100*r['rate']:.0f}%, {100*lo:.0f}–{100*hi:.0f})"


def _num(x, fmt="{:.3f}") -> str:
    return "—" if x is None else fmt.format(x)


def markdown(rep: dict) -> str:
    """Short table: one row per speed."""
    lines = [
        f"# Gate G3: {'PASS' if rep['gate_g3_pass'] else 'FAIL'}"
        + ("" if rep["claim_ready"] else f" (scout: min n={rep['min_n_per_bin']} < {CLAIM_N})"),
        "",
        "| speed | success off | success on | timing-fail off | timing-fail on | jerk off | jerk on | path off | path on | zero |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["per_speed"]:
        b, t = r.get("baseline") or {}, r.get("treatment") or {}
        lines.append(
            f"| {100*r['speed_m_s']:.0f} cm/s | {_pct(b.get('success'))} | {_pct(t.get('success'))} | "
            f"{_pct(b.get('timing_fail'))} | {_pct(t.get('timing_fail'))} | "
            f"{_num(b.get('jerk_rms'), '{:.1f}')} | {_num(t.get('jerk_rms'), '{:.1f}')} | "
            f"{_num(b.get('path_len_m'))} | {_num(t.get('path_len_m'))} | "
            f"{_pct((r.get('zero') or {}).get('success'))} |"
        )
    lines += ["", "Checks: " + ", ".join(f"{k}={v}" for k, v in rep["checks"].items()),
              f"Timing failures = {' + '.join(rep['timing_failures'])}; pooled kept fraction "
              f"{_num(rep['timing_fail_pooled']['fraction_kept'], '{:.2f}')} (pass ≤ {TIMING_KEEP_MAX})."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", required=True, help="eval run folder, reflex off")
    p.add_argument("--treatment", required=True, help="eval run folder, reflex on")
    p.add_argument("--zero", default=None, help="optional eval run folder with --zero-conditioning")
    p.add_argument("-o", "--output", required=True, help="output folder")
    a = p.parse_args()
    rep = score(load_run(a.baseline), load_run(a.treatment), load_run(a.zero) if a.zero else None)
    rep["runs"] = {"baseline": a.baseline, "treatment": a.treatment, "zero": a.zero}
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_g3_report.json").write_text(json.dumps(rep, indent=2))
    (out / "gate_g3_report.md").write_text(markdown(rep))
    print(markdown(rep))


if __name__ == "__main__":
    main()
