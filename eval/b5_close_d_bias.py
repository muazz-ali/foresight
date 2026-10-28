"""B5: no hand-trailing bias — mean d at CLOSE is 0 ± 2 mm.

Plain words: after subtracting the expert close forecast (τ = approach pad +
dt), the leftover offset d = action − p̂ should centre on zero. A systematic
lead/lag (hand trailing the object, or over-leading) shows up as a non-zero
mean. Same d as B4; this gate is on the **mean**, not the spread.

Isaac-free. Pass: every scored (speed) bin and the pool have
|mean(d_x)|, |mean(d_y)|, |mean(d_z)| ≤ 2 mm.

Usage:
  PYTHONPATH=. python eval/b5_close_d_bias.py
  PYTHONPATH=. python eval/b5_close_d_bias.py --root data/eval/A1_velocity
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b4_dk_invariance import (  # noqa: E402
    _stats_mm,
    expert_close_tau_s,
    one_demo_close,
)
from interfaces.config import load_yaml  # noqa: E402

BIAS_MAX_MM = 2.0
MIN_N = 3


def _bias_pass(bias_mm: list[float] | np.ndarray, *, lim_mm: float = BIAS_MAX_MM) -> bool:
    """True when each axis |mean| is inside ±lim_mm."""
    b = np.asarray(bias_mm, dtype=np.float64).reshape(-1)
    if b.size < 3 or not np.all(np.isfinite(b)):
        return False
    return bool(np.all(np.abs(b[:3]) <= float(lim_mm) + 1e-9))


def summarize(
    rows: list[dict],
    *,
    label: str,
    lim_mm: float = BIAS_MAX_MM,
    min_n: int = MIN_N,
) -> dict:
    """Pooled + per-speed mean d; pass if every scored bin is inside ±lim."""
    closed = [r for r in rows if r.get("closed")]
    d = np.asarray([r["d_m"] for r in closed], dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(d), axis=1)
    d = d[finite]
    closed_f = [r for r, ok in zip(closed, finite) if ok]

    pooled = _stats_mm(d) if d.size else _stats_mm(np.zeros((0, 3)))
    pass_pooled = bool(d.shape[0] >= 2 and _bias_pass(pooled["bias_mm"], lim_mm=lim_mm))

    by_speed: dict[float | None, list[np.ndarray]] = defaultdict(list)
    for r in closed_f:
        spd = r.get("speed_m_s", float("nan"))
        key = round(float(spd), 2) if np.isfinite(spd) else None
        by_speed[key].append(np.asarray(r["d_m"], dtype=np.float64))

    bins: list[dict] = []
    for key in sorted(by_speed, key=lambda k: ((k is None), k if k is not None else -1.0)):
        a = np.stack(by_speed[key])
        st = _stats_mm(a)
        scored = bool(a.shape[0] >= min_n)
        bins.append(
            {
                "speed_m_s": key,
                "n": int(a.shape[0]),
                "bias_mm": st["bias_mm"],
                "sigma_mm": st["sigma_mm"],
                "scored": scored,
                "pass": bool(scored and _bias_pass(st["bias_mm"], lim_mm=lim_mm)),
            }
        )
    scored = [b for b in bins if b["scored"]]
    multi = len(scored) >= 2
    pass_speeds = bool(scored) and all(bool(b["pass"]) for b in scored)
    return {
        "check": "close_d_bias",
        "label": label,
        "episodes": len(rows),
        "closed": int(d.shape[0]),
        "lim_mm": float(lim_mm),
        "pooled": pooled,
        "speed_bins": bins,
        "pass_pooled": pass_pooled,
        "pass_speeds": pass_speeds if multi else pass_pooled,
        "pass": bool(pass_pooled and (pass_speeds if multi else True)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Demo root with .h5 (repeatable). Default: A1_velocity + smoke",
    )
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--lim-mm", type=float, default=BIAS_MAX_MM)
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "B5_close_d_bias" / "b5_report.json"),
    )
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    tau_s = expert_close_tau_s(cfg)
    lim = float(args.lim_mm)

    roots = args.root or [
        str(ROOT / "data" / "eval" / "A1_velocity"),
        str(ROOT / "data" / "smoke"),
    ]
    reports: list[dict] = []
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            continue
        files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
        rows = [one_demo_close(f, tau_s=tau_s) for f in files]
        rep = summarize(rows, label=str(root_p), lim_mm=lim, min_n=int(args.min_n))
        rep["tau_s"] = float(tau_s)
        reports.append(rep)
        print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(reports) and all(bool(r["pass"]) for r in reports)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lim_mm": lim, "tau_s": float(tau_s), "roots": reports, "pass": all_pass}
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B5: no hand-trailing bias (mean d at close)",
        "",
        f"Gate: |mean(d_xyz)| ≤ {lim:.1f} mm after forecast(τ={1000.0 * tau_s:.0f} ms).",
        "A growing mean with speed usually means τ is wrong (hand trails or leads).",
        "",
    ]
    for r in reports:
        lines.append(f"## `{Path(r['label']).name}` — {'PASS' if r['pass'] else 'FAIL'}")
        lines.append("")
        pb = r["pooled"]["bias_mm"]
        lines.append(
            f"Pooled n={r['closed']}: bias = "
            f"[{pb[0]:.2f}, {pb[1]:.2f}, {pb[2]:.2f}] mm "
            f"({'yes' if r['pass_pooled'] else 'no'})"
        )
        lines.append("")
        lines.append("| speed cm/s | n | bias x/y/z mm | pass |")
        lines.append("|---|---|---|---|")
        for b in r["speed_bins"]:
            spd = "—" if b["speed_m_s"] is None else f"{100.0 * b['speed_m_s']:.0f}"
            tag = "" if b["scored"] else " (n small)"
            bb = b["bias_mm"]
            lines.append(
                f"| {spd} | {b['n']} | "
                f"{bb[0]:.2f} / {bb[1]:.2f} / {bb[2]:.2f} | "
                f"{'yes' if b['pass'] else 'no'}{tag} |"
            )
        lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
