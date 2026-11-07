"""A2: measured Δ vs injected Δ — bias ≈ 0, σ_Δ ≤ 5 ms.

Inject a known delay (100 / 200 / 300 ms). Recover Δ from timestamps the same
way the claim loop does:

  measured = apply_at − t0          (bookkeeping)
  measured = release_time − t0      (enforce_apply_at hold, 30 Hz steps)

Also checks that the 5-number pack lead uses that cycle's reported Δ
(p̂x − px ≈ vx · reported_Δ).

Isaac-free. Pass is on the timestamp recoveries, not on reported−applied
jitter (yaml still draws ~N(0, 10 ms) on reported).

Usage:
  PYTHONPATH=. python eval/a2_delta_inject.py
  PYTHONPATH=. python eval/a2_delta_inject.py --delays-ms 100 200 300 --n 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.delay_injection import DelayInjection  # noqa: E402
from interfaces.forecast import conditioning_at_measured_delta  # noqa: E402
from interfaces.state import ObjectState  # noqa: E402

DT_S = 1.0 / 30.0
DEFAULT_DELAYS_MS = (100, 200, 300)
BIAS_MAX_MS = 1.0  # ≈ 0
SIGMA_MAX_MS = 5.0
PACK_LEAD_MAX_MM = 0.1  # pack lead must match v·Δ within 0.1 mm


def _moving_state(v_x: float = 0.20, *, t: float = 0.0) -> ObjectState:
    """Constant-speed object for pack-lead check (m/s)."""
    return ObjectState(
        position=np.array([0.50, 0.0, 0.03], dtype=np.float64),
        velocity=np.array([float(v_x), 0.0, 0.0], dtype=np.float64),
        covariance=np.full(6, 1e-6, dtype=np.float64),
        timestamp=float(t),
        valid=True,
    )


def _stats_ms(err_s: np.ndarray) -> dict:
    """Bias / σ of error (s → ms)."""
    e = np.asarray(err_s, dtype=np.float64).reshape(-1)
    if e.size == 0:
        return {
            "n": 0,
            "bias_ms": float("nan"),
            "sigma_ms": float("nan"),
            "pass": False,
        }
    bias = float(np.mean(e))
    sigma = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
    ok = abs(bias) * 1000.0 <= BIAS_MAX_MS and sigma * 1000.0 <= SIGMA_MAX_MS
    return {
        "n": int(e.size),
        "bias_ms": 1000.0 * bias,
        "sigma_ms": 1000.0 * sigma,
        "pass": bool(ok),
    }


def run_cell(
    inject_s: float,
    *,
    n: int = 1000,
    dt: float = DT_S,
    seed: int = 0,
    jitter_std_s: float = 0.010,
    v_x: float = 0.20,
) -> dict:
    """One inject cell: bookkeeping + 30 Hz release + pack lead.

    Call once per injected delay. Returns a dict with bias/σ and pass flags.
    """
    delay = DelayInjection(
        jitter_std_s=float(jitter_std_s),
        eval_delta_s=float(inject_s),
        seed=int(seed),
    )
    delay.set_eval_delta(float(inject_s))

    # --- (1) Bookkeeping: measured = apply_at − t0 ---
    book_err = []
    report_err = []
    pack_err_m = []
    t0 = 0.0
    for i in range(int(n)):
        s = delay.sample(t0, mode="eval")
        measured = float(s.apply_at) - float(s.t0)
        book_err.append(measured - float(inject_s))
        report_err.append(float(s.reported_delta_s) - float(inject_s))

        # Pack lead must use *reported* Δ (what the policy is told).
        st = _moving_state(v_x, t=t0)
        cond = conditioning_at_measured_delta(st, float(s.reported_delta_s))
        lead = float(cond[0]) - float(st.position[0])
        want = float(v_x) * float(s.reported_delta_s)
        pack_err_m.append(lead - want)

        t0 += float(dt)

    # --- (2) Enforce-apply pipeline (mirrors sim/collect.py hold) ---
    # Sample at capture time t0≈now; hold until now >= apply_at; measure
    # release_time − t0 vs inject.
    delay2 = DelayInjection(
        jitter_std_s=float(jitter_std_s),
        eval_delta_s=float(inject_s),
        seed=int(seed) + 17,
    )
    delay2.set_eval_delta(float(inject_s))
    pending: list[tuple[float, float]] = []  # (apply_at, t0)
    release_err = []
    now = 0.0
    # Run long enough that every sample gets a chance to release.
    n_steps = int(n) + int(np.ceil(float(inject_s) / float(dt))) + 2
    samples_left = int(n)
    for _ in range(n_steps):
        if samples_left > 0:
            t_cap = float(now)  # last_capture_t0 ≈ sim time before step
            samp = delay2.sample(t_cap, mode="eval")
            pending.append((float(samp.apply_at), float(samp.t0)))
            samples_left -= 1
        still: list[tuple[float, float]] = []
        for apply_at, t_cap in pending:
            if now + 1e-12 >= float(apply_at):
                measured = float(now) - float(t_cap)
                release_err.append(measured - float(inject_s))
            else:
                still.append((apply_at, t_cap))
        pending = still
        now += float(dt)

    book = _stats_ms(np.asarray(book_err))
    release = _stats_ms(np.asarray(release_err))
    reported = _stats_ms(np.asarray(report_err))
    pack_e = np.asarray(pack_err_m, dtype=np.float64)
    pack_ok = bool(pack_e.size > 0 and float(np.max(np.abs(pack_e))) * 1000.0 <= PACK_LEAD_MAX_MM)

    cell_pass = bool(book["pass"] and release["pass"] and pack_ok)
    return {
        "inject_ms": 1000.0 * float(inject_s),
        "n": int(n),
        "dt_ms": 1000.0 * float(dt),
        "jitter_std_ms": 1000.0 * float(jitter_std_s),
        "bookkeeping_apply_at_minus_t0": book,
        "enforce_release_minus_t0": release,
        "reported_minus_inject": reported,
        "pack_lead_err_max_mm": 1000.0 * float(np.max(np.abs(pack_e))) if pack_e.size else float("nan"),
        "pack_lead_ok": pack_ok,
        "pass": cell_pass,
        "note": (
            "Gate is bookkeeping + enforce-release vs inject (|bias|≤1 ms, σ≤5 ms) "
            "and pack lead. reported−inject σ ≈ yaml jitter (~10 ms), not gated."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delays-ms", type=float, nargs="+", default=list(DEFAULT_DELAYS_MS))
    ap.add_argument("--n", type=int, default=1000, help="cycles per inject cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--jitter-ms",
        type=float,
        default=10.0,
        help="reported−applied noise std (yaml default 10). Does not affect apply_at.",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "A2_delta" / "a2_report.json"),
    )
    args = ap.parse_args()

    cells = []
    for ms in args.delays_ms:
        cell = run_cell(
            float(ms) * 1e-3,
            n=int(args.n),
            seed=int(args.seed) + int(ms),
            jitter_std_s=float(args.jitter_ms) * 1e-3,
        )
        cells.append(cell)
        print(json.dumps(cell), flush=True)

    all_pass = all(bool(c["pass"]) for c in cells)
    report = {
        "gate": {
            "bias_max_ms": BIAS_MAX_MS,
            "sigma_max_ms": SIGMA_MAX_MS,
            "pack_lead_max_mm": PACK_LEAD_MAX_MM,
        },
        "cells": cells,
        "pass": bool(all_pass),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# A2: measured Δ vs injected Δ",
        "",
        f"Gate: |bias| ≤ {BIAS_MAX_MS:.0f} ms, σ ≤ {SIGMA_MAX_MS:.0f} ms "
        f"(apply_at−t0 and enforce release−t0). Pack lead within {PACK_LEAD_MAX_MM} mm.",
        "",
        "| inject ms | book bias/σ ms | release bias/σ ms | reported σ ms (info) | pack lead | pass |",
        "|---|---|---|---|---|---|",
    ]
    for c in cells:
        b = c["bookkeeping_apply_at_minus_t0"]
        r = c["enforce_release_minus_t0"]
        info = c["reported_minus_inject"]
        lines.append(
            f"| {c['inject_ms']:.0f} | {b['bias_ms']:+.3f} / {b['sigma_ms']:.3f} | "
            f"{r['bias_ms']:+.3f} / {r['sigma_ms']:.3f} | {info['sigma_ms']:.2f} | "
            f"{c['pack_lead_err_max_mm']:.4f} mm | {'yes' if c['pass'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
