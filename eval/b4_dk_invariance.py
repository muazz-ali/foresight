"""B4: d_k = target_k − p̂_k stays the same across speeds, delays, heights.

Why: speed goes into p̂ via v, delay via Δ, height via p̂z. The relative
approach offset must not soak those up (claim: change p̂, never d).

Checks (Isaac-free):

  1. Synthetic grid — expert rides with a fixed offset d_true. Rebuild absolute
     targets over (v, Δ, z). Relabel → recovered d matches d_true; σ(d) ≤
     s_xy/2 and s_z/2. Absolute targets must move a lot (contrast). Naive
     d = target − p_now (no Δ) fails the delay×speed grid.

  2. Demo scan (optional --root) — at first CLOSE, d = action − forecast(τ)
     where τ is the expert close lead (approach pad; no +dt). Pooled and
     per-speed σ ≤ budgets. Absolute Z must span with object height (same
     fixed-height fingerprint as B2). Delay invariance is covered by (1);
     absolute demo actions are not delay-relabeled.

Usage:
  PYTHONPATH=. python eval/b4_dk_invariance.py
  PYTHONPATH=. python eval/b4_dk_invariance.py --root data/eval/A1_velocity
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interfaces.config import load_yaml  # noqa: E402
from interfaces.forecast import DEFAULT_DT_S, forecast_horizon  # noqa: E402
from interfaces.state import oracle_from_gt  # noqa: E402
from sim.state_machine import STAGE_CLOSE  # noqa: E402

# Claim grid (same spirit as foresight_plan.md speed×delay table).
_SPEEDS_M_S = (0.0, 0.10, 0.20, 0.30)
_DELTAS_S = (0.0, 0.100, 0.200, 0.300)
_HEIGHTS_M = (0.020, 0.035, 0.050, 0.080)
# Fixed approach offset the synthetic expert always uses (metres).
_D_TRUE = np.array([0.0, 0.0, 0.030], dtype=np.float64)
# Absolute XYZ must move at least this much across the synthetic grid.
_ABS_SPAN_MIN_M = 0.040
MIN_N_SPEED = 3  # speed bins need this many closes to vote


def _budgets_m(cfg: dict) -> tuple[float, float]:
    """XY / Z half-slack for d spread (metres).

    XY from grasp_xyz_ok_m / 2; Z from claim_height.z_slack / 2 (same as B2).
    """
    smp = cfg.get("state_machine_params") if isinstance(cfg.get("state_machine_params"), dict) else {}
    xyz_ok = smp.get("grasp_xyz_ok_m")
    if xyz_ok is None:
        mm = smp.get("grasp_xyz_ok_mm", [10.0, 10.0, 12.0])
        s_xy = 0.5 * float(mm[0]) * 1e-3
    else:
        s_xy = 0.5 * float(np.asarray(xyz_ok, dtype=np.float64).reshape(-1)[0])
    ch = cfg.get("claim_height") if isinstance(cfg.get("claim_height"), dict) else {}
    s_z = float(ch.get("z_slack_m", ch.get("z_slack_mm", 5.0) * 1e-3))
    return float(s_xy), 0.5 * float(s_z)


def expert_close_tau_s(cfg: dict) -> float:
    """CLOSE aim lead: ``robot_approach_pad_s`` only — no hand-written ``+dt``.

    The expert holds hand XY at CLOSE (no forecast lead), so ``+dt`` here put a
    fake ``−v·dt`` (~10 mm at 30 cm/s) into every recovered d.
    """
    smp = cfg.get("state_machine_params") if isinstance(cfg.get("state_machine_params"), dict) else {}
    pad = smp.get("robot_approach_pad_s", 0.0)
    return float(pad)


def _sigma(xs: np.ndarray) -> float:
    """Sample std; 0 when fewer than 2 finite values."""
    x = np.asarray(xs, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def _stats_mm(d: np.ndarray) -> dict:
    """Bias / σ / abs-max for a (N, 3) cloud of d, in millimetres."""
    a = np.asarray(d, dtype=np.float64).reshape(-1, 3)
    a = a[np.all(np.isfinite(a), axis=1)]
    if a.size == 0:
        return {
            "n": 0,
            "bias_mm": [float("nan")] * 3,
            "sigma_mm": [float("nan")] * 3,
            "abs_max_mm": [float("nan")] * 3,
        }
    return {
        "n": int(a.shape[0]),
        "bias_mm": (1000.0 * np.mean(a, axis=0)).tolist(),
        "sigma_mm": (1000.0 * np.std(a, axis=0, ddof=1 if a.shape[0] > 1 else 0)).tolist(),
        "abs_max_mm": (1000.0 * np.max(np.abs(a), axis=0)).tolist(),
    }


def synthetic_grid(
    *,
    s_xy: float,
    s_z: float,
    dt: float = DEFAULT_DT_S,
    d_true: np.ndarray = _D_TRUE,
) -> dict:
    """Build tracking-expert targets over (v, Δ, z); check recovered d is flat.

    Call with budgets from ``_budgets_m``. Returns a pass/fail report dict.
    """
    d_true = np.asarray(d_true, dtype=np.float64).reshape(3)
    rel_rows: list[np.ndarray] = []
    abs_rows: list[np.ndarray] = []
    naive_rows: list[np.ndarray] = []  # target − p_now (wrong: absorbs Δ·v)
    detail: list[dict] = []

    for v in _SPEEDS_M_S:
        for delta in _DELTAS_S:
            for z in _HEIGHTS_M:
                p0 = np.array([0.50, 0.0, float(z)], dtype=np.float64)
                vel = np.array([float(v), 0.0, 0.0], dtype=np.float64)
                state = oracle_from_gt(p0, vel, 0.0)
                phat = forecast_horizon(state, float(delta), dt=dt, n=1, p_hat_z=float(z))[0]
                target = phat + d_true
                rel = target - phat
                naive = target - p0  # no delay in the subtract → leaks v·Δ into d
                rel_rows.append(rel)
                abs_rows.append(target)
                naive_rows.append(naive)
                detail.append(
                    {
                        "speed_m_s": float(v),
                        "delta_s": float(delta),
                        "height_m": float(z),
                        "d_mm": (1000.0 * rel).tolist(),
                        "abs_mm": (1000.0 * target).tolist(),
                        "naive_d_mm": (1000.0 * naive).tolist(),
                    }
                )

    rel_a = np.stack(rel_rows)
    abs_a = np.stack(abs_rows)
    naive_a = np.stack(naive_rows)
    err = rel_a - d_true[None, :]
    sigma = np.array([_sigma(rel_a[:, i]) for i in range(3)])
    abs_span = np.array([float(np.max(abs_a[:, i]) - np.min(abs_a[:, i])) for i in range(3)])
    naive_sigma = np.array([_sigma(naive_a[:, i]) for i in range(3)])

    pass_match = bool(np.max(np.abs(err)) <= 1e-12)
    pass_sigma = bool(sigma[0] <= s_xy + 1e-12 and sigma[1] <= s_xy + 1e-12 and sigma[2] <= s_z + 1e-12)
    # Absolute hand target must move with speed/height (otherwise the grid is fake).
    pass_abs_moves = bool(abs_span[0] >= _ABS_SPAN_MIN_M and abs_span[2] >= _ABS_SPAN_MIN_M)
    # Naive (no-Δ) d_x must blow up with v·Δ — proves the contrast is real.
    pass_naive_leaks = bool(naive_sigma[0] >= s_xy * 4.0)

    return {
        "check": "synthetic_grid",
        "n": int(rel_a.shape[0]),
        "d_true_mm": (1000.0 * d_true).tolist(),
        "s_xy_mm": 1000.0 * s_xy,
        "s_z_mm": 1000.0 * s_z,
        "rel": _stats_mm(rel_a),
        "abs_span_mm": (1000.0 * abs_span).tolist(),
        "naive_sigma_mm": (1000.0 * naive_sigma).tolist(),
        "max_abs_err_mm": 1000.0 * float(np.max(np.abs(err))),
        "pass_match": pass_match,
        "pass_sigma": pass_sigma,
        "pass_abs_moves": pass_abs_moves,
        "pass_naive_leaks": pass_naive_leaks,
        "pass": bool(pass_match and pass_sigma and pass_abs_moves and pass_naive_leaks),
        "detail": detail,
    }


def _first_stage_idx(stage: np.ndarray, want: int) -> int | None:
    hits = np.flatnonzero(np.asarray(stage, dtype=np.int64) == int(want))
    return int(hits[0]) if hits.size else None


def _sigma_pass(sigma_m: np.ndarray, *, s_xy: float, s_z: float) -> bool:
    """True when each axis σ is inside its half-slack budget."""
    s = np.asarray(sigma_m, dtype=np.float64).reshape(3)
    return bool(s[0] <= s_xy + 1e-12 and s[1] <= s_xy + 1e-12 and s[2] <= s_z + 1e-12)


def one_demo_close(path: str, *, tau_s: float) -> dict:
    """One episode: d at first CLOSE = action − forecast(expert τ).

    Uses the same lead the SM applies in CLOSE (``tau``), so recovered d is the
    residual approach offset — not the raw aim lead.
    """
    p = Path(path)
    meta: dict = {}
    jp = p.with_suffix(".json")
    if jp.exists():
        meta = json.loads(jp.read_text())
    cat = str(meta.get("category") or meta.get("object_category") or "unknown")
    half = float(meta.get("object_half_height") or float("nan"))
    out: dict = {
        "path": path,
        "category": cat,
        "object_half_height_m": half,
        "closed": False,
        "d_m": [float("nan")] * 3,
        "action_m": [float("nan")] * 3,
        "phat_m": [float("nan")] * 3,
        "speed_m_s": float("nan"),
        "tau_s": float(tau_s),
    }
    with h5py.File(path, "r") as h:
        need = ("stage", "action_pos", "object_pos", "object_vel", "timestamp")
        if any(k not in h for k in need):
            return out
        stage = h["stage"][:]
        action = np.asarray(h["action_pos"][:], dtype=np.float64)
        obj = np.asarray(h["object_pos"][:], dtype=np.float64)
        vel = np.asarray(h["object_vel"][:], dtype=np.float64)
        ts = np.asarray(h["timestamp"][:], dtype=np.float64)
        if "speed_m_s" in h:
            speed = float(np.asarray(h["speed_m_s"][...]).reshape(-1)[0])
        else:
            speed = float(meta.get("speed_m_s") or float("nan"))

    i = _first_stage_idx(stage, STAGE_CLOSE)
    if i is None:
        return out
    # collect logs the world *after* the step, but action_pos[i] was planned from
    # the state before it = the row logged at i−1. Pair them, else d gains −v·dt.
    j = max(i - 1, 0)
    state = oracle_from_gt(obj[j], vel[j], float(ts[j]))
    phat = forecast_horizon(
        state, float(tau_s), dt=DEFAULT_DT_S, n=1, p_hat_z=float(obj[j, 2])
    )[0]
    act = action[i]
    d = act - phat
    out.update(
        {
            "closed": True,
            "d_m": d.tolist(),
            "action_m": act.tolist(),
            "phat_m": phat.tolist(),
            "speed_m_s": speed,
            "z_grasp_m": float(obj[i, 2]),
        }
    )
    return out


def summarize_demos(
    close_rows: list[dict],
    *,
    s_xy: float,
    s_z: float,
    label: str,
    min_n: int = MIN_N_SPEED,
) -> dict:
    """Pooled + per-speed CLOSE d; pass if spreads stay inside budgets."""
    closed = [r for r in close_rows if r.get("closed")]
    d = np.asarray([r["d_m"] for r in closed], dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(d), axis=1)
    d = d[finite]
    closed_f = [r for r, ok in zip(closed, finite) if ok]

    act_z = np.asarray([r["action_m"][2] for r in closed_f], dtype=np.float64)
    zg = np.asarray([r.get("z_grasp_m", float("nan")) for r in closed_f], dtype=np.float64)
    ee_span = float(np.max(act_z) - np.min(act_z)) if act_z.size else 0.0
    zg_span = float(np.max(zg) - np.min(zg)) if zg.size and np.all(np.isfinite(zg)) else 0.0
    fixed_like = bool(zg_span >= 0.010 and ee_span < 0.5 * zg_span)

    close_stats = _stats_mm(d) if d.size else _stats_mm(np.zeros((0, 3)))
    close_sigma = np.array(close_stats["sigma_mm"], dtype=np.float64) * 1e-3
    pass_pooled = bool(
        d.shape[0] >= 2 and _sigma_pass(close_sigma, s_xy=s_xy, s_z=s_z) and not fixed_like
    )

    by_speed: dict[float | None, list[np.ndarray]] = defaultdict(list)
    for r in closed_f:
        spd = r.get("speed_m_s", float("nan"))
        key = round(float(spd), 2) if np.isfinite(spd) else None
        by_speed[key].append(np.asarray(r["d_m"], dtype=np.float64))

    speed_bins: list[dict] = []
    for key in sorted(by_speed, key=lambda k: ((k is None), k if k is not None else -1.0)):
        a = np.stack(by_speed[key])
        st = _stats_mm(a)
        sig = np.array(st["sigma_mm"], dtype=np.float64) * 1e-3
        scored = bool(a.shape[0] >= min_n)
        speed_bins.append(
            {
                "speed_m_s": key,
                "n": int(a.shape[0]),
                "sigma_mm": st["sigma_mm"],
                "bias_mm": st["bias_mm"],
                "scored": scored,
                "pass": bool(scored and _sigma_pass(sig, s_xy=s_xy, s_z=s_z)),
            }
        )
    scored_bins = [b for b in speed_bins if b["scored"]]
    pass_speeds = bool(scored_bins) and all(bool(b["pass"]) for b in scored_bins)
    # One speed only (e.g. smoke): pooled gate is enough; multi-speed needs bins.
    multi = len(scored_bins) >= 2
    pass_all = bool(pass_pooled and (pass_speeds if multi else True))

    return {
        "check": "demos",
        "label": label,
        "episodes": len(close_rows),
        "closed": int(d.shape[0]),
        "s_xy_mm": 1000.0 * s_xy,
        "s_z_mm": 1000.0 * s_z,
        "close": close_stats,
        "action_z_span_mm": 1000.0 * ee_span,
        "z_grasp_span_mm": 1000.0 * zg_span,
        "fixed_height_like": fixed_like,
        "speed_bins": speed_bins,
        "pass_pooled": pass_pooled,
        "pass_speeds": pass_speeds if multi else pass_pooled,
        "pass": pass_all,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Demo root with .h5 (repeatable). Default: data/smoke",
    )
    ap.add_argument(
        "--skip-demos",
        action="store_true",
        help="Only run the synthetic (v, Δ, z) grid.",
    )
    ap.add_argument("--min-n", type=int, default=MIN_N_SPEED)
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "B4_dk_invariance" / "b4_report.json"),
    )
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    s_xy, s_z = _budgets_m(cfg)
    tau_s = expert_close_tau_s(cfg)

    syn = synthetic_grid(s_xy=s_xy, s_z=s_z)
    # Drop bulky per-cell detail from stdout; keep it in the JSON.
    syn_print = {k: v for k, v in syn.items() if k != "detail"}
    print(json.dumps(syn_print, indent=2), flush=True)

    demo_reps: list[dict] = []
    if not args.skip_demos:
        roots = args.root or [
            str(ROOT / "data" / "eval" / "A1_velocity"),
            str(ROOT / "data" / "smoke"),
        ]
        for root in roots:
            root_p = Path(root)
            if not root_p.exists():
                continue
            files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
            close_rows = [one_demo_close(f, tau_s=tau_s) for f in files]
            rep = summarize_demos(
                close_rows,
                s_xy=s_xy,
                s_z=s_z,
                label=str(root_p),
                min_n=int(args.min_n),
            )
            rep["tau_s"] = float(tau_s)
            demo_reps.append(rep)
            print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(syn["pass"]) and (
        True if args.skip_demos else (bool(demo_reps) and all(r["pass"] for r in demo_reps))
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "s_xy_mm": 1000.0 * s_xy,
        "s_z_mm": 1000.0 * s_z,
        "tau_s": float(tau_s),
        "synthetic": syn,
        "demos": demo_reps,
        "pass": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B4: d_k invariance across speeds, delays, heights",
        "",
        f"Gate: σ(d_xy) ≤ {1000.0 * s_xy:.1f} mm, σ(d_z) ≤ {1000.0 * s_z:.1f} mm "
        f"(half of grasp_xyz_ok_m / z_slack). Absolute targets must move; "
        f"naive (no-Δ) d must leak. Demo CLOSE uses expert τ={1000.0 * tau_s:.0f} ms.",
        "",
        "| check | detail | pass |",
        "|---|---|---|",
        f"| synthetic grid | n={syn['n']}, "
        f"σ_rel={['%.2f' % x for x in syn['rel']['sigma_mm']]} mm, "
        f"abs_span={['%.1f' % x for x in syn['abs_span_mm']]} mm, "
        f"naive_σ={['%.1f' % x for x in syn['naive_sigma_mm']]} mm | "
        f"{'yes' if syn['pass'] else 'no'} |",
    ]
    for r in demo_reps:
        sig = r.get("close", {}).get("sigma_mm", [float("nan")] * 3)
        lines.append(
            f"| demos `{Path(r['label']).name}` | "
            f"close n={r.get('closed', 0)}, σ={['%.2f' % x for x in sig]} mm, "
            f"fixed_like={r.get('fixed_height_like')} | "
            f"{'yes' if r['pass'] else 'no'} |"
        )
        for b in r.get("speed_bins") or []:
            spd = "—" if b["speed_m_s"] is None else f"{100.0 * b['speed_m_s']:.0f}"
            tag = "" if b["scored"] else " (n small)"
            lines.append(
                f"| ↳ {spd} cm/s | n={b['n']}, σ={['%.2f' % x for x in b['sigma_mm']]} mm | "
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
