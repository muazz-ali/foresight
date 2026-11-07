"""A3: forecast error of p̂_k vs the true position at τ_k, in units of σ_forecast.  Pass: within 3σ.

For each chunk step k (0 … n−1) the policy's targets assume the object is at
    p̂_k = p + v·τ_k,   τ_k = Δ + k·h          (interfaces.forecast.forecast_horizon)
where p, v come from the live static-camera filter (perception.StaticCamEstimator, current config)
at the moment the chunk is planned. We log, per axis,
    z = (p̂_k − p_true(t + τ_k)) / σ_forecast(τ_k)
for every cell of the speed × delay grid in foresight_plan.md.

σ_forecast (primary) = the filter covariance carried along the constant-velocity forecast:
    σ²(τ) = P_pp + 2τ·P_pv + τ²·P_vv           (per axis; no added process noise)
Also reported, for contrast, the σ that ObjectState.predict() puts in the message today
(P_pp + 1e-4·τ, which ignores velocity uncertainty).

Pass per cell: ≥ 99% of samples have |z_x| ≤ 3 and |z_y| ≤ 3 (Gaussian: 99.46%).
Also shown: mean z (bias) and std z (1.0 = the filter's σ is honest; > 1 = overconfident).

Offline, no Isaac. Reads the A1 recordings (expert throws at fixed speed, current setup). The delay
only shifts the comparison time, so one recording serves every Δ. Forecasts are made at every frame
with a valid filter state after warm-up; a pair is used only while the object is still free at
t + τ_k (before hold / lift / off-table), so long horizons have fewer samples (reported per k).

Usage: python eval/a3_forecast_error.py [--root data/eval/A1_velocity] [--cells 0.10:100 ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

from interfaces.config import load_yaml  # noqa: E402
from interfaces.forecast import DEFAULT_CHUNK, forecast_horizon  # noqa: E402
from perception.estimator import StaticCamEstimator  # noqa: E402

import g2_perception_report as g2  # noqa: E402  (episode reader + stop rules)
from a1_velocity_ratio import preflight  # noqa: E402  (fails loudly on setup mismatch)

GRID = ("0.10:100", "0.10:300", "0.20:100", "0.20:300", "0.30:200", "0.30:300")  # foresight_plan.md
WARMUP_S = 0.20
PASS_FRAC = 0.99
_CFG = None


def _init():
    global _CFG
    cv2.setNumThreads(1)
    _CFG = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")


def replay(path: str) -> dict:
    """Run the filter over one episode; keep its state (p, v, P) per frame and the truth."""
    ep = g2._read_episode(path)
    ts, orc, rgb = ep["ts"], ep["oracle"], ep["rgb"]
    n = len(ts)
    out = {"path": path, "ts": ts.tolist(), "truth": orc[:, :2].tolist(), "states": [], "start_ok": False}
    if n < 10:
        return out
    est = StaticCamEstimator.from_cfg(_CFG)
    if ep["xy_bounds"] is not None:
        est.search_xy_bounds = tuple(map(tuple, ep["xy_bounds"]))
    out["start_ok"] = bool(est.start(rgb[0], orc[0, :3]))
    t_first = None
    for t in range(n):
        st = est.step(rgb[t], ts[t])
        if est.last_uv is not None and t_first is None:
            t_first = ts[t]
        if st is None or not st.valid or t_first is None or ts[t] - t_first < WARMUP_S:
            continue
        C = np.asarray(st.covariance, dtype=np.float64)
        C = C.reshape(6, 6) if C.size == 36 else np.diag(C.reshape(-1)[:6])
        out["states"].append({
            "i": t, "t": float(ts[t]),
            "p": [float(x) for x in st.position], "v": [float(x) for x in st.velocity],
            # per axis (x, y): P_pp, P_pv, P_vv  — filter state order [x, y, z, vx, vy, vz]
            "Pxx": [C[0, 0], C[0, 3], C[3, 3]], "Pyy": [C[1, 1], C[1, 4], C[4, 4]],
        })
    return out


def cell_stats(eps, speed, delta_s, h, n_k):
    zs, zs_msg, errs, per_k = [], [], [], np.zeros(n_k, int)
    err_k = [[] for _ in range(n_k)]
    for ep in eps:
        ts = np.asarray(ep["ts"]); tr = np.asarray(ep["truth"])
        if len(ts) < 2:
            continue
        for s in ep["states"]:
            from interfaces.state import ObjectState
            st = ObjectState(np.array(s["p"]), np.array(s["v"]), np.zeros((6, 6)), s["t"], True)
            fc = forecast_horizon(st, delta_s, dt=h, n=n_k)[:, :2]
            for k in range(n_k):
                tau = delta_s + k * h
                tq = s["t"] + tau
                if tq > ts[-1] + 1e-9:
                    break  # object no longer free (or episode over) at t + τ_k
                true = np.array([np.interp(tq, ts, tr[:, 0]), np.interp(tq, ts, tr[:, 1])])
                e = fc[k] - true
                sig = [np.sqrt(max(P[0] + 2 * tau * P[1] + tau * tau * P[2], 1e-12)) for P in (s["Pxx"], s["Pyy"])]
                sig_msg = [np.sqrt(max(P[0] + 1e-4 * tau, 1e-12)) for P in (s["Pxx"], s["Pyy"])]
                zs.append(e / np.array(sig)); zs_msg.append(e / np.array(sig_msg)); errs.append(np.linalg.norm(e))
                per_k[k] += 1
                err_k[k].append(float(np.linalg.norm(e)))
    if not zs:
        return {"speed": speed, "delta_ms": delta_s * 1000, "n": 0}
    z = np.array(zs); zm = np.array(zs_msg); err = np.array(errs)
    inside = np.all(np.abs(z) <= 3, axis=1)
    inside_m = np.all(np.abs(zm) <= 3, axis=1)
    return {
        "speed": speed, "delta_ms": delta_s * 1000, "n": int(len(z)),
        "frac_within_3sigma": float(inside.mean()),
        "z_mean_xy": z.mean(0).tolist(), "z_std_xy": z.std(0).tolist(),
        "abs_z_p99": float(np.percentile(np.abs(z).max(1), 99)),
        "err_mm_median": float(np.median(err) * 1000), "err_mm_p95": float(np.percentile(err, 95) * 1000),
        "msg_sigma_frac_within_3sigma": float(inside_m.mean()),
        "k_coverage": per_k.tolist(),
        "err_mm_by_k": {str(k): [float(np.median(v) * 1000), float(np.percentile(v, 95) * 1000)] for k, v in enumerate(err_k) if len(v) >= 20},
        "k_max_covered": int(np.max(np.nonzero(per_k)[0])) if per_k.any() else -1,
        "pass": bool(inside.mean() >= PASS_FRAC),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "eval" / "A1_velocity"))
    ap.add_argument("--cells", nargs="+", default=list(GRID), help="speed:delay_ms")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "A3_forecast"))
    a = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    sim = cfg.get("sim", {})
    if "dt" not in sim:
        sys.exit("A3 stop: no sim.dt in sim/phase0_cfg.yaml; cannot place τ_k = Δ + k·h.")
    h = float(sim["dt"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cells = [(float(c.split(":")[0]), float(c.split(":")[1]) / 1000.0) for c in a.cells]
    speeds = sorted({s for s, _ in cells})
    eps_by_speed = {}
    with Pool(a.workers, initializer=_init) as pool:
        for sp in speeds:
            vdir = Path(a.root) / f"v{sp:.2f}"
            files = sorted(glob.glob(str(vdir / "**" / "*.h5"), recursive=True))
            if not files:
                sys.exit(f"A3 stop: no recordings for {sp:.2f} m/s under {vdir}. Record them with the current setup first.")
            preflight(files)
            eps = pool.map(replay, files)
            n_ok = sum(e["start_ok"] for e in eps)
            if n_ok < 0.5 * len(eps):
                sys.exit(f"A3 stop at {sp:.2f} m/s: tracker started in only {n_ok}/{len(eps)} episodes; "
                         "camera config likely does not match these recordings.")
            eps_by_speed[sp] = eps
    report = [cell_stats(eps_by_speed[s], s, d, h, a.chunk) for s, d in cells]
    (out / "a3_report.json").write_text(json.dumps(report, indent=1))
    lines = ["# A3: forecast error of p̂_k vs true position at τ_k = Δ + k·h", "",
             f"h = {h*1000:.1f} ms, chunk = {a.chunk}. z = (p̂_k − p_true) / σ_forecast per axis; "
             "σ² = P_pp + 2τP_pv + τ²P_vv. Pass: ≥ 99% of samples with |z| ≤ 3 on both axes.", "",
             "| speed | Δ ms | samples | k covered | within 3σ | mean z (x, y) | std z (x, y) | |z| p99 | error mm median / p95 | within 3σ with today's message σ | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report:
        if not r["n"]:
            lines.append(f"| {r['speed']:.2f} | {r['delta_ms']:.0f} | 0 | – | – | – | – | – | – | – | no data |")
            continue
        lines.append(f"| {r['speed']:.2f} | {r['delta_ms']:.0f} | {r['n']} | 0–{r['k_max_covered']} | {r['frac_within_3sigma']*100:.1f}% | "
                     f"{r['z_mean_xy'][0]:+.2f}, {r['z_mean_xy'][1]:+.2f} | {r['z_std_xy'][0]:.2f}, {r['z_std_xy'][1]:.2f} | "
                     f"{r['abs_z_p99']:.2f} | {r['err_mm_median']:.1f} / {r['err_mm_p95']:.1f} | "
                     f"{r['msg_sigma_frac_within_3sigma']*100:.1f}% | {'yes' if r['pass'] else 'no'} |")
    lines += ["", "**Overall: " + ("PASS" if all(r.get("pass") for r in report) else "FAIL") + "**"]
    (out / "REPORT_auto.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
