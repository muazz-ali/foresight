#!/usr/bin/env python3
"""P# conditioning-signal diagnostic (scratch under tools/diag/).

Asks: do the live conditioning numbers carry information, independent of policy use?
Read-only on production trees. Writes report under tools/diag/out/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from interfaces.state import ObjectState, conditioning_vector, oracle_from_gt  # noqa: E402
from policy.features import CONDITIONING_DIM, CONDITIONING_NAMES, PROPRIO_DIM  # noqa: E402
from policy.transforms import expand_state_stats  # noqa: E402

OUT_DIR = ROOT / "tools" / "diag" / "out"
DATA_P1 = ROOT / "data" / "p1"
LEROBOT_P1 = ROOT / "data" / "lerobot" / "p1"

# Layout from interfaces/state.py:101-123 + features.py:23-36.
# Frame: ObjectState.position docstring = "meters, world/table frame" (state.py:23);
# scene feeds Isaac root_pos_w / root_lin_vel_w (sim/scene.py:537-548).
# workspace_scale default 1.0 → units remain meters / m/s (state.py:105-121).
LAYOUT = [
    # idx, name, frame, units, source_note
    (0, "p_hat_x", "Isaac world (root_pos_w)", "m / workspace_scale", "state.predict(delta).position[0]"),
    (1, "p_hat_y", "Isaac world (root_pos_w)", "m / workspace_scale", "state.predict(delta).position[1]"),
    (2, "v_hat_x", "FD / root vel", "m/s / workspace_scale", "state.velocity[0]"),
    (3, "v_hat_y", "FD / root vel", "m/s / workspace_scale", "state.velocity[1]"),
]

STD_EPS = 1e-6
UNIQUE_LOW = 10


def _list_success_h5(root: Path) -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    for js in sorted(root.rglob("*.json")):
        if "speed_" not in str(js.parent):
            continue
        try:
            meta = json.loads(js.read_text())
        except json.JSONDecodeError:
            continue
        h5 = js.with_suffix(".h5")
        if not h5.is_file():
            continue
        if not meta.get("success", False):
            continue
        out.append((h5, meta))
    return out


def _dim_stats(X: np.ndarray) -> list[dict]:
    """Per-dim stats over rows of X (N, 12)."""
    rows = []
    for i in range(CONDITIONING_DIM):
        col = X[:, i]
        finite = np.isfinite(col)
        n = col.size
        n_bad = int((~finite).sum())
        good = col[finite]
        if good.size == 0:
            rows.append(
                {
                    "idx": i,
                    "name": CONDITIONING_NAMES[i],
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "pct_zeros": 100.0,
                    "pct_nan_inf": 100.0 * n_bad / max(n, 1),
                    "n_unique": 0,
                    "flag": "NAN",
                }
            )
            continue
        # unique with light rounding to absorb float noise
        rounded = np.round(good, decimals=8)
        n_unique = int(np.unique(rounded).size)
        std = float(good.std())
        pct_zeros = 100.0 * float(np.sum(np.abs(good) < 1e-12)) / good.size
        flag = "OK"
        if n_bad:
            flag = "NAN"
        elif std < STD_EPS or n_unique < UNIQUE_LOW:
            flag = "LOW_VAR"
        rows.append(
            {
                "idx": i,
                "name": CONDITIONING_NAMES[i],
                "mean": float(good.mean()),
                "std": std,
                "min": float(good.min()),
                "max": float(good.max()),
                "pct_zeros": pct_zeros,
                "pct_nan_inf": 100.0 * n_bad / max(n, 1),
                "n_unique": n_unique,
                "flag": flag,
            }
        )
    return rows


def _episode_table(h5_path: Path, meta: dict, *, stride: int = 5) -> dict:
    with h5py.File(h5_path, "r") as f:
        cond = np.asarray(f["conditioning"], dtype=np.float64)
        vel = np.asarray(f["object_vel"], dtype=np.float64)
        pos = np.asarray(f["object_pos"], dtype=np.float64)
        ts = np.asarray(f["timestamp"], dtype=np.float64)
    speed = float(meta.get("speed_m_s", float("nan")))
    rows = []
    for t in range(0, cond.shape[0], stride):
        vnorm = float(np.linalg.norm(vel[t]))
        rows.append(
            {
                "t": t,
                "time_s": float(ts[t]),
                "v_norm": vnorm,
                "p_hat": cond[t, 0:3].tolist(),
                "p": pos[t].tolist(),
                "v_hat": cond[t, 3:6].tolist(),
                "delta": float(cond[t, 6]),
                "delta_p_hat_minus_p": (cond[t, 0:3] - pos[t]).tolist(),
            }
        )
    return {
        "path": str(h5_path),
        "speed_m_s": speed,
        "delta_meta": meta.get("delta"),
        "n_frames": int(cond.shape[0]),
        "rows": rows,
        "p_hat_std": cond[:, 0:3].std(axis=0).tolist(),
        "v_hat_std": cond[:, 3:6].std(axis=0).tolist(),
        "v_norm_mean": float(np.linalg.norm(vel, axis=1).mean()),
        "v_norm_max": float(np.linalg.norm(vel, axis=1).max()),
    }


def _predict_rmse(h5_path: Path) -> dict:
    """Compare logged p̂(t)=p+v*Δ vs GT position at nearest timestamp ≈ t+Δ."""
    with h5py.File(h5_path, "r") as f:
        cond = np.asarray(f["conditioning"], dtype=np.float64)
        oracle = np.asarray(f["oracle_state"], dtype=np.float64)
        pos = np.asarray(f["object_pos"], dtype=np.float64)
        vel = np.asarray(f["object_vel"], dtype=np.float64)
        ts = np.asarray(f["timestamp"], dtype=np.float64)

    # Recompute conditioning from oracle to confirm producer fidelity.
    recon_err = []
    for t in range(oracle.shape[0]):
        st = ObjectState(
            position=oracle[t, 0:3],
            velocity=oracle[t, 3:6],
            covariance=oracle[t, 6:12],
            timestamp=float(oracle[t, 12]),
            valid=bool(oracle[t, 13] > 0.5),
        )
        delta = float(cond[t, 6])
        recon = conditioning_vector(st, delta=delta)
        recon_err.append(np.linalg.norm(recon - cond[t]))
    recon_err = np.asarray(recon_err)

    # Future GT: find index with timestamp closest to t + delta
    errs = []
    deltas_used = []
    for t in range(pos.shape[0]):
        delta = float(cond[t, 6])
        target = ts[t] + delta
        # only if target within episode
        if target > ts[-1] + 1e-6:
            continue
        j = int(np.argmin(np.abs(ts - target)))
        if abs(ts[j] - target) > 0.021:  # > ~half frame @ 25Hz
            continue
        pred = cond[t, 0:3]
        gt = pos[j]
        errs.append(pred - gt)
        deltas_used.append(delta)
        # also check CV identity: p+v*d should match pred
    if not errs:
        return {
            "path": str(h5_path),
            "n_pairs": 0,
            "rmse_m": None,
            "corr": None,
            "recon_max_l2": float(recon_err.max()) if recon_err.size else None,
        }
    E = np.asarray(errs)
    rmse = float(np.sqrt(np.mean(E**2)))
    # Pearson corr per XYZ between pred and gt for paired frames
    preds = []
    gts = []
    for t in range(pos.shape[0]):
        delta = float(cond[t, 6])
        target = ts[t] + delta
        if target > ts[-1] + 1e-6:
            continue
        j = int(np.argmin(np.abs(ts - target)))
        if abs(ts[j] - target) > 0.021:
            continue
        preds.append(cond[t, 0:3])
        gts.append(pos[j])
    P = np.asarray(preds)
    G = np.asarray(gts)
    corrs = []
    for k in range(3):
        if P[:, k].std() < 1e-12 or G[:, k].std() < 1e-12:
            corrs.append(float("nan"))
        else:
            corrs.append(float(np.corrcoef(P[:, k], G[:, k])[0, 1]))
    # CV residual: ||p̂ - (p+vΔ)|| should be ~0 for oracle CV producer
    cv_res = []
    for t in range(pos.shape[0]):
        delta = float(cond[t, 6])
        cv = pos[t] + vel[t] * delta
        cv_res.append(np.linalg.norm(cond[t, 0:3] - cv))
    return {
        "path": str(h5_path),
        "n_pairs": int(len(errs)),
        "rmse_m": rmse,
        "rmse_xyz_m": np.sqrt(np.mean(E**2, axis=0)).tolist(),
        "corr_xyz": corrs,
        "mean_abs_err_m": float(np.mean(np.abs(E))),
        "cv_identity_max_l2": float(np.max(cv_res)),
        "recon_max_l2": float(recon_err.max()),
        "delta_unique": sorted(set(float(d) for d in deltas_used)),
    }


def _verdict_row(stat: dict, idx: int) -> str:
    """INFORMATIVE | CONSTANT | DEGENERATE | MISLABELED-FRAME | NAN."""
    if stat["flag"] == "NAN" or stat["pct_nan_inf"] > 0:
        return "NAN"
    if stat["std"] < STD_EPS:
        return "CONSTANT"
    if stat["n_unique"] < UNIQUE_LOW:
        return "DEGENERATE"
    # Frame check: XY should be workspace-scale meters (~0.1–1.0 for table)
    if idx in (0, 1):
        if abs(stat["mean"]) > 50 or (abs(stat["mean"]) < 1e-6 and stat["std"] < 1e-4):
            return "MISLABELED-FRAME"
    return "INFORMATIVE"


def _load_lerobot_cond_stats() -> dict | None:
    """Aggregate conditioning stats from episodes_stats.jsonl if present."""
    path = LEROBOT_P1 / "meta" / "episodes_stats.jsonl"
    info = LEROBOT_P1 / "meta" / "info.json"
    if not path.is_file():
        return None
    means, stds, mins, maxs, counts = [], [], [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        st = d["stats"]["observation.conditioning"]
        means.append(st["mean"])
        stds.append(st["std"])
        mins.append(st["min"])
        maxs.append(st["max"])
        counts.append(st["count"][0] if isinstance(st["count"], list) else st["count"])
    info_d = json.loads(info.read_text()) if info.is_file() else {}
    # Weighted pool approx via concat of per-ep means is wrong for true global;
    # we only report what is on disk + episode count.
    return {
        "path": str(path),
        "n_episodes_in_lerobot": int(info_d.get("total_episodes", len(means))),
        "n_frames_in_lerobot": int(info_d.get("total_frames", -1)),
        "note": "lerobot/p1 may be a stub; H5 pool is authoritative for Phase-1 demos",
        "per_episode_std_mean": np.mean(np.asarray(stds), axis=0).tolist() if stds else None,
        "per_episode_std_max": np.max(np.asarray(stds), axis=0).tolist() if stds else None,
    }


def _norm_stats_from_h5_pool(cond_mean: np.ndarray, cond_std: np.ndarray, cond_min: np.ndarray, cond_max: np.ndarray) -> dict:
    """Simulate expand_state_stats merge for Model B conditioning dims."""
    fake_stats = {
        "observation.state": {
            "min": np.zeros(PROPRIO_DIM, np.float32),
            "max": np.ones(PROPRIO_DIM, np.float32),
            "mean": np.zeros(PROPRIO_DIM, np.float32),
            "std": np.ones(PROPRIO_DIM, np.float32),
            "count": np.array([1]),
        },
        "observation.conditioning": {
            "min": cond_min.astype(np.float32),
            "max": cond_max.astype(np.float32),
            "mean": cond_mean.astype(np.float32),
            "std": cond_std.astype(np.float32),
            "count": np.array([1]),
        },
    }
    merged = expand_state_stats(fake_stats, "B")
    st = merged["observation.state"]
    cond_std_m = st["std"][PROPRIO_DIM:]
    zero_std_dims = [i for i, s in enumerate(cond_std_m) if float(s) < STD_EPS]
    return {
        "transforms_expand_state_stats": "policy/transforms.py:165-190",
        "appends_cond": "policy/transforms.py:128-139 (train) / scripts/eval_policy.py:184-190 (eval)",
        "cond_std_in_merged_state": cond_std_m.tolist(),
        "zero_std_cond_dims": zero_std_dims,
        "zero_std_names": [CONDITIONING_NAMES[i] for i in zero_std_dims],
        "normalization_mapping_checkpoint": "STATE: MEAN_STD (runs/p1_b/pretrained_model/config.json)",
        "risk": "zero-std dims → (x-mean)/std blows up or is guarded; either way no live signal",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = _list_success_h5(DATA_P1)
    print(f"success episodes with h5: {len(pairs)}")

    # Sample all success episodes (2000) — streaming accumulate
    # For memory: reservoir of frames up to ~200k
    chunks: list[np.ndarray] = []
    n_frames = 0
    speeds = []
    for h5, meta in pairs:
        with h5py.File(h5, "r") as f:
            c = np.asarray(f["conditioning"], dtype=np.float64)
        chunks.append(c)
        n_frames += c.shape[0]
        speeds.append(float(meta.get("speed_m_s", np.nan)))
    X = np.concatenate(chunks, axis=0)
    assert X.shape[1] == CONDITIONING_DIM, X.shape
    print(f"pooled frames: {X.shape[0]} from {len(pairs)} episodes")

    dim_stats = _dim_stats(X)
    verdicts = [_verdict_row(s, s["idx"]) for s in dim_stats]

    # Pick one v≈0 and one v≥0.20
    ep0 = next((p for p in pairs if abs(float(p[1].get("speed_m_s", -1))) < 1e-9), None)
    ep20 = next((p for p in pairs if float(p[1].get("speed_m_s", -1)) >= 0.20), None)
    tables = {}
    rmse_reports = {}
    if ep0:
        tables["v0"] = _episode_table(ep0[0], ep0[1])
        rmse_reports["v0"] = _predict_rmse(ep0[0])
    if ep20:
        tables["v20"] = _episode_table(ep20[0], ep20[1])
        rmse_reports["v20"] = _predict_rmse(ep20[0])

    # Pool RMSE over a stratified sample of episodes
    sample = []
    by_speed: dict[float, list] = {}
    for h5, meta in pairs:
        sp = round(float(meta.get("speed_m_s", -1)), 2)
        by_speed.setdefault(sp, []).append((h5, meta))
    for sp, lst in sorted(by_speed.items()):
        sample.extend(lst[:3])  # up to 3 per speed bin
    rmse_pool = []
    corr_pool = []
    recon_max = []
    cv_max = []
    for h5, _ in sample:
        r = _predict_rmse(h5)
        if r["n_pairs"] and r["rmse_m"] is not None:
            rmse_pool.append(r["rmse_m"])
            corr_pool.append(r["corr_xyz"])
        if r.get("recon_max_l2") is not None:
            recon_max.append(r["recon_max_l2"])
        if r.get("cv_identity_max_l2") is not None:
            cv_max.append(r["cv_identity_max_l2"])

    cond_mean = X.mean(0)
    cond_std = X.std(0)
    cond_min = X.min(0)
    cond_max = X.max(0)
    norm = _norm_stats_from_h5_pool(cond_mean, cond_std, cond_min, cond_max)
    lerobot_meta = _load_lerobot_cond_stats()

    # Sanity: live 4-D pack matches CV XY predict
    st = oracle_from_gt(np.array([0.4, 0.0, 0.05]), np.array([0.2, 0.0, 0.0]), 0.0)
    vec = conditioning_vector(st, delta=0.25)
    assert abs(vec[0] - (0.4 + 0.2 * 0.25)) < 1e-12
    assert abs(vec[2] - 0.2) < 1e-12
    assert vec.shape[0] == CONDITIONING_DIM

    # Top-line decision
    informative = sum(1 for v in verdicts if v == "INFORMATIVE")
    # All 4 live dims are the payload
    payload_ok = all(verdicts[i] == "INFORMATIVE" for i in range(CONDITIONING_DIM))
    # RMSE: oracle CV on table motion should be small; high RMSE → NOISY
    mean_rmse = float(np.mean(rmse_pool)) if rmse_pool else float("nan")
    if not payload_ok:
        top = "SIGNAL-DEAD"
    elif mean_rmse == mean_rmse and mean_rmse > 0.05:  # >5 cm
        top = "SIGNAL-NOISY"
    elif mean_rmse == mean_rmse:
        top = "SIGNAL-ALIVE"
    else:
        top = "UNKNOWN"

    report = {
        "top_line": top,
        "layout_cite": {
            "conditioning_vector": "interfaces/state.py:101-123",
            "names": "policy/features.py:23-36",
            "object_state_frame": "interfaces/state.py:23-24 (world/table); sim/scene.py:537-548 (root_pos_w / root_lin_vel_w)",
            "workspace_scale_default": "interfaces/state.py:105,114 → 1.0 (no rescaling in collect.py:77)",
            "collect_producer": "sim/collect.py:77,89",
            "coast_time_hardcoded": "removed from Phase-1 live pack (was always 0.0)",
        },
        "layout": [
            {
                "idx": i,
                "name": n,
                "frame": fr,
                "units": u,
                "source": src,
                "verdict": verdicts[i],
                **{k: dim_stats[i][k] for k in ("mean", "std", "min", "max", "pct_zeros", "pct_nan_inf", "n_unique")},
            }
            for i, n, fr, u, src in LAYOUT
        ],
        "pool": {
            "n_episodes": len(pairs),
            "n_frames": int(n_frames),
            "speed_min": float(np.nanmin(speeds)),
            "speed_max": float(np.nanmax(speeds)),
            "data_root": str(DATA_P1),
        },
        "episode_tables": tables,
        "predictor_vs_gt": {
            "per_episode_examples": rmse_reports,
            "sample_n_episodes": len(sample),
            "sample_mean_rmse_m": mean_rmse,
            "sample_median_rmse_m": float(np.median(rmse_pool)) if rmse_pool else None,
            "sample_corr_xyz_nanmean": np.nanmean(np.asarray(corr_pool), axis=0).tolist() if corr_pool else None,
            "recon_max_l2_over_sample": float(np.max(recon_max)) if recon_max else None,
            "cv_identity_max_l2_over_sample": float(np.max(cv_max)) if cv_max else None,
            "note": "Predictor here is constant-velocity oracle (state.predict); not a learned filter.",
        },
        "transforms_and_norm": norm,
        "lerobot_p1_meta": lerobot_meta,
        "informative_count": informative,
        "smoke_vec_example": vec.tolist(),
    }

    out_json = OUT_DIR / "conditioning_signal_alive.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_json}")

    # Human-readable markdown
    lines = []
    lines.append(f"# Conditioning signal diagnostic\n")
    lines.append(f"**TOP-LINE: {top}**\n")
    lines.append(f"Pool: {len(pairs)} success H5 episodes, {n_frames} frames under `{DATA_P1}`.\n")
    lines.append(f"## {CONDITIONING_DIM}-D live layout + verdict\n")
    lines.append("| idx | name | frame | units | mean | std | min | max | %zero | %nan/inf | n_unique | verdict |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in report["layout"]:
        lines.append(
            f"| {row['idx']} | `{row['name']}` | {row['frame']} | {row['units']} | "
            f"{row['mean']:.6g} | {row['std']:.6g} | {row['min']:.6g} | {row['max']:.6g} | "
            f"{row['pct_zeros']:.2f} | {row['pct_nan_inf']:.2f} | {row['n_unique']} | **{row['verdict']}** |"
        )
    lines.append("\nCites: `interfaces/state.py:101-123`, `policy/features.py:23-36`, "
                 "`interfaces/state.py:23-24`, `sim/scene.py:537-548`, `sim/collect.py:77`.\n")
    lines.append("## Predictor vs GT future\n")
    lines.append(
        f"Sample mean RMSE (p̂ vs logged pos at t+Δ): **{mean_rmse:.6f} m** "
        f"over {len(sample)} episodes; "
        f"recon_max_l2={report['predictor_vs_gt']['recon_max_l2_over_sample']}; "
        f"cv_identity_max_l2={report['predictor_vs_gt']['cv_identity_max_l2_over_sample']}.\n"
    )
    lines.append("## Norm / append\n")
    lines.append(
        f"Append: `{norm['appends_cond']}`. "
        f"Zero-std cond dims after expand_state_stats: {norm['zero_std_names']}.\n"
    )
    out_md = OUT_DIR / "conditioning_signal_alive.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")
    print("TOP-LINE:", top)
    for row in report["layout"]:
        print(f"  [{row['idx']}] {CONDITIONING_NAMES[row['idx']]}: {row['verdict']} std={row['std']:.4g} uniq={row['n_unique']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
