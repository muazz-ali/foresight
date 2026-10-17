#!/usr/bin/env python3
"""Part B: dataset statistics from p3_throw H5 demos (shortcut-model evidence).

Hard rules: no train, no Isaac. Reads policy.transforms for noise only (import).
Writes: foresight_verification/partials/part_b.{json,md} + figures/*.png
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

# Repo root on path so we can import policy.transforms (read-only).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from policy.transforms import AugConfig, augment_conditioning  # noqa: E402

SEED = 0
FPS = 25.0
DELTA_S = 0.25
LEAD_FRAMES = int(round(DELTA_S * FPS))  # 6
K_LIST = (1, 2, 4, 8, 12, 16, 24)
CHUNK_IDX = (0, 10, 25, 49)
POS_NOISE = 0.015  # m
VEL_NOISE = 0.03  # m/s
M_TO_CM = 100.0
MS_TO_CMS = 100.0

DATA_GLOB = "data/p3_throw/*/speed_*/*.h5"
OUT_DIR = ROOT / "foresight_verification"
PARTIALS = OUT_DIR / "partials"
FIGURES = OUT_DIR / "figures"

STAGE_NAMES = {
    0: "start",
    1: "intercept",
    2: "go_down",
    3: "close",
    4: "lift",
    5: "carry",
    6: "lower",
    7: "open",
    8: "home",
    9: "done",
}

AUG = AugConfig(
    seed=SEED,
    delta_min=DELTA_S,
    delta_max=DELTA_S,
    pos_noise_std=POS_NOISE,
    vel_noise_std=VEL_NOISE,
)


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _round_dict(d: dict, nd: int = 6) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _round_dict(v, nd)
        elif isinstance(v, (list, tuple)):
            out[k] = [
                round(float(x), nd) if isinstance(x, (float, np.floating)) and math.isfinite(float(x)) else x
                for x in v
            ]
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            out[k] = round(fv, nd) if math.isfinite(fv) else None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def discover_episodes() -> list[dict]:
    """Prefer success+clean sidecar JSON; record where we looked if missing."""
    eps = []
    n_h5 = 0
    n_no_json = 0
    n_reject = 0
    for h5_path in sorted((ROOT / "data/p3_throw").glob("*/speed_*/*.h5")):
        n_h5 += 1
        js = h5_path.with_suffix(".json")
        if not js.is_file():
            n_no_json += 1
            continue
        meta = json.loads(js.read_text())
        if not (bool(meta.get("success")) and bool(meta.get("clean"))):
            n_reject += 1
            continue
        speed_folder = h5_path.parent.name  # speed_0.10
        speed_key = speed_folder.replace("speed_", "")
        usd = meta.get("usd") or ""
        obj = meta.get("category") or (Path(usd).stem if usd else "UNKNOWN")
        eps.append(
            {
                "h5": h5_path,
                "json": js,
                "seed": int(meta.get("seed", 0)),
                "speed_key": speed_key,
                "speed_m_s": float(meta.get("speed_m_s", 0.0)),
                "object": str(obj),
                "usd_stem": Path(usd).stem if usd else "UNKNOWN",
            }
        )
    # LeRobot count (reference only)
    lr_info = ROOT / "data/lerobot/p3_throw/meta/info.json"
    lerobot = {"path": str(lr_info), "unknown": "info.json missing"}
    if lr_info.is_file():
        info = json.loads(lr_info.read_text())
        lerobot = {
            "path": str(lr_info),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fps": info.get("fps"),
        }
    return eps, {
        "n_h5_scanned": n_h5,
        "n_no_sidecar_json": n_no_json,
        "n_not_success_clean": n_reject,
        "n_success_clean": len(eps),
        "lerobot": lerobot,
        "sidecar_fields_used": ["success", "clean", "seed", "speed_m_s", "category", "usd"],
        "h5_glob": DATA_GLOB,
    }


def first_close_idx(gripper: np.ndarray) -> int | None:
    hit = np.where(np.asarray(gripper, dtype=np.float64) < 0.5)[0]
    return int(hit[0]) if hit.size else None


def action_at(action_pos: np.ndarray, i: int, chunk_idx: int) -> np.ndarray:
    """H5 stores single-frame action_pos (T,3), not a chunk series.

    Chunk index j → action_pos[min(i+j, T-1)] (clipped to episode).
    """
    t = action_pos.shape[0]
    j = min(max(i + chunk_idx, 0), t - 1)
    return np.asarray(action_pos[j, :2], dtype=np.float64)


def noisy_cond(oracle_row: np.ndarray, ep_seed: int, frame_i: int) -> np.ndarray:
    rng = np.random.default_rng([SEED, int(ep_seed), int(frame_i)])
    return augment_conditioning(oracle_row, rng=rng, cfg=AUG)


def collect_frames(eps: list[dict]) -> tuple[dict, dict]:
    """Return records keyed by group, plus field/verify notes."""
    # records[group][k_tag] -> list of dicts with arrays in meters / m/s
    records: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    notes = {
        "action_chunk": (
            "H5 has action_pos (T,3) single-frame only (looked: dataset 'action_pos'). "
            "Chunk indices 10/25/49 use action_pos[i+idx] clipped to [0,T-1]."
        ),
        "p_definition": None,
        "fields_ok": [],
        "fields_missing_examples": [],
        "n_episodes_used": 0,
        "n_episodes_no_close": 0,
        "n_episodes_too_short": 0,
        "cond_verify": {},
    }
    required = (
        "action_pos",
        "ee_pos",
        "object_pos",
        "object_vel",
        "conditioning",
        "oracle_state",
        "gripper",
        "stage",
    )
    cond_abs_err = []
    lead_abs_err = []

    for ep in eps:
        with h5py.File(ep["h5"], "r") as f:
            missing = [k for k in required if k not in f]
            if missing:
                notes["fields_missing_examples"].append(
                    {"file": str(ep["h5"]), "missing": missing}
                )
                continue
            action_pos = np.asarray(f["action_pos"][:], dtype=np.float64)
            ee_pos = np.asarray(f["ee_pos"][:], dtype=np.float64)
            object_pos = np.asarray(f["object_pos"][:], dtype=np.float64)
            object_vel = np.asarray(f["object_vel"][:], dtype=np.float64)
            conditioning = np.asarray(f["conditioning"][:], dtype=np.float64)
            oracle = np.asarray(f["oracle_state"][:], dtype=np.float64)
            gripper = np.asarray(f["gripper"][:], dtype=np.float64)
            stage = np.asarray(f["stage"][:], dtype=np.int64)
            T = action_pos.shape[0]
            # Verify p: saved conditioning vs CV (o+v*Δ) and vs object_pos[t+6]
            p_cv = object_pos[:, :2] + object_vel[:, :2] * DELTA_S
            cond_abs_err.append(float(np.max(np.abs(conditioning[:, :2] - p_cv))))
            if T > LEAD_FRAMES:
                lead_abs_err.append(
                    float(np.max(np.abs(conditioning[:-LEAD_FRAMES, :2] - object_pos[LEAD_FRAMES:, :2])))
                )

            close_i = first_close_idx(gripper)
            if close_i is None:
                notes["n_episodes_no_close"] += 1
                continue
            if close_i < 1:
                notes["n_episodes_too_short"] += 1
                continue

            notes["n_episodes_used"] += 1
            groups = [
                "pooled",
                f"speed:{ep['speed_key']}",
                f"object:{ep['object']}",
            ]
            max_k = min(max(K_LIST), close_i)
            for k in range(1, max_k + 1):
                i = close_i - k
                h = ee_pos[i, :2]
                o = object_pos[i, :2]
                v = object_vel[i, :2]
                p = conditioning[i, :2].copy()  # noise-free CV future XY
                noisy = noisy_cond(oracle[i], ep["seed"], i)
                p_hat = noisy[0:2]
                v_hat = noisy[2:4]
                y0 = action_at(action_pos, i, 0)
                y_extra = {j: action_at(action_pos, i, j) for j in CHUNK_IDX if j != 0}
                st = int(stage[i])
                row = {
                    "k": k,
                    "y": y0,
                    "y_extra": y_extra,
                    "h": h,
                    "p": p,
                    "o": o,
                    "v": v,
                    "p_hat": p_hat,
                    "v_hat": v_hat,
                    "stage": st,
                }
                k_tags = []
                if k in K_LIST:
                    k_tags.append(f"k={k}")
                if 1 <= k <= 8:
                    k_tags.append("k=1..8")
                if 1 <= k <= 16:
                    k_tags.append("k=1..16")
                for g in groups:
                    for tag in k_tags:
                        records[g][tag].append(row)

    notes["fields_ok"] = list(required)
    notes["cond_verify"] = {
        "fps_assumed": FPS,
        "delta_s": DELTA_S,
        "lead_frames": LEAD_FRAMES,
        "saved_conditioning_vs_o_plus_v_delta_max_abs_m": _f(max(cond_abs_err) if cond_abs_err else None),
        "saved_conditioning_vs_object_pos_t_plus_6_max_abs_m": _f(
            max(lead_abs_err) if lead_abs_err else None
        ),
        "decision": (
            "p := saved conditioning XY (noise-free). Equals o + v·Δ exactly. "
            "object_pos[t+6] does NOT match (throw / non-CV motion). "
            "Looked: H5 'conditioning', 'object_pos', 'object_vel', attrs conditioning_layout."
        ),
    }
    notes["p_definition"] = notes["cond_verify"]["decision"]
    notes["noise"] = {
        "fn": "policy.transforms.augment_conditioning",
        "pos_noise_std_m": POS_NOISE,
        "vel_noise_std_m_s": VEL_NOISE,
        "rng": "np.random.default_rng([0, episode_seed, frame_i])",
        "source_yaml": "policy/configs/smolvla.yaml",
    }
    if notes["fields_missing_examples"]:
        notes["fields_missing_examples"] = notes["fields_missing_examples"][:5]
    return records, notes


def _stack(rows: list[dict], key: str) -> np.ndarray:
    return np.stack([r[key] for r in rows], axis=0)


def ls_fit_yh(yh: np.ndarray, v: np.ndarray, k: np.ndarray) -> dict:
    """(y-h) ~ a + b·v + c·k  (per axis already). Returns coeffs, R², σ_u_cond."""
    n = yh.shape[0]
    if n < 4:
        return {"a": None, "b": None, "c": None, "R2": None, "sigma_u_cond": None, "n": n}
    X = np.column_stack([np.ones(n), v, k.astype(np.float64)])
    coef, _, _, _ = np.linalg.lstsq(X, yh, rcond=None)
    pred = X @ coef
    resid = yh - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yh - yh.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(n - 3, 1)
    sigma = float(np.sqrt(ss_res / dof))
    return {
        "a": _f(coef[0]),
        "b": _f(coef[1]),
        "c": _f(coef[2]),
        "R2": _f(r2),
        "sigma_u_cond": _f(sigma),
        "n": n,
    }


def regress_y_on_pho(y: np.ndarray, p: np.ndarray, h: np.ndarray, o: np.ndarray) -> dict:
    """y ~ β0 + β_p p + β_h h + β_o o; report β_h ± 95% CI."""
    n = y.shape[0]
    if n < 5:
        return {"beta_h": None, "ci95": None, "n": n}
    X = np.column_stack([np.ones(n), p, h, o])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 4:
        return {"beta_h": _f(beta[2]), "ci95": None, "n": n, "note": "rank-deficient"}
    pred = X @ beta
    resid = y - pred
    dof = max(n - 4, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(max(sigma2 * xtx_inv[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        return {"beta_h": _f(beta[2]), "ci95": None, "n": n, "note": "XtX singular"}
    half = 1.96 * se
    bh = float(beta[2])
    return {
        "beta_h": _f(bh),
        "se": _f(se),
        "ci95": [_f(bh - half), _f(bh + half)],
        "beta_p": _f(beta[1]),
        "beta_o": _f(beta[3]),
        "beta0": _f(beta[0]),
        "n": n,
    }


def multimodality_flag(x: np.ndarray, n_bins: int = 40) -> dict:
    """Simple peak count on histogram; flag if ≥2 prominent peaks."""
    if x.size < 30:
        return {"multimodal": False, "n_peaks": 0, "note": "too_few_samples"}
    hist, edges = np.histogram(x, bins=n_bins)
    # smooth lightly
    ker = np.array([1, 2, 1], dtype=np.float64)
    ker /= ker.sum()
    sm = np.convolve(hist.astype(np.float64), ker, mode="same")
    peaks = []
    thr = 0.15 * sm.max() if sm.max() > 0 else 0
    for i in range(1, len(sm) - 1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] >= thr:
            peaks.append(i)
    return {
        "multimodal": len(peaks) >= 2,
        "n_peaks": len(peaks),
        "peak_centers": [_f(0.5 * (edges[i] + edges[i + 1])) for i in peaks[:8]],
    }


def axis_block(rows: list[dict], axis: int, units: str) -> dict:
    """Compute B1–B8 pieces for one axis. Internal SI → report cm / cm/s."""
    if not rows:
        return {"unknown": "no frames", "n": 0}
    y = _stack(rows, "y")[:, axis]
    h = _stack(rows, "h")[:, axis]
    p = _stack(rows, "p")[:, axis]
    o = _stack(rows, "o")[:, axis]
    v = _stack(rows, "v")[:, axis]
    p_hat = _stack(rows, "p_hat")[:, axis]
    v_hat = _stack(rows, "v_hat")[:, axis]
    k = np.array([r["k"] for r in rows], dtype=np.float64)

    # Convert position axes to cm; velocity to cm/s for reporting.
    if units == "cm":
        scale = M_TO_CM
        y_r, h_r, p_r, o_r = y * scale, h * scale, p * scale, o * scale
        p_hat_r = p_hat * scale
        v_r, v_hat_r = v * MS_TO_CMS, v_hat * MS_TO_CMS
    else:
        scale = MS_TO_CMS
        y_r = y * M_TO_CM
        h_r = h * M_TO_CM
        p_r = p * M_TO_CM
        o_r = o * M_TO_CM
        p_hat_r = p_hat * M_TO_CM
        v_r, v_hat_r = v * scale, v_hat * scale

    yh = y_r - h_r
    d = float(np.mean(yh))
    sigma_u = float(np.std(yh - d, ddof=1)) if yh.size > 1 else 0.0
    fit = ls_fit_yh(yh, v_r, k)  # v in cm/s for position residual fit
    # For velocity-related B3 term use m then scale
    yp = y_r - p_r
    yo = y_r - o_r
    ph_y = p_hat_r - y_r
    ph_p = p_hat_r - p_r
    # v̂·Δ − v·Δ in cm (position units)
    vdelta = (v_hat - v) * DELTA_S * M_TO_CM

    sigma_n = float(np.std(ph_y, ddof=1)) if ph_y.size > 1 else 0.0
    sigma_ph_p = float(np.std(ph_p, ddof=1)) if ph_p.size > 1 else 0.0
    sigma_vdelta = float(np.std(vdelta, ddof=1)) if vdelta.size > 1 else 0.0

    # B4 residuals corr
    r1 = yh - d
    r2 = ph_y
    r3 = (v_hat_r - v_r)  # cm/s
    R = np.corrcoef(np.vstack([r1, r2, r3]))

    # B5
    sk = float(skew(r1, bias=False)) if r1.size > 2 else None
    ku = float(kurtosis(r1, fisher=True, bias=False)) if r1.size > 3 else None
    mm = multimodality_flag(r1)

    # B7
    std_y = float(np.std(y_r, ddof=1)) if y_r.size > 1 else 0.0
    std_p = float(np.std(p_r, ddof=1)) if p_r.size > 1 else 0.0

    # B8
    def weights(su, sn, d_val):
        if su is None or sn is None or su <= 0 or sn <= 0:
            return {"lambda_h": None, "lambda_n": None, "w_h": None, "w_n": None, "e_star": None}
        lh = 1.0 / (su**2)
        ln = 1.0 / (sn**2)
        wh = lh / (lh + ln)
        wn = ln / (lh + ln)
        e_star = d_val * lh / ln
        return {
            "lambda_h": _f(lh),
            "lambda_n": _f(ln),
            "w_h": _f(wh),
            "w_n": _f(wn),
            "e_star": _f(e_star),
            "arithmetic": (
                f"λ_h=1/σ_u²=1/{su:.6g}²={lh:.6g}; "
                f"λ_n=1/σ_n²=1/{sn:.6g}²={ln:.6g}; "
                f"w_h=λ_h/(λ_h+λ_n)={wh:.6g}; w_n=λ_n/(λ_h+λ_n)={wn:.6g}; "
                f"e*=d·λ_h/λ_n={d_val:.6g}·{lh:.6g}/{ln:.6g}={e_star:.6g}"
            ),
        }

    b8 = weights(sigma_u, sigma_n, d)
    su_c = fit.get("sigma_u_cond")
    b8_cond = weights(su_c, sigma_n, d) if su_c else {"unknown": "no sigma_u_cond"}

    # chunk extras (mean y_j - h) for documentation
    chunk = {}
    for j in CHUNK_IDX:
        if j == 0:
            ys = y_r
        else:
            ys = np.array(
                [
                    (
                        r["y_extra"][j][axis] * M_TO_CM
                        if j in r["y_extra"]
                        else float("nan")
                    )
                    for r in rows
                ]
            )
        chunk[str(j)] = {
            "mean_y_minus_h_cm": _f(float(np.nanmean(ys - h_r))),
            "std_y_minus_h_cm": _f(float(np.nanstd(ys - h_r, ddof=1))) if np.sum(np.isfinite(ys)) > 1 else None,
        }

    return {
        "n": int(len(rows)),
        "units": "cm" if units == "cm" else "cm_and_cm_s",
        "B1": {
            "d_cm": _f(d),
            "sigma_u_cm": _f(sigma_u),
            "fit_yh_a_b_v_c_k": fit,
        },
        "B2": {
            "mean_y_minus_p_cm": _f(float(np.mean(yp))),
            "std_y_minus_p_cm": _f(float(np.std(yp, ddof=1)) if yp.size > 1 else 0.0),
            "mean_y_minus_o_cm": _f(float(np.mean(yo))),
            "std_y_minus_o_cm": _f(float(np.std(yo, ddof=1)) if yo.size > 1 else 0.0),
        },
        "B3": {
            "sigma_n_std_phat_minus_y_cm": _f(sigma_n),
            "std_phat_minus_p_cm": _f(sigma_ph_p),
            "std_vhat_delta_minus_v_delta_cm": _f(sigma_vdelta),
            "delta_s": DELTA_S,
        },
        "B4": {
            "labels": ["y-h-d", "phat-y", "vhat-v"],
            "corr": [[_f(R[i, j]) for j in range(3)] for i in range(3)],
            "note": "residuals in cm / cm/s",
        },
        "B5": {
            "skewness_yh_minus_d": _f(sk),
            "excess_kurtosis_yh_minus_d": _f(ku),
            "multimodality": mm,
        },
        "B7": {"std_y_cm": _f(std_y), "std_p_cm": _f(std_p)},
        "B8": {**b8, "sigma_u_cm": _f(sigma_u), "sigma_n_cm": _f(sigma_n), "d_cm": _f(d)},
        "B8_sigma_u_cond": {
            **(b8_cond if isinstance(b8_cond, dict) else {}),
            "sigma_u_cond_cm": su_c,
            "sigma_n_cm": _f(sigma_n),
            "d_cm": _f(d),
        },
        "chunk_index_action_pos": chunk,
        "_resid_yh_d": r1,  # for histograms; stripped later
    }


def stage_block(rows: list[dict], axis: int) -> dict:
    by_stage: dict[int, list] = defaultdict(list)
    for r in rows:
        by_stage[int(r["stage"])].append(r)
    out = {}
    for st, rs in sorted(by_stage.items()):
        y = _stack(rs, "y")[:, axis] * M_TO_CM
        p = _stack(rs, "p")[:, axis] * M_TO_CM
        h = _stack(rs, "h")[:, axis] * M_TO_CM
        o = _stack(rs, "o")[:, axis] * M_TO_CM
        out[STAGE_NAMES.get(st, str(st))] = regress_y_on_pho(y, p, h, o)
    return out


def analyze_group(rows: list[dict]) -> dict:
    ax = {}
    for axis, name in enumerate(("x", "y")):
        blk = axis_block(rows, axis, "cm")
        resid = blk.pop("_resid_yh_d", None)
        blk["B6_per_stage"] = stage_block(rows, axis)
        ax[name] = blk
        ax[name]["_hist_resid"] = resid
    return ax


def save_histograms(pooled_k18: dict, tag: str = "pooled_k1_8") -> list[str]:
    paths = []
    FIGURES.mkdir(parents=True, exist_ok=True)
    for axis in ("x", "y"):
        resid = pooled_k18[axis].get("_hist_resid")
        if resid is None or len(resid) == 0:
            continue
        fig, axp = plt.subplots(figsize=(6, 4))
        axp.hist(resid, bins=50, color="#3d5a80", edgecolor="white", linewidth=0.4)
        axp.set_xlabel(f"(y−h−d)_{axis} [cm]")
        axp.set_ylabel("count")
        axp.set_title(f"Part B residual histogram {tag} axis={axis}")
        fig.tight_layout()
        p = FIGURES / f"part_b_hist_{tag}_{axis}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p.relative_to(ROOT)))
    return paths


def strip_hist(obj):
    if isinstance(obj, dict):
        return {k: strip_hist(v) for k, v in obj.items() if k != "_hist_resid"}
    if isinstance(obj, list):
        return [strip_hist(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def summary_row(ax_blk: dict) -> dict:
    return {
        "d_cm": ax_blk["B1"]["d_cm"],
        "sigma_u_cm": ax_blk["B1"]["sigma_u_cm"],
        "sigma_n_cm": ax_blk["B3"]["sigma_n_std_phat_minus_y_cm"],
        "w_h": ax_blk["B8"]["w_h"],
        "w_n": ax_blk["B8"]["w_n"],
        "e_star_cm": ax_blk["B8"]["e_star"],
        "n": ax_blk["n"],
    }


def write_md(result: dict, path: Path) -> None:
    s = result["summary_pooled_k1_8"]
    lines = [
        "# Part B — dataset statistics (p3_throw demos)",
        "",
        "Seed=0. Units: cm / cm/s. Noise: `augment_conditioning` "
        f"pos_std={POS_NOISE} m, vel_std={VEL_NOISE} m/s.",
        "",
        "## Data",
        "",
        f"- H5 scanned: {result['data']['n_h5_scanned']}",
        f"- success+clean used: {result['data']['n_success_clean']}",
        f"- episodes with frames: {result['notes']['n_episodes_used']}",
        f"- LeRobot episodes (ref): {result['data']['lerobot'].get('total_episodes', 'UNKNOWN')}",
        "",
        "## Conditioning check",
        "",
        f"- {result['notes']['cond_verify']['decision']}",
        f"- max |cond − (o+vΔ)| m = {result['notes']['cond_verify']['saved_conditioning_vs_o_plus_v_delta_max_abs_m']}",
        f"- max |cond − object_pos[t+6]| m = {result['notes']['cond_verify']['saved_conditioning_vs_object_pos_t_plus_6_max_abs_m']}",
        "",
        "## Summary table — pooled k=1..8",
        "",
        "| axis | d (cm) | σ_u (cm) | σ_n (cm) | w_h | e* (cm) | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in ("x", "y"):
        r = s[axis]
        lines.append(
            f"| {axis} | {r['d_cm']} | {r['sigma_u_cm']} | {r['sigma_n_cm']} | "
            f"{r['w_h']} | {r['e_star_cm']} | {r['n']} |"
        )
    lines += [
        "",
        "## B8 arithmetic (pooled k=1..8)",
        "",
    ]
    for axis in ("x", "y"):
        ar = result["by_group"]["pooled"]["k=1..8"][axis]["B8"].get("arithmetic")
        lines.append(f"- **{axis}:** {ar}")
        arc = result["by_group"]["pooled"]["k=1..8"][axis]["B8_sigma_u_cond"].get("arithmetic")
        if arc:
            lines.append(f"  - with σ_u_cond: {arc}")
    lines += [
        "",
        "## Action chunk note",
        "",
        result["notes"]["action_chunk"],
        "",
        "## Figures",
        "",
    ]
    for fig in result.get("figures", []):
        lines.append(f"- `{fig}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    PARTIALS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Discovering episodes…", flush=True)
    eps, data_notes = discover_episodes()
    print(f"  success+clean: {len(eps)}", flush=True)

    print("Collecting pre-close frames…", flush=True)
    records, notes = collect_frames(eps)
    print(
        f"  episodes used={notes['n_episodes_used']} "
        f"no_close={notes['n_episodes_no_close']}",
        flush=True,
    )

    # Analyze: pooled all k tags; per-speed and per-object for k=1..8 and each k
    by_group: dict = {}
    # Priority groups for full detail
    priority_groups = ["pooled"] + sorted(
        g for g in records if g.startswith("speed:")
    )
    # objects: only k=1..8 to keep JSON size sane
    object_groups = sorted(g for g in records if g.startswith("object:"))

    print("Analyzing groups…", flush=True)
    for g in priority_groups:
        by_group[g] = {}
        for tag in sorted(records[g].keys(), key=lambda s: (0 if s.startswith("k=") else 1, s)):
            rows = records[g][tag]
            print(f"  {g} {tag} n={len(rows)}", flush=True)
            by_group[g][tag] = analyze_group(rows)

    by_group["per_object_k1_8"] = {}
    for g in object_groups:
        rows = records[g].get("k=1..8", [])
        if not rows:
            continue
        by_group["per_object_k1_8"][g.replace("object:", "")] = analyze_group(rows)

    # Histograms from pooled k=1..8
    figs = save_histograms(by_group["pooled"]["k=1..8"], "pooled_k1_8")
    # also per-speed for a couple extremes
    for sk in ("speed:0.00", "speed:0.20"):
        if sk in by_group and "k=1..8" in by_group[sk]:
            figs += save_histograms(by_group[sk]["k=1..8"], sk.replace(":", "_") + "_k1_8")

    summary = {
        "x": summary_row(by_group["pooled"]["k=1..8"]["x"]),
        "y": summary_row(by_group["pooled"]["k=1..8"]["y"]),
    }

    # Per-speed B8 snapshot
    per_speed_b8 = {}
    for g in priority_groups:
        if not g.startswith("speed:"):
            continue
        if "k=1..8" not in by_group[g]:
            continue
        per_speed_b8[g.replace("speed:", "")] = {
            "x": summary_row(by_group[g]["k=1..8"]["x"]),
            "y": summary_row(by_group[g]["k=1..8"]["y"]),
        }

    result = {
        "part": "B",
        "seed": SEED,
        "data": data_notes,
        "notes": notes,
        "summary_pooled_k1_8": summary,
        "per_speed_k1_8": per_speed_b8,
        "figures": figs,
        "by_group": strip_hist(by_group),
    }
    result = json.loads(json.dumps(result, default=lambda o: _f(o) if isinstance(o, float) else str(o)))

    out_json = PARTIALS / "part_b.json"
    out_md = PARTIALS / "part_b.md"
    out_json.write_text(json.dumps(result, indent=2))
    write_md(result, out_md)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    print("SUMMARY pooled k=1..8:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
