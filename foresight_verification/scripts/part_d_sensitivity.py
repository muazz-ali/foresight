#!/usr/bin/env python3
"""Part D — inference-only sensitivity probe (D1–D5).

No training. No Isaac. Writes:
  foresight_verification/partials/part_d.json
  foresight_verification/partials/part_d.md

Reuses the same observation build as eval/probe_cond_gain.py
(pack_proprio + cond_to_model + action_from_model). Perturbations are in
raw metres / m/s *before* the policy's internal normalization.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PARTIALS = ROOT / "foresight_verification" / "partials"
CHUNK_IDX = (0, 10, 25, 49)
PRIMARY_DELTAS_CM = (0.5, 1.0, 1.5)
EXTRA_DELTA_CM = 3.0
ALL_DELTAS_CM = PRIMARY_DELTAS_CM + (EXTRA_DELTA_CM,)
DELTA_LEAD_S = 0.25  # plan Δ; v̂ ± δ/Δ shifts implied lead by ±δ
NOISE_SEEDS_SPREAD = 5
K_RANGE = range(1, 17)  # before-close offsets


def _load(checkpoint: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from policy.transforms import load_frames

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda" if device.startswith("cuda") else "cpu"
    pol = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval()
    return pol, cfg, load_frames(checkpoint)


def _img(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1).float() / 255.0


def _language(h5: str) -> str:
    from policy.features import default_language

    js = Path(h5).with_suffix(".json")
    m = json.loads(js.read_text()) if js.exists() else {}
    if m.get("language"):
        return m["language"]
    return default_language(m.get("category"), m.get("container_category"))


def _meta(h5: str) -> dict:
    js = Path(h5).with_suffix(".json")
    return json.loads(js.read_text()) if js.exists() else {}


def _speed_bin(speed: float) -> str:
    """Round to cm/s label used in paths (0.00, 0.08, …)."""
    if not np.isfinite(speed):
        return "nan"
    return f"{round(float(speed), 2):.2f}"


def list_clean_candidates(h5_glob: str) -> list[dict]:
    """Every (episode, k) pre-close frame on clean successes."""
    out: list[dict] = []
    for h5 in sorted(glob.glob(h5_glob)):
        m = _meta(h5)
        if not m.get("success") or m.get("clean_reject"):
            continue
        cat = m.get("category") or m.get("object") or "unknown"
        with h5py.File(h5, "r") as f:
            if "static_cam_rgb" not in f or "conditioning" not in f:
                continue
            g = np.asarray(f["gripper"][:])
            if not (g < 0.5).any():
                continue
            ic = int(np.argmax(g < 0.5))
            speed = float(np.asarray(f["speed_m_s"]).reshape(-1)[0]) if "speed_m_s" in f else float("nan")
            for k in K_RANGE:
                i = ic - k
                if i < 0:
                    continue
                out.append(
                    {
                        "h5": h5,
                        "frame": i,
                        "k": k,
                        "speed": speed,
                        "speed_bin": _speed_bin(speed),
                        "category": cat,
                        "close_idx": ic,
                    }
                )
    return out


def stratified_sample(cands: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by (speed_bin, category); round-robin until n frames."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cands:
        buckets[(c["speed_bin"], c["category"])].append(c)
    keys = sorted(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
    picked: list[dict] = []
    ptr = {k: 0 for k in keys}
    while len(picked) < n and keys:
        progressed = False
        for key in list(keys):
            i = ptr[key]
            if i >= len(buckets[key]):
                keys.remove(key)
                continue
            picked.append(buckets[key][i])
            ptr[key] = i + 1
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break
    rng.shuffle(picked)
    return picked[:n]


def _make_noise(cfg, device: str, seed: int, n: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    one = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g)
    return one.repeat(n, 1, 1).to(device)


def _forward_batch(pol, cfg, frames, device, lang, static, wrist, cases, noise):
    """cases: list of (hand_xyz, cond_w). Returns world actions (N, T, 8)."""
    from policy.convert_h5 import pack_proprio
    from policy.transforms import action_from_model, cond_to_model

    quat = cases[0][2]
    grip = cases[0][3]
    states, s_imgs, w_imgs = [], [], []
    s_t, w_t = _img(static), _img(wrist)
    for hand, cw, _, _ in cases:
        st = pack_proprio(hand, quat, grip)
        states.append(np.r_[st, cond_to_model(cw, hand, frames["cond_frame"])])
        s_imgs.append(s_t)
        w_imgs.append(w_t)
    n = len(cases)
    batch = {
        "observation.state": torch.tensor(np.stack(states), dtype=torch.float32, device=device),
        "observation.images.static_cam": torch.stack(s_imgs).to(device),
        "observation.images.wrist_cam": torch.stack(w_imgs).to(device),
        "task": [lang if lang.endswith("\n") else lang + "\n"] * n,
    }
    if noise.shape[0] != n:
        noise = noise[:1].repeat(n, 1, 1)
    with torch.inference_mode():
        out = pol.predict_action_chunk(batch, noise=noise).float().cpu().numpy()
    worlds = []
    for (hand, *_), chunk in zip(cases, out):
        worlds.append(action_from_model(chunk, hand, frames["action_frame"]))
    return np.stack(worlds, axis=0)


def _forward_chunked(pol, cfg, frames, device, lang, static, wrist, cases, noise, micro: int):
    """Run cases in micro-batches; reuse the same noise row for every case."""
    base_noise = noise[:1]
    parts = []
    for i in range(0, len(cases), micro):
        sl = cases[i : i + micro]
        nse = base_noise.repeat(len(sl), 1, 1)
        parts.append(_forward_batch(pol, cfg, frames, device, lang, static, wrist, sl, nse))
    return np.concatenate(parts, axis=0)


def build_perturbation_cases(ee, quat, grip, cond_w, deltas_m: list[float]):
    """Named cases for D1–D5. Returns list[(name, hand, cond, quat, grip)]."""
    ee = np.asarray(ee, dtype=np.float64).copy()
    cond = np.asarray(cond_w, dtype=np.float64).copy()
    cases = [("base", ee, cond, quat, grip)]
    for dm in deltas_m:
        dv = dm / DELTA_LEAD_S
        tag = f"{dm * 100:.1f}cm"
        # D1 hand ±
        for ax, name in ((0, "x"), (1, "y")):
            for sign, sname in ((+1, "p"), (-1, "m")):
                h = ee.copy()
                h[ax] += sign * dm
                cases.append((f"D1_hand_{name}{sname}_{tag}", h, cond, quat, grip))
        # D2 p̂ ±
        for ax, name in ((0, "x"), (1, "y")):
            for sign, sname in ((+1, "p"), (-1, "m")):
                c = cond.copy()
                c[ax] += sign * dm
                cases.append((f"D2_phat_{name}{sname}_{tag}", ee, c, quat, grip))
        # D3 v̂ ± δ/Δ
        for ax, name in ((0, "x"), (1, "y")):
            for sign, sname in ((+1, "p"), (-1, "m")):
                c = cond.copy()
                c[2 + ax] += sign * dv
                cases.append((f"D3_vhat_{name}{sname}_{tag}", ee, c, quat, grip))
        # D4 hand + p̂ together ± (same axis)
        for ax, name in ((0, "x"), (1, "y")):
            for sign, sname in ((+1, "p"), (-1, "m")):
                h = ee.copy()
                c = cond.copy()
                h[ax] += sign * dm
                c[ax] += sign * dm
                cases.append((f"D4_both_{name}{sname}_{tag}", h, c, quat, grip))
    return cases


def _central(ap: np.ndarray, am: np.ndarray, delta: float) -> np.ndarray:
    return (ap - am) / (2.0 * delta)


def sensitivities_from_worlds(names: list[str], worlds: np.ndarray, deltas_m: list[float]) -> dict:
    """Compute S at CHUNK_IDX from named world action chunks."""
    by = {n: w for n, w in zip(names, worlds)}
    out: dict = {}
    for dm in deltas_m:
        tag = f"{dm * 100:.1f}cm"
        key = f"delta_{tag}"
        block: dict = {"delta_m": dm, "delta_cm": dm * 100, "axes": {}}
        for ax, aname in ((0, "x"), (1, "y")):
            other = 1 - ax
            oname = "y" if ax == 0 else "x"
            S = {}
            for ci in CHUNK_IDX:
                # D1
                hp = by[f"D1_hand_{aname}p_{tag}"][ci]
                hm = by[f"D1_hand_{aname}m_{tag}"][ci]
                Sp = by[f"D2_phat_{aname}p_{tag}"][ci]
                Sm = by[f"D2_phat_{aname}m_{tag}"][ci]
                vp = by[f"D3_vhat_{aname}p_{tag}"][ci]
                vm = by[f"D3_vhat_{aname}m_{tag}"][ci]
                bp = by[f"D4_both_{aname}p_{tag}"][ci]
                bm = by[f"D4_both_{aname}m_{tag}"][ci]
                # same-axis gains (dimensionless for hand/p̂; lead-equivalent for v̂)
                S_h = float(_central(hp[ax], hm[ax], dm))
                S_p = float(_central(Sp[ax], Sm[ax], dm))
                S_v = float(_central(vp[ax], vm[ax], dm))  # /δ lead-equivalent
                S_both = float(_central(bp[ax], bm[ax], dm))
                # cross-axis: Δa_other / Δinput_ax
                S_h_cross = float(_central(hp[other], hm[other], dm))
                S_p_cross = float(_central(Sp[other], Sm[other], dm))
                S_v_cross = float(_central(vp[other], vm[other], dm))
                S[f"chunk_{ci}"] = {
                    "S_h": S_h,
                    "S_p": S_p,
                    "S_v": S_v,
                    "S_both": S_both,
                    "S_h_plus_S_p": S_h + S_p,
                    "S_both_minus_sum": S_both - (S_h + S_p),
                    "D5_S_h_cross": S_h_cross,  # Δa_other / Δh_ax
                    "D5_S_p_cross": S_p_cross,
                    "D5_S_v_cross": S_v_cross,
                    "cross_output_axis": oname,
                }
            block["axes"][aname] = S
        out[key] = block
    return out


def noise_spread(worlds5: np.ndarray) -> dict:
    """worlds5: (5, T, 8) unperturbed actions under 5 noise seeds."""
    # std of a_x, a_y at CHUNK_IDX across seeds, in cm
    out = {}
    for ci in CHUNK_IDX:
        xy = worlds5[:, ci, 0:2] * 100.0  # cm
        out[f"chunk_{ci}"] = {
            "std_ax_cm": float(xy[:, 0].std(ddof=1)) if len(xy) > 1 else 0.0,
            "std_ay_cm": float(xy[:, 1].std(ddof=1)) if len(xy) > 1 else 0.0,
            "mean_ax_cm": float(xy[:, 0].mean()),
            "mean_ay_cm": float(xy[:, 1].mean()),
            "range_ax_cm": float(xy[:, 0].max() - xy[:, 0].min()),
            "range_ay_cm": float(xy[:, 1].max() - xy[:, 1].min()),
        }
    return out


def _agg(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    ci = 1.96 * se
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": [mean - ci, mean + ci],
        "mean_pm_std": f"{mean:.4f}±{std:.4f}",
        "ci95_str": f"[{mean - ci:.4f}, {mean + ci:.4f}]",
    }


def aggregate_rows(rows: list[dict], deltas_cm: tuple[float, ...]) -> dict:
    """mean±std and 95% CI of S at each chunk, per k and per speed (and pooled)."""
    metrics = ("S_h", "S_p", "S_v", "S_both", "D5_S_h_cross", "D5_S_p_cross")
    tables: dict = {}
    for dcm in deltas_cm:
        tag = f"{dcm:.1f}cm"
        dkey = f"delta_{tag}"
        tables[dkey] = {"pooled": {}, "per_k": {}, "per_speed": {}, "per_k_speed": {}}

        def collect(filter_fn):
            bucket: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                sens = r["sensitivity"].get(dkey)
                if sens is None:
                    continue
                if not filter_fn(r):
                    continue
                for axis in ("x", "y"):
                    for ci in CHUNK_IDX:
                        cell = sens["axes"][axis][f"chunk_{ci}"]
                        for m in metrics:
                            bucket[f"{m}_{axis}_c{ci}"].append(cell[m])
            return {k: _agg(v) for k, v in sorted(bucket.items())}

        tables[dkey]["pooled"] = collect(lambda _r: True)
        for k in K_RANGE:
            tables[dkey]["per_k"][str(k)] = collect(lambda r, kk=k: r["k"] == kk)
        speeds = sorted({r["speed_bin"] for r in rows})
        for sb in speeds:
            tables[dkey]["per_speed"][sb] = collect(lambda r, s=sb: r["speed_bin"] == s)
        for k in K_RANGE:
            for sb in speeds:
                tables[dkey]["per_k_speed"][f"k{k}_v{sb}"] = collect(
                    lambda r, kk=k, s=sb: r["k"] == kk and r["speed_bin"] == s
                )
    return tables


def compare_to_part_b(summary_pooled: dict) -> dict:
    pb = PARTIALS / "part_b.json"
    if not pb.exists():
        return {
            "status": "PENDING",
            "notes": "foresight_verification/partials/part_b.json missing; cannot compare S to w_h/w_n",
        }
    b = json.loads(pb.read_text())
    # Part B stores B8-style weights under summary_pooled_k1_8.{x,y}.w_h / w_n
    pooled_b = b.get("summary_pooled_k1_8") or b.get("B8") or b.get("b8") or {}
    w_h = w_n = None
    w_h_y = w_n_y = None
    if isinstance(pooled_b, dict):
        if "w_h" in pooled_b:
            w_h, w_n = pooled_b.get("w_h"), pooled_b.get("w_n")
        x = pooled_b.get("x") if isinstance(pooled_b.get("x"), dict) else None
        y = pooled_b.get("y") if isinstance(pooled_b.get("y"), dict) else None
        if x:
            w_h, w_n = x.get("w_h"), x.get("w_n")
        if y:
            w_h_y, w_n_y = y.get("w_h"), y.get("w_n")
    dkey = "delta_1.5cm"
    pooled = summary_pooled.get(dkey, {}).get("pooled", {})
    S_h = pooled.get("S_h_x_c0", {})
    S_p = pooled.get("S_p_x_c0", {})
    S_h_y = pooled.get("S_h_y_c0", {})
    S_p_y = pooled.get("S_p_y_c0", {})
    return {
        "status": "ok" if w_h is not None and w_n is not None else "PARTIAL",
        "part_b_path": str(pb),
        "source_key": "summary_pooled_k1_8",
        "w_h_x": w_h,
        "w_n_x": w_n,
        "w_h_y": w_h_y,
        "w_n_y": w_n_y,
        # aliases used by merge_report / headline
        "w_h": w_h,
        "w_n": w_n,
        "S_h_x_chunk0_delta1.5cm": S_h,
        "S_p_x_chunk0_delta1.5cm": S_p,
        "S_h_y_chunk0_delta1.5cm": S_h_y,
        "S_p_y_chunk0_delta1.5cm": S_p_y,
        "expect": "S_h ≈ w_h, S_p ≈ w_n (B8 / summary_pooled_k1_8)",
        "delta_S_h_minus_w_h_x": (None if not S_h or S_h.get("mean") is None or w_h is None else float(S_h["mean"]) - float(w_h)),
        "delta_S_p_minus_w_n_x": (None if not S_p or S_p.get("mean") is None or w_n is None else float(S_p["mean"]) - float(w_n)),
    }


def write_md(payload: dict, path: Path) -> None:
    s = payload["summary"]
    cmp_ = payload.get("compare_B8") or {}
    lines = [
        "# Part D — Inference sensitivity (D1–D5)",
        "",
        f"- Checkpoint: `{s.get('checkpoint')}`",
        f"- Frames probed: **{s.get('n_frames')}** (target ≥{s.get('n_target')}); seed={s.get('seed')}",
        f"- Device: `{s.get('device')}`; micro-batch={s.get('micro_batch')}",
        f"- Status: `{s.get('status', 'DONE')}`",
        f"- Δ lead = {DELTA_LEAD_S} s; primary δ={list(PRIMARY_DELTAS_CM)} cm; extra δ={EXTRA_DELTA_CM} cm",
        f"- Chunk indices: {list(CHUNK_IDX)}; k=1..16 before close",
        f"- Frames config: `{s.get('frames')}`",
        "",
        "## Headline (δ=1.5 cm, chunk 0, pooled)",
        "",
    ]
    pooled = (s.get("tables") or {}).get("delta_1.5cm", {}).get("pooled") or {}
    for key in ("S_h_x_c0", "S_h_y_c0", "S_p_x_c0", "S_p_y_c0", "S_both_x_c0", "S_v_x_c0"):
        cell = pooled.get(key) or {}
        lines.append(
            f"- **{key}**: {cell.get('mean_pm_std', 'n/a')} "
            f"(95% CI {cell.get('ci95_str', 'n/a')}, n={cell.get('n')})"
        )
    lines += ["", "## Compare to B8 (w_h / w_n)", ""]
    if cmp_.get("status") == "PENDING":
        lines.append(f"- **PENDING**: {cmp_.get('notes')}")
    else:
        lines.append(f"- status: {cmp_.get('status')} (from `{cmp_.get('source_key')}`)")
        lines.append(f"- predicted w_h_x / w_n_x = {cmp_.get('w_h_x')} / {cmp_.get('w_n_x')}")
        lines.append(f"- predicted w_h_y / w_n_y = {cmp_.get('w_h_y')} / {cmp_.get('w_n_y')}")
        sh = cmp_.get("S_h_x_chunk0_delta1.5cm") or {}
        sp = cmp_.get("S_p_x_chunk0_delta1.5cm") or {}
        lines.append(f"- measured S_h_x @c0 δ=1.5cm = {sh.get('mean_pm_std')} CI {sh.get('ci95_str')}")
        lines.append(f"- measured S_p_x @c0 δ=1.5cm = {sp.get('mean_pm_std')} CI {sp.get('ci95_str')}")
        lines.append(f"- S_h − w_h (x) = {cmp_.get('delta_S_h_minus_w_h_x')}")
        lines.append(f"- S_p − w_n (x) = {cmp_.get('delta_S_p_minus_w_n_x')}")
        lines.append("- Expectation: S_h ≈ w_h, S_p ≈ w_n.")
    lines += [
        "",
        "## Noise spread (unperturbed, 5 seeds)",
        "",
        f"- Mean std a_x @c0: {s.get('noise_spread_summary', {}).get('std_ax_cm_c0_mean')} cm",
        f"- Mean std a_y @c0: {s.get('noise_spread_summary', {}).get('std_ay_cm_c0_mean')} cm",
        "",
        "## Notes",
        "",
        "- Perturbations applied in raw metres (and m/s for v̂) before policy normalization.",
        "- Flow-matching noise fixed per frame across all D1–D5 cases; 5 seeds only for unperturbed spread.",
        "- S_v uses lead-equivalent denominator δ (v̂ shifted by ±δ/Δ).",
        "- δ=3.0 cm table is under `tables.delta_3.0cm` (outside train noise support).",
        "- Full per-k / per-speed grids: see `part_d.json`.",
        "",
    ]
    if s.get("oom_fallback"):
        lines += ["## OOM / load fallback", "", f"- {s['oom_fallback']}", ""]
    if s.get("errors"):
        lines += ["## Errors", ""] + [f"- {e}" for e in s["errors"]] + [""]
    path.write_text("\n".join(lines))


def pick_device(preferred: str) -> str:
    if not preferred.startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    # Prefer the free-er of cuda:0 / cuda:1 when user asked auto.
    if preferred in ("cuda", "cuda:auto"):
        best, best_free = "cuda:0", -1
        for i in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best = f"cuda:{i}"
        return best
    return preferred


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/p3_throw/pretrained_model")
    p.add_argument("--h5", default="data/p3_throw/*/speed_*/*.h5")
    p.add_argument("--n", type=int, default=512, help="target held-out frames (≥500)")
    p.add_argument("--min-n", type=int, default=100, help="OOM fallback floor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--out-json", default=str(PARTIALS / "part_d.json"))
    p.add_argument("--out-md", default=str(PARTIALS / "part_d.md"))
    a = p.parse_args()

    device = pick_device(a.device)
    errors: list[str] = []
    oom_note = None
    n_target = a.n

    print(f"[part_d] listing candidates under {a.h5} …", flush=True)
    cands = list_clean_candidates(a.h5)
    print(f"[part_d] candidates={len(cands)}", flush=True)
    sample = stratified_sample(cands, n_target, a.seed)
    print(
        f"[part_d] stratified sample={len(sample)} "
        f"speeds={len({r['speed_bin'] for r in sample})} "
        f"cats={len({r['category'] for r in sample})}",
        flush=True,
    )

    print(f"[part_d] loading {a.checkpoint} on {device} …", flush=True)
    try:
        pol, cfg, frames = _load(a.checkpoint, device)
    except torch.cuda.OutOfMemoryError as e:
        oom_note = f"OOM on load ({e}); falling back to CPU"
        errors.append(oom_note)
        device = "cpu"
        pol, cfg, frames = _load(a.checkpoint, device)
    except Exception as e:
        errors.append(f"load failed: {e}")
        traceback.print_exc()
        payload = {
            "summary": {
                "checkpoint": a.checkpoint,
                "n_frames": 0,
                "n_target": n_target,
                "seed": a.seed,
                "device": device,
                "errors": errors,
                "oom_fallback": oom_note,
                "status": "FAILED_LOAD",
            },
            "compare_B8": compare_to_part_b({"delta_1.5cm": {"pooled": {}}}),
            "rows": [],
        }
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(payload, indent=2))
        write_md(payload, Path(a.out_md))
        raise SystemExit(1)

    deltas_m = [c / 100.0 for c in ALL_DELTAS_CM]
    # case count sanity
    n_cases = len(build_perturbation_cases(np.zeros(3), np.zeros(4), 1.0, np.zeros(4), deltas_m))
    print(f"[part_d] cases/frame (fixed noise)={n_cases}; micro={a.micro_batch}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    micro = a.micro_batch

    def run_one(rec: dict, frame_seed: int) -> dict | None:
        nonlocal micro
        with h5py.File(rec["h5"], "r") as f:
            i = rec["frame"]
            ee = np.asarray(f["ee_pos"][i], dtype=np.float64)
            quat = np.asarray(f["ee_quat"][i], dtype=np.float64)
            grip = float(f["gripper"][i])
            cond_w = np.asarray(f["conditioning"][i], dtype=np.float64)
            static = f["static_cam_rgb"][i]
            wrist = f["wrist_cam_rgb"][i]
        lang = _language(rec["h5"])
        named = build_perturbation_cases(ee, quat, grip, cond_w, deltas_m)
        names = [n for n, *_ in named]
        # cases for forward: (hand, cond, quat, grip)
        cases = [(h, c, q, g) for _, h, c, q, g in named]

        # fixed noise for all perturbations
        while True:
            try:
                noise = _make_noise(cfg, device, frame_seed, 1)
                worlds = _forward_chunked(
                    pol, cfg, frames, device, lang, static, wrist, cases, noise, micro
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if micro > 1:
                    micro = max(1, micro // 2)
                    print(f"[part_d] OOM → micro_batch={micro}", flush=True)
                    continue
                raise

        sens = sensitivities_from_worlds(names, worlds, deltas_m)

        # 5-seed unperturbed spread (reuse base case only)
        base_case = [cases[0]]
        spread_worlds = []
        for s in range(NOISE_SEEDS_SPREAD):
            nse = _make_noise(cfg, device, frame_seed + 1000 + s, 1)
            w = _forward_chunked(
                pol, cfg, frames, device, lang, static, wrist, base_case, nse, micro
            )
            spread_worlds.append(w[0])
        spread = noise_spread(np.stack(spread_worlds, axis=0))

        return {
            **rec,
            "language": lang,
            "sensitivity": sens,
            "noise_spread": spread,
            "noise_seed": frame_seed,
        }

    idx = 0
    while idx < len(sample):
        rec = sample[idx]
        frame_seed = a.seed + 17 * idx + rec["k"] * 10007
        try:
            row = run_one(rec, frame_seed)
            if row is not None:
                rows.append(row)
            idx += 1
            if len(rows) % 25 == 0 or len(rows) == 1:
                dt = time.time() - t0
                rate = len(rows) / max(dt, 1e-6)
                eta = (len(sample) - len(rows)) / max(rate, 1e-6)
                print(
                    f"[part_d] {len(rows)}/{len(sample)} frames  "
                    f"{rate:.2f}/s  ETA {eta / 60:.1f} min  micro={micro}",
                    flush=True,
                )
                # mid-run checkpoint so a kill still leaves usable partials
                if len(rows) >= a.min_n and len(rows) % 50 == 0:
                    try:
                        _tables = aggregate_rows(rows, ALL_DELTAS_CM)
                        _payload = {
                            "summary": {
                                "checkpoint": a.checkpoint,
                                "frames": frames,
                                "n_frames": len(rows),
                                "n_target": n_target,
                                "seed": a.seed,
                                "device": device,
                                "micro_batch": micro,
                                "tables": _tables,
                                "headline_delta_1.5cm_chunk0": {
                                    "S_h_x": _tables["delta_1.5cm"]["pooled"].get("S_h_x_c0"),
                                    "S_p_x": _tables["delta_1.5cm"]["pooled"].get("S_p_x_c0"),
                                },
                                "status": "IN_PROGRESS",
                                "elapsed_s": dt,
                            },
                            "compare_B8": compare_to_part_b(_tables),
                        }
                        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
                        Path(a.out_json).write_text(json.dumps(_payload, indent=2, default=str))
                        write_md(_payload, Path(a.out_md))
                        print(f"[part_d] checkpointed partial → {a.out_json}", flush=True)
                    except Exception as ck_err:
                        print(f"[part_d] checkpoint write failed (continuing): {ck_err}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            oom_note = (
                f"OOM mid-run after {len(rows)} frames ({e}); "
                f"keeping ≥{a.min_n} if possible, else truncate"
            )
            errors.append(oom_note)
            print(f"[part_d] {oom_note}", flush=True)
            torch.cuda.empty_cache()
            if len(rows) >= a.min_n:
                break
            # shrink remaining sample and continue on smaller set
            remain = max(a.min_n - len(rows), 0)
            sample = sample[: idx + remain]
            print(f"[part_d] truncated remaining sample → total target {len(sample)}", flush=True)
            continue
        except Exception as e:
            errors.append(f"frame {rec['h5']}#{rec['frame']}: {e}")
            traceback.print_exc()
            idx += 1
            continue

    tables = aggregate_rows(rows, ALL_DELTAS_CM)
    # noise spread summary
    ns_ax = [r["noise_spread"]["chunk_0"]["std_ax_cm"] for r in rows]
    ns_ay = [r["noise_spread"]["chunk_0"]["std_ay_cm"] for r in rows]
    noise_summary = {
        "std_ax_cm_c0_mean": float(np.mean(ns_ax)) if ns_ax else None,
        "std_ay_cm_c0_mean": float(np.mean(ns_ay)) if ns_ay else None,
        "std_ax_cm_c0_agg": _agg(ns_ax),
        "std_ay_cm_c0_agg": _agg(ns_ay),
    }

    # stratification counts
    strat = {
        "by_speed": {sb: sum(1 for r in rows if r["speed_bin"] == sb) for sb in sorted({r["speed_bin"] for r in rows})},
        "by_category": {
            c: sum(1 for r in rows if r["category"] == c) for c in sorted({r["category"] for r in rows})
        },
        "by_k": {str(k): sum(1 for r in rows if r["k"] == k) for k in K_RANGE},
    }

    summary = {
        "checkpoint": a.checkpoint,
        "frames": frames,
        "n_frames": len(rows),
        "n_target": n_target,
        "seed": a.seed,
        "device": device,
        "micro_batch": micro,
        "delta_lead_s": DELTA_LEAD_S,
        "primary_deltas_cm": list(PRIMARY_DELTAS_CM),
        "extra_delta_cm": EXTRA_DELTA_CM,
        "chunk_indices": list(CHUNK_IDX),
        "k_range": list(K_RANGE),
        "stratification": strat,
        "tables": tables,
        "noise_spread_summary": noise_summary,
        "headline_delta_1.5cm_chunk0": {
            "S_h_x": tables["delta_1.5cm"]["pooled"].get("S_h_x_c0"),
            "S_p_x": tables["delta_1.5cm"]["pooled"].get("S_p_x_c0"),
            "S_h_y": tables["delta_1.5cm"]["pooled"].get("S_h_y_c0"),
            "S_p_y": tables["delta_1.5cm"]["pooled"].get("S_p_y_c0"),
            "S_both_x": tables["delta_1.5cm"]["pooled"].get("S_both_x_c0"),
            "S_v_x": tables["delta_1.5cm"]["pooled"].get("S_v_x_c0"),
        },
        "oom_fallback": oom_note,
        "errors": errors,
        "elapsed_s": time.time() - t0,
        "source": "foresight_verification/scripts/part_d_sensitivity.py",
        "method": {
            "S_h": "central diff Δa_axis / (2δ) with hand ±δ",
            "S_p": "central diff Δa_axis / (2δ) with p̂ ±δ",
            "S_v": "central diff Δa_axis / (2δ) with v̂ ±δ/Δ (lead-equivalent)",
            "S_both": "hand and p̂ ±δ together; expect ≈ S_h+S_p",
            "D5": "cross-axis Δa_other / (2δ) under same-axis input pert",
            "noise": "fixed flow-matching noise per frame across perts; 5 seeds for unperturbed spread",
            "units": "raw metres before normalization",
        },
    }
    compare = compare_to_part_b(tables)
    # Drop bulky per-frame sensitivity from default JSON? Keep compact: store rows without full grids if huge.
    # Keep rows but they are needed for audit; 512 * small dict is fine.
    compact_rows = []
    for r in rows:
        compact_rows.append(
            {
                "h5": r["h5"],
                "frame": r["frame"],
                "k": r["k"],
                "speed": r["speed"],
                "speed_bin": r["speed_bin"],
                "category": r["category"],
                "noise_seed": r["noise_seed"],
                "noise_spread_c0": r["noise_spread"]["chunk_0"],
                # keep only δ=1.5 and 3.0 at chunk0 for row-level audit
                "S_1.5cm_c0": {
                    ax: r["sensitivity"]["delta_1.5cm"]["axes"][ax]["chunk_0"] for ax in ("x", "y")
                },
                "S_3.0cm_c0": {
                    ax: r["sensitivity"]["delta_3.0cm"]["axes"][ax]["chunk_0"] for ax in ("x", "y")
                },
            }
        )

    payload = {
        "summary": summary,
        "compare_B8": compare,
        "D1": {"name": "hand ±δ → S_h", "tables_key": "tables.*.pooled.S_h_*"},
        "D2": {"name": "p̂ ±δ → S_p", "tables_key": "tables.*.pooled.S_p_*"},
        "D3": {"name": "v̂ ±δ/Δ → S_v (lead-equivalent /δ)", "tables_key": "tables.*.pooled.S_v_*"},
        "D4": {"name": "hand+p̂ ±δ → S_both", "tables_key": "tables.*.pooled.S_both_*"},
        "D5": {"name": "cross-axis", "tables_key": "tables.*.pooled.D5_*"},
        "rows": compact_rows,
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(payload, indent=2, default=str))
    write_md(payload, Path(a.out_md))
    print(json.dumps({"summary_headline": summary["headline_delta_1.5cm_chunk0"], "compare_B8": compare, "n": len(rows)}, indent=2))
    print(f"[part_d] wrote {a.out_json} and {a.out_md}", flush=True)


if __name__ == "__main__":
    main()
