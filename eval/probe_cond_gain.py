"""Does the model aim with the 4 numbers? Offline shift test (no Isaac).

For saved frames just before the grasp, ask the model for its action chunk, then
ask again with one input changed and see how far the aimed point (mean XY of the
first ``--k`` actions, world frame) moves:

  cond gain  — move p̂ by +3 cm. The model aims with the numbers ⇒ gain ≈ 1.
  hand gain  — tell the model its hand is 3 cm elsewhere (object unchanged).
               Copying the hand ("hand + offset") ⇒ gain ≈ 1. Wanted: ≈ 0.
  static / wrist gray — swap that camera for a gray frame; aim shift in cm.
  E0 additions (all in-distribution unless noted):
  cond_swap  — the 4 numbers from a *different* episode (a real, plausible object).
               follow = projection of the aim shift on (donor p̂ - own p̂) / |.|² ⇒ 1 = follows.
  static/wrist_swap — that camera's frame from a different episode (aim shift, cm).
  lang_*     — other object name / generic / blank / unrelated task text (aim shift, cm).
  noise_floor — same inputs, different flow-matching noise (aim shift, cm): the
               sensitivity any input change must beat to count as "used".

Inputs are rebuilt exactly as eval does (``cond_to_model`` / ``action_from_model``
with the checkpoint's saved frames), so abs and rel/delta checkpoints compare 1:1.

Usage:
  python eval/probe_cond_gain.py --checkpoint runs/X/pretrained_model \
      --h5 'data/p3_throw/*/speed_*/*.h5' --n 40 --device cuda:0 -o out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHIFT_M = 0.03


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


def probe_frame(pol, cfg, frames, f, i: int, lang: str, device: str, k: int, seed: int, donor: dict | None = None,
                langs: dict | None = None) -> dict:
    """All perturbations for one frame in one batched forward pass."""
    from policy.convert_h5 import pack_proprio
    from policy.transforms import action_from_model, cond_to_model

    ee = np.asarray(f["ee_pos"][i], dtype=np.float64)
    quat, grip = f["ee_quat"][i], float(f["gripper"][i])
    cond_w = np.asarray(f["conditioning"][i], dtype=np.float64)  # world p̂, v̂ fed at the time
    static, wrist = f["static_cam_rgb"][i], f["wrist_cam_rgb"][i]
    gray_s, gray_w = np.full_like(static, 128), np.full_like(wrist, 128)
    sx, sy = np.array([SHIFT_M, 0.0]), np.array([0.0, SHIFT_M])

    # (name, hand the model is told, world cond, static, wrist)
    d = donor or {"cond": cond_w, "static": static, "wrist": wrist}
    cond_swap = np.r_[np.asarray(d["cond"], dtype=np.float64)[0:2], cond_w[2:4]]  # donor p̂, own v̂
    cases = [
        ("base", ee, cond_w, static, wrist),
        ("cond_swap", ee, cond_swap, static, wrist),
        ("static_swap", ee, cond_w, d["static"], wrist),
        ("wrist_swap", ee, cond_w, static, d["wrist"]),
        ("cond_x", ee, cond_w + np.r_[sx, 0, 0], static, wrist),
        ("cond_y", ee, cond_w + np.r_[sy, 0, 0], static, wrist),
        ("hand_x", ee + np.r_[sx, 0], cond_w, static, wrist),
        ("hand_y", ee + np.r_[sy, 0], cond_w, static, wrist),
        ("static_gray", ee, cond_w, gray_s, wrist),
        ("wrist_gray", ee, cond_w, static, gray_w),
    ]
    states, s_imgs, w_imgs = [], [], []
    for _, hand, cw, si, wi in cases:
        st = pack_proprio(hand, quat, grip)
        states.append(np.r_[st, cond_to_model(cw, hand, frames["cond_frame"])])
        s_imgs.append(_img(si))
        w_imgs.append(_img(wi))
    case_lang = [lang] * len(cases)
    for name, text in (langs or {}).items():  # language counterfactuals: same state and images as base
        cases.append((name, ee, cond_w, static, wrist))
        st = pack_proprio(ee, quat, grip)
        states.append(np.r_[st, cond_to_model(cond_w, ee, frames["cond_frame"])])
        s_imgs.append(_img(static))
        w_imgs.append(_img(wrist))
        case_lang.append(text)
    cases.append(("noise_floor", ee, cond_w, static, wrist))
    st = pack_proprio(ee, quat, grip)
    states.append(np.r_[st, cond_to_model(cond_w, ee, frames["cond_frame"])])
    s_imgs.append(_img(static))
    w_imgs.append(_img(wrist))
    case_lang.append(lang)
    n = len(cases)
    batch = {
        "observation.state": torch.tensor(np.stack(states), dtype=torch.float32, device=device),
        "observation.images.static_cam": torch.stack(s_imgs).to(device),
        "observation.images.wrist_cam": torch.stack(w_imgs).to(device),
        "task": [t if t.endswith("\n") else t + "\n" for t in case_lang],
    }
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g).repeat(n, 1, 1)
    noise[-1] = torch.randn((cfg.chunk_size, cfg.max_action_dim), generator=g)  # noise_floor case
    noise = noise.to(device)
    with torch.inference_mode():
        out = pol.predict_action_chunk(batch, noise=noise).float().cpu().numpy()
    aims = {}
    for (name, hand, *_), chunk in zip(cases, out):
        world = action_from_model(chunk, hand, frames["action_frame"])
        aims[name] = world[:k, 0:2].mean(0)
    base = aims["base"]
    obj = np.asarray(f["object_pos"][i, 0:2], dtype=np.float64)
    label = np.asarray(f["action_pos"][i : i + k, 0:2], dtype=np.float64).mean(0)
    return {
        "aim_minus_obj_cm": ((base - obj) * 100).tolist(),
        "aim_minus_label_cm": ((base - label) * 100).tolist(),
        "cond_gain": [float((aims["cond_x"] - base)[0] / SHIFT_M), float((aims["cond_y"] - base)[1] / SHIFT_M)],
        "hand_gain": [float((aims["hand_x"] - base)[0] / SHIFT_M), float((aims["hand_y"] - base)[1] / SHIFT_M)],
        "static_gray_cm": float(np.linalg.norm(aims["static_gray"] - base) * 100),
        "wrist_gray_cm": float(np.linalg.norm(aims["wrist_gray"] - base) * 100),
        "cond_swap_follow": _follow(aims["cond_swap"] - base, cond_swap[0:2] - cond_w[0:2]),
        "cond_swap_dist_cm": float(np.linalg.norm(cond_swap[0:2] - cond_w[0:2]) * 100),
        "static_swap_cm": float(np.linalg.norm(aims["static_swap"] - base) * 100),
        "wrist_swap_cm": float(np.linalg.norm(aims["wrist_swap"] - base) * 100),
        "noise_floor_cm": float(np.linalg.norm(aims["noise_floor"] - base) * 100),
        **{f"{name}_cm": float(np.linalg.norm(aims[name] - base) * 100) for name in (langs or {})},
    }


def _follow(shift: np.ndarray, target: np.ndarray) -> float:
    """How much of the move toward ``target`` the aim made (1 = all of it)."""
    d2 = float(target @ target)
    return float(shift @ target / d2) if d2 > 1e-8 else float("nan")


def _language(h5: str) -> str:
    from policy.features import default_language

    js = Path(h5).with_suffix(".json")
    m = json.loads(js.read_text()) if js.exists() else {}
    if m.get("language"):
        return m["language"]
    if m.get("category") is None:  # eval files: identity lives in metrics.jsonl
        mf = Path(h5).parent / "metrics.jsonl"
        if mf.exists():
            for line in mf.read_text().splitlines():
                r = json.loads(line)
                if Path(r.get("h5", "")).name == Path(h5).name:
                    m = r
                    break
    return default_language(m.get("category"), m.get("container_category"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5", required=True, help="glob of episode .h5 files (demos or eval)")
    p.add_argument("--n", type=int, default=40, help="episodes to sample")
    p.add_argument("--before-close", type=int, nargs="+", default=[8, 16], help="frames before the close")
    p.add_argument("--k", type=int, default=8, help="actions averaged into the aim (= replan steps)")
    p.add_argument("--success-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    files = sorted(glob.glob(a.h5))
    random.Random(a.seed).shuffle(files)
    pol, cfg, frames = _load(a.checkpoint, a.device)
    from policy.features import default_language

    # Pass 1: pick frames. Pass 2: probe each frame with a donor from another episode.
    picks = []
    for h5 in files:
        if len(picks) >= a.n * len(a.before_close):
            break
        if a.success_only:
            js = Path(h5).with_suffix(".json")
            m = json.loads(js.read_text()) if js.exists() else {}
            if not m.get("success") or m.get("clean_reject"):
                continue
        with h5py.File(h5, "r") as f:
            if "static_cam_rgb" not in f or "conditioning" not in f:
                continue
            g = f["gripper"][:]
            if not (g < 0.5).any():
                continue
            ic = int(np.argmax(g < 0.5))
            for b in a.before_close:
                if ic - b >= 0:
                    picks.append((h5, ic - b, b))
    rows = []
    for j, (h5, i, b) in enumerate(picks):
        dh5, di, _ = picks[(j + len(a.before_close)) % len(picks)]  # a frame from another episode
        if dh5 == h5:
            dh5, di, _ = picks[(j + 2 * len(a.before_close) + 1) % len(picks)]
        with h5py.File(dh5, "r") as fd:
            donor = {"cond": fd["conditioning"][di][:], "static": fd["static_cam_rgb"][di][:], "wrist": fd["wrist_cam_rgb"][di][:]}
        lang = _language(h5)
        other = "tomato" if "tangerine" not in lang else "lemon"
        langs = {
            "lang_obj": re.sub(r"Pick the (.+?) and", f"Pick the {other} and", lang),
            "lang_generic": "Pick the object and place it in the container.",
            "lang_blank": "",
            "lang_other": "Open the drawer and push the button.",
        }
        with h5py.File(h5, "r") as f:
            speed = float(np.asarray(f["speed_m_s"]).reshape(-1)[0]) if "speed_m_s" in f else float("nan")
            r = probe_frame(pol, cfg, frames, f, i, lang, a.device, a.k, a.seed + j, donor=donor, langs=langs)
        r.update(h5=h5, frame=i, before_close=b, speed=speed, donor=dh5, lang=lang)
        rows.append(r)

    def med(key, j=None):
        v = [r[key] if j is None else r[key][j] for r in rows]
        return float(np.median(v)) if v else float("nan")

    summary = {
        "checkpoint": a.checkpoint,
        "frames": frames,
        "n_frames": len(rows),
        "cond_gain_xy": [med("cond_gain", 0), med("cond_gain", 1)],
        "hand_gain_xy": [med("hand_gain", 0), med("hand_gain", 1)],
        "static_gray_cm": med("static_gray_cm"),
        "wrist_gray_cm": med("wrist_gray_cm"),
        "cond_swap_follow": med("cond_swap_follow"),
        "cond_swap_dist_cm": med("cond_swap_dist_cm"),
        "static_swap_cm": med("static_swap_cm"),
        "wrist_swap_cm": med("wrist_swap_cm"),
        "noise_floor_cm": med("noise_floor_cm"),
        **{f"{k}_cm": med(f"{k}_cm") for k in ("lang_obj", "lang_generic", "lang_blank", "lang_other")},
        "p90": {k: float(np.percentile([r[k] for r in rows], 90)) for k in
                ("noise_floor_cm", "static_gray_cm", "wrist_gray_cm", "static_swap_cm", "wrist_swap_cm",
                 "lang_obj_cm", "lang_generic_cm", "lang_blank_cm", "lang_other_cm")} if rows else {},
        "abs_aim_minus_obj_cm": float(np.median([np.linalg.norm(r["aim_minus_obj_cm"]) for r in rows])) if rows else None,
        "abs_aim_minus_label_cm": float(np.median([np.linalg.norm(r["aim_minus_label_cm"]) for r in rows])) if rows else None,
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
