#!/usr/bin/env python3
"""Scratch diag: does Model B conditioning reach the policy input tensor?

No Isaac. Reconstructs eval_policy.build_obs_batch conditioning hops and
checks runs/p1_b checkpoint input dim vs a fake batch's observation.state.
Read-only on production trees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from interfaces.state import ObjectState, conditioning_vector  # noqa: E402
from policy.features import CONDITIONING_DIM, PROPRIO_DIM, STATE_DIM_B  # noqa: E402
from policy.convert_h5 import pack_proprio  # noqa: E402


def _fake_object() -> ObjectState:
    # Matches typical mid-speed table object (not from a live scene).
    cov = np.diag([1.25e-4] * 3 + [1e-4] * 3).astype(np.float64)
    return ObjectState(
        position=np.array([0.45, -0.05, 0.035], dtype=np.float64),
        velocity=np.array([0.12, -0.08, 0.0], dtype=np.float64),
        covariance=cov,
        timestamp=0.0,
        valid=True,
    )


def build_cond(*, zero: bool, delta: float = 0.25) -> np.ndarray:
    """Mirror scripts/eval_policy.py build_obs_batch conditioning."""
    if zero:
        return np.zeros(CONDITIONING_DIM, dtype=np.float32)
    return conditioning_vector(_fake_object(), delta=delta).astype(np.float32)


def build_state(cond: np.ndarray | None) -> np.ndarray:
    proprio = pack_proprio(
        np.array([0.4, 0.0, 0.3], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        1.0,
    )
    if cond is None:
        return proprio.astype(np.float32)
    return np.concatenate([proprio, cond], axis=0).astype(np.float32)


def report_arm(name: str, cond: np.ndarray) -> dict:
    out = {
        "arm": name,
        "cond_shape": list(cond.shape),
        "cond_dtype": str(cond.dtype),
        "cond_l2": float(np.linalg.norm(cond)),
        "cond_all_zero": bool(np.allclose(cond, 0.0)),
        "cond_preview": cond.tolist(),
    }
    state = build_state(cond)
    out["state_shape"] = list(state.shape)
    out["state_dtype"] = str(state.dtype)
    out["state_l2"] = float(np.linalg.norm(state))
    out["state_tail_equals_cond"] = bool(np.allclose(state[-CONDITIONING_DIM:], cond))
    print(
        f"[{name}] cond shape={out['cond_shape']} dtype={out['cond_dtype']} "
        f"L2={out['cond_l2']:.6g} all_zero={out['cond_all_zero']}"
    )
    print(
        f"[{name}] state shape={out['state_shape']} dtype={out['state_dtype']} "
        f"L2={out['state_l2']:.6g} tail==cond={out['state_tail_equals_cond']}"
    )
    return out


def load_ckpt_meta(ckpt: Path) -> dict:
    cfg = json.loads((ckpt / "config.json").read_text())
    shape = cfg["input_features"]["observation.state"]["shape"]
    model_type = (ckpt.parent / "model_type.txt").read_text().strip()
    return {
        "checkpoint": str(ckpt),
        "model_type.txt": model_type,
        "input_features.observation.state.shape": shape,
        "max_state_dim": cfg.get("max_state_dim"),
        "n_action_steps": cfg.get("n_action_steps"),
    }


def try_one_forward(ckpt: Path, state: np.ndarray, device: str = "cpu") -> dict:
    """One policy.select_action; print tensor that enters the call."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.to(device)
    policy.eval()
    policy.reset()

    # Dummy images matching checkpoint visual shape (3,360,480).
    img = torch.zeros(1, 3, 360, 480, device=device, dtype=torch.float32)
    state_t = torch.as_tensor(state, device=device).unsqueeze(0)
    batch = {
        "observation.state": state_t,
        "observation.images.static_cam": img,
        "observation.images.wrist_cam": img.clone(),
        "task": ["Pick the object and place it in the container.\n"],
    }
    print(
        f"[forward] observation.state shape={tuple(state_t.shape)} "
        f"dtype={state_t.dtype} L2={float(state_t.norm()):.6g} "
        f"tail_zero={bool(torch.allclose(state_t[0, -CONDITIONING_DIM:], torch.zeros(CONDITIONING_DIM, device=device)))}"
    )
    with torch.inference_mode():
        act = policy.select_action(batch)
    return {
        "obs_state_shape": list(state_t.shape),
        "obs_state_dtype": str(state_t.dtype),
        "obs_state_l2": float(state_t.norm().item()),
        "action_shape": list(act.shape),
        "action_finite": bool(torch.isfinite(act).all().item()),
    }


def main() -> None:
    print("CONST PROPRIO_DIM=", PROPRIO_DIM, "CONDITIONING_DIM=", CONDITIONING_DIM, "STATE_DIM_B=", STATE_DIM_B)
    b = report_arm("B", build_cond(zero=False))
    bz = report_arm("B_zero", build_cond(zero=True))

    ckpt_b = ROOT / "runs" / "p1_b" / "pretrained_model"
    ckpt_a = ROOT / "runs" / "p1_a" / "pretrained_model"
    meta_b = load_ckpt_meta(ckpt_b)
    meta_a = load_ckpt_meta(ckpt_a)
    print("[ckpt A]", json.dumps(meta_a))
    print("[ckpt B]", json.dumps(meta_b))

    train_dim_b = int(meta_b["input_features.observation.state.shape"][0])
    eval_dim_b = int(b["state_shape"][0])
    print(f"[dim] train_B={train_dim_b} eval_B_state={eval_dim_b} match={train_dim_b == eval_dim_b}")
    print(
        f"[dim] train_A={meta_a['input_features.observation.state.shape'][0]} "
        f"model_type A/B={meta_a['model_type.txt']}/{meta_b['model_type.txt']}"
    )

    # Mid5 logged evidence (no Isaac re-run).
    mid5_b = ROOT / "data/p1/reeval_n8_mid5/eval_b/eval_B_v0.20_s9200_conditioning.json"
    mid5_bz = ROOT / "data/p1/reeval_n8_mid5/eval_b_zero/eval_B_v0.20_s9200_conditioning.json"
    if mid5_b.is_file():
        jb = json.loads(mid5_b.read_text())
        print(
            f"[mid5 B] zero_flag={jb['zero_conditioning']} "
            f"l2_mean_replan={jb['l2_mean_replan']:.6g} all_zero={jb['all_zero']} "
            f"first_replan_len={len(jb['first_replan'])}"
        )
    if mid5_bz.is_file():
        jz = json.loads(mid5_bz.read_text())
        print(
            f"[mid5 B_zero] zero_flag={jz['zero_conditioning']} "
            f"l2_mean_replan={jz['l2_mean_replan']:.6g} all_zero={jz['all_zero']} "
            f"first_replan_len={len(jz['first_replan'])}"
        )

    forward = None
    try:
        forward = try_one_forward(ckpt_b, build_state(build_cond(zero=False)), device="cpu")
        print("[forward B ok]", json.dumps(forward))
        forward_z = try_one_forward(ckpt_b, build_state(build_cond(zero=True)), device="cpu")
        print("[forward B_zero ok]", json.dumps(forward_z))
    except Exception as ex:
        print(f"[forward] SKIPPED/FAILED: {type(ex).__name__}: {ex}")
        forward = {"error": str(ex)}

    out = {
        "B": b,
        "B_zero": bz,
        "ckpt_A": meta_a,
        "ckpt_B": meta_b,
        "train_eval_dim_match": train_dim_b == eval_dim_b,
        "forward": forward,
    }
    out_path = ROOT / "tools" / "diag" / "p1_conditioning_wiring_out.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
