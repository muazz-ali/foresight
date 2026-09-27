"""Training helpers: fixed look-ahead + sensor noise; pack Model A vs B inputs.

“Conditioning” = the 4 live future numbers (XY p̂, XY v̂).
“Oracle state” = perfect sim pose saved each frame (14-D; Δ rebuild uses it).
Δ defaults come from yaml (``conditioning_delta_s`` / ``smolvla_b.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from interfaces.config import policy_aug_defaults
from interfaces.state import ObjectState, conditioning_vector
from policy.features import CONDITIONING_DIM, PROPRIO_DIM, STATE_DIM_B


class _SizedMapDataset(Protocol):
    """Map-style dataset: torch.utils.data.Dataset does not declare __len__."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


try:
    _AUG_YAML = policy_aug_defaults()
except OSError:
    _AUG_YAML = {
        "delta_min": 0.25,
        "delta_max": 0.25,
        "pos_noise_std": 0.015,
        "vel_noise_std": 0.03,
    }


@dataclass
class AugConfig:
    """Fixed Δ + sensor noise for Model B (values from ``smolvla_b.yaml``).

    Keep ``delta_min == delta_max`` while Δ is not packed — a spread only smears p̂. 
    Widen / pack Δ as a 5th number when deploy latency varies.
    """

    delta_min: float = _AUG_YAML["delta_min"]
    delta_max: float = _AUG_YAML["delta_max"]
    pos_noise_std: float = _AUG_YAML["pos_noise_std"]
    vel_noise_std: float = _AUG_YAML["vel_noise_std"]
    seed: int = 0
    zero_conditioning: bool = False  # eval sanity for B


def oracle_flat_to_state(flat: np.ndarray | torch.Tensor) -> ObjectState:
    """Convert Oracle state to ObjectState."""
    x = np.asarray(flat, dtype=np.float64).reshape(-1)
    assert x.shape[0] >= 14, x.shape
    return ObjectState(
        position=x[0:3],
        velocity=x[3:6],
        covariance=x[6:12],
        timestamp=float(x[12]),
        valid=bool(x[13] > 0.5),
    )


def augment_conditioning(
    oracle_flat: np.ndarray | torch.Tensor,
    *,
    rng: np.random.Generator,
    cfg: AugConfig,
) -> np.ndarray:
    """Rebuild 4-D live conditioning at cfg Δ with Gaussian noise on XY p̂/v̂."""
    if cfg.zero_conditioning:
        return np.zeros(CONDITIONING_DIM, dtype=np.float64)

    state = oracle_flat_to_state(oracle_flat)
    delta = float(rng.uniform(cfg.delta_min, cfg.delta_max))
    cond = conditioning_vector(state, delta=delta).astype(np.float64)
    # Layout: [p_hat_x, p_hat_y, v_hat_x, v_hat_y]
    cond[0:2] += rng.normal(0.0, cfg.pos_noise_std, size=2)
    cond[2:4] += rng.normal(0.0, cfg.vel_noise_std, size=2)
    return cond


class PolicyDataset(Dataset):
    """Wrap LeRobotDataset: Model A keeps proprio; Model B appends aug conditioning."""

    def __init__(
        self,
        base: _SizedMapDataset,
        *,
        model: str = "A",
        aug: AugConfig | None = None,
        apply_aug: bool = True,
    ):
        if model not in ("A", "B"):
            raise ValueError(f"model must be A or B, got {model}")
        self.base = base
        self.model = model
        self.aug = aug or AugConfig()
        self.apply_aug = bool(apply_aug)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.base[idx])
        state = item["observation.state"]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().float()
        else:
            state = torch.as_tensor(state, dtype=torch.float32)
        # torch.initial_seed() differs per DataLoader worker and is redrawn each
        # epoch, so noise is fresh on every visit yet reproducible from --seed.
        # With num_workers=0 nothing varies it and the draw stays frozen.
        rng = np.random.default_rng(
            [self.aug.seed, int(idx), int(torch.initial_seed()) & 0xFFFFFFFF]
        )

        # LeRobot delta_timestamps may yield state as (T, D) — use last frame for proprio/cond.
        if state.ndim == 2:
            state_last = state[-1]
            state_seq = state
        else:
            state_last = state
            state_seq = None

        if self.model == "B":
            if self.apply_aug and "observation.oracle_state" in item:
                oracle = item["observation.oracle_state"]
                if isinstance(oracle, torch.Tensor):
                    oracle = oracle.detach().cpu().numpy()
                oracle = np.asarray(oracle)
                if oracle.ndim == 2:
                    oracle = oracle[-1]
                cond = augment_conditioning(oracle, rng=rng, cfg=self.aug)
            elif "observation.conditioning" in item:
                cond = item["observation.conditioning"]
                if isinstance(cond, torch.Tensor):
                    cond = cond.detach().cpu().numpy()
                cond = np.asarray(cond)
                if cond.ndim == 2:
                    cond = cond[-1]
                if self.aug.zero_conditioning:
                    cond = np.zeros_like(cond)
            else:
                raise KeyError("Model B needs observation.oracle_state or observation.conditioning")
            cond_t = torch.as_tensor(cond, dtype=torch.float32).reshape(CONDITIONING_DIM)
            if state_seq is not None:
                # Broadcast same aug conditioning across obs history (usually T=1).
                cond_rep = cond_t.unsqueeze(0).expand(state_seq.shape[0], -1)
                item["observation.state"] = torch.cat(
                    [state_seq.reshape(-1, PROPRIO_DIM), cond_rep], dim=-1
                )
            else:
                item["observation.state"] = torch.cat(
                    [state_last.reshape(PROPRIO_DIM), cond_t], dim=-1
                )
            assert item["observation.state"].shape[-1] == STATE_DIM_B
        else:
            item["observation.state"] = state if state.ndim == 2 else state.reshape(PROPRIO_DIM)

        # Drop aux keys so Normalize / SmolVLA only see state + images + action.
        item.pop("observation.conditioning", None)
        item.pop("observation.oracle_state", None)
        return item


def policy_features_for_model(raw_features: dict, model: str) -> dict:
    """Shrink dataset meta features to what SmolVLA should consume."""
    out = {}
    for k, v in raw_features.items():
        if k in ("observation.conditioning", "observation.oracle_state"):
            continue
        if k.startswith("observation.") or k == "action" or k.startswith("observation.images"):
            out[k] = dict(v)
    if model == "B":
        st = dict(out["observation.state"])
        st["shape"] = (STATE_DIM_B,)
        st["names"] = list(st.get("names") or []) + [f"cond_{i}" for i in range(CONDITIONING_DIM)]
        out["observation.state"] = st
    return out


def expand_state_stats(stats: dict, model: str) -> dict:
    """Pad observation.state stats for Model B concatenated dims."""
    if model != "B" or "observation.state" not in stats:
        # Drop aux from stats copy.
        return {k: v for k, v in stats.items() if k not in ("observation.conditioning", "observation.oracle_state")}

    out = {k: v for k, v in stats.items() if k not in ("observation.conditioning", "observation.oracle_state")}
    st = {k: np.asarray(v) for k, v in stats["observation.state"].items()}
    cond = stats.get("observation.conditioning")
    if cond is None:
        # Identity-ish stats for cond dims.
        zeros = np.zeros(CONDITIONING_DIM, dtype=np.float32)
        ones = np.ones(CONDITIONING_DIM, dtype=np.float32)
        cond_stats = {"min": zeros, "max": ones, "mean": zeros, "std": ones, "count": st.get("count")}
    else:
        cond_stats = {k: np.asarray(v) for k, v in cond.items()}

    merged = {}
    for key in ("min", "max", "mean", "std"):
        merged[key] = np.concatenate(
            [st[key].reshape(-1)[:PROPRIO_DIM], cond_stats[key].reshape(-1)[:CONDITIONING_DIM]]
        ).astype(np.float32)
    if "count" in st:
        merged["count"] = st["count"]
    out["observation.state"] = merged
    return out
