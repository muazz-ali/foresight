"""Training helpers: fixed look-ahead + sensor noise; pack 12-D policy state.

“Conditioning” = the 4 live future numbers (XY p̂, XY v̂):
  where the object will be (XY), and how fast it moves (XY).
“Oracle state” = perfect sim pose saved each frame (14-D; Δ rebuild uses it).
Δ defaults come from yaml (``conditioning_delta_s`` / ``smolvla.yaml``).

Why this file exists:
  * The saved data has clean, perfect numbers. A real camera + filter gives
    slightly wrong ones. So at train time we rebuild the 4 numbers every time
    we read a frame and add a little random noise. This gives the model
    examples of small input errors during training.
  * We glue the 4 numbers onto the 8 robot numbers -> the 12-number state.
    Eval / deploy must build the same 12 numbers in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from interfaces.config import policy_aug_defaults
from interfaces.state import ObjectState, conditioning_vector
from policy.features import CONDITIONING_DIM, PROPRIO_DIM, STATE_DIM


class _SizedMapDataset(Protocol):
    """Map-style dataset: torch.utils.data.Dataset does not declare __len__."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


# Default Δ and noise sizes come from policy/configs/smolvla.yaml.
# If the yaml cannot be read, use the same numbers as the yaml has today.
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
    """Fixed Δ + sensor noise (values from ``smolvla.yaml``).

    Keep ``delta_min == delta_max`` while Δ is not packed — a spread only smears p̂.
    Widen / pack Δ as a 5th number when deploy latency varies.
    """

    # Look-ahead time in seconds: "where will the object be this far ahead?"
    delta_min: float = _AUG_YAML["delta_min"]
    delta_max: float = _AUG_YAML["delta_max"]
    # Noise size (std dev) added to the 4 numbers: meters, and meters/second.
    pos_noise_std: float = _AUG_YAML["pos_noise_std"]
    vel_noise_std: float = _AUG_YAML["vel_noise_std"]
    seed: int = 0  # makes the noise repeatable
    zero_conditioning: bool = False  # eval sanity: force zeros on the 4 cond dims
    # Frames (see POLICY_FRAMES): "abs" = world numbers as before.
    # cond_frame "rel": p̂ minus hand XY ("object will be 2 cm ahead of you").
    # action_frame "delta": hand XYZ targets minus hand XYZ at the obs frame.
    cond_frame: str = "abs"
    action_frame: str = "abs"
    # Noise (m) on the hand XYZ the model reads. Rel cond / delta action use the
    # clean hand, so this only stops "copy my hand + offset" shortcuts.
    ee_noise_std: float = 0.0


POLICY_FRAMES = {"cond_frame": ("abs", "rel"), "action_frame": ("abs", "delta")}
# Saved next to each checkpoint so eval rebuilds the same inputs / outputs.
FRAMES_FILE = "foresight_policy.json"


def cond_to_model(cond_abs: np.ndarray, ee_xyz: np.ndarray, cond_frame: str) -> np.ndarray:
    """World 4-number pack → what the model reads. Train and eval share this."""
    out = np.asarray(cond_abs, dtype=np.float64).copy()
    if cond_frame == "rel":
        out[0:2] -= np.asarray(ee_xyz, dtype=np.float64).reshape(-1)[0:2]
    elif cond_frame != "abs":
        raise ValueError(f"cond_frame must be abs|rel, got {cond_frame!r}")
    return out


def action_from_model(chunk: np.ndarray, ee_xyz: np.ndarray, action_frame: str) -> np.ndarray:
    """Model action chunk (T, 8) → world hand targets. Inverse of the train-side shift."""
    out = np.asarray(chunk, dtype=np.float64).copy()
    if action_frame == "delta":
        out[..., 0:3] += np.asarray(ee_xyz, dtype=np.float64).reshape(-1)[0:3]
    elif action_frame != "abs":
        raise ValueError(f"action_frame must be abs|delta, got {action_frame!r}")
    return out


def load_frames(checkpoint: str | Path) -> dict[str, Any]:
    """Frames a checkpoint was trained with. Old checkpoints (no file) = abs / abs."""
    import json

    p = Path(checkpoint) / FRAMES_FILE
    d = json.loads(p.read_text()) if p.is_file() else {}
    return {"cond_frame": d.get("cond_frame", "abs"), "action_frame": d.get("action_frame", "abs"), **d}


def oracle_flat_to_state(flat: np.ndarray | torch.Tensor) -> ObjectState:
    """Convert Oracle state to ObjectState."""
    # Slots: pos(0:3) vel(3:6) cov(6:12) time(12) valid(13). Same as ORACLE_NAMES.
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
    # Eval-only switch (see AugConfig). Training never sets it.
    if cfg.zero_conditioning:
        return np.zeros(CONDITIONING_DIM, dtype=np.float64)

    state = oracle_flat_to_state(oracle_flat)
    delta = float(rng.uniform(cfg.delta_min, cfg.delta_max))
    # Same helper the filter path uses, so train and deploy build the 4 numbers alike.
    cond = conditioning_vector(state, delta=delta).astype(np.float64)
    # Layout: [p_hat_x, p_hat_y, v_hat_x, v_hat_y]
    # Shake the clean numbers a bit, like a real tracker would.
    cond[0:2] += rng.normal(0.0, cfg.pos_noise_std, size=2)
    cond[2:4] += rng.normal(0.0, cfg.vel_noise_std, size=2)
    return cond


class PolicyDataset(Dataset):
    """Wrap LeRobotDataset: append aug conditioning to proprio → 12-D state.

    Each read: take the 8 robot numbers, make 4 fresh noisy object numbers,
    glue them (robot first, object last) -> 12 numbers. Drop the side data.
    """

    def __init__(
        self,
        base: _SizedMapDataset,
        *,
        aug: AugConfig | None = None,
        apply_aug: bool = True,
    ):
        self.base = base
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

        if self.apply_aug and "observation.oracle_state" in item:
            oracle = item["observation.oracle_state"]
            if isinstance(oracle, torch.Tensor):
                oracle = oracle.detach().cpu().numpy()
            oracle = np.asarray(oracle)
            if oracle.ndim == 2:
                oracle = oracle[-1]
            cond = augment_conditioning(oracle, rng=rng, cfg=self.aug)
        elif "observation.conditioning" in item:
            # No oracle to rebuild from: use the saved clean 4 numbers as they are.
            cond = item["observation.conditioning"]
            if isinstance(cond, torch.Tensor):
                cond = cond.detach().cpu().numpy()
            cond = np.asarray(cond)
            if cond.ndim == 2:
                cond = cond[-1]
            if self.aug.zero_conditioning:
                cond = np.zeros_like(cond)
        else:
            raise KeyError(
                "PolicyDataset needs observation.oracle_state or observation.conditioning"
            )
        # Frames: rel cond and delta action both use the clean hand pose.
        ee_clean = state_last[0:3].numpy().astype(np.float64)
        if not self.aug.zero_conditioning:
            cond = cond_to_model(cond, ee_clean, self.aug.cond_frame)
        if self.aug.action_frame != "abs" and "action" in item:
            act = torch.as_tensor(item["action"], dtype=torch.float32).clone()
            act[..., 0:3] -= torch.as_tensor(ee_clean, dtype=torch.float32)
            item["action"] = act
        if self.apply_aug and self.aug.ee_noise_std > 0:
            # Noisy copy of the hand XYZ only in what the model reads.
            shake = torch.as_tensor(
                rng.normal(0.0, self.aug.ee_noise_std, size=3), dtype=torch.float32
            )
            if state_seq is not None:
                state_seq = state_seq.clone()
                state_seq[..., 0:3] += shake
            else:
                state_last = state_last.clone()
                state_last[0:3] += shake
        cond_t = torch.as_tensor(cond, dtype=torch.float32).reshape(CONDITIONING_DIM)
        # Glue: [8 robot | 4 object] -> 12. Order matters; infer must match.
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
        assert item["observation.state"].shape[-1] == STATE_DIM

        # Drop aux keys so Normalize / SmolVLA only see state + images + action.
        item.pop("observation.conditioning", None)
        item.pop("observation.oracle_state", None)
        return item


def policy_features_for_model(raw_features: dict) -> dict:
    """Shrink dataset meta features to what SmolVLA should consume.

    Tells the model: state is 12 wide, not 8. Side data is hidden.
    """
    out = {}
    for k, v in raw_features.items():
        if k in ("observation.conditioning", "observation.oracle_state"):
            continue
        if k.startswith("observation.") or k == "action" or k.startswith("observation.images"):
            out[k] = dict(v)
    st = dict(out["observation.state"])
    st["shape"] = (STATE_DIM,)
    st["names"] = list(st.get("names") or []) + [f"cond_{i}" for i in range(CONDITIONING_DIM)]
    out["observation.state"] = st
    return out


def expand_state_stats(stats: dict) -> dict:
    """Pad observation.state stats for concatenated proprio + conditioning dims.

    The model scales each input with saved min/max/mean/std. Saved stats only
    cover 8 robot numbers, so we add the 4 object numbers' stats to make 12.
    The same stats must be used at train and infer.
    """
    if "observation.state" not in stats:
        return {
            k: v
            for k, v in stats.items()
            if k not in ("observation.conditioning", "observation.oracle_state")
        }

    out = {
        k: v
        for k, v in stats.items()
        if k not in ("observation.conditioning", "observation.oracle_state")
    }
    st = {k: np.asarray(v) for k, v in stats["observation.state"].items()}
    cond = stats.get("observation.conditioning")
    if cond is None:
        # No saved stats for the 4 numbers: use a simple 0/1 fallback.
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


def frame_stats(
    stats: dict,
    dataset_root: str | Path,
    *,
    cond_frame: str,
    action_frame: str,
    chunk_size: int,
    n_samples: int = 400_000,
    seed: int = 0,
) -> dict:
    """Replace normalization stats for rel cond / delta action (abs frames: unchanged).

    ``stats`` is the 12-D output of ``expand_state_stats``. Reads only the number
    columns of the LeRobot parquet files (no images), so it takes seconds.
    Delta action = action[t+k] - hand[t], k in [0, chunk), clipped at episode end,
    the same pairing the chunked training samples use.
    """
    if cond_frame == "abs" and action_frame == "abs":
        return stats
    import pyarrow.parquet as pq

    files = sorted(Path(dataset_root).glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset_root}/data")
    states, conds, acts, ep_start, ep_len = [], [], [], [], []
    n = 0
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "observation.conditioning", "action"])
        s = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
        states.append(s)
        conds.append(np.stack(t.column("observation.conditioning").to_numpy(zero_copy_only=False)))
        acts.append(np.stack(t.column("action").to_numpy(zero_copy_only=False)))
        ep_start.append(np.full(len(s), n))
        ep_len.append(np.full(len(s), len(s)))
        n += len(s)
    S, C, A = np.concatenate(states), np.concatenate(conds), np.concatenate(acts)
    start, length = np.concatenate(ep_start), np.concatenate(ep_len)

    def _summ(x: np.ndarray) -> dict:
        return {
            "min": x.min(0).astype(np.float32),
            "max": x.max(0).astype(np.float32),
            "mean": x.mean(0).astype(np.float32),
            "std": np.maximum(x.std(0), 1e-6).astype(np.float32),
        }

    out = dict(stats)
    st = {k: np.asarray(v).copy() for k, v in stats["observation.state"].items()}
    if cond_frame == "rel":
        rel = C.copy()
        rel[:, 0:2] -= S[:, 0:2]
        cs = _summ(rel)
        for k in ("min", "max", "mean", "std"):
            st[k][PROPRIO_DIM:] = cs[k]
    out["observation.state"] = st
    if action_frame == "delta":
        rng = np.random.default_rng(seed)
        t = rng.integers(0, len(S), size=min(n_samples, len(S)))
        k = rng.integers(0, chunk_size, size=len(t))
        j = np.minimum(t + k, start[t] + length[t] - 1)
        d = A[j].copy()
        d[:, 0:3] -= S[t, 0:3]
        a_old = {kk: np.asarray(v) for kk, v in stats["action"].items()}
        a_new = _summ(d)
        # Rotation / gripper columns keep their original stats (std 0 stays 0).
        for kk in ("min", "max", "mean", "std"):
            col = a_old[kk].astype(np.float32).copy()
            col[0:3] = a_new[kk][0:3]
            a_new[kk] = col
        if "count" in a_old:
            a_new["count"] = a_old["count"]
        out["action"] = a_new
    return out
