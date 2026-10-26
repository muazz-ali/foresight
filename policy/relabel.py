"""Step 4: relative actions to the forecast, and delay-aware demo relabel.

Plain words:
  * Absolute expert target at slot k is ``target_k`` (hand XYZ + quat + gripper).
  * Forecast at that slot is ``p̂_k`` (object place when the motion should land).
  * Relative label: ``d_k = target_k_xyz − p̂_k`` (orientation + gripper unchanged).
  * To play back: ``target_k_xyz = p̂_k + d_k``.

Relabel (no new demos): for a frame at time t, draw Δ ∈ [0, 200] ms, build the
forecast for that Δ, and take expert motions from t+Δ onward as the chunk label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interfaces.forecast import DEFAULT_CHUNK, DEFAULT_DT_S, forecast_horizon
from interfaces.state import ObjectState, oracle_from_gt

ACTION_DIM = 8  # ee_xyz(3) + quat_wxyz(4) + gripper(1)


def actions_to_forecast_rel(
    targets: np.ndarray,
    forecasts: np.ndarray,
) -> np.ndarray:
    """Absolute chunk (T, 8) → relative to object forecast (T, 8).

    Only XYZ becomes ``target − p̂``. Quat + gripper stay as-is.
    """
    targets = np.asarray(targets, dtype=np.float64)
    forecasts = np.asarray(forecasts, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != ACTION_DIM:
        raise ValueError(f"targets must be (T, {ACTION_DIM}), got {targets.shape}")
    if forecasts.shape != (targets.shape[0], 3):
        raise ValueError(
            f"forecasts must be (T, 3) matching targets T, got {forecasts.shape}"
        )
    out = targets.copy()
    out[:, 0:3] = targets[:, 0:3] - forecasts
    return out


def actions_from_forecast_rel(
    rel: np.ndarray,
    forecasts: np.ndarray,
) -> np.ndarray:
    """Inverse of ``actions_to_forecast_rel``: relative chunk → world hand targets."""
    rel = np.asarray(rel, dtype=np.float64)
    forecasts = np.asarray(forecasts, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[1] != ACTION_DIM:
        raise ValueError(f"rel must be (T, {ACTION_DIM}), got {rel.shape}")
    if forecasts.shape != (rel.shape[0], 3):
        raise ValueError(
            f"forecasts must be (T, 3) matching rel T, got {forecasts.shape}"
        )
    out = rel.copy()
    out[:, 0:3] = rel[:, 0:3] + forecasts
    return out


def _state_at(episode: dict[str, Any], i: int) -> ObjectState:
    """Build ObjectState from logged oracle arrays at frame i."""
    pos = np.asarray(episode["object_pos"][i], dtype=np.float64)
    vel = np.asarray(episode["object_vel"][i], dtype=np.float64)
    ts = float(np.asarray(episode["timestamp"][i]))
    return oracle_from_gt(pos, vel, ts, valid=True)


def relabel_frame(
    episode: dict[str, Any],
    frame_i: int,
    *,
    delta_s: float,
    dt: float = DEFAULT_DT_S,
    chunk: int = DEFAULT_CHUNK,
) -> dict[str, Any]:
    """Relabel one demo frame with a chosen delay Δ.

    Returns dict with ``delta_s``, ``forecasts`` (chunk, 3), ``action_abs``
    (chunk, 8), ``action_rel`` (chunk, 8). Raises if not enough future frames.
    """
    T = len(episode["timestamp"])
    # First motion lands at t+Δ → frame index offset ≈ round(Δ/dt).
    offset = int(round(float(delta_s) / float(dt)))
    start = int(frame_i) + offset
    if start + chunk > T:
        raise ValueError(
            f"need {chunk} actions from index {start}, episode length {T}"
        )
    state = _state_at(episode, frame_i)
    forecasts = forecast_horizon(state, float(delta_s), dt=dt, n=chunk)
    # Expert absolute targets: logged action_pos + action_quat + gripper.
    pos = np.asarray(episode["action_pos"][start : start + chunk], dtype=np.float64)
    quat = np.asarray(episode["action_quat"][start : start + chunk], dtype=np.float64)
    grip = np.asarray(episode["gripper"][start : start + chunk], dtype=np.float64).reshape(
        chunk, 1
    )
    action_abs = np.concatenate([pos, quat, grip], axis=1)
    action_rel = actions_to_forecast_rel(action_abs, forecasts)
    return {
        "delta_s": float(delta_s),
        "forecasts": forecasts,
        "action_abs": action_abs,
        "action_rel": action_rel,
        "frame_i": int(frame_i),
        "action_start": int(start),
    }


def relabel_frame_random_delta(
    episode: dict[str, Any],
    frame_i: int,
    rng: np.random.Generator,
    *,
    delta_min_s: float = 0.0,
    delta_max_s: float = 0.200,
    dt: float = DEFAULT_DT_S,
    chunk: int = DEFAULT_CHUNK,
) -> dict[str, Any]:
    """Same as ``relabel_frame`` but Δ drawn uniformly in [min, max]."""
    delta = float(rng.uniform(float(delta_min_s), float(delta_max_s)))
    return relabel_frame(episode, frame_i, delta_s=delta, dt=dt, chunk=chunk)
