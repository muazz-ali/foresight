"""Forecast helpers for the main-claim pack (Steps 3–5).

Plain words: given object place + speed now, and a measured delay Δ, say where
the object will be when each motion in a chunk lands.

  p̂_k = p + v · (Δ + k · dt)     for k = 0 … n-1   (k is the chunk index)

The 5-number pack uses p̂_0 (XYZ) + v (XY). p̂z is grasp height (usually latched; vz = 0). 
Delay is baked into XY p̂ — not a separate number.
"""

from __future__ import annotations

import numpy as np

from interfaces.state import ObjectState, conditioning_vector

# Claim default control step (30 Hz). Callers may pass their cfg dt instead.
DEFAULT_DT_S = 1.0 / 30.0
DEFAULT_CHUNK = 25  # SmolVLA claim chunk_size


def forecast_horizon(
    state: ObjectState,
    delta: float,
    *,
    dt: float = DEFAULT_DT_S,
    n: int = DEFAULT_CHUNK,
    p_hat_z: float | None = None,
) -> np.ndarray:
    """World XYZ of the object/grasp at each chunk slot: shape (n, 3).

    Slot k lands at time ``timestamp + Δ + k·dt`` under constant velocity.
    XY coasts with v; Z stays at ``p_hat_z`` (or ``state.position[2]``) because
    vz = 0. Relative actions: ``d_k = target_k − p̂_k`` in 3D.
    """
    p = np.asarray(state.position, dtype=np.float64).reshape(3).copy()
    v = np.asarray(state.velocity, dtype=np.float64).reshape(3).copy()
    if p_hat_z is not None and np.isfinite(p_hat_z):
        p[2] = float(p_hat_z)
        v[2] = 0.0
    delta = float(delta)
    dt = float(dt)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    leads = delta + dt * np.arange(n, dtype=np.float64)
    return p[None, :] + v[None, :] * leads[:, None]


def conditioning_at_measured_delta(
    state: ObjectState,
    delta: float,
    *,
    workspace_scale: float = 1.0,
    p_hat_z: float | None = None,
) -> np.ndarray:
    """5-number pack using this cycle's measured Δ (reported delay).

    Layout: [p̂x, p̂y, p̂z, vx, vy].
    """
    return conditioning_vector(
        state,
        float(delta),
        workspace_scale=workspace_scale,
        p_hat_z=p_hat_z,
    )
