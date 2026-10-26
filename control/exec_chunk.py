"""Step 5: turn a relative action chunk into timed world targets.

Plain words:
  1. Motions from a frame at t0 start at ``apply_at = t0 + applied_Δ``.
  2. Stale slots (time already past) are dropped via ``ActionBuffer``.
  3. Each slot: ``target_k = p̂_k + d_k`` (forecast + relative action).
"""

from __future__ import annotations

import numpy as np

from control.buffer import ActionBuffer
from interfaces.forecast import DEFAULT_DT_S, forecast_horizon
from interfaces.state import ObjectState
from policy.relabel import actions_from_forecast_rel


def push_forecast_chunk(
    buf: ActionBuffer,
    *,
    rel_actions: np.ndarray,
    state: ObjectState,
    delta_s: float,
    apply_at: float,
    now: float,
    dt: float = DEFAULT_DT_S,
) -> np.ndarray:
    """Push relative chunk as world targets into ``buf``. Returns absolute chunk.

    ``apply_at`` is the sim time of slot 0 (usually ``t0 + applied_Δ``).
    ``now`` drops any slot whose time has already passed.
    """
    rel = np.asarray(rel_actions, dtype=np.float64)
    forecasts = forecast_horizon(state, float(delta_s), dt=dt, n=len(rel))
    abs_chunk = actions_from_forecast_rel(rel, forecasts)
    buf.push_chunk(
        abs_chunk,
        t0=float(apply_at),
        dt=float(dt),
        now=None if now is None else float(now),
    )
    return abs_chunk
