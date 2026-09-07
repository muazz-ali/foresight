"""Scripted on-table object motion + bounce-aware forecast.

Kinematic on-table motion keeps speed pinned for Gate G0.
``kinematic_forecast`` rolls the same bounce rules the scene uses.
"""

from __future__ import annotations

import numpy as np


def kinematic_advance(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    *,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance XY at constant velocity; bounce at table bounds if given."""
    pos = np.asarray(position, dtype=np.float64).reshape(3).copy()
    vel = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    pos[:2] = pos[:2] + vel[:2] * float(dt)
    if xy_bounds is not None:
        (xmin, xmax), (ymin, ymax) = xy_bounds
        if pos[0] < xmin or pos[0] > xmax:
            vel[0] = -vel[0]
            pos[0] = float(np.clip(pos[0], xmin, xmax))
        if pos[1] < ymin or pos[1] > ymax:
            vel[1] = -vel[1]
            pos[1] = float(np.clip(pos[1], ymin, ymax))
    return pos, vel


def kinematic_forecast(
    position: np.ndarray,
    velocity: np.ndarray,
    horizon_s: float,
    dt: float,
    *,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll scripted motion forward ``horizon_s`` (same bounce rules as the sim)."""
    pos = np.asarray(position, dtype=np.float64).reshape(3).copy()
    vel = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    h = float(horizon_s)
    step = float(dt)
    if h <= 0.0 or step <= 0.0:
        return pos, vel
    # Exact step count; leftover fractional dt so horizon matches wall clock.
    n_full = int(h // step)
    rem = h - n_full * step
    for _ in range(n_full):
        pos, vel = kinematic_advance(pos, vel, step, xy_bounds=xy_bounds)
    if rem > 1e-12:
        pos, vel = kinematic_advance(pos, vel, rem, xy_bounds=xy_bounds)
    return pos, vel
