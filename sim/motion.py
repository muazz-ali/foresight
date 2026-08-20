"""Scripted on-table object motion + lead aiming.

Kinematic on-table motion keeps speed pinned for Gate G0.
Velocity lead is clamped so PhysX contact jitter cannot corrupt grasp aim.
"""

from __future__ import annotations

import numpy as np


def clamp_lead_velocity(
    velocity: np.ndarray,
    *,
    v_static: float,
    lead_max: float,
    lookahead_s: float,
    near_contact: bool = False,
) -> np.ndarray:
    """Return lead offset ``clamp(v) * lookahead`` (meters).

    Pass yaml ``state_machine_params`` (no hidden defaults).
    - ``‖v‖ < v_static`` or ``near_contact`` → zero lead (static / contact gate).
    - else clamp lead vector length to ``lead_max``.
    """
    v = np.asarray(velocity, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(v))
    if near_contact or speed < float(v_static):
        return np.zeros(3, dtype=np.float64)
    lead = v * float(lookahead_s)
    lead_norm = float(np.linalg.norm(lead))
    if lead_norm > float(lead_max) and lead_norm > 1e-9:
        lead = lead * (float(lead_max) / lead_norm)
    return lead


def aim_position(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    hover_z: float,
    lookahead_s: float,
    v_static: float,
    lead_max: float,
    near_contact: bool = False,
    z_mode: str = "hover",
) -> np.ndarray:
    """World aim point for approach (hover) or descend (object height).

    Pass yaml ``state_machine_params`` (no hidden defaults).
    """
    p = np.asarray(position, dtype=np.float64).reshape(3).copy()
    lead = clamp_lead_velocity(
        velocity,
        v_static=v_static,
        lead_max=lead_max,
        lookahead_s=lookahead_s,
        near_contact=near_contact,
    )
    aim = p + lead
    if z_mode == "hover":
        aim[2] = p[2] + float(hover_z)
    else:
        aim[2] = p[2]
    return aim


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
