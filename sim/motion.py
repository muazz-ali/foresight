"""Scripted on-table object motion + bounce-aware forecast.

Moves the object in a straight line and bounces it off the table edges.
Runs every sim step (scene) and when we ask "where will it be soon?".
Why: the object speed must stay exactly what we asked for (Gate G0).
Must not: touch PhysX, the robot, or any policy code. Pure math only.
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
    """Move the object one small step in a straight line (XY only).

    ``xy_bounds`` set (floor / bounce box): hitting an edge flips that speed.
    ``xy_bounds=None`` (house / throw): NO bounce, it coasts straight on and
    can leave the band. This is the one place that decides how the object
    moves, so the sim and the forecast below always agree.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3).copy()
    vel = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    pos[:2] = pos[:2] + vel[:2] * float(dt)
    if xy_bounds is not None:
        (xmin, xmax), (ymin, ymax) = xy_bounds
        # Past an edge: reverse that axis and clamp, so it never leaves the table.
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
    """Play the object's motion forward ``horizon_s`` seconds and return where it ends up.

    Uses the same bounce rules as the sim, so a forecast near an edge is right.
    Does not change the real object.
    """
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


# --- Physics substeps (pure math, so it can be tested without Isaac) ---------


def physics_substeps(cfg: dict) -> int:
    """How many physics steps to run per control step.

    House / throw mode reads ``sim.physics_substeps`` (1 if the key is missing).
    Floor mode (``scene.use_house: false``) is ALWAYS 1: old G1 / G2b numbers
    were measured with one 25 Hz physics step per control step, so that world
    must not change.
    """
    if not bool((cfg.get("scene") or {}).get("use_house", False)):
        return 1
    n = int((cfg.get("sim") or {}).get("physics_substeps", 1))
    if n < 1:
        raise ValueError(f"sim.physics_substeps must be >= 1, got {n}")
    return n


def substep_times(t0: float, dt: float, n: int) -> list[float]:
    """Sim time at the START of each of the ``n`` physics substeps of one control step.

    ``t0`` = sim time when the control step begins, ``dt`` = control step.
    Substep k starts at ``t0 + k * dt / n``; the control step ends at ``t0 + dt``.
    """
    h = float(dt) / int(n)
    return [float(t0) + k * h for k in range(int(n))]
