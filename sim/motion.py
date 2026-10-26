"""Scripted on-table object motion + coast forecast.

Moves the object in a straight line. A miss coasts off the table.
Runs every sim step (scene) and when we ask "where will it be soon?".
Why: the object speed must stay exactly what we asked for (Gate G0 / G3).
Must not: touch PhysX, the robot, or any policy code. Pure math only.
"""

from __future__ import annotations

import numpy as np


def kinematic_advance(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the object one small step in a straight line (XY only).

    Coasts straight; can leave the play band / table.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3).copy()
    vel = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    pos[:2] = pos[:2] + vel[:2] * float(dt)
    return pos, vel


def kinematic_forecast(
    position: np.ndarray,
    velocity: np.ndarray,
    horizon_s: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Play the object's motion forward ``horizon_s`` seconds and return where it ends up.

    Same coast rules as the sim. Does not change the real object.
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
        pos, vel = kinematic_advance(pos, vel, step)
    if rem > 1e-12:
        pos, vel = kinematic_advance(pos, vel, rem)
    return pos, vel


# --- Physics substeps (pure math, so it can be tested without Isaac) ---------


def physics_substeps(cfg: dict) -> int:
    """How many physics steps to run per control step.

    Reads ``sim.physics_substeps`` (1 if the key is missing).
    """
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
