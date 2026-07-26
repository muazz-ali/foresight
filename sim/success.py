"""Foresight success criterion + failure taxonomy (plan §8).

Success = place stage reached + object lifted + held at end.
Taxonomy checks stage / lift diagnostics *before* off-table.
"""

from __future__ import annotations

import numpy as np

from sim.expert import STAGE_APPROACH, STAGE_DONE, STAGE_LIFT, STAGE_PLACE

FAILURE_TAXONOMY = (
    "never-engaged",
    "too-late",
    "early-close",
    "wrong-object",
    "grasp-slip",
    "off-table",
    "safety-abort",
    "timeout",
    "success",
)


def episode_success(
    episode: dict | None,
    *,
    z_lift_min: float = 0.12,
    z_end_min: float = 0.10,
    place_stage: int = STAGE_PLACE,
) -> bool:
    """Plan-aligned expert success: grasp completed + object lifted + held."""
    if episode is None:
        return False
    sm = np.asarray(episode.get("sm_state", []))
    pos = np.asarray(episode.get("object_pos", np.zeros((0, 3))))
    if len(sm) == 0 or len(pos) == 0:
        return False
    z = pos[:, 2]
    lifted = float(z.max()) >= float(z_lift_min)
    finished = int(sm.max()) >= int(place_stage)
    held_at_end = (
        float(z[-5:].mean()) >= float(z_end_min)
        if len(z) >= 5
        else float(z[-1]) >= float(z_end_min)
    )
    return bool(finished and lifted and held_at_end)


def classify_failure(
    *,
    success: bool,
    max_sm_state: int,
    object_z_min: float,
    object_z_max: float,
    object_z_end: float,
    table_z: float = 0.0,
    off_table_margin: float = 0.04,
) -> str:
    """Coarse taxonomy for Gate diagnostics.

    Order: success → never-engaged → too-late → grasp-slip → early-close →
    off-table (only if object fell below table) → timeout.
    """
    if success:
        return "success"
    if max_sm_state <= STAGE_APPROACH:
        return "never-engaged"
    if max_sm_state < STAGE_LIFT:
        return "too-late"
    if max_sm_state < STAGE_PLACE:
        # Reached lift but never place — slip or early drop.
        if object_z_max < 0.10:
            return "early-close"
        return "grasp-slip"
    if object_z_min < (table_z - off_table_margin):
        return "off-table"
    if max_sm_state < STAGE_DONE:
        return "timeout"
    # Reached DONE stage ints but success criterion failed (drop at end).
    if object_z_end < 0.10:
        return "grasp-slip"
    return "timeout"


def episode_diagnostics(episode: dict | None) -> dict:
    """Extract max_sm / z_max / z_end for logging."""
    if episode is None:
        return {
            "max_sm": 0,
            "z_min": 0.0,
            "z_max": 0.0,
            "z_end": 0.0,
        }
    sm = np.asarray(episode.get("sm_state", [0]))
    pos = np.asarray(episode.get("object_pos", [[0.0, 0.0, 0.0]]))
    z = pos[:, 2] if len(pos) else np.array([0.0])
    return {
        "max_sm": int(sm.max()) if len(sm) else 0,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_end": float(z[-1]),
    }
