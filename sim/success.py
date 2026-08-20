"""Success check + failure labels (plan §8).

Success = done stage reached + object lifted + (held high at end OR placed on container).
We check stage / lift height *before* calling something “off table.”
"""

from __future__ import annotations

import numpy as np

from sim.state_machine import (
    STAGE_APPROACH,
    STAGE_DONE,
    STAGE_HOLD_IN_BOX,
    STAGE_LIFT,
)

FAILURE_LABELS = (
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

POLICY_FAILURE_LABELS = (
    "never-held",
    "held-no-lift",
    "lifted-no-place",
    "success",
)


def _stages(episode: dict | None) -> np.ndarray:
    if episode is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray(episode.get("stage", []), dtype=np.int64)


def episode_success(
    episode: dict | None,
    *,
    z_lift_min: float = 0.12,
    z_end_min: float = 0.10,
    place_stage: int = STAGE_DONE,
    container_xy: np.ndarray | None = None,
    container_xy_tol: float = 0.12,
    z_placed_min: float = 0.02,
) -> bool:
    """Plan-aligned expert success: grasp + lift + hold, or place onto container."""
    if episode is None:
        return False
    stages = _stages(episode)
    pos = np.asarray(episode.get("object_pos", np.zeros((0, 3))))
    if len(stages) == 0 or len(pos) == 0:
        return False
    z = pos[:, 2]
    lifted = float(z.max()) >= float(z_lift_min)
    finished = int(stages.max()) >= int(place_stage)
    held_at_end = (
        float(z[-5:].mean()) >= float(z_end_min)
        if len(z) >= 5
        else float(z[-1]) >= float(z_end_min)
    )
    placed_on_container = False
    if container_xy is not None and finished:
        xy = pos[-1, :2]
        cxy = np.asarray(container_xy, dtype=np.float64).reshape(2)
        near = float(np.linalg.norm(xy - cxy)) <= float(container_xy_tol)
        above_table = float(z[-1]) >= float(z_placed_min)
        placed_on_container = bool(near and above_table)
    if (
        not placed_on_container
        and container_xy is None
        and "container_pos" in episode
        and finished
    ):
        cpos = np.asarray(episode["container_pos"])
        if len(cpos) and np.isfinite(cpos[-1, 0]):
            xy = pos[-1, :2]
            near = float(np.linalg.norm(xy - cpos[-1, :2])) <= float(container_xy_tol)
            above_table = float(z[-1]) >= float(z_placed_min)
            placed_on_container = bool(near and above_table)
    return bool(finished and lifted and (held_at_end or placed_on_container))


def classify_failure(
    *,
    success: bool,
    max_stage: int,
    object_z_min: float,
    object_z_max: float,
    object_z_end: float,
    table_z: float = 0.0,
    off_table_margin: float = 0.04,
) -> str:
    """Why the expert run failed (or success).

    Order: success → never-engaged → too-late → grasp-slip → early-close →
    off-table (only if object fell below table) → timeout.
    """
    if success:
        return "success"
    if max_stage <= STAGE_APPROACH:
        return "never-engaged"
    if max_stage < STAGE_LIFT:
        return "too-late"
    if max_stage < STAGE_HOLD_IN_BOX:
        if object_z_max < 0.10:
            return "early-close"
        return "grasp-slip"
    if object_z_min < (table_z - off_table_margin):
        return "off-table"
    if max_stage < STAGE_DONE:
        return "timeout"
    if object_z_end < 0.10:
        return "grasp-slip"
    return "timeout"


def classify_policy_failure(
    *,
    success: bool,
    holding: np.ndarray | None,
    object_z_max: float,
    z_lift_min: float = 0.12,
) -> str:
    """Why a policy run failed. Uses holding (0/1), not expert stages."""
    if success:
        return "success"
    held = False
    if holding is not None and len(np.asarray(holding)):
        held = bool(np.any(np.asarray(holding) > 0.5))
    if not held:
        return "never-held"
    if float(object_z_max) < float(z_lift_min):
        return "held-no-lift"
    return "lifted-no-place"


def episode_diagnostics(episode: dict | None) -> dict:
    """Extract max_stage / z_max / z_end for logging."""
    if episode is None:
        return {
            "max_stage": 0,
            "z_min": 0.0,
            "z_max": 0.0,
            "z_end": 0.0,
        }
    stages = _stages(episode)
    pos = np.asarray(episode.get("object_pos", [[0.0, 0.0, 0.0]]))
    z = pos[:, 2] if len(pos) else np.array([0.0])
    return {
        "max_stage": int(stages.max()) if len(stages) else 0,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_end": float(z[-1]),
    }
