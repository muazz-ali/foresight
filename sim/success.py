"""Success check + failure labels (plan §8).

Success = done stage reached + object lifted + (held high at end OR placed on container).
We check stage / lift height *before* calling something “off table.”

Runs after an episode ends, on the logged arrays. Never during the sim loop.
Two label sets: expert runs use stages; policy runs only know "holding" 0/1.
Must not: guess failures from one loose height check (it hides the real cause).
"""

from __future__ import annotations

import numpy as np

from sim.state_machine import (
    STAGE_DONE,
    STAGE_HOME,
    STAGE_OPEN,
    STAGE_SCHEMA,
    StageSchema,
)

# Expert-run labels. Use these only when real stage numbers are logged.
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

# Policy-run labels. A policy has no stages, only "holding" 0/1 and height,
# so the expert labels above (early-close, grasp-slip...) do not apply.
# The plan calls the first two never-glued and glued-no-lift — same meaning.
POLICY_FAILURE_LABELS = (
    "never-held",
    "held-no-lift",
    "lifted-no-place",
    "success",
)


def _stages(episode: dict | None) -> np.ndarray:
    """Stage ids per step as an int array (empty if no episode)."""
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
    z_placed_max_above_container: float = 0.05,
) -> bool:
    """Did the expert win? Yes if it reached the last stage, lifted the object,
    and either kept it up at the end or left it on the container.

    "On the container" means near in XY and low enough to be inside/on it —
    not hovering at carry height after a stalled lower skipped the open stage.

    This is OUR rule (logged), not any outside "picked" flag.
    """
    if episode is None:
        return False
    stages = _stages(episode)
    pos = np.asarray(episode.get("object_pos", np.zeros((0, 3))))
    if len(stages) == 0 or len(pos) == 0:
        return False
    z = pos[:, 2]
    lifted = float(z.max()) >= float(z_lift_min)
    finished = int(stages.max()) >= int(place_stage)
    if not (finished and lifted):
        return False
    # Average the last 5 steps so one noisy frame cannot decide "held high".
    held_at_end = (
        float(z[-5:].mean()) >= float(z_end_min)
        if len(z) >= 5
        else float(z[-1]) >= float(z_end_min)
    )
    # Resolve container pose (caller XY and/or logged per-step pose).
    cpos = None
    if "container_pos" in episode:
        raw = np.asarray(episode["container_pos"])
        if len(raw) and np.isfinite(raw[-1, 0]):
            cpos = raw[-1]
    if container_xy is not None:
        cxy = np.asarray(container_xy, dtype=np.float64).reshape(2)
    elif cpos is not None:
        cxy = cpos[:2]
    else:
        cxy = None

    # Over the bowl: must sit low (in/on it). A stalled lower that freezes the
    # object mid-air still has high z_end — that must NOT count as success via
    # the "held high" branch.
    if cxy is not None:
        near = float(np.linalg.norm(pos[-1, :2] - cxy)) <= float(container_xy_tol)
        if near:
            above_table = float(z[-1]) >= float(z_placed_min)
            if cpos is not None:
                low_enough = float(z[-1]) <= float(cpos[2]) + float(
                    z_placed_max_above_container
                )
            else:
                # No container height logged: reject carry-height hover via z_end_min.
                low_enough = float(z[-1]) < float(z_end_min)
            return bool(above_table and low_enough)
    return bool(held_at_end)


def classify_failure(
    *,
    success: bool,
    max_stage: int,
    object_z_min: float,
    object_z_max: float,
    object_z_end: float,
    table_z: float = 0.0,
    off_table_margin: float = 0.04,
    z_lift_min: float = 0.12,
    schema: StageSchema | None = None,
    off_table: bool = False,
    abort_reason: str | None = None,
) -> str:
    """Give one reason an expert run failed (or "success").

    Physics grasp (no weld): stages say how far the expert got, height says if
    anything was really lifted. ``early-close`` = fingers shut but the object
    never rose (missed / bumped away); ``grasp-slip`` = it rose (or the run
    ended low) but never made it to the box; ``too-late`` = never got to lift.
    ``off_table`` = left the table footprint (even mid-chase). ``abort_reason``
    (ee off table / object through box or base) → ``safety-abort``.

    Order: success → safety-abort → never-engaged → off-table → too-late →
    grasp-slip → early-close → timeout.
    Pass ``z_lift_min=expert.lift_height`` when the expert is available.
    ``schema`` maps stage ids to meaning; pass ``expert.stage_schema`` so
    numbering stays correct if stages change.
    """
    sch = schema if schema is not None else STAGE_SCHEMA
    if success:
        return "success"
    if abort_reason:
        return "safety-abort"
    # Stuck at the first stage: the hand never got to the object.
    if max_stage <= sch.engaged:
        # Object already gone counts as off-table, not "never tried".
        if off_table or object_z_min < (table_z - off_table_margin):
            return "off-table"
        return "never-engaged"
    # Fell off while chasing / before lift — name the fall, not "too-late".
    if off_table or object_z_min < (table_z - off_table_margin):
        if max_stage < sch.lift:
            return "off-table"
    # Got close but never reached the lift stage: too slow to catch it.
    if max_stage < sch.lift:
        return "too-late"
    # Reached lift, never placed. Height tells us if it ever left the table.
    if max_stage < sch.placed:
        if object_z_max < float(z_lift_min):
            return "early-close"
        return "grasp-slip"
    # Only call it off-table if the scene says it left the table, or it truly dropped below it.
    if off_table or object_z_min < (table_z - off_table_margin):
        return "off-table"
    if max_stage < sch.done:
        return "timeout"
    # Finished all stages but the object ended low: it slipped out.
    if object_z_end < float(z_lift_min):
        return "grasp-slip"
    return "timeout"


def classify_policy_failure(
    *,
    success: bool,
    holding: np.ndarray | None,
    object_z_max: float,
    z_lift_min: float = 0.12,
) -> str:
    """Give one reason a policy run failed (or "success").

    Uses "holding" 0/1 and peak height only. No expert stages here.
    """
    if success:
        return "success"
    # Did the object ever count as held? If not, the policy never grabbed it.
    held = False
    if holding is not None and len(np.asarray(holding)):
        held = bool(np.any(np.asarray(holding) > 0.5))
    if not held:
        return "never-held"
    # Held but never rose to lift height.
    if float(object_z_max) < float(z_lift_min):
        return "held-no-lift"
    return "lifted-no-place"


def episode_diagnostics(
    episode: dict | None,
    *,
    z_lift_min: float = 0.12,
) -> dict:
    """Pull the numbers we log next to every result: max stage, min/max/end height.

    They let a human check why a label was chosen (no blind "off-table").

    ``z_lift_min`` is echoed for gate logs; pass ``expert.lift_height`` when available.
    """
    if episode is None:
        return {
            "max_stage": 0,
            "z_min": 0.0,
            "z_max": 0.0,
            "z_end": 0.0,
            "z_lift_min": float(z_lift_min),
        }
    stages = _stages(episode)
    pos = np.asarray(episode.get("object_pos", [[0.0, 0.0, 0.0]]))
    z = pos[:, 2] if len(pos) else np.array([0.0])
    return {
        "max_stage": int(stages.max()) if len(stages) else 0,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_end": float(z[-1]),
        "z_lift_min": float(z_lift_min),
    }


def release_disturbed(stage, obj_pos, ee_pos, steps: int = 4) -> bool:
    """True if the hand pulled or lifted the placed object as go-home began.

    First ``steps`` frames of go-home: the object moved > 1 cm in XY along the
    hand's direction, or rose > 1 cm. (g0_object_smoke: 34% of successes did
    this before the straight-up retreat in ``_home_target``.)
    """
    stage = np.asarray(stage)
    obj = np.asarray(obj_pos, dtype=np.float64)
    ee = np.asarray(ee_pos, dtype=np.float64)
    idx = np.where(stage == STAGE_HOME)[0]
    if len(idx) == 0:
        return False
    a = int(idx[0])
    b = min(a + steps, len(stage) - 1)
    d_obj = obj[b, :2] - obj[a, :2]
    d_ee = ee[b, :2] - ee[a, :2]
    rise = float(obj[a : b + 1, 2].max() - obj[a, 2])
    n_obj, n_ee = float(np.linalg.norm(d_obj)), float(np.linalg.norm(d_ee))
    # The hand must itself move sideways (> 5 mm) for "along" to mean a drag;
    # otherwise a fruit rolling inside the box under a still hand looks dragged.
    along = n_obj > 0.01 and n_ee > 0.005 and float(d_obj @ d_ee) / (n_obj * n_ee) > 0.7
    return bool(along or rise > 0.01)


def clean_reject_reason(episode: dict, retries: int | None) -> str | None:
    """Why a successful demo is not clean (None = clean).

    Clean = first-try grasp (no slip + re-chase), a real open stage, and the
    hand leaves without dragging the object. Used at collect time (target
    counts clean wins) and at convert time (``--clean-only``).
    """
    if int(retries or 0) > 0:
        return "retried"
    stage = np.asarray(episode["stage"])
    if STAGE_OPEN not in stage:
        return "no-open-stage"
    if release_disturbed(stage, episode["object_pos"], episode["ee_pos"]):
        return "release-drag"
    return None
