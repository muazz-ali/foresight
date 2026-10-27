"""B1 success + F1–F7 failure modes (expert only).

Success (all must hold):
  - force-place off
  - gripper closed on the object (both fingers in contact — proxy: gap shrinks
    onto the object and holding sticks for the hold window)
  - object lifted ≥ 5 cm above the table
  - held ≥ 1.0 s after lift without slipping > 10 mm in the hand frame

Failure modes: first match wins (F1…F7).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.state_machine import STAGE_CLOSE, STAGE_LIFT

# B1 latch numbers (do not loosen without a new gate version).
LIFT_MIN_M = 0.05
HOLD_MIN_S = 1.0
SLIP_MAX_M = 0.010
CLOSE_XY_EMPTY_M = 0.005
KNOCK_M = 0.010
CLOSED_ABOVE_SZ_M = 0.005  # s_z default; caller may override


FAILURE_MODES = (
    "unreachable",
    "never_closed",
    "closed_above",
    "closed_empty_xy",
    "knocked",
    "slipped",
    "other",
)


def _arr(ep: dict, key: str) -> np.ndarray:
    return np.asarray(ep.get(key, []), dtype=np.float64)


def b1_success(
    episode: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
    *,
    z_table: float = 0.0,
    force_place: bool | None = None,
    dt: float = 1.0 / 30.0,
    s_z_m: float = CLOSED_ABOVE_SZ_M,
) -> tuple[bool, str | None]:
    """Return (ok, failure_mode). failure_mode is None on success.

    May read live logged GT (object/ee) — this is the success judge, so leak
    audit allows it. Does not change G0 ``episode_success``.
    """
    meta = meta or {}
    if force_place is None:
        force_place = bool(meta.get("force_place", False))
    if force_place:
        return False, "other"

    if episode is None:
        return False, "other"

    stages = np.asarray(episode.get("stage", []), dtype=np.int64)
    obj = _arr(episode, "object_pos")
    ee = _arr(episode, "ee_pos")
    grip = _arr(episode, "gripper").reshape(-1)
    if stages.size == 0 or obj.size == 0 or ee.size == 0:
        return False, "other"

    n = min(len(stages), len(obj), len(ee))
    stages, obj, ee = stages[:n], obj[:n], ee[:n]
    if grip.size:
        grip = grip[:n]

    # First CLOSE frame. Not ``>= STAGE_CLOSE``: HOME (8) / DONE (9) are larger,
    # so an episode that never closed and went home was scored at its first
    # go-home frame as closed_above / closed_empty_xy.
    closed = np.where(stages == STAGE_CLOSE)[0]
    if closed.size == 0:
        # Never entered CLOSE.
        if bool(meta.get("unreachable", False)):
            return False, "unreachable"
        # Aim went outside reach / speed → F1 when tagged; else never_closed.
        return False, "never_closed"

    c0 = int(closed[0])
    # Knocked: object left its coast path by > 10 mm while the hand was near.
    # (Raw per-frame travel is ~v·dt — at 30 cm/s that is already ~10 mm/frame,
    # so we compare to the constant-velocity prediction, not to absolute travel.)
    if c0 > 2 and "object_vel" in episode:
        vel = _arr(episode, "object_vel")
        n_v = min(c0, len(vel))
        for i in range(1, n_v):
            hand_near = float(np.linalg.norm(ee[i, :2] - obj[i, :2])) < 0.03
            if not hand_near:
                continue
            pred = obj[i - 1, :2] + vel[i - 1, :2] * float(dt)
            err = float(np.linalg.norm(obj[i, :2] - pred))
            if err > float(KNOCK_M) + 1e-9:
                return False, "knocked"

    xy_err = float(np.linalg.norm(ee[c0, :2] - obj[c0, :2]))
    # Height vs object centre at close. ``ee_pos`` is the pinch frame
    # (tcp_offset Z = 103.4 mm pad midpoint), so z_err > 0 means the pinch is
    # above the centre (closed_above). Not meta["z_grasp"]: that is the expert's
    # value at episode END (object already in the box).
    z_err = float(ee[c0, 2] - obj[c0, 2])

    if abs(z_err) > float(s_z_m) + 1e-9 and z_err > 0:
        return False, "closed_above"
    if xy_err > float(CLOSE_XY_EMPTY_M) + 1e-9:
        return False, "closed_empty_xy"

    # Lift ≥ 5 cm above table.
    z = obj[:, 2]
    z_max = float(np.max(z))
    if z_max < float(z_table) + float(LIFT_MIN_M) - 1e-9:
        # Closed but never lifted → slipped or other.
        if int(np.max(stages)) >= STAGE_LIFT:
            return False, "slipped"
        return False, "slipped"

    # Hold ≥ 1 s after first lift frame, slip ≤ 10 mm in hand frame.
    lift_idx = np.where(z >= float(z_table) + float(LIFT_MIN_M))[0]
    if lift_idx.size == 0:
        return False, "slipped"
    i0 = int(lift_idx[0])
    n_hold = max(1, int(round(float(HOLD_MIN_S) / max(float(dt), 1e-6))))
    if i0 + n_hold > n:
        return False, "slipped"
    # Hand-frame slip: object − ee relative to the lift-start offset.
    rel0 = obj[i0] - ee[i0]
    slipped = False
    for i in range(i0, min(n, i0 + n_hold)):
        rel = obj[i] - ee[i]
        if float(np.linalg.norm(rel - rel0)) > float(SLIP_MAX_M) + 1e-9:
            slipped = True
            break
        # Dropped below lift height during hold.
        if float(obj[i, 2]) < float(z_table) + float(LIFT_MIN_M) - 1e-9:
            slipped = True
            break
    if slipped:
        return False, "slipped"

    # Both-finger contact proxy: gripper commanded closed and object stayed near hand.
    if grip.size and float(grip[c0]) > 0.25:
        return False, "never_closed"

    return True, None


def classify_b1(
    episode: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score one episode for the B1 JSONL row extras."""
    ok, mode = b1_success(episode, meta, **kwargs)
    return {
        "success": bool(ok),
        "failure_mode": None if ok else (mode or "other"),
    }
