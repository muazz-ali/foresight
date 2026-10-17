"""Thin glue: ReflexClose on top of an expert (or policy) hand command.

Used in throw / physics-grasp episodes. Far from the object the reflex stays
off and the caller keeps driving. Near the object it steers XY at the meeting
point and can force a close.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from control.reflex import ReflexClose, ReflexOut
from interfaces.state import ObjectState


# Close (3) only in the happy path. Stage 2 (go down) is unused but kept if entered.
# Do not run during stage 1 intercept — that would fight the chase aim.
_REFLEX_STAGES = frozenset({2, 3})


def overlay_reflex(
    cmd: Any,
    reflex: ReflexClose,
    state: ObjectState,
    ee_pos: np.ndarray,
    *,
    now: float | None = None,
    ee_vel: np.ndarray | None = None,
) -> tuple[Any, ReflexOut | None]:
    """Return (possibly updated command, reflex out or None).

    ``cmd`` must look like ``sim.state_machine.ExpertCommand`` (position,
    gripper_open, attach_object, kinematic_object, stage, …). Frozen dataclasses
    are rebuilt with ``replace``; plain namespaces are mutated in place.

    Steer XY at the meeting point in close (stage 3; unused stage 2 if entered).
    Only force a close once the expert is already in CLOSE.
    """
    if int(cmd.stage) not in _REFLEX_STAGES:
        return cmd, None
    # Object already off the table — do not chase a fall.
    if float(state.position[2]) < -0.02:
        return cmd, None
    t = float(state.timestamp if now is None else now)
    out = reflex.step(state, ee_pos, t, ee_vel=ee_vel)
    if not out.active:
        return cmd, out

    pos = np.asarray(cmd.position, dtype=np.float64).reshape(3).copy()
    pos[0], pos[1] = float(out.target[0]), float(out.target[1])
    grip = float(cmd.gripper_open)
    attach = bool(cmd.attach_object)
    kin = bool(cmd.kinematic_object)
    if out.close and int(cmd.stage) == 3:
        grip = 0.0
        attach = True
        kin = False

    updates = {
        "position": pos,
        "gripper_open": grip,
        "attach_object": attach,
        "kinematic_object": kin,
    }
    if hasattr(cmd, "__dataclass_fields__"):
        from dataclasses import replace

        return replace(cmd, **updates), out
    for k, v in updates.items():
        setattr(cmd, k, v)
    return cmd, out
