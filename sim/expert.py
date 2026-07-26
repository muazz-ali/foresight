"""Foresight four-stage pick expert on GT ObjectState (plan §4).

Stages (logged ints, Gate G0 success needs max >= PLACE):
  RESET(0) → APPROACH(1) → DESCEND(2) → SETTLE(3) → CLOSE(4)
  → LIFT(5) → PLACE(6) → DONE(7)

Fixes vs DynamicVLA PickStateMachine:
  - clamp / zero velocity lead near contact and when nearly static
  - XY+Z gates before descend and before close (no timer-only grasp)
  - continuous XY tracking during descend for moving objects
  - kinematic attach after gated close (scripted expert; avoids PhysX latch holes)
  - drop (detach) → back to APPROACH
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.state import ObjectState
from sim.motion import aim_position

STAGE_RESET = 0
STAGE_APPROACH = 1
STAGE_DESCEND = 2
STAGE_SETTLE = 3
STAGE_CLOSE = 4
STAGE_LIFT = 5
STAGE_PLACE = 6
STAGE_DONE = 7

STAGE_NAMES = {
    STAGE_RESET: "reset",
    STAGE_APPROACH: "approach_ahead",
    STAGE_DESCEND: "descend",
    STAGE_SETTLE: "settle",
    STAGE_CLOSE: "close",
    STAGE_LIFT: "lift",
    STAGE_PLACE: "place",
    STAGE_DONE: "done",
}

STAGE_APPROACH_NAME = "approach_ahead"
STAGE_GRASP_LIFT_NAME = "grasp_lift"
STAGE_PLACE_NAME = "place"
STAGE_RESET_NAME = "reset"


def stage_to_plan_name(stage: int) -> str:
    if stage <= STAGE_APPROACH:
        return STAGE_APPROACH_NAME if stage == STAGE_APPROACH else STAGE_RESET_NAME
    if stage <= STAGE_LIFT:
        return STAGE_GRASP_LIFT_NAME
    if stage <= STAGE_PLACE:
        return STAGE_PLACE_NAME
    return STAGE_PLACE_NAME


@dataclass
class ExpertCommand:
    """Cartesian EE target (world) + gripper + object-drive flags."""

    position: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)
    gripper_open: float  # 1=open, 0=closed
    stage: int
    kinematic_object: bool  # True while expert drives table motion
    attach_object: bool  # True while object is welded to EE (held)
    freeze_object: bool = False  # hold XY during settle/close


class PickExpert:
    """Scripted pick-place expert reading ground-truth ObjectState."""

    def __init__(self, cfg: dict):
        e = cfg["expert"]
        r = cfg["robot"]
        self.lookahead_s = float(e["lookahead_s"])
        self.hover_z = float(e["hover_z"])
        self.v_static = float(e["v_static"])
        self.lead_max = float(e["lead_max"])
        self.approach_xy_tol = float(e["approach_xy_tol"])
        self.approach_z_tol = float(e["approach_z_tol"])
        self.grasp_xy_tol = float(e["grasp_xy_tol"])
        self.grasp_z_tol = float(e["grasp_z_tol"])
        self.settle_steps = int(e["settle_steps"])
        self.close_steps = int(e["close_steps"])
        self.lift_height = float(e["lift_height"])
        self.drop_ee_dist = float(e["drop_ee_dist"])
        self.init_pose = np.asarray(r["init_pose"], dtype=np.float64)
        self.place_pose = np.asarray(r["place_pose"], dtype=np.float64)
        self._down_quat = self.init_pose[3:].copy()
        self.reset()

    def reset(self) -> None:
        self.stage = STAGE_RESET
        self._timer = 0
        self._lift_z: float | None = None
        self._holding = False
        self._grasp_xy: np.ndarray | None = None

    def step(
        self,
        state: ObjectState,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
    ) -> ExpertCommand:
        del ee_quat
        ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        obj = np.asarray(state.position, dtype=np.float64).reshape(3)
        vel = np.asarray(state.velocity, dtype=np.float64).reshape(3)

        if self._holding and self.stage >= STAGE_LIFT:
            if float(np.linalg.norm(ee - obj)) > self.drop_ee_dist:
                self._holding = False
                self._lift_z = None
                self._grasp_xy = None
                self.stage = STAGE_APPROACH
                self._timer = 0

        grip = 1.0
        kin = True
        attach = False
        freeze = False
        target = ee.copy()
        quat = self._down_quat.copy()

        if self.stage == STAGE_RESET:
            target = self.init_pose[:3].copy()
            quat = self.init_pose[3:].copy()
            self._timer += 1
            if self._timer >= 5:
                self.stage = STAGE_APPROACH
                self._timer = 0

        elif self.stage == STAGE_APPROACH:
            target = aim_position(
                obj,
                vel,
                hover_z=self.hover_z,
                lookahead_s=self.lookahead_s,
                v_static=self.v_static,
                lead_max=self.lead_max,
                near_contact=False,
                z_mode="hover",
            )
            hover_z = float(obj[2] + self.hover_z)
            err_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
            err_z = abs(float(ee[2] - hover_z))
            if err_xy < self.approach_xy_tol and err_z < self.approach_z_tol:
                self.stage = STAGE_DESCEND
                self._timer = 0

        elif self.stage == STAGE_DESCEND:
            target = aim_position(
                obj,
                vel,
                hover_z=self.hover_z,
                lookahead_s=self.lookahead_s,
                v_static=self.v_static,
                lead_max=self.lead_max,
                near_contact=False,
                z_mode="grasp",
            )
            err_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
            err_z = abs(float(ee[2] - target[2]))
            if err_xy < self.grasp_xy_tol and err_z < self.grasp_z_tol:
                self.stage = STAGE_SETTLE
                self._timer = 0

        elif self.stage == STAGE_SETTLE:
            # Freeze conveyor — plan §4 "stabilize" before grasp.
            freeze = True
            target = obj.copy()
            self._timer += 1
            if self._timer >= self.settle_steps:
                err_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
                err_z = abs(float(ee[2] - obj[2]))
                if err_xy < self.grasp_xy_tol and err_z < self.grasp_z_tol:
                    self.stage = STAGE_CLOSE
                    self._timer = 0
                elif self._timer > self.settle_steps * 4:
                    self.stage = STAGE_APPROACH
                    self._timer = 0

        elif self.stage == STAGE_CLOSE:
            freeze = True
            grip = 0.0
            target = obj.copy()
            self._timer += 1
            if self._timer >= self.close_steps:
                err_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
                if err_xy < self.grasp_xy_tol * 2.0:
                    self._holding = True
                    attach = True
                    freeze = False
                    self._grasp_xy = ee[:2].copy()
                    self._lift_z = float(ee[2]) + self.lift_height
                    self.stage = STAGE_LIFT
                    self._timer = 0
                else:
                    self.stage = STAGE_APPROACH
                    self._timer = 0

        elif self.stage == STAGE_LIFT:
            kin = False
            grip = 0.0
            attach = True
            if self._lift_z is None:
                self._lift_z = float(ee[2]) + self.lift_height
            target = ee.copy()
            if self._grasp_xy is not None:
                target[0] = float(self._grasp_xy[0])
                target[1] = float(self._grasp_xy[1])
            target[2] = self._lift_z
            if float(ee[2]) >= self._lift_z - 0.03 and float(obj[2]) >= self._lift_z - 0.08:
                self.stage = STAGE_PLACE
                self._timer = 0

        elif self.stage == STAGE_PLACE:
            kin = False
            grip = 0.0
            attach = True
            target = self.place_pose[:3].copy()
            quat = self.place_pose[3:].copy()
            err = float(np.linalg.norm(ee - target))
            if err < 0.04:
                self._timer += 1
                if self._timer >= 8:
                    grip = 1.0
                    attach = False
                    self._holding = False
                    self.stage = STAGE_DONE
                    self._timer = 0
            else:
                self._timer = 0

        else:  # DONE
            kin = False
            grip = 1.0
            attach = False
            target = self.place_pose[:3].copy()
            quat = self.place_pose[3:].copy()

        return ExpertCommand(
            position=target,
            quat_wxyz=quat,
            gripper_open=float(grip),
            stage=int(self.stage),
            kinematic_object=bool(kin and not attach),
            attach_object=bool(attach),
            freeze_object=bool(freeze and not attach),
        )
