"""Foresight expert — intercept-servo pick-place.
intercept: predict where the object will be and move hand there
servo: move the hand to that predicted position

Stages: 0 start → 1 intercept → 2 go down → 3 close → 4 lift → 5 carry
→ 6 lower → 7 open → 8 home → 9 done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from interfaces.state import ObjectState
from sim.motion import kinematic_forecast
from sim.state_machine_logs import STAGE_PLAIN, STAGE_WHY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSchema:
    """Which stage id means what, so success/logging work for any state machine."""

    engaged: int  # reached the object at all
    lift: int  # object left the table
    placed: int  # object is at the drop point
    done: int  # episode finished cleanly
    names: dict  # stage id -> spoken name
    why: dict  # stage id -> one-line explanation


STAGE_RESET = 0
STAGE_APPROACH = 1
STAGE_DESCEND = 2
STAGE_CLOSE = 3
STAGE_LIFT = 4
STAGE_CARRY = 5
STAGE_LOWER = 6
STAGE_OPEN = 7
STAGE_HOME = 8
STAGE_DONE = 9

STAGE_SCHEMA = StageSchema(
    engaged=STAGE_APPROACH,
    lift=STAGE_LIFT,
    placed=STAGE_OPEN,
    done=STAGE_DONE,
    names=STAGE_PLAIN,
    why=STAGE_WHY,
)


def _require(cfg: dict, *keys: str) -> Any:
    """Nested yaml lookup. Logs the missing path and raises KeyError."""
    node: Any = cfg
    path: list[str] = []
    for k in keys:
        path.append(str(k))
        try:
            node = node[k]
        except (KeyError, TypeError):
            logger.error("missing required yaml key: %s", ".".join(path))
            raise KeyError(".".join(path)) from None
    return node


def sim_dt(cfg: dict) -> float:
    """Sim step size (s) from required ``sim.dt``."""
    return float(_require(cfg, "sim", "dt"))


def inference_window_steps(cfg: dict, dt: float | None = None) -> int:
    """Max episode length in sim steps from ``state_machine_params.inference_window_s``."""
    step_dt = float(dt if dt is not None else sim_dt(cfg))
    window_s = float(_require(cfg, "state_machine_params", "inference_window_s"))
    return max(1, int(round(window_s / step_dt)))


@dataclass
class ExpertCommand:
    """Cartesian EE target (world) + gripper + object-drive flags."""

    position: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)
    gripper_open: float  # 1=open, 0=closed
    stage: int
    kinematic_object: bool  # True while we slide the object on the table
    attach_object: bool  # True while the object is held (snapped to the hand)
    freeze_object: bool = False  # pause object pose (open / home / done)

class PickPlaceStateMachine:
    """Intercept-servo pick-place expert reading ground-truth ObjectState."""

    stage_schema = STAGE_SCHEMA
    stage_done = STAGE_DONE

    def __init__(self, cfg: dict, *, dt: float | None = None):
        """Load ``state_machine_params`` + ``robot.init_pose``. ``dt`` overrides ``sim.dt``."""
        self._p = dict(_require(cfg, "state_machine_params"))
        robot = _require(cfg, "robot")
        scene = cfg.get("scene") or {}

        self.dt = float(dt if dt is not None else sim_dt(cfg))

        # --- timings -----------------------------------------------------
        self.settle_s = self._f("settle_s")
        self.close_gripper_s = self._f("close_gripper_s")
        self.inference_window_s = self._f("inference_window_s")
        self.stage_timeout_s = self._f("stage_timeout_s")
        self.grasp_stage_timeout_s = self._f("grasp_stage_timeout_s", self.stage_timeout_s * 3.0)
        self.track_hold_s = self._f("track_hold_s")
        self.bounce_guard_s = self._f("bounce_guard_s")

        # --- interception ------------------------------------------------
        self.lookahead_s = self._f("lookahead_s")  # (plan: 0.25 s)
        self.conditioning_delta_s = self._f("conditioning_delta_s", self.lookahead_s)
        self.servo_lag_s = self._f("servo_lag_s")  # calibrate; see tools/diag/calibrate_servo_lag.py
        self.ee_speed_max = self._f("ee_speed_max")
        self.intercept_iters = int(self._p.get("intercept_iters", 3))

        # --- tolerances ---------------------------------------------------
        self.servo_xy_tol = self._f("servo_xy_tol")
        self.approach_z_tol = self._f("approach_z_tol")
        self.track_xy_tol = self._f("track_xy_tol")
        self.grasp_xy_tol = self._f("grasp_xy_tol")
        self.grasp_z_tol = self._f("grasp_z_tol")
        self.grasp_tol_step_k = self._f("grasp_tol_step_k")
        self.place_xy_tol = self._f("place_xy_tol")
        self.home_z_tol = self._f("home_z_tol")
        self.ee_settle_tol = self._f("ee_settle_tol")

        # --- heights ------------------------------------------------------
        self.hover_z = self._f("hover_z")
        self.z_release_radius = self._f("z_release_radius")
        self.lift_height = self._f("lift_height")
        self.drop_z_gap = self._f("drop_z_gap")

        # --- retries ------------------------------------------------------
        self.max_retries = int(self._p.get("max_retries", 3))

        self.init_pose = np.asarray(robot["init_pose"], dtype=np.float64)
        self._down_quat = self.init_pose[3:].copy()

        bounds = scene.get("object_xy_bounds")
        self.xy_bounds = (
            ((float(bounds[0][0]), float(bounds[0][1])), (float(bounds[1][0]), float(bounds[1][1])))
            if bounds is not None
            else None
        )

        self.carry_pose = np.concatenate([np.zeros(3), self._down_quat])
        self._drop_pose: np.ndarray | None = None
        self.reset()

    # ------------------------------------------------------------------ cfg
    def _f(self, key: str, default: float | None = None) -> float:
        """Required float from the merged param block; no silent zeros."""
        if key in self._p:
            return float(self._p[key])
        if default is None:
            logger.error("missing required yaml key: state_machine_params.%s", key)
            raise KeyError(f"state_machine_params.{key}")
        return float(default)

    @property
    def tau(self) -> float:
        """Effective command-to-motion lag (servo lag + one zero-order-hold step)."""
        return self.servo_lag_s + self.dt

    # --------------------------------------------------------------- public
    def set_box_poses(
        self,
        carry_pos: np.ndarray,
        quat_wxyz: np.ndarray | None = None,
        *,
        drop_pos: np.ndarray | None = None,
    ) -> None:
        """Carry pose (box XY + clear Z); ``drop_pos`` is inside the box."""
        p = np.asarray(carry_pos, dtype=np.float64).reshape(3)
        q = self._down_quat.copy() if quat_wxyz is None else np.asarray(
            quat_wxyz, dtype=np.float64
        ).reshape(4)
        self.carry_pose = np.concatenate([p, q])
        if drop_pos is None:
            self._drop_pose = p.copy()
        else:
            d = np.asarray(drop_pos, dtype=np.float64).reshape(3)
            d[0], d[1] = p[0], p[1]
            self._drop_pose = d

    def reset(self) -> None:
        self.stage = STAGE_RESET
        self.timer_s = 0.0
        self.lock_s = 0.0
        self.retries = 0
        self._blocked_s = 0.0
        self._z_frac = 1.0
        self.lift_z: float | None = None
        self.grasp_xy: np.ndarray | None = None
        # Grasp quality, surfaced in meta so a loosened tolerance cannot hide.
        self.grasp_err_xy: float | None = None
        self.grasp_err_z: float | None = None
        self.grasp_speed: float | None = None

    def grasp_report(self) -> dict:
        """Per-episode grasp quality. ``None`` until CLOSE fires."""
        return {
            "grasp_err_xy": self.grasp_err_xy,
            "grasp_err_z": self.grasp_err_z,
            "grasp_speed_m_s": self.grasp_speed,
            "retries": int(self.retries),
        }

    # -------------------------------------------------------------- helpers
    def _forecast(self, pos: np.ndarray, vel: np.ndarray, horizon_s: float) -> np.ndarray:
        """Where the object will be in ``horizon_s`` — with wall bounces.

        Uses the same ``kinematic_forecast`` the scene integrates with, so the
        prediction and the world cannot disagree.
        """
        if horizon_s <= 0.0:
            return np.asarray(pos, dtype=np.float64).reshape(3).copy()
        p, _ = kinematic_forecast(pos, vel, float(horizon_s), self.dt, xy_bounds=self.xy_bounds)
        return p

    def _intercept_lead_s(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        vel: np.ndarray,
        goal_z: float | None = None,
    ) -> float:
        """Lookahead horizon: hand travel time, fixed-point solved, capped at ``lookahead_s``.

        Travel time is measured to where the object *will be*, without ``tau``:

        ``tau`` is a command offset, not distance the hand has to cover — in
        steady state the hand sits on the object while the command runs ahead.
        Folding ``tau`` into the distance would leave a floor of ``v * tau`` and
        the horizon would never collapse.

        Far from the object the 0.25 s cap binds → aim a full look-ahead ahead.
        On top of the object travel time → 0 → the lead collapses to ``tau`` and
        the hand rides the object instead of running ahead of it.
        """
        gz = float(obj[2] + self.hover_z if goal_z is None else goal_z)
        t_travel = 0.0
        for _ in range(max(1, self.intercept_iters)):
            aim = self._forecast(obj, vel, min(t_travel, self.lookahead_s))
            goal = np.array([aim[0], aim[1], gz], dtype=np.float64)
            t_travel = float(np.linalg.norm(ee - goal)) / max(self.ee_speed_max, 1e-6)
        return min(t_travel, self.lookahead_s) + self.tau

    def _grasp_tol_xy(self, vel: np.ndarray) -> float:
        """Grasp gate cannot be tighter than one control step of object travel."""
        step = self.grasp_tol_step_k * float(np.linalg.norm(vel[:2])) * self.dt
        return max(self.grasp_xy_tol, step)

    def _time_to_wall_s(self, pos: np.ndarray, vel: np.ndarray) -> float:
        """Seconds until the object reverses off a play-band wall (inf if never)."""
        if self.xy_bounds is None:
            return float("inf")
        best = float("inf")
        for i, (lo, hi) in enumerate(self.xy_bounds):
            v = float(vel[i])
            if abs(v) < 1e-6:
                continue
            bound = hi if v > 0.0 else lo
            t = (bound - float(pos[i])) / v
            if t >= 0.0:
                best = min(best, t)
        return best

    def _descend_time_s(self, ee: np.ndarray, obj: np.ndarray) -> float:
        """Seconds of Z still to burn before the grasp gate can pass.

        First-order settle of the remaining height to ``grasp_z_tol``. Because
        APPROACH already walks Z down with the XY lock, this is usually one or
        two steps, not a fresh ``hover_z`` descent.
        """
        dz = max(0.0, float(ee[2] - obj[2]))
        if dz <= self.grasp_z_tol:
            return self.dt
        return self.tau * float(np.log(dz / self.grasp_z_tol))

    def _grasp_window_s(self, ee: np.ndarray | None = None, obj: np.ndarray | None = None) -> float:
        """Straight-line time the grasp needs before the object reverses.

        Only the *residual descend* has to fit. Once CLOSE attaches, the object
        is welded to the hand and stops following the table, so nothing after
        attach cares about walls. ``bounce_guard_s`` is the floor.
        """
        need = self.dt if ee is None or obj is None else self._descend_time_s(ee, obj)
        return max(self.bounce_guard_s, need)

    def _bounce_clear(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        ee: np.ndarray | None = None,
    ) -> bool:
        """Enough straight-line time left to finish the descend without a reversal."""
        if float(np.linalg.norm(vel[:2])) < self.ee_settle_tol:
            return True
        return self._time_to_wall_s(pos, vel) > self._grasp_window_s(ee, pos)

    def _retract_z(self) -> float:
        """after object lift, go to drop location"""
        way_back_z = float(self.carry_pose[2])
        lift_z = float(self.lift_z) if self.lift_z is not None else way_back_z
        return max(lift_z, way_back_z)

    def _drop_target(self) -> np.ndarray:
        """drop inside the box"""
        if self._drop_pose is None:
            return self.carry_pose[:3].copy()
        p = self._drop_pose.copy()
        p[0], p[1] = float(self.carry_pose[0]), float(self.carry_pose[1])
        if self.lift_z is not None:
            p[2] = min(float(p[2]), self._retract_z() - self.approach_z_tol)
        return p

    def _at_home(self, ee: np.ndarray) -> bool:
        """at home: close enough to the home position"""
        home = self.init_pose[:3]
        return (
            float(np.linalg.norm(ee[:2] - home[:2])) < self.servo_xy_tol
            and abs(float(ee[2] - home[2])) < self.home_z_tol
        )

    def _home_target(self, ee: np.ndarray) -> np.ndarray:
        """go home: climb clear before crossing XY — straight lines from drop Z hit the table."""
        home = self.init_pose[:3].copy()
        clear_z = max(float(home[2]), float(self.carry_pose[2]), self._retract_z())
        err_xy = float(np.linalg.norm(ee[:2] - home[:2]))
        if err_xy >= self.servo_xy_tol and float(ee[2]) < clear_z - self.home_z_tol:
            t = ee.copy()
            t[2] = clear_z
            return t
        return home

    def _in_container(self, obj: np.ndarray) -> bool:
        """in container: close enough to the drop location"""
        return float(np.linalg.norm(obj[:2] - self.carry_pose[:2])) < self.place_xy_tol

    def _restart_approach(self, why: str) -> None:
        """Bounded retry: a missed grasp goes back to intercept, not straight home."""
        self.retries += 1
        self.timer_s = 0.0
        self.lock_s = 0.0
        self._z_frac = 1.0
        if self.retries > self.max_retries:
            logger.info("    %s; out of retries (%d), going home", why, self.max_retries)
            self.stage = STAGE_HOME
        else:
            logger.info("    %s; re-intercepting (retry %d/%d)", why, self.retries, self.max_retries)
            self.stage = STAGE_APPROACH

    # ------------------------------------------------------------------ core state machine logic 
    def step(
        self,
        state: ObjectState,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
    ) -> ExpertCommand:
        """Advance one control step. Returns the EE command for this tick.
        
        This is the core state machine logic that runs every control step.
        It takes the current state of the object, the end effector position and
        orientation, and returns the next end effector command.
        """
        del ee_quat

        ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        obj = np.asarray(state.position, dtype=np.float64).reshape(3)
        vel = np.asarray(state.velocity, dtype=np.float64).reshape(3)
        speed = float(np.linalg.norm(vel))
        prev_stage = self.stage

        grip, kin, attach, freeze = 1.0, True, False, False
        target = ee.copy()
        quat = self._down_quat.copy()

        self.timer_s += self.dt

        if self.stage == STAGE_RESET:
            target = self.init_pose[:3].copy()
            if self.timer_s >= self.settle_s:
                self.stage = STAGE_APPROACH
                self.timer_s = 0.0
                self.lock_s = 0.0

        elif self.stage == STAGE_APPROACH:
            # Intercept: aim where the object will be when the hand arrives.
            # Z converges *with* XY, not after it. A flat hover followed by a
            # separate descend spends ~0.3 s of straight-line time that a fast
            # object simply does not have between wall bounces; blending means
            # the hand is already on the object the moment XY locks.
            err_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
            frac = float(np.clip(err_xy / max(self.z_release_radius, 1e-6), 0.0, 1.0))
            # Ratchet: height only ever comes down within one approach attempt.
            # Without it a fast object's XY error oscillates on every wall
            # bounce, the Z target pumps up and down, and Z never settles.
            self._z_frac = min(self._z_frac, frac)
            target_z = obj[2] + self.hover_z * self._z_frac

            lead_s = self._intercept_lead_s(ee, obj, vel, goal_z=target_z)
            aim = self._forecast(obj, vel, lead_s)
            target = np.array([aim[0], aim[1], target_z], dtype=np.float64)

            # Gate in the object's frame — with lag compensation this closes.
            err_z = abs(float(ee[2] - target_z))
            locked = err_xy < self.track_xy_tol and err_z < self.approach_z_tol
            self.lock_s = self.lock_s + self.dt if locked else 0.0

            if self.lock_s >= self.track_hold_s and not self._bounce_clear(obj, vel, ee):
                self._blocked_s += self.dt
                if self._blocked_s >= 1.0:
                    self._blocked_s = 0.0
                    logger.info(
                        "    locked, but waiting for a straight run: need %.2f s, "
                        "wall in %.2f s (widen scene.object_xy_bounds, cut "
                        "z_release_radius, or cut servo_lag_s)",
                        self._grasp_window_s(ee, obj),
                        self._time_to_wall_s(obj, vel),
                    )
            if self.lock_s >= self.track_hold_s and self._bounce_clear(obj, vel, ee):
                logger.info(
                    "    locked over object: XY %.1f cm, speed %.0f cm/s, "
                    "lead %.0f ms, wall in %.2f s",
                    100.0 * err_xy,
                    100.0 * speed,
                    1000.0 * lead_s,
                    self._time_to_wall_s(obj, vel),
                )
                self.stage = STAGE_DESCEND
                self.timer_s = 0.0
            elif self.timer_s > self.grasp_stage_timeout_s:
                self._restart_approach("could not lock onto the object")

        elif self.stage == STAGE_DESCEND:
            # Keep riding the object in XY (lag lead only), drive Z onto it.
            aim = self._forecast(obj, vel, self.tau)
            target = np.array([aim[0], aim[1], obj[2]], dtype=np.float64)

            tol_xy = self._grasp_tol_xy(vel)
            err_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
            err_z = abs(float(ee[2] - obj[2]))
            if err_xy < tol_xy and err_z < self.grasp_z_tol:
                self.grasp_err_xy = err_xy
                self.grasp_err_z = err_z
                self.grasp_speed = speed
                self.grasp_xy = obj[:2].copy()
                self.lift_z = float(ee[2]) + self.lift_height
                self.stage = STAGE_CLOSE
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                self._restart_approach(f"descend stalled at XY {100.0 * err_xy:.1f} cm")

        elif self.stage == STAGE_CLOSE:
            # Attach welds the object to the hand, so it co-moves. No freeze
            # needed and none used: freezing a fast object is a teleport-stop.
            grip = 0.0
            attach = True
            kin = False
            target = ee.copy()
            target[2] = obj[2]
            if self.timer_s >= self.close_gripper_s:
                self.stage = STAGE_LIFT
                self.timer_s = 0.0

        elif self.stage == STAGE_LIFT:
            grip, attach, kin = 0.0, True, False
            if self.lift_z is None:
                self.lift_z = float(ee[2]) + self.lift_height
            target = ee.copy()
            if self.grasp_xy is not None:
                target[0], target[1] = float(self.grasp_xy[0]), float(self.grasp_xy[1])
            target[2] = self.lift_z
            if float(ee[2]) >= self.lift_z - self.approach_z_tol:
                self.stage = STAGE_CARRY
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                logger.info("    lift stalled; going home")
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        elif self.stage == STAGE_CARRY:
            grip, attach, kin = 0.0, True, False
            target = self.carry_pose[:3].copy()
            target[2] = self._retract_z()
            quat = self.carry_pose[3:].copy()
            err_xy = float(np.linalg.norm(ee[:2] - target[:2]))
            err_z = abs(float(ee[2] - target[2]))
            if err_xy < self.servo_xy_tol and err_z < self.approach_z_tol:
                self.stage = STAGE_LOWER
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                logger.info("    carry stalled; going home")
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        elif self.stage == STAGE_LOWER:
            grip, attach, kin = 0.0, True, False
            target = self._drop_target()
            quat = self.carry_pose[3:].copy()
            err_xy = float(np.linalg.norm(ee[:2] - target[:2]))
            err_z = abs(float(ee[2] - target[2]))
            if err_xy < self.servo_xy_tol and err_z < self.approach_z_tol:
                self.stage = STAGE_OPEN
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                logger.info("    lower stalled; going home")
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        elif self.stage == STAGE_OPEN:
            # Release + pin so the object cannot bounce back out of the box.
            grip, attach, kin, freeze = 1.0, False, False, True
            target = self._drop_target()
            quat = self.carry_pose[3:].copy()
            if self.timer_s >= self.close_gripper_s and self._in_container(obj):
                self.stage = STAGE_HOME
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        elif self.stage == STAGE_HOME:
            grip, attach, kin, freeze = 1.0, False, False, True
            target = self._home_target(ee)
            quat = self.init_pose[3:].copy()
            if self._at_home(ee) and self._in_container(obj):
                self.stage = STAGE_DONE
                self.timer_s = 0.0

        else:  # DONE
            grip, attach, kin, freeze = 1.0, False, False, True
            target = self.init_pose[:3].copy()
            quat = self.init_pose[3:].copy()

        if self.stage != prev_stage:
            logger.info(
                "  %-16s  hand (%.3f, %.3f, %.3f)   object (%.3f, %.3f, %.3f)"
                "   gap %.1f cm   speed %.1f cm/s   %s",
                f"{STAGE_PLAIN.get(self.stage, self.stage)} ({self.stage})",
                *ee,
                *obj,
                100.0 * float(np.linalg.norm(ee - obj)),
                100.0 * speed,
                "holding" if attach else "not holding",
            )
            if self.stage == STAGE_CLOSE:
                logger.info(
                    "    grasp error XY %.1f cm, Z %.1f cm at %.0f cm/s (gate %.1f cm)",
                    100.0 * (self.grasp_err_xy or 0.0),
                    100.0 * (self.grasp_err_z or 0.0),
                    100.0 * (self.grasp_speed or 0.0),
                    100.0 * self._grasp_tol_xy(vel),
                )

        return ExpertCommand(
            position=target,
            quat_wxyz=quat,
            gripper_open=float(grip),
            stage=int(self.stage),
            kinematic_object=bool(kin and not attach),
            attach_object=bool(attach),
            freeze_object=bool(freeze),
        )
