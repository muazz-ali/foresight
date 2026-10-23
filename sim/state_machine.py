"""Foresight expert — intercept-servo pick-place.
intercept: predict where the object will be and move hand there
servo: move the hand to that predicted position

Stages: 0 start → 1 intercept → 3 close → 4 lift → 5 carry → 6 lower
→ 7 open → 8 home → 9 done. Stage 2 (go down) is unused: intercept closes
when the grasp_xyz_ok gate passes (spoken names in state_machine_logs.py).

What it is: the scripted "teacher". It reads the true object state from the sim
and tells the hand where to go. Its demos train the policy and set the G0 score.
It runs once per control step, inside the episode loop (sim/collect.py).
It must NOT use DynamicVLA code or its "object_picked" check. Success is judged
elsewhere (sim/success.py), never inside this file.
The plan's expert is four big steps (approach, grab+lift, place, reset). Here
they are split into stage ids 0–9 so logs show exactly where a run got stuck.
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

    # The success / failure labels (sim/success.py) read these ids instead of
    # hard-coding numbers, so a different expert could reuse the same labels.
    engaged: int  # reached the object at all
    lift: int  # object left the table
    placed: int  # object is at the drop point
    done: int  # episode finished cleanly
    names: dict  # stage id -> spoken name
    why: dict  # stage id -> one-line explanation


# Stage ids. Order matters: logs and "max stage reached" compare these numbers.
# Same ids as STAGE_PLAIN in sim/state_machine_logs.py — change both together.
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

# "engaged" = we started chasing (stage 1). "lift" = object is up (4).
# "placed" = fingers opened over the box (7). "done" = back home (9).
STAGE_SCHEMA = StageSchema(
    engaged=STAGE_APPROACH,
    lift=STAGE_LIFT,
    placed=STAGE_OPEN,
    done=STAGE_DONE,
    names=STAGE_PLAIN,
    why=STAGE_WHY,
)



def _yaw_about_z(q_down_wxyz: np.ndarray, heading: float) -> np.ndarray:
    """Rotate the hand about world Z so the pad axis (hand local x = world x at yaw 0)
    points along ``heading``. The gripper is symmetric, so use the smaller turn."""
    psi = (heading + np.pi / 2) % np.pi - np.pi / 2  # wrap to [-90, 90) deg
    h = 0.5 * psi
    qz = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = np.asarray(q_down_wxyz, dtype=np.float64)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

def _require(cfg: dict, *keys: str) -> Any:
    """Nested yaml lookup. Logs the missing path and raises KeyError.

    Why: a missing setting must stop the run loudly, not turn into a silent 0.
    """
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
    """Max episode length in sim steps from ``state_machine_params.inference_window_s``.

    Why: gives every episode a hard time cap, so a stuck run still ends.
    """
    step_dt = float(dt if dt is not None else sim_dt(cfg))
    window_s = float(_require(cfg, "state_machine_params", "inference_window_s"))
    return max(1, int(round(window_s / step_dt)))


@dataclass
class ExpertCommand:
    """One tick of orders: where the hand goes, open or closed, and how the sim moves the object.

    The three object flags tell the scene who moves the object this step:
    the table slide (kinematic), the hand (attach), or nobody (freeze).
    """

    position: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)
    gripper_open: float  # 1=open, 0=closed
    stage: int
    kinematic_object: bool  # True while we slide the object on the table
    attach_object: bool  # True while the object is held (snapped to the hand)
    freeze_object: bool = False  # pause object pose (open / home / done)

class PickPlaceStateMachine:
    """The scripted teacher. Reads true object state, walks the 9 stages, returns hand orders.

    One instance is reused across episodes; call ``reset()`` between them.
    """

    stage_schema = STAGE_SCHEMA
    stage_done = STAGE_DONE

    def __init__(self, cfg: dict, *, dt: float | None = None):
        """Load ``state_machine_params`` + ``robot.init_pose``. ``dt`` overrides ``sim.dt``.

        All tuning numbers come from the yaml (sim/phase0_cfg.yaml), not from
        code, so a run can be repeated from its config alone.
        """
        self._p = dict(_require(cfg, "state_machine_params"))
        robot = _require(cfg, "robot")
        scene = cfg.get("scene") or {}

        self.dt = float(dt if dt is not None else sim_dt(cfg))
        # E6 option (default off = old behaviour): "along" turns the hand about world Z
        # so the finger pads line up with the object's motion during approach/close/lift.
        self.grasp_yaw_mode = str(self._p.get("grasp_yaw_mode", "fixed"))
        self._grasp_quat: np.ndarray | None = None

        # --- timings -----------------------------------------------------
        self.settle_s = self._f("settle_s")
        self.close_gripper_s = self._f("close_gripper_s")
        self.inference_window_s = self._f("inference_window_s")
        self.stage_timeout_s = self._f("stage_timeout_s")
        # Chasing a fast object takes longer than other stages: give the
        # approach 3x the normal stage time unless the yaml says otherwise.
        self.grasp_stage_timeout_s = self._f("grasp_stage_timeout_s", self.stage_timeout_s * 3.0)
        # Min straight-line time we need before the object hits a wall and turns.
        self.bounce_guard_s = self._f("bounce_guard_s")

        # --- interception ------------------------------------------------
        self.lookahead_s = self._f("lookahead_s")  # (plan: 0.25 s)
        # The look-ahead we write into the policy's 4 numbers. Kept separate from
        # the aim look-ahead above; falls back to it if the yaml has no value.
        self.conditioning_delta_s = self._f("conditioning_delta_s", self.lookahead_s)
        # Pad on aim for tip lag (not a wait). Calibrate: tools/diag/calibrate_servo_lag.py
        self.robot_approach_pad_s = self._f("robot_approach_pad_s")
        self.ee_speed_max = self._f("ee_speed_max")
        # Passes of "guess travel time -> re-aim -> guess again" (default 3).
        self.intercept_iters = int(self._p.get("intercept_iters", 3))

        # --- tolerances ---------------------------------------------------
        # One chase→close gate: |dx|, |dy|, |dz| (metres).
        ok = self._p.get("grasp_xyz_ok")
        if ok is None or len(ok) != 3:
            logger.error("missing required yaml key: state_machine_params.grasp_xyz_ok")
            raise KeyError("state_machine_params.grasp_xyz_ok")
        self.grasp_xyz_ok = np.asarray(ok, dtype=np.float64).reshape(3)
        self.grasp_xy_tol = float(max(self.grasp_xyz_ok[0], self.grasp_xyz_ok[1]))
        self.grasp_z_tol = float(self.grasp_xyz_ok[2])
        # Carry / lower / lift *arrival* height — looser than the grasp close gate.
        # Held hands settle ~12–15 mm above the drop command; 15 mm made lowers stall.
        self.place_z_tol = self._f("place_z_tol", 0.03)
        self.servo_xy_tol = self._f("servo_xy_tol")
        self.place_xy_tol = self._f("place_xy_tol")
        self.home_z_tol = self._f("home_z_tol")
        self.ee_settle_tol = self._f("ee_settle_tol")

        # --- heights ------------------------------------------------------
        self.lift_height = self._f("lift_height")
        self.drop_z_gap = self._f("drop_z_gap")
        # Wrist depth close gate (m). 0 = off (use GT hand-object Z). Needs a depth reading.
        self.close_depth_m = self._f("close_depth_m", 0.0)
        # Held object farther than this from the hand = it slipped out (physics grasp).
        self.slip_dist_m = self._f("slip_dist_m", 0.10)
        # After release: rise this far straight up (fixed XY) before heading home,
        # leading Z by at most release_step_m per tick so the hand does not swing.
        self.release_retreat_m = self._f("release_retreat_m", 0.10)
        self.release_step_m = self._f("release_step_m", 0.02)

        # --- retries ------------------------------------------------------
        # How many times a missed grab may go back to stage 1 before giving up.
        self.max_retries = int(self._p.get("max_retries", 3))

        # init_pose = (x, y, z, quat wxyz): the home spot. Its quat is "gripper
        # pointing down", which we reuse for every stage that does not turn the hand.
        self.init_pose = np.asarray(robot["init_pose"], dtype=np.float64)
        self._down_quat = self.init_pose[3:].copy()

        # Walls of the object's play area. Used to predict bounces. None = no walls.
        # House / throw mode: NO bounce. The object coasts straight, so there is no
        # wall to wait for (_bounce_clear is always true) and the forecast is constant velocity.
        bounds = scene.get("object_xy_bounds")
        self.throw_mode = bool(scene.get("use_house", False))
        self.xy_bounds = (
            ((float(bounds[0][0]), float(bounds[0][1])), (float(bounds[1][0]), float(bounds[1][1])))
            if bounds is not None and not self.throw_mode
            else None
        )
        # Real PhysX grasp (no weld) — same default rule as sim/scene.py.
        self.physics_grasp = not bool(scene.get("use_weld_grasp", not self.throw_mode))
        if self.physics_grasp:
            # Fingers need the longer dwell before lift starts.
            self.close_gripper_s = self._f("physics_close_s")

        # Placeholders. The episode loop calls set_box_poses() each episode with
        # the real box location.
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
        """Effective command-to-motion lag (approach pad + one zero-order-hold step)."""
        return self.robot_approach_pad_s + self.dt

    # --------------------------------------------------------------- public
    def set_box_poses(
        self,
        carry_pos: np.ndarray,
        quat_wxyz: np.ndarray | None = None,
        *,
        drop_pos: np.ndarray | None = None,
    ) -> None:
        """Carry pose (box XY + clear Z); ``drop_pos`` is inside the box.

        Called once per episode because the box moves between episodes.
        If no ``drop_pos`` is given, we drop at the carry point.
        """
        p = np.asarray(carry_pos, dtype=np.float64).reshape(3)
        q = self._down_quat.copy() if quat_wxyz is None else np.asarray(
            quat_wxyz, dtype=np.float64
        ).reshape(4)
        self.carry_pose = np.concatenate([p, q])
        if drop_pos is None:
            self._drop_pose = p.copy()
        else:
            d = np.asarray(drop_pos, dtype=np.float64).reshape(3)
            # Keep only the drop HEIGHT from the caller; XY always matches carry,
            # so carry and lower stages line up over the same spot.
            d[0], d[1] = p[0], p[1]
            self._drop_pose = d

    def reset(self) -> None:
        """Back to stage 0 and clear all per-episode memory (timers, retries, grasp stats)."""
        self._grasp_quat = None
        self.stage = STAGE_RESET
        self.timer_s = 0.0  # time spent in the current stage
        self.retries = 0
        self._blocked_s = 0.0  # only throttles the "waiting for a straight run" log
        self.lift_z: float | None = None
        self.grasp_xy: np.ndarray | None = None
        self._home_anchor: np.ndarray | None = None  # hand pose when go-home began
        self._open_anchor: np.ndarray | None = None  # hand pose when open began
        # Grasp quality, surfaced in meta so a loosened tolerance cannot hide.
        self.grasp_err_xy: float | None = None
        self.grasp_err_z: float | None = None
        self.grasp_speed: float | None = None

    def grasp_report(self) -> dict:
        """Per-episode grasp quality. ``None`` until CLOSE fires.

        Why: goes into the logs so a loose tolerance cannot hide a sloppy grab.
        """
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
        gz = float(obj[2] if goal_z is None else goal_z)
        t_travel = 0.0
        for _ in range(max(1, self.intercept_iters)):
            aim = self._forecast(obj, vel, min(t_travel, self.lookahead_s))
            goal = np.array([aim[0], aim[1], gz], dtype=np.float64)
            t_travel = float(np.linalg.norm(ee - goal)) / max(self.ee_speed_max, 1e-6)
        return min(t_travel, self.lookahead_s) + self.tau

    def _grasp_gate_ok(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        wrist_depth_m: float | None = None,
    ) -> tuple[bool, float, float]:
        """True when the hand is inside ``grasp_xyz_ok`` (ready to close).

        XY uses |dx| and |dy|. Z always uses GT |dz| — never wrist depth alone.
        Wrist depth used to *replace* GT Z and caused both early closes (wrist
        under threshold while |dz| was still ~17 mm) and stuck retries (GT fine
        but wrist stuck ~6 mm above the depth gate). ``wrist_depth_m`` is kept
        for the call site / logs only. Returns (ok, err_xy_norm, err_z).
        """
        del wrist_depth_m  # logged elsewhere; must not override GT Z
        err_dx = abs(float(ee[0] - obj[0]))
        err_dy = abs(float(ee[1] - obj[1]))
        err_z = abs(float(ee[2] - obj[2]))
        err_xy = float(np.hypot(err_dx, err_dy))
        xy_ok = err_dx <= float(self.grasp_xyz_ok[0]) and err_dy <= float(
            self.grasp_xyz_ok[1]
        )
        low_enough = err_z <= float(self.grasp_xyz_ok[2])
        return xy_ok and low_enough, err_xy, err_z

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

    def _close_time_s(self, ee: np.ndarray, obj: np.ndarray) -> float:
        """Seconds of Z still to burn before the grasp gate can pass.

        First-order settle of the remaining height to ``grasp_z_tol``. Chase aims
        at object Z, so this is usually one or two steps once XY is close.
        """
        dz = max(0.0, float(ee[2] - obj[2]))
        if dz <= self.grasp_z_tol:
            return self.dt
        return self.tau * float(np.log(dz / self.grasp_z_tol))

    def _grasp_window_s(self, ee: np.ndarray | None = None, obj: np.ndarray | None = None) -> float:
        """Straight-line time the grasp needs before the object reverses.

        Only the leftover Z drop has to fit. Once CLOSE attaches, the object
        is welded to the hand and stops following the table, so nothing after
        attach cares about walls. ``bounce_guard_s`` is the floor.
        """
        need = self.dt if ee is None or obj is None else self._close_time_s(ee, obj)
        return max(self.bounce_guard_s, need)

    def _bounce_clear(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        ee: np.ndarray | None = None,
    ) -> bool:
        """Enough straight-line time left to close without a wall reversal."""
        if float(np.linalg.norm(vel[:2])) < self.ee_settle_tol:
            return True
        return self._time_to_wall_s(pos, vel) > self._grasp_window_s(ee, pos)

    def _retract_z(self) -> float:
        """Height to travel at while holding the object: the higher of lift height and carry height.

        Why: never dip below where we lifted, so the held object clears the box rim.
        """
        way_back_z = float(self.carry_pose[2])
        lift_z = float(self.lift_z) if self.lift_z is not None else way_back_z
        return max(lift_z, way_back_z)

    def _drop_target(self) -> np.ndarray:
        """Where the hand goes to let go: over the box, low enough to be inside it."""
        if self._drop_pose is None:
            return self.carry_pose[:3].copy()
        p = self._drop_pose.copy()
        p[0], p[1] = float(self.carry_pose[0]), float(self.carry_pose[1])
        if self.lift_z is not None:
            # Never ask for a drop point higher than the travel height minus
            # one tolerance; keeps "lower" a real downward move.
            p[2] = min(float(p[2]), self._retract_z() - self.place_z_tol)
        return p

    def _at_home(self, ee: np.ndarray) -> bool:
        """True when the hand is back at the start spot (XY and height both close)."""
        home = self.init_pose[:3]
        return (
            float(np.linalg.norm(ee[:2] - home[:2])) < self.servo_xy_tol
            and abs(float(ee[2] - home[2])) < self.home_z_tol
        )

    def _home_target(self, ee: np.ndarray) -> np.ndarray:
        """go home: retreat straight up off the released object, then climb, then cross.

        Physics grasp does not pin the placed object, so open fingers that move
        sideways sweep it out of the box. Leg 1 holds the XY where home started
        and leads Z by a small step (a big Z jump makes DiffIK swing ~5 cm in XY)
        until the fingers clear the object. Leg 2 climbs to a safe height, leg 3
        crosses to home.
        """
        if self._home_anchor is None:
            self._home_anchor = ee.copy()
        anchor = self._home_anchor
        top = float(anchor[2]) + self.release_retreat_m
        # Done within one step of the top: a lagging arm only approaches it
        # asymptotically, so an exact ">= top" would never finish.
        if float(ee[2]) < top - self.release_step_m:
            t = anchor.copy()
            t[2] = min(float(ee[2]) + self.release_step_m, top)
            return t
        home = self.init_pose[:3].copy()
        clear_z = max(float(home[2]), float(self.carry_pose[2]), self._retract_z())
        err_xy = float(np.linalg.norm(ee[:2] - home[:2]))
        # Two-leg trip: first straight up, then across. Only go across once high enough.
        if err_xy >= self.servo_xy_tol and float(ee[2]) < clear_z - self.home_z_tol:
            t = ee.copy()
            t[2] = clear_z
            return t
        return home

    def _in_container(self, obj: np.ndarray) -> bool:
        """True when the object's XY is over the box. This is the "it landed" check for stages 7 and 8."""
        return float(np.linalg.norm(obj[:2] - self.carry_pose[:2])) < self.place_xy_tol

    def _restart_approach(self, why: str) -> None:
        """Bounded retry: a missed grasp goes back to intercept, not straight home.

        Why: one bad try should not end the episode. After ``max_retries`` we
        give up and go home so the run still finishes.
        """
        self.retries += 1
        self.timer_s = 0.0
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
        wrist_depth_m: float | None = None,
    ) -> ExpertCommand:
        """Advance one control step. Returns the EE command for this tick.
        
        This is the core state machine logic that runs every control step.
        It takes the current state of the object, the end effector position and
        orientation, and returns the next end effector command.

        Input is the TRUE object state from the sim (not a camera guess). Each
        stage block below sets the hand target and decides when to move on.
        There is no "drop -> stage 1" shortcut: a miss goes to
        ``_restart_approach``, which re-enters stage 1 (until retries run out).
        ``wrist_depth_m``: nearest thing in the wrist view (NaN / None = no reading).
        """
        # Hand rotation is not read: every stage picks its own target rotation.
        del ee_quat

        ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        obj = np.asarray(state.position, dtype=np.float64).reshape(3)
        vel = np.asarray(state.velocity, dtype=np.float64).reshape(3)
        speed = float(np.linalg.norm(vel))
        prev_stage = self.stage

        # Defaults each tick: fingers open, object slides on the table, not held,
        # not frozen. Stages below override only what they need.
        grip, kin, attach, freeze = 1.0, True, False, False
        target = ee.copy()  # default = stay put
        quat = self._down_quat.copy()
        if self.grasp_yaw_mode == "along" and self.stage in (STAGE_APPROACH, STAGE_DESCEND, STAGE_CLOSE, STAGE_LIFT):
            if self.stage in (STAGE_APPROACH, STAGE_DESCEND) and speed > 0.02:
                self._grasp_quat = _yaw_about_z(self._down_quat, float(np.arctan2(vel[1], vel[0])))
            if self._grasp_quat is not None:
                quat = self._grasp_quat.copy()

        self.timer_s += self.dt

        # Stage 0 — START: sit at home while the object starts moving, then go.
        if self.stage == STAGE_RESET:
            target = self.init_pose[:3].copy()
            if self.timer_s >= self.settle_s:
                self.stage = STAGE_APPROACH
                self.timer_s = 0.0

        elif self.stage in (STAGE_APPROACH, STAGE_DESCEND):
            # Chase: aim where the object will be when the hand arrives. Target Z
            # is the object itself (no hover). One grasp_xyz_ok gate → close.
            # STAGE_DESCEND is unused in the happy path; if entered, same logic.
            target_z = float(obj[2])
            lead_s = self._intercept_lead_s(ee, obj, vel, goal_z=target_z)
            aim = self._forecast(obj, vel, lead_s)
            target = np.array([aim[0], aim[1], target_z], dtype=np.float64)

            gate_ok, err_xy, err_z = self._grasp_gate_ok(ee, obj, wrist_depth_m)
            if gate_ok and not self._bounce_clear(obj, vel, ee):
                self._blocked_s += self.dt
                if self._blocked_s >= 1.0:
                    self._blocked_s = 0.0
                    logger.info(
                        "    in grasp box, but waiting for a straight run: need %.2f s, "
                        "wall in %.2f s (widen scene.object_xy_bounds or cut "
                        "tip_lag_pad_ms)",
                        self._grasp_window_s(ee, obj),
                        self._time_to_wall_s(obj, vel),
                    )
            if gate_ok and self._bounce_clear(obj, vel, ee):
                logger.info(
                    "    grasp gate: XY %.1f cm, Z %.1f cm, speed %.0f cm/s, "
                    "lead %.0f ms, wall in %.2f s",
                    100.0 * err_xy,
                    100.0 * err_z,
                    100.0 * speed,
                    1000.0 * lead_s,
                    self._time_to_wall_s(obj, vel),
                )
                self.grasp_err_xy = err_xy
                self.grasp_err_z = err_z
                self.grasp_speed = speed
                self.grasp_xy = obj[:2].copy()
                self.lift_z = float(ee[2]) + self.lift_height
                self.stage = STAGE_CLOSE
                self.timer_s = 0.0
            elif self.timer_s > self.grasp_stage_timeout_s:
                self._restart_approach("could not reach grasp gate")

        # Stage 3 — CLOSE FINGERS: shut the gripper. ``attach`` means "grasp now":
        # weld mode glues the object to the hand; physics mode frees it to PhysX
        # and the fingers squeeze it. Same flag, the scene decides (sim/scene.py).
        elif self.stage == STAGE_CLOSE:
            # No freeze needed and none used: freezing a fast object is a teleport-stop.
            grip = 0.0
            attach = True
            kin = False
            target = ee.copy()
            target[2] = obj[2]
            if self.timer_s >= self.close_gripper_s:
                self.stage = STAGE_LIFT
                self.timer_s = 0.0

        # Physics grasp can lose the object. Held but far from the hand → try again.
        # (Weld mode never trips this: the object sits within 1 cm of the hand.)
        elif self.stage in (STAGE_LIFT, STAGE_CARRY, STAGE_LOWER) and (
            float(np.linalg.norm(obj - ee)) > self.slip_dist_m
        ):
            self._restart_approach(
                f"object slipped ({100.0 * float(np.linalg.norm(obj - ee)):.0f} cm from hand)"
            )

        # Stages 4-6 (lift, carry, lower): the object is held, so no table slide.
        # If any of them stalls, give up and go home (no retry once holding).
        # Stage 4 — LIFT: straight up from the grab spot.
        elif self.stage == STAGE_LIFT:
            grip, attach, kin = 0.0, True, False
            if self.lift_z is None:
                self.lift_z = float(ee[2]) + self.lift_height
            target = ee.copy()
            if self.grasp_xy is not None:
                target[0], target[1] = float(self.grasp_xy[0]), float(self.grasp_xy[1])
            target[2] = self.lift_z
            if float(ee[2]) >= self.lift_z - self.place_z_tol:
                self.stage = STAGE_CARRY
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                logger.info("    lift stalled; going home")
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        # Stage 5 — CARRY: fly high, over the box.
        elif self.stage == STAGE_CARRY:
            grip, attach, kin = 0.0, True, False
            target = self.carry_pose[:3].copy()
            target[2] = self._retract_z()
            quat = self.carry_pose[3:].copy()
            err_xy = float(np.linalg.norm(ee[:2] - target[:2]))
            err_z = abs(float(ee[2] - target[2]))
            if err_xy < self.servo_xy_tol and err_z < self.place_z_tol:
                self.stage = STAGE_LOWER
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                logger.info("    carry stalled; going home")
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        # Stage 6 — LOWER: come down inside the box.
        elif self.stage == STAGE_LOWER:
            grip, attach, kin = 0.0, True, False
            target = self._drop_target()
            quat = self.carry_pose[3:].copy()
            err_xy = float(np.linalg.norm(ee[:2] - target[:2]))
            err_z = abs(float(ee[2] - target[2]))
            if err_xy < self.servo_xy_tol and err_z < self.place_z_tol:
                self.stage = STAGE_OPEN
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                # Blocked (object on the floor / finger on the rim) but already
                # over the box: release here so the demo still shows a real open.
                if err_xy < self.servo_xy_tol:
                    logger.info("    lower blocked over the box; releasing here")
                    self.stage = STAGE_OPEN
                else:
                    logger.info("    lower stalled off the box; going home")
                    self.stage = STAGE_HOME
                self.timer_s = 0.0

        # Stage 7 — OPEN FINGERS: let go over the box. Move on once the fingers
        # had time to open AND the object sits over the box (or time runs out).
        elif self.stage == STAGE_OPEN:
            # Release. Weld mode pins the object; physics mode lets it drop.
            # Hold the hand where the open began: pushing on toward the drop
            # point while the fingers part shoves the object sideways.
            grip, attach, kin, freeze = 1.0, False, False, True
            if self._open_anchor is None:
                self._open_anchor = ee.copy()
            target = self._open_anchor.copy()
            quat = self.carry_pose[3:].copy()
            if self.timer_s >= self.close_gripper_s and self._in_container(obj):
                self.stage = STAGE_HOME
                self.timer_s = 0.0
            elif self.timer_s > self.stage_timeout_s:
                self.stage = STAGE_HOME
                self.timer_s = 0.0

        # Stage 8 — GO HOME: retreat straight up off the object, then home (see
        # _home_target). Done only if the hand is home AND the object is still in the box.
        elif self.stage == STAGE_HOME:
            grip, attach, kin, freeze = 1.0, False, False, True
            target = self._home_target(ee)
            quat = self.init_pose[3:].copy()
            if self._at_home(ee) and self._in_container(obj):
                self.stage = STAGE_DONE
                self.timer_s = 0.0

        else:  # Stage 9 — DONE: park at home. The episode loop stops on this stage.
            grip, attach, kin, freeze = 1.0, False, False, True
            target = self.init_pose[:3].copy()
            quat = self.init_pose[3:].copy()

        # Anchors live only while their stage runs (a retry can re-enter later).
        if self.stage != STAGE_OPEN:
            self._open_anchor = None
        if self.stage != STAGE_HOME:
            self._home_anchor = None

        # Log only on a stage change, so the human log reads like a short story.
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
                    "    grasp error XY %.1f cm, Z %.1f cm at %.0f cm/s "
                    "(gate dx/dy/dz %.1f/%.1f/%.1f cm)",
                    100.0 * (self.grasp_err_xy or 0.0),
                    100.0 * (self.grasp_err_z or 0.0),
                    100.0 * (self.grasp_speed or 0.0),
                    100.0 * float(self.grasp_xyz_ok[0]),
                    100.0 * float(self.grasp_xyz_ok[1]),
                    100.0 * float(self.grasp_xyz_ok[2]),
                )

        # kinematic only when not holding: a held object follows the hand instead.
        return ExpertCommand(
            position=target,
            quat_wxyz=quat,
            gripper_open=float(grip),
            stage=int(self.stage),
            kinematic_object=bool(kin and not attach),
            attach_object=bool(attach),
            freeze_object=bool(freeze),
        )
