"""Foresight expert — intercept-servo pick-place.
intercept: predict where the object will be and move hand there
servo: move the hand to that predicted position

Stages: 0 start → 1 intercept → 2 close → 3 lift → 4 carry → 5 lower
→ 6 open → 7 home → 8 done. Intercept closes when the grasp_xyz_ok gate
passes (spoken names in state_machine_logs.py).

What it is: the scripted "teacher". It reads the true object state from the sim
and tells the hand where to go. Its demos train the policy and set the G0 score.
It runs once per control step, inside the episode loop (sim/collect.py).
It must NOT use DynamicVLA code or its "object_picked" check. Success is judged
elsewhere (sim/success.py), never inside this file.
The plan's expert is four big steps (approach, grab+lift, place, reset). Here
they are split into stage ids 0–8 so logs show exactly where a run got stuck.
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
STAGE_CLOSE = 2
STAGE_LIFT = 3
STAGE_CARRY = 4
STAGE_LOWER = 5
STAGE_OPEN = 6
STAGE_HOME = 7
STAGE_DONE = 8

# Which chase-aim formula ran this tick. Index = int code logged per tick
# (h5 ``aim_branch``). "none" = no chase aim computed (other stages / bail-outs).
AIM_BRANCHES = ("none", "far_chase", "settle", "forecast_off")

# "engaged" = we started chasing (stage 1). "lift" = object is up (3).
# "placed" = fingers opened over the box (6). "done" = back home (8).
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


def episode_max_steps(cfg: dict, dt: float | None = None) -> int:
    """Max episode length in sim steps from ``state_machine_params.episode_max_s``.

    Why: gives every episode a hard time cap, so a stuck run still ends.
    """
    step_dt = float(dt if dt is not None else sim_dt(cfg))
    window_s = float(_require(cfg, "state_machine_params", "episode_max_s"))
    return max(1, int(round(window_s / step_dt)))


@dataclass
class StateMachineCommand:
    """One tick of orders: where the hand goes, open or closed, and how the sim moves the object.

    The three object flags tell the scene who moves the object this step:
    the table slide (kinematic), the hand (attach), or nobody (freeze).

    ``velocity`` is v̂_obj for DiffIK feedforward: the same object coast
    velocity ``_forecast`` uses (not FD of consecutive aims). Zeros when the
    aim is a fixed pose (home / open / done).
    """

    position: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)
    gripper_open: float  # 1=open, 0=closed
    stage: int
    kinematic_object: bool  # True while we slide the object on the table
    attach_object: bool  # True while the object is held (snapped to the hand)
    freeze_object: bool = False  # pause object pose (open / home / done)
    velocity: np.ndarray | None = None  # (3,) m/s = v̂_obj; None → zeros at scene

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
        # Yaml robot_base_safe_zone (mm), stored here in metres. 0 = off.
        self.robot_base_safe_zone_m = float(self._p.get("robot_base_safe_zone_m", 0.0))
        # Diag only (smoke_physics_grasp.py --miss): stay at stage 0, never chase.
        self.hold_home = False

        # --- timings -----------------------------------------------------
        # Weld-only dwell. Physics grasp replaces this with SM_hold_to_close_fingers_s.
        self.close_gripper_s = self._f("hold_closed_weld_s", 0.1)
        # Only clock. The episode loop stops after this many seconds.
        self.episode_max_s = self._f("episode_max_s")
        # --- interception ------------------------------------------------
        self.max_aim_ahead_s = self._f("max_aim_ahead_s")  # (plan: 0.25 s)
        # The look-ahead we write into the policy's 4 numbers. Kept separate from
        # the aim look-ahead above; falls back to it if the yaml has no value.
        self.policy_look_ahead_s = self._f("policy_look_ahead_s", self.max_aim_ahead_s)
        # Extra aim time. 0: joint speed feed already covers fingertip lag.
        self.robot_approach_pad_s = self._f("robot_approach_pad_s", 0.0)
        self.robot_speed_max_m_s = self._f("robot_speed_max_m_s")
        # Passes of "guess travel time -> re-aim -> guess again" (default 3).
        self.aim_refine_iters = int(self._p.get("aim_refine_iters", 3))

        # --- tolerances ---------------------------------------------------
        # One chase→close gate: |dx|, |dy|, |dz| (metres).
        ok = self._p.get("grasp_xyz_ok_m")
        if ok is None or len(ok) != 3:
            logger.error("missing required yaml key: state_machine_params.grasp_xyz_ok_m")
            raise KeyError("state_machine_params.grasp_xyz_ok_m")
        self.grasp_xyz_ok_m = np.asarray(ok, dtype=np.float64).reshape(3)
        self.grasp_xy_tol = float(max(self.grasp_xyz_ok_m[0], self.grasp_xyz_ok_m[1]))
        self.grasp_z_tol = float(self.grasp_xyz_ok_m[2])
        # B5: CLOSE only when |dx|,|dy| ≤ this (default 2 mm). Chase uses grasp_xyz_ok.
        self.close_xy_ok_m = float(self._p.get("close_xy_ok_m", 0.002))
        # Ride correction gain: inside the chase box aim += k·(object_now − hand).
        # The fixed tip-lag lead leaves a speed/heading-dependent residual (2–4 mm
        # at 30 cm/s on diagonal throws) that sat just outside close_xy_ok, so the
        # hand never closed (B1 smoke). 0 = off (old behaviour).
        self.ride_k = float(self._p.get("ride_correction_gain", 0.0))
        # CLOSE also needs |dz| ≤ close_z_ok (default = the 12 mm chase-box z).
        # The 2 mm XY check closes on the same step it passes. Riding inside the
        # 10 mm box is what matches speed, so there is no second drift check.
        self.close_z_ok_m = float(self._p.get("close_z_ok_m", self.grasp_z_tol))
        # One arrival box for lift / carry / lower / home XY: |dx|, |dy|, |dz|.
        # Held hands settle ~12–15 mm above a drop command; 15 mm made lowers stall.
        self.arrived_m = self._f("arrived_xyz_m", 0.03)
        self.place_z_tol = self.arrived_m
        # "In the bowl" is the same box around the bowl center. Release Z is separate.
        self.place_xy_tol = self.arrived_m
        # Ball around the start pose. Inside it, go-home can finish.
        self.home_radius_m = self._f("home_radius_m", 0.10)

        # --- heights ------------------------------------------------------
        self.lift_height_m = self._f("lift_height_m")
        # Open pose: bowl center, this far under the rim. Scene builds the point.
        self.release_below_rim_m = self._f("release_below_rim_m", 0.01)
        # Floor for chase/close aim Z (ride-contact: avocado/peach tip through table).
        self.min_aim_z_above_table = float(scene.get("min_aim_z_above_table_m", 0.015))
        # Claim p̂z / z_grasp (collect ≥20 cm, lock median <10 cm). See grasp_height.
        from interfaces.grasp_height import ZLatch

        self._z_latch = ZLatch.from_cfg(cfg)
        self.z_grasp: float | None = None
        # Approx half-height for GT top when wrist depth is missing (sphere on table).
        # Overwritten each episode via set_object_half_height(scene mesh size).
        self._obj_half_h = float((scene.get("object_radius") or 0.03))
        # Expert aim Z = object centre (reliable demos). Pack p̂z can come from
        # wrist depth latch so the policy trains on estimated height, not GT.
        ch = cfg.get("claim_height") if isinstance(cfg.get("claim_height"), dict) else {}
        # Aim Z = object centre by default (reliable demos).
        self.close_at_object_center = bool(ch.get("close_at_object_center", True))
        # Pack p̂z from wrist depth latch when true (not GT centre).
        self.use_object_height_from_depth = bool(
            ch.get("use_object_height_from_depth", True)
        )
        self.depth_rmse_max_m = float(ch.get("depth_rmse_max_m", 0.01))
        # A4: add onto wrist z_top (yaml depth_z_bias_m).
        self.depth_z_bias_m = float(ch.get("depth_z_bias_m", 0.0))
        # Depth top → grasp Z uses half-height (object centre), not a yaml insert.
        self._z_latch.insert_m = float(self._obj_half_h)
        # Fallback if live cam pose is missing (hand cam above TCP).
        tcp = robot.get("tcp_offset") or [0.0, 0.0, 0.1034]
        self._wrist_cam_above_tcp_m = float(tcp[2]) if len(tcp) >= 3 else 0.1034
        # Wrist point cloud (A4): intrinsics + mount + TCP from the yaml.
        from interfaces.grasp_height import wrist_intrinsics

        wc = cfg.get("wrist_camera") or {}
        self._wrist_K = wrist_intrinsics(cfg)
        self._tcp_offset_m = np.asarray(tcp, dtype=np.float64).reshape(3)
        self._wrist_pos_m = np.asarray(wc.get("pos", [0.065, 0.0, 0.0]), dtype=np.float64)
        self._wrist_quat = np.asarray(
            wc.get("quat_wxyz", [0.0, 0.7071068, 0.7071068, 0.0]), dtype=np.float64
        )
        self.p_hat_z: float | None = None  # pack height (depth or centre)
        # Wrist depth close gate (m). 0 = off (use GT / latched z_grasp). Needs a depth reading.
        self.close_depth_m = self._f("close_depth_m", 0.0)
        # Held object farther than this from the hand = it slipped out (physics grasp).
        self.robot_distance_to_object_m = self._f("robot_distance_to_object_m", 0.10)
        # After the drop: rise this far above the bowl rim (XY held) before going home.
        self.move_up_m = self._f("move_up_above_container_m", 0.05)

        # init_pose = (x, y, z, quat wxyz): the home spot. Its quat is "gripper
        # pointing down", which we reuse for every stage that does not turn the hand.
        self.init_pose = np.asarray(robot["init_pose"], dtype=np.float64)
        self._down_quat = self.init_pose[3:].copy()

        # Real PhysX grasp (no weld) — same default as sim/scene.py.
        self.physics_grasp = not bool(scene.get("use_weld_grasp", False))
        # Finger travel time (robot). Physics close waits this long before freeing
        # the object to PhysX so the pads are already shut (else ≥50 cm/s slips).
        self.finger_close_s = float(robot.get("gripper_close_time_s", 0.180))
        # Pad-contact time for the predictive close gate (cmd + travel, no force).
        from sim.gripper_actuator import nominal_t_contact_s

        ga = robot.get("gripper_actuator") if isinstance(robot.get("gripper_actuator"), dict) else {}
        # Prefer explicit contact time; else actuator nominal (≈0.13 s).
        if "t_contact_s" in ga:
            self.t_contact_s = float(ga["t_contact_s"])
        else:
            kwargs = {}
            if "v_finger_m_s" in ga:
                kwargs["v_finger_m_s"] = float(ga["v_finger_m_s"])
            if "t_cmd_s" in ga:
                kwargs["t_cmd_s"] = float(ga["t_cmd_s"])
            self.t_contact_s = float(nominal_t_contact_s(**kwargs))
        if self.physics_grasp:
            # Same yaml wait: release holds this long; close may be extended below.
            self.release_hold_s = self._f("SM_hold_to_close_fingers_s")
            self.close_gripper_s = self.release_hold_s
            # Dwell must cover finger travel + a squeeze beat before the lift.
            if self.close_gripper_s < self.finger_close_s + 0.15:
                self.close_gripper_s = self.finger_close_s + 0.15
        else:
            self.release_hold_s = self.close_gripper_s

        # B1: count every GT fallback (z_top / XY / vel / Δ). Must stay 0 for a valid run.
        self.gt_fallback = {"z_top": 0, "object_xy": 0, "velocity": 0, "delta": 0}
        # When True, delayed GT top is the oracle z channel (not a fallback).
        self.oracle_z_top: bool = False
        # Ride |Δv| samples while chasing (C30 pre-check).
        self.ride_dv_cms: list[float] = []
        # Pre-shape open fraction (1 = fully open). Set per episode from w_pre.
        self.pre_shape_open: float = 1.0
        self.w_pre_clamped: bool = False
        self.w_pre_m: float | None = None

        # Placeholders. The episode loop calls set_box_poses() each episode with
        # the real box location.
        self.carry_pose = np.concatenate([np.zeros(3), self._down_quat])
        self._drop_pose: np.ndarray | None = None
        # Table XY for clipping intercept aims (set each episode from the scene).
        # Without this, a long lead at ≥50 cm/s pulls the hand off the table.
        # Use the full table footprint — not the thin play band.
        self._aim_xy: tuple[tuple[float, float], tuple[float, float]] | None = None
        # Looser box for "object escaped — stop chase". Aim is inset for hand
        # safety; escape must still allow the far-edge spawn rim (smoke lemon
        # seed 50194 sat 6 mm past inset xmax and went home on frame 1).
        self._escape_xy: tuple[tuple[float, float], tuple[float, float]] | None = None
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

    def set_aim_xy_bounds(self, bounds) -> None:
        """Clip intercept / close aims to the table footprint.

        Call once per episode with the scene table XY box (not the thin play
        band). Stops a long look-ahead from commanding the hand off the table
        at ≥50 cm/s (``ee-off-table`` abort).
        """
        if bounds is None:
            self._aim_xy = None
            return
        bx, by = bounds[0], bounds[1]
        self._aim_xy = (
            (float(bx[0]), float(bx[1])),
            (float(by[0]), float(by[1])),
        )

    def set_escape_xy_bounds(self, bounds) -> None:
        """Where the object may still be chased (full table / play, not aim inset).

        Call once per episode after ``set_aim_xy_bounds``. If unset, chase escape
        falls back to the aim box (old behaviour). 
        Prefer the raw table XY so a spawn on the far rim is not treated as "already escaped."
        """
        if bounds is None:
            self._escape_xy = None
            return
        bx, by = bounds[0], bounds[1]
        self._escape_xy = (
            (float(bx[0]), float(bx[1])),
            (float(by[0]), float(by[1])),
        )

    def _object_left_catch_box(self, obj: np.ndarray) -> bool:
        """True if the object left the catch region — stop chase, go home.

        Uses ``_escape_xy`` when set; else the tighter aim box. Near-base and
        off-table misses should trip this; far-rim spawns should not.
        """
        box = self._escape_xy if self._escape_xy is not None else self._aim_xy
        if box is None:
            return False
        (ax0, ax1), (ay0, ay1) = box
        x, y = float(obj[0]), float(obj[1])
        return not (ax0 <= x <= ax1 and ay0 <= y <= ay1)

    def _object_in_base_zone(
        self,
        obj: np.ndarray,
        vel: np.ndarray,
        delta_meas_s: float | None,
    ) -> bool:
        """True when the object is inside the base circle (``robot_base_safe_zone_m``).

        Call during the chase. A delayed frame is moved forward by ``delta_meas_s``
        first, so the check uses where the object is now.
        """
        r = float(self.robot_base_safe_zone_m)
        if r <= 0.0:
            return False
        p = obj if delta_meas_s is None else self._forecast(obj, vel, float(delta_meas_s))
        return float(np.hypot(float(p[0]), float(p[1]))) <= r

    def _clip_aim_xy(self, aim: np.ndarray) -> np.ndarray:
        """Keep aim XY on the table when bounds are set."""
        out = np.asarray(aim, dtype=np.float64).reshape(3).copy()
        if self._aim_xy is None:
            return out
        (xmin, xmax), (ymin, ymax) = self._aim_xy
        out[0] = float(np.clip(out[0], xmin, xmax))
        out[1] = float(np.clip(out[1], ymin, ymax))
        return out

    def set_object_half_height(
        self, half_h: float, *, z_table: float | None = None
    ) -> None:
        """Use this episode's mesh half-height for GT top and close insert.

        Call once per episode after ``scene.reset_episode`` 
        (meta has ``object_half_height``). Sets insert ≈ half-height so ``z_grasp`` lands
        near the object centre — 10 mm insert was closing in the air above fruit.
        Pass ``z_table`` (scene table top) so the clear floor matches the real
        table, not a stale yaml height.
        """
        h = float(half_h)
        if not np.isfinite(h) or h <= 0.0:
            return
        self._obj_half_h = h
        self._z_latch.insert_m = h
        if z_table is not None and np.isfinite(float(z_table)):
            self._z_latch.z_table = float(z_table)

    def reset(self) -> None:
        """Back to stage 0 and clear all per-episode memory (timers, grasp stats)."""
        self._grasp_quat = None
        self.stage = STAGE_RESET
        self.timer_s = 0.0  # time spent in the current stage
        # Kept at 0 so old logs still have the field. The expert does not re-chase.
        self.retries = 0
        self.lift_z: float | None = None
        self.grasp_xy: np.ndarray | None = None
        self._home_anchor: np.ndarray | None = None  # hand pose when go-home began
        self._home_crossing = False  # go-home leg 3 started (latched; see _home_target)
        self._open_anchor: np.ndarray | None = None  # hand pose when open began
        self._lift_anchor: np.ndarray | None = None  # hand pose when lift began
        self._z_latch.reset()
        self.z_grasp: float | None = None  # expert aim height
        self.p_hat_z: float | None = None  # policy pack height
        # Grasp quality, surfaced in meta so a loosened tolerance cannot hide.
        self.grasp_err_xy: float | None = None
        self.grasp_err_z: float | None = None
        self.grasp_speed: float | None = None
        self._prev_ee: np.ndarray | None = None  # last tick's hand (B3 speed gate)
        # Hand poses, one per tick (newest last): the slip check compares a delayed
        # object frame with the hand at that frame's time, not the live hand.
        self._ee_hist: list[np.ndarray] = []
        self.gt_fallback = {"z_top": 0, "object_xy": 0, "velocity": 0, "delta": 0}
        self.ride_dv_cms = []
        # Per-close diagnostics for B1 JSONL.
        self.close_xy_err_m: float | None = None
        self.close_z_err_m: float | None = None
        self.dv_at_close_m_s: float | None = None
        self.t_lag_meas_s: float | None = None
        # Chase stopped by the reach rule → B1 judge labels it F1 "unreachable".
        self.reach_stopped = False
        self.aim_branch = "none"  # last tick's chase-aim branch (AIM_BRANCHES)
        self.p_now = np.full(3, np.nan)  # last chase tick's object-now (F2)

    def set_pre_shape(self, open_frac: float, *, w_pre_m: float | None = None, clamped: bool = False) -> None:
        """Set approach gripper opening (0=shut, 1=full). Call once per episode from w_pre."""
        self.pre_shape_open = float(np.clip(open_frac, 0.0, 1.0))
        self.w_pre_m = None if w_pre_m is None else float(w_pre_m)
        self.w_pre_clamped = bool(clamped)

    def grasp_report(self) -> dict:
        """Per-episode grasp quality. ``None`` until CLOSE fires.

        Why: goes into the logs so a loose tolerance cannot hide a sloppy grab.
        """
        return {
            "grasp_err_xy": self.grasp_err_xy,
            "grasp_err_z": self.grasp_err_z,
            "grasp_speed_m_s": self.grasp_speed,
            "z_grasp": self.z_grasp,
            "retries": int(self.retries),
            "gt_fallback": dict(self.gt_fallback),
            "gt_fallback_count": int(sum(self.gt_fallback.values())),
            "ride_dv_cms_median": (
                float(np.median(self.ride_dv_cms)) if self.ride_dv_cms else None
            ),
            "w_pre_m": self.w_pre_m,
            "w_pre_clamped": bool(self.w_pre_clamped),
            "close_xy_err_m": self.close_xy_err_m,
            "close_z_err_m": self.close_z_err_m,
            "dv_at_close_m_s": self.dv_at_close_m_s,
            "t_lag_meas_s": self.t_lag_meas_s,
            "unreachable": bool(self.reach_stopped),
        }

    # -------------------------------------------------------------- helpers
    def _forecast(self, pos: np.ndarray, vel: np.ndarray, horizon_s: float) -> np.ndarray:
        """Where the object will be in ``horizon_s`` (straight coast).

        Uses the same ``kinematic_forecast`` the scene integrates with, so the
        prediction and the world cannot disagree.
        """
        if horizon_s <= 0.0:
            return np.asarray(pos, dtype=np.float64).reshape(3).copy()
        p, _ = kinematic_forecast(pos, vel, float(horizon_s), self.dt)
        return p

    def _intercept_travel_s(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        vel: np.ndarray,
        goal_z: float | None = None,
    ) -> float:
        """Hand travel time to the coasting object, capped at ``max_aim_ahead_s``.

        Fixed-point: guess time → where object will be → recompute time.
        No ``tau`` / Δ pad here — callers add tip_lag only when needed.
        Pass ``obj`` = object-*now* (already unwrapped by Δ) so Δ is not
        double-counted on top of travel time.
        """
        gz = float(obj[2] if goal_z is None else goal_z)
        t_travel = 0.0
        for _ in range(max(1, self.aim_refine_iters)):
            aim = self._forecast(obj, vel, min(t_travel, self.max_aim_ahead_s))
            goal = np.array([aim[0], aim[1], gz], dtype=np.float64)
            t_travel = float(np.linalg.norm(ee - goal)) / max(self.robot_speed_max_m_s, 1e-6)
        return min(t_travel, self.max_aim_ahead_s)

    def _intercept_lead_s(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        vel: np.ndarray,
        goal_z: float | None = None,
    ) -> float:
        """Far-chase lead from *now*: travel time + ``robot_approach_pad_s``.

        Same for every Δ (pass ``obj`` = object-now). No extra ``dt``: the old
        Δ=0-only ``+dt`` aimed one tick further ahead than the delayed path.
        Far → look-ahead cap + pad. On top of the object → collapses to the pad
        so the hand rides instead of running a fixed 0.25 s ahead.
        """
        return self._intercept_travel_s(ee, obj, vel, goal_z=goal_z) + float(
            self.robot_approach_pad_s
        )

    def _grasp_gate_ok(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        wrist_depth_m: float | None = None,
        *,
        goal_z: float | None = None,
    ) -> tuple[bool, float, float]:
        """True when the hand is inside the grasp box (ready to close).

        XY: |dx|, |dy| vs ``grasp_xyz_ok``. Z: vs ``goal_z`` (z_grasp / p̂z)
        with tol = ``grasp_xyz_ok[2]`` (floor 2 mm). 
        Do not clamp with claim ``z_slack/2`` — that made a 15 mm yaml gate into 2.5 mm and missed near-hits at ≥30 cm/s.
        Never uses wrist depth alone as the Z gate
        (that caused early closes). ``wrist_depth_m`` is logs-only.
        """
        del wrist_depth_m
        gz = float(obj[2] if goal_z is None else goal_z)
        err_dx = abs(float(ee[0] - obj[0]))
        err_dy = abs(float(ee[1] - obj[1]))
        err_z = abs(float(ee[2]) - gz)
        err_xy = float(np.hypot(err_dx, err_dy))
        xy_ok = err_dx <= float(self.grasp_xyz_ok_m[0]) and err_dy <= float(
            self.grasp_xyz_ok_m[1]
        )
        # Honor yaml Z; floor 2 mm so a zero knob cannot freeze the expert.
        z_tol = max(float(self.grasp_xyz_ok_m[2]), 0.002)
        low_enough = err_z <= z_tol
        return xy_ok and low_enough, err_xy, err_z

    def _plausible_z_top(self, z_top: float, ee_z: float) -> bool:
        """True if a depth/GT top height is usable (not under table / above the hand)."""
        if not np.isfinite(z_top):
            return False
        z_lo = float(self._z_latch.z_table) - 0.01
        # Top must sit between table and roughly the hand (depth bugs go negative).
        return z_lo < float(z_top) < float(ee_z) + 0.05

    def _depth_z_top(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        wrist_depth: np.ndarray | None,
        ee_quat: np.ndarray | None,
        static_depth: np.ndarray | None,
        static_cam_z: float | None,
        wrist_cam_z: float | None,
    ) -> float:
        """One fused object-top Z from wrist (+ optional static). NaN if unusable.

        Prefers live wrist cam world Z. Rejects tops that disagree with GT by more
        than ~1.5 radii (sim sanity). Static is backup only — its formula is rough.
        """
        from interfaces.grasp_height import (
            fuse_z_top,
            z_top_from_gt_center,
            z_top_from_static_depth,
            z_top_from_wrist_cloud,
        )

        ee_z = float(ee[2])
        wrist_z = float("nan")
        if wrist_depth is not None and ee_quat is not None:
            # Whole-image lift around the object XY (not a fixed finger window):
            # sees the object from the start of the approach (A4).
            f_px, cx, cy = self._wrist_K
            wrist_z = z_top_from_wrist_cloud(
                wrist_depth,
                ee,
                ee_quat,
                obj[:2],
                f_px=f_px,
                cx=cx,
                cy=cy,
                z_table=float(self._z_latch.z_table),
                tcp_offset_m=self._tcp_offset_m,
                wrist_pos_m=self._wrist_pos_m,
                wrist_quat_wxyz=self._wrist_quat,
                z_bias_m=float(self.depth_z_bias_m),
            )
            if not self._plausible_z_top(wrist_z, ee_z):
                wrist_z = float("nan")
        static_z = float("nan")
        if static_depth is not None and static_cam_z is not None:
            static_z = z_top_from_static_depth(static_depth, cam_z=float(static_cam_z))
            if not self._plausible_z_top(static_z, ee_z):
                static_z = float("nan")
        gt_top = z_top_from_gt_center(float(obj[2]), half_height_m=float(self._obj_half_h))
        # Wrist preferred: static alone was latching ~18 cm mid-air closes.
        if np.isfinite(wrist_z) and np.isfinite(static_z):
            gap = ee_z - float(wrist_z)
            z_top = fuse_z_top(
                wrist_z,
                static_z,
                gap_m=gap,
                latch_hi_m=float(self._z_latch.latch_hi_m),
            )
        elif np.isfinite(wrist_z):
            z_top = float(wrist_z)
        else:
            z_top = float("nan")
        if not self._plausible_z_top(z_top, ee_z):
            return float("nan")
        if abs(float(z_top) - gt_top) > max(0.02, 1.5 * float(self._obj_half_h)):
            return float("nan")
        return float(z_top)

    def _update_z_grasp(
        self,
        ee: np.ndarray,
        obj: np.ndarray,
        wrist_depth: np.ndarray | None = None,
        ee_quat: np.ndarray | None = None,
        static_depth: np.ndarray | None = None,
        static_cam_z: float | None = None,
        wrist_cam_z: float | None = None,
    ) -> float:
        """Refresh expert aim Z and pack p̂z; return the aim height.

        * Expert aim (``z_grasp``): object centre when ``close_at_object_center``.
          DiffIK TCP = pad pinch (``tcp_offset`` Z = 103.4 mm), so commanding
          ``z_grasp`` puts the pinch on the object centre — not the old 107 mm TCP.
        * Pack (``p_hat_z``): wrist-depth latch when ``use_object_height_from_depth``
          (gap ≥ 7 cm collect / lock below) so training sees estimated height, not GT.
        """
        center = float(obj[2])
        ee_z = float(ee[2])
        z_top = self._depth_z_top(
            ee, obj, wrist_depth, ee_quat, static_depth, static_cam_z, wrist_cam_z
        )
        if np.isfinite(z_top):
            depth_zg = self._z_latch.update(z_top=z_top, ee_z=ee_z)
        elif bool(getattr(self, "oracle_z_top", False)):
            # B1 oracle: delayed GT centre+half-height is the intended z_top
            # channel (same as delayed XY/vel), not a sensor fallback.
            from interfaces.grasp_height import z_top_from_gt_center

            gt_top = z_top_from_gt_center(center, half_height_m=float(self._obj_half_h))
            depth_zg = self._z_latch.update(z_top=gt_top, ee_z=ee_z)
        else:
            # No good depth: GT top keeps the latch alive — counts as fallback.
            from interfaces.grasp_height import z_top_from_gt_center

            self.gt_fallback["z_top"] = int(self.gt_fallback.get("z_top", 0)) + 1
            gt_top = z_top_from_gt_center(center, half_height_m=float(self._obj_half_h))
            depth_zg = self._z_latch.update(z_top=gt_top, ee_z=ee_z)

        if self.use_object_height_from_depth and np.isfinite(depth_zg):
            # Close height from top − insert ≈ centre when insert ≈ half-height.
            self.p_hat_z = float(depth_zg)
        else:
            self.p_hat_z = center

        if self.close_at_object_center:
            self.z_grasp = center
        else:
            self.z_grasp = float(self.p_hat_z)
        return float(self.z_grasp)

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

    def _xyz_close(self, a: np.ndarray, b: np.ndarray, axes: tuple[int, ...] = (0, 1, 2)) -> bool:
        """True when every listed axis of ``a`` and ``b`` is inside ``arrived_m``.

        One number gates X, Y, and Z together. Pass ``axes=(0, 1)`` for XY only.
        """
        d = np.abs(np.asarray(a, dtype=np.float64).reshape(-1)[:3] - np.asarray(b, dtype=np.float64).reshape(-1)[:3])
        return bool(np.all(d[list(axes)] < self.arrived_m))

    def _at_home(self, ee: np.ndarray) -> bool:
        """True when the hand is inside ``home_radius_m`` of the start pose.

        Callers also require the object to still be in the bowl before the episode ends.
        """
        home = self.init_pose[:3]
        return float(np.linalg.norm(np.asarray(ee[:3], dtype=np.float64) - home)) < self.home_radius_m

    def _clear_z(self) -> float:
        """Height to clear the bowl before moving sideways: rim plus ``move_up_m``.

        The drop point sits ``release_below_rim_m`` under the rim, so the rim is
        that gap above the drop height.
        """
        if self._drop_pose is None:
            return float(self.init_pose[2])
        rim = float(self._drop_pose[2]) + float(self.release_below_rim_m)
        return rim + float(self.move_up_m)

    def _home_target(self, ee: np.ndarray) -> np.ndarray:
        """After the drop: rise above the rim (XY held), then go to the start pose.

        Sideways motion while the fingers are still in the bowl sweeps the object out.
        Once the hand is within ``arrived_m`` of the rise height, head home.
        The crossing flag stays on so a small dip does not send the hand back up.
        """
        if self._home_anchor is None:
            self._home_anchor = ee.copy()
        top = self._clear_z()
        if not self._home_crossing and float(ee[2]) < top - self.arrived_m:
            t = self._home_anchor.copy()
            t[2] = top
            return t
        self._home_crossing = True
        return self.init_pose[:3].copy()

    def _in_container(self, obj: np.ndarray) -> bool:
        """True when the object's XY is over the box. This is the "it landed" check for stages 7 and 8."""
        return float(np.linalg.norm(obj[:2] - self.carry_pose[:2])) < self.place_xy_tol

    # ------------------------------------------------------------------ core state machine logic 
    def step(
        self,
        state: ObjectState,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        wrist_depth_m: float | None = None,
        wrist_depth_img: np.ndarray | None = None,
        static_depth_img: np.ndarray | None = None,
        static_cam_z: float | None = None,
        wrist_cam_z: float | None = None,
        *,
        delta_meas_s: float | None = None,
        forecast_off: bool = False,
        ee_vel: np.ndarray | None = None,
    ) -> StateMachineCommand:
        """Advance one control step. Returns the EE command for this tick.

        ``state`` must be the delayed observation pack for B1 (GT at t-Δ), not
        live GT. Live proprio (``ee_pos`` / ``ee_quat`` / ``ee_vel``) is fine.

        ``delta_meas_s``: Δ_meas (F1: measured frame age + η). The only Δ the
        expert sees: p_now = p_delayed + v·Δ_meas (F2) for chase, ride, gate
        and reach stop (do not add Δ again on top of travel). Ride aim = p_now.
        ``forecast_off``: aim at the delayed position p.
        ``ee_vel``: Isaac EE-link world velocity for the close-speed gate
        (preferred). Falls back to FD of ``ee_pos`` only when omitted (unit tests).
        """
        ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        ee_q = np.asarray(ee_quat, dtype=np.float64).reshape(4)
        obj = np.asarray(state.position, dtype=np.float64).reshape(3)
        vel = np.asarray(state.velocity, dtype=np.float64).reshape(3)
        speed = float(np.linalg.norm(vel))
        prev_stage = self.stage
        # Close-gate hand speed: Isaac EE link vel when given; else FD (tests).
        if ee_vel is not None:
            v_hand = np.asarray(ee_vel, dtype=np.float64).reshape(3)
        else:
            v_hand = None if self._prev_ee is None else (ee - self._prev_ee) / max(self.dt, 1e-6)
        self._prev_ee = ee.copy()
        self.aim_branch = "none"  # set below only if a chase aim is computed
        self.p_now = np.full(3, np.nan)  # logged object-now (chase ticks only)
        self._ee_hist.append(ee.copy())
        del self._ee_hist[:-90]  # keep 3 s at 30 Hz (Δ ≤ 300 ms needs 10)
        # Hand when the observed frame was taken (Δ_meas ago; live when no delay).
        k_obs = 0 if delta_meas_s is None else int(round(float(delta_meas_s) / self.dt))
        ee_obs = self._ee_hist[-1 - min(k_obs, len(self._ee_hist) - 1)]

        # Defaults each tick: pre-shape open (not fully open), object slides,
        # not held, not frozen. Stages below override only what they need.
        grip, kin, attach, freeze = float(self.pre_shape_open), True, False, False
        target = ee.copy()  # default = stay put
        quat = self._down_quat.copy()
        # Stage 0 is not a pause. The chase runs on this same tick.
        if self.stage == STAGE_RESET and not self.hold_home:
            self.stage = STAGE_APPROACH
        if self.grasp_yaw_mode == "along" and self.stage in (STAGE_APPROACH, STAGE_CLOSE, STAGE_LIFT):
            if self.stage == STAGE_APPROACH and speed > 0.02:
                self._grasp_quat = _yaw_about_z(self._down_quat, float(np.arctan2(vel[1], vel[0])))
            if self._grasp_quat is not None:
                quat = self._grasp_quat.copy()

        self.timer_s += self.dt

        # Stage 0 only when hold_home is set (diag --miss): command home, do not chase.
        if self.stage == STAGE_RESET:
            target = self.init_pose[:3].copy()

        elif self.stage == STAGE_APPROACH:
            # Chase: aim XY where the object will be; Z = latched z_grasp (p̂z).
            # Height lives in p̂, not in a fixed desk height. Close when the gate passes.
            #
            # If the object left the catch box (past keepout / off table), do
            # not chase into ee-off-table — go home and let success label it.
            # Catch box ≠ aim inset (far-rim spawn must still be chaseable).
            if self._object_left_catch_box(obj):
                logger.info(
                    "    object left catch box at (%.3f, %.3f); going home",
                    float(obj[0]),
                    float(obj[1]),
                )
                self.stage = STAGE_HOME
                self.timer_s = 0.0
                target = self._home_target(ee)
            elif self._object_in_base_zone(obj, vel, delta_meas_s):
                # Object is inside the base circle: stop the chase and go home.
                logger.info(
                    "    object inside %.0f cm of the base at (%.3f, %.3f); going home",
                    100.0 * self.robot_base_safe_zone_m,
                    float(obj[0]),
                    float(obj[1]),
                )
                self.reach_stopped = True
                self.stage = STAGE_HOME
                self.timer_s = 0.0
                target = self._home_target(ee)
            else:
                target_z = self._update_z_grasp(
                    ee,
                    obj,
                    wrist_depth=wrist_depth_img,
                    ee_quat=ee_q,
                    static_depth=static_depth_img,
                    static_cam_z=static_cam_z,
                    wrist_cam_z=wrist_cam_z,
                )
                # Never aim tips through the table (ride-contact avocado/peach dives).
                z_floor = float(self._z_latch.z_table) + float(self.min_aim_z_above_table)
                if target_z < z_floor:
                    target_z = z_floor
                # Delayed obs is at t−Δ. p_now = p + v·Δ_meas (F2): one Δ for
                # chase, ride and gate; the real robot has only the measured age.
                # Sign: e = hand − object; along-track + = hand ahead of object.
                if bool(forecast_off) or delta_meas_s is None:
                    obj_now = obj
                else:
                    obj_now = self._forecast(obj, vel, float(delta_meas_s))
                self.p_now = np.asarray(obj_now, dtype=np.float64).copy()
                near_ok, err_xy, err_z = self._grasp_gate_ok(
                    ee, obj_now, wrist_depth_m, goal_z=target_z
                )
                # Age of the observation (0 when live). Only used for the lead
                # log line — the aim below has no Δ fork: obj_now already
                # unwraps Δ once, so Δ=0 and Δ>0 run the same formulas.
                obs_age_s = 0.0 if delta_meas_s is None else float(delta_meas_s)
                # Ride: reference is p_now (unwrap Δ once). Settle aim = p_now,
                # no hand-written +dt: with DiffIK FF (v̂_obj) a +v·dt aim parks
                # the hand v·dt/(1+ride_k) ahead — outside the 2 mm close box
                # (C30 +3.5 mm ride lead). An older bare-p_now run trailed
                # −5 mm (before FF); if it trails again, tune the FF τ, not a
                # tick here. Far chase: p_now + v·(travel + pad). Do NOT add Δ
                # again. forecast_off → delayed p.
                # ``aim_branch`` is logged per tick (code-path flag, not a
                # post-hoc label).
                if bool(forecast_off):
                    self.aim_branch = "forecast_off"
                    lead_s = 0.0
                    aim = np.array(obj, dtype=np.float64).copy()
                elif near_ok:
                    self.aim_branch = "settle"
                    lead_s = obs_age_s
                    aim = np.array(obj_now, dtype=np.float64).copy()
                else:
                    self.aim_branch = "far_chase"
                    lead_from_now = self._intercept_lead_s(
                        ee, obj_now, vel, goal_z=target_z
                    )
                    aim = np.array(
                        self._forecast(obj_now, vel, lead_from_now),
                        dtype=np.float64,
                    )
                    lead_s = obs_age_s + float(lead_from_now)
                if near_ok and not bool(forecast_off):
                    # gap · û > 0 ⇒ object ahead of hand (hand behind).
                    gap = obj_now[:2] - ee[:2]
                    inside_close = (
                        abs(float(gap[0])) <= float(self.close_xy_ok_m)
                        and abs(float(gap[1])) <= float(self.close_xy_ok_m)
                    )
                    # Inside the close box: aim = object_now (no hold). The old
                    # hold (aim = hand) froze the entry offset (C30 v5: egg
                    # +2.4 mm); with joint velocity FF the hand no longer hunts.
                    if not inside_close and self.ride_k > 0.0:
                        # Soft pull while still outside the close box.
                        aim[:2] = aim[:2] + self.ride_k * gap
                aim = self._clip_aim_xy(aim)
                target = np.array([aim[0], aim[1], target_z], dtype=np.float64)

                err_dx = abs(float(ee[0] - obj_now[0]))
                err_dy = abs(float(ee[1] - obj_now[1]))
                # C30 pre-check: score |Δv| only in the tight ride (≤ 3× close_xy),
                # not while still catching up inside the larger chase box.
                if (
                    near_ok
                    and v_hand is not None
                    and err_dx <= 3.0 * float(self.close_xy_ok_m)
                    and err_dy <= 3.0 * float(self.close_xy_ok_m)
                ):
                    self.ride_dv_cms.append(
                        100.0 * float(np.linalg.norm(v_hand[:2] - vel[:2]))
                    )
                # Close when the hand is inside the 2 mm box. The 10 mm box already
                # put the hand on the object's speed, so this is the only close line.
                settled = (
                    near_ok
                    and err_dx <= float(self.close_xy_ok_m)
                    and err_dy <= float(self.close_xy_ok_m)
                    and err_z <= float(self.close_z_ok_m)
                )
                if settled:
                    logger.info(
                        "    grasp gate: XY %.1f cm, Z %.1f cm, speed %.0f cm/s, "
                        "settle lead %.0f ms",
                        100.0 * err_xy,
                        100.0 * err_z,
                        100.0 * speed,
                        1000.0 * lead_s,
                    )
                    self.grasp_err_xy = err_xy
                    self.grasp_err_z = err_z
                    self.grasp_speed = speed
                    self.grasp_xy = obj[:2].copy()
                    self.close_xy_err_m = float(err_xy)
                    self.close_z_err_m = float(err_z)
                    if v_hand is not None:
                        self.dv_at_close_m_s = float(np.linalg.norm(v_hand[:2] - vel[:2]))
                    # Tip lag ≈ how far the hand trails / speed (when moving).
                    # Ahead sign: (hand − obj)·û > 0 ⇒ hand ahead; lag = −ahead.
                    if speed > 0.02 and v_hand is not None:
                        vhat = vel[:2] / max(speed, 1e-6)
                        ahead = float(np.dot((ee[:2] - obj_now[:2]), vhat))
                        self.t_lag_meas_s = (-ahead) / speed
                    self.lift_z = float(ee[2]) + self.lift_height_m
                    self.stage = STAGE_CLOSE
                    self.timer_s = 0.0
                    # Same-tick CLOSE: elif below will not run this frame, so shut
                    # the command now (else the log still shows pre-shape open).
                    grip = 0.0
                    # Hold hand XY (FF coasts with v̂_obj). A base_tau forecast
                    # snap here shoved s1003 ~2 mm off and the egg slipped.
                    z_close = float(self.z_grasp) if self.z_grasp is not None else float(obj[2])
                    target = np.array([ee[0], ee[1], z_close], dtype=np.float64)
                    if self.physics_grasp:
                        attach, kin = False, True
                    else:
                        attach, kin = True, False

        # Stage 3 — CLOSE FINGERS: shut the gripper. ``attach`` means "grasp now":
        # weld mode glues the object to the hand; physics mode frees it to PhysX
        # and the fingers squeeze it. Same flag, the scene decides (sim/scene.py).
        elif self.stage == STAGE_CLOSE:
            # Hold hand XY; DiffIK FF keeps ~v_obj while the fingers shut.
            # Do not snap to forecast(base_tau) — that jerked s1003 off-center.
            grip = 0.0
            z_close = float(self.z_grasp) if self.z_grasp is not None else float(obj[2])
            target = np.array([ee[0], ee[1], z_close], dtype=np.float64)
            if self.physics_grasp and self.timer_s < self.finger_close_s:
                # Keep the scripted coast until pads are closed. Freeing on tick 0
                # left a 50 cm/s shear and every close slipped at ~10 cm.
                attach, kin = False, True
            else:
                attach, kin = True, False
            if self.timer_s >= self.close_gripper_s:
                self.stage = STAGE_LIFT
                self.timer_s = 0.0

        # Physics grasp can lose the object. Held but far from the hand → go home.
        # No re-chase: the episode has one clock (episode_max_s).
        # Delayed obs: a held object trails the live hand by v_hand·Δ (10–25 cm at
        # Δ = 300 ms while carrying), so compare with the hand at the frame's time.
        elif self.stage in (STAGE_LIFT, STAGE_CARRY, STAGE_LOWER) and (
            float(np.linalg.norm(obj - ee_obs)) > self.robot_distance_to_object_m
        ):
            gap_cm = 100.0 * float(np.linalg.norm(obj - ee_obs))
            logger.info("    object slipped (%.0f cm from hand); going home", gap_cm)
            self.stage = STAGE_HOME
            self.timer_s = 0.0
            target = self._home_target(ee)

        # Stages 4-6 (lift, carry, lower): the object is held, so no table slide.
        # Stage 4 — LIFT: straight up from the grab spot.
        elif self.stage == STAGE_LIFT:
            grip, attach, kin = 0.0, True, False
            if self.lift_z is None:
                self.lift_z = float(ee[2]) + self.lift_height_m
            # Straight up from where the hand was when LIFT began. Snapping XY to
            # grasp_xy (object at CLOSE) sheared s1003: free-on-contact left a ~9 mm
            # pinch offset, then the hand yanked back to the old XY and the egg
            # slipped. Tracking the live hand instead had no XY pull at all, so any
            # sideways drift was accepted (15 mm, C30 s1001). Anchor = no jump, and
            # still a restoring pull.
            if self._lift_anchor is None:
                self._lift_anchor = ee.copy()
            target = self._lift_anchor.copy()
            target[2] = self.lift_z
            if self._xyz_close(ee, target):
                self.stage = STAGE_CARRY
                self.timer_s = 0.0

        # Stage 5 — CARRY: fly high, over the box.
        elif self.stage == STAGE_CARRY:
            grip, attach, kin = 0.0, True, False
            target = self.carry_pose[:3].copy()
            target[2] = self._retract_z()
            quat = self.carry_pose[3:].copy()
            if self._xyz_close(ee, target):
                self.stage = STAGE_LOWER
                self.timer_s = 0.0

        # Stage 6 — LOWER: come down inside the box.
        elif self.stage == STAGE_LOWER:
            grip, attach, kin = 0.0, True, False
            target = self._drop_target()
            quat = self.carry_pose[3:].copy()
            if self._xyz_close(ee, target):
                self.stage = STAGE_OPEN
                self.timer_s = 0.0

        # Stage 7 — OPEN FINGERS: let go over the box. Move on once the fingers
        # had time to open AND the object sits over the box.
        elif self.stage == STAGE_OPEN:
            # Release. Weld mode pins the object; physics mode lets it drop.
            # Hold the hand where the open began: pushing on toward the drop
            # point while the fingers part shoves the object sideways.
            grip, attach, kin, freeze = 1.0, False, False, True
            if self._open_anchor is None:
                self._open_anchor = ee.copy()
            target = self._open_anchor.copy()
            quat = self.carry_pose[3:].copy()
            # Stay put for release_hold_s (yaml 300 ms) so the object can leave the fingers.
            if self.timer_s >= self.release_hold_s and self._in_container(obj):
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

        # Anchors live only while their stage runs.
        if self.stage != STAGE_OPEN:
            self._open_anchor = None
        if self.stage != STAGE_LIFT:
            self._lift_anchor = None
        if self.stage != STAGE_HOME:
            self._home_anchor = None
            self._home_crossing = False

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
                    100.0 * float(self.grasp_xyz_ok_m[0]),
                    100.0 * float(self.grasp_xyz_ok_m[1]),
                    100.0 * float(self.grasp_xyz_ok_m[2]),
                )

        # DiffIK FF: pass v̂_obj (observed/coast object velocity — same vector
        # ``_forecast`` integrates). Never FD of aim. Fixed poses → v=0.
        # Only while the object moves on its own (chase, and CLOSE until the
        # attach). Once held, v̂_obj IS the hand velocity: feeding it forward
        # made the arm aim 1/K ahead of its own motion — carry overshot to 31 cm
        # and dove to 8 cm into the bowl (C30 s1000/s1001).
        if self.stage == STAGE_APPROACH:
            cmd_vel = vel.copy()  # v̂_obj
        elif self.stage == STAGE_CLOSE and kin and not attach:
            # XY only: the squeeze pops the object up, and a z FF fed that pop
            # back into the hand (pop 13.5 mm at Δ0 vs 2.4 mm at Δ300, v8).
            cmd_vel = np.array([vel[0], vel[1], 0.0], dtype=np.float64)
        else:
            cmd_vel = np.zeros(3, dtype=np.float64)

        # kinematic only when not holding: a held object follows the hand instead.
        return StateMachineCommand(
            position=target,
            quat_wxyz=quat,
            gripper_open=float(grip),
            stage=int(self.stage),
            kinematic_object=bool(kin and not attach),
            attach_object=bool(attach),
            freeze_object=bool(freeze),
            velocity=cmd_vel,
        )
