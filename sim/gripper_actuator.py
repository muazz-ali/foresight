"""Physical Franka Hand close model for B1 (SPEC-DERIVED).

Plain words: fingers do not teleport shut. Each finger moves at most 50 mm/s,
the command waits 30 ms, then pads squeeze for ~50 ms of force build-up.

  t_close ≈ t_cmd + s / v_finger + t_force
  nominal: 30 + 5/50 + 50 = 180 ms  (s = 5 mm per finger from w_pre)

Use ``GripperActuator`` in the scene instead of snapping joints when
``robot.gripper_actuator.enabled`` is true. Log width vs time to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Spec + assumed latencies (replace with measurements on the real Hand).
DEFAULT_V_FINGER_M_S = 0.050  # 50 mm/s per finger
DEFAULT_T_CMD_S = 0.030
DEFAULT_T_FORCE_S = 0.050
DEFAULT_SLACK_PER_FINGER_M = 0.005  # s in w_pre = w_obj + 2·s


def nominal_t_contact_s(
    *,
    travel_per_finger_m: float = DEFAULT_SLACK_PER_FINGER_M,
    v_finger_m_s: float = DEFAULT_V_FINGER_M_S,
    t_cmd_s: float = DEFAULT_T_CMD_S,
) -> float:
    """Time from close command to pad contact (no force ramp).

    Nominal: 30 ms cmd + 5 mm / 50 mm/s = **130 ms**. Use this in the
    predictive close gate (``e + Δv·t_contact``), not the full 180 ms squeeze.
    """
    v = float(v_finger_m_s)
    if v <= 0.0:
        raise ValueError(f"v_finger_m_s must be > 0, got {v}")
    return float(t_cmd_s) + float(travel_per_finger_m) / v


def nominal_t_close_s(
    *,
    travel_per_finger_m: float = DEFAULT_SLACK_PER_FINGER_M,
    v_finger_m_s: float = DEFAULT_V_FINGER_M_S,
    t_cmd_s: float = DEFAULT_T_CMD_S,
    t_force_s: float = DEFAULT_T_FORCE_S,
) -> float:
    """Expected close time from w_pre to contact+squeeze (≈180 ms nominal)."""
    return nominal_t_contact_s(
        travel_per_finger_m=travel_per_finger_m,
        v_finger_m_s=v_finger_m_s,
        t_cmd_s=t_cmd_s,
    ) + float(t_force_s)


@dataclass
class GripperActuator:
    """Rate-limited finger width with command delay and force ramp.

    Call ``set_command(open_m)`` when the expert changes the open width.
    Call ``step(dt)`` every physics/control tick; read ``width_m``.
    """

    v_finger_m_s: float = DEFAULT_V_FINGER_M_S
    t_cmd_s: float = DEFAULT_T_CMD_S
    t_force_s: float = DEFAULT_T_FORCE_S
    width_open_m: float = 0.080
    width_closed_m: float = 0.0
    width_m: float = 0.080
    _cmd_width_m: float = 0.080
    _pending_cmd_m: float | None = None
    _cmd_delay_left_s: float = 0.0
    _force_left_s: float = 0.0
    _closing: bool = False
    log_t_s: list[float] = field(default_factory=list)
    log_width_m: list[float] = field(default_factory=list)
    log_cmd_m: list[float] = field(default_factory=list)
    _sim_t: float = 0.0
    _close_start_t: float | None = None
    _contact_t: float | None = None
    _close_done_t: float | None = None

    @classmethod
    def from_cfg(cls, cfg: dict) -> GripperActuator:
        """Build from ``robot.gripper_actuator`` (+ open/close widths).

        Yaml is SI: ``gripper_open`` is one-finger width (full gap = 2×),
        ``v_finger_m_s`` / ``t_cmd_s`` / ``t_force_s`` on the actuator block.
        """
        robot = cfg.get("robot") if isinstance(cfg, dict) else {}
        robot = robot if isinstance(robot, dict) else {}
        ga = robot.get("gripper_actuator") if isinstance(robot.get("gripper_actuator"), dict) else {}
        # gripper_open is one finger; actuator tracks full gap.
        open_m = 2.0 * float(robot.get("gripper_open", 0.040))
        close_m = float(robot.get("gripper_close", 0.0))
        return cls(
            v_finger_m_s=float(ga.get("v_finger_m_s", DEFAULT_V_FINGER_M_S)),
            t_cmd_s=float(ga.get("t_cmd_s", DEFAULT_T_CMD_S)),
            t_force_s=float(ga.get("t_force_s", DEFAULT_T_FORCE_S)),
            width_open_m=float(open_m),
            width_closed_m=float(close_m),
            width_m=float(open_m),
            _cmd_width_m=float(open_m),
        )

    def reset(self, width_m: float | None = None) -> None:
        w = float(self.width_open_m if width_m is None else width_m)
        self.width_m = w
        self._cmd_width_m = w
        self._pending_cmd_m = None
        self._cmd_delay_left_s = 0.0
        self._force_left_s = 0.0
        self._closing = False
        self.log_t_s.clear()
        self.log_width_m.clear()
        self.log_cmd_m.clear()
        self._sim_t = 0.0
        self._close_start_t = None
        self._contact_t = None
        self._close_done_t = None

    def set_command(self, width_m: float) -> None:
        """Request a new gap (metres). Closing starts the cmd-delay timer."""
        w = float(np.clip(width_m, self.width_closed_m, self.width_open_m))
        if abs(w - self._cmd_width_m) < 1e-6 and self._pending_cmd_m is None:
            return
        # Narrower = closing.
        if w < self.width_m - 1e-6:
            self._pending_cmd_m = w
            self._cmd_delay_left_s = float(self.t_cmd_s)
            if self._close_start_t is None:
                self._close_start_t = float(self._sim_t)
            self._closing = True
        else:
            # Opening: apply after the same small cmd delay.
            self._pending_cmd_m = w
            self._cmd_delay_left_s = float(self.t_cmd_s)
            self._closing = False

    def step(self, dt: float) -> float:
        """Advance fingers by ``dt`` seconds; return measured gap (m)."""
        dt = float(dt)
        self._sim_t += dt
        if self._pending_cmd_m is not None:
            self._cmd_delay_left_s -= dt
            if self._cmd_delay_left_s <= 0.0:
                self._cmd_width_m = float(self._pending_cmd_m)
                self._pending_cmd_m = None
                self._cmd_delay_left_s = 0.0

        # Travel toward command at v_finger per finger → gap rate = 2·v.
        gap_rate = 2.0 * float(self.v_finger_m_s)
        err = float(self._cmd_width_m) - float(self.width_m)
        max_step = gap_rate * dt
        if abs(err) <= max_step:
            self.width_m = float(self._cmd_width_m)
            if self._closing and self._contact_t is None and abs(err) < 1e-5:
                # Reached the commanded (contact) width → start force ramp.
                self._contact_t = float(self._sim_t)
                self._force_left_s = float(self.t_force_s)
        else:
            self.width_m = float(self.width_m) + float(np.sign(err)) * max_step

        if self._contact_t is not None and self._close_done_t is None:
            self._force_left_s -= dt
            if self._force_left_s <= 0.0:
                self._close_done_t = float(self._sim_t)

        self.log_t_s.append(float(self._sim_t))
        self.log_width_m.append(float(self.width_m))
        self.log_cmd_m.append(float(self._cmd_width_m))
        return float(self.width_m)

    def measured_close_time_s(self) -> float | None:
        """Time from close command to end of force ramp, or None if unfinished."""
        if self._close_start_t is None or self._close_done_t is None:
            return None
        return float(self._close_done_t) - float(self._close_start_t)

    def open_fraction(self) -> float:
        """Map current width to expert-style open∈[0,1]."""
        span = float(self.width_open_m) - float(self.width_closed_m)
        if span <= 1e-9:
            return 0.0
        return float(
            np.clip(
                (float(self.width_m) - float(self.width_closed_m)) / span,
                0.0,
                1.0,
            )
        )

    def width_log_arrays(self) -> dict[str, np.ndarray]:
        """Columnar width-vs-time for the t_close verify."""
        return {
            "t_s": np.asarray(self.log_t_s, dtype=np.float64),
            "width_m": np.asarray(self.log_width_m, dtype=np.float64),
            "cmd_width_m": np.asarray(self.log_cmd_m, dtype=np.float64),
        }
