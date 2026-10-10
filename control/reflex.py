"""Reflex close (Job 3): last ~10 cm, no learning.

Steer the hand at where the object will be. Close the gripper when the hand
will touch it sooner than the gripper needs to close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.state import ObjectState


@dataclass
class ReflexOut:
    active: bool  # False = object not near / not trusted: let the policy drive
    target: np.ndarray  # (3,) meeting point (world frame)
    vel_cmd: np.ndarray  # (3,) hand velocity command, m/s
    close: bool  # True = close the gripper now (latched until reset)
    tta: float  # time to arrival, seconds (inf if not closing in)


class ReflexClose:
    def __init__(self, close_time: float = 0.15, radius: float = 0.10, vmax: float = 0.5,
                 kp: float = 8.0, touch_dist: float = 0.015, max_lead: float = 0.5):
        """close_time: seconds the gripper needs to close. radius: reflex starts inside this.
        vmax: hand speed limit. touch_dist: close anyway if this near. max_lead: cap on look-ahead."""
        self.close_time, self.radius, self.vmax = close_time, radius, vmax
        self.kp, self.touch_dist, self.max_lead = kp, touch_dist, max_lead
        self.closed = False

    def reset(self) -> None:
        self.closed = False

    def step(self, obj: ObjectState, ee_pos, now: float, ee_vel=None) -> ReflexOut:
        ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        ev = np.zeros(3) if ee_vel is None else np.asarray(ee_vel, dtype=np.float64).reshape(3)
        zero = np.zeros(3)
        if not obj.valid:  # never act on a bad reading
            return ReflexOut(False, ee.copy(), zero, self.closed, np.inf)
        age = now - obj.timestamp
        if age < 0:
            raise ValueError(f"object timestamp {obj.timestamp} is after now {now}")
        p = obj.position + obj.velocity * age  # object now
        r = p - ee
        dist = float(np.linalg.norm(r))
        if dist > self.radius and not self.closed:
            return ReflexOut(False, p, zero, False, np.inf)

        # Meeting point: guess arrival time at full speed, look ahead that far, repeat.
        tau = 0.0
        for _ in range(3):
            meet = p + obj.velocity * tau
            tau = min(float(np.linalg.norm(meet - ee)) / self.vmax, self.max_lead)
        meet = p + obj.velocity * tau

        # Time to arrival: distance / closing speed (how fast the gap shrinks right now).
        rel_v = obj.velocity - ev
        closing = -float(r @ rel_v) / dist if dist > 1e-9 else 0.0
        tta = dist / closing if closing > 1e-6 else np.inf

        if dist <= self.touch_dist or tta < self.close_time:
            self.closed = True

        cmd = self.kp * (meet - ee) + obj.velocity  # PD toward meeting point + match its drift
        speed = float(np.linalg.norm(cmd))
        if speed > self.vmax:
            cmd *= self.vmax / speed
        return ReflexOut(True, meet, cmd, self.closed, tta)
