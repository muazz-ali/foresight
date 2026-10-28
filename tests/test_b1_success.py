"""Isaac-free: B1 success latch and failure labels."""

from __future__ import annotations

import numpy as np

from eval.b1_success import b1_success
from sim.state_machine import STAGE_APPROACH, STAGE_CARRY, STAGE_CLOSE, STAGE_HOME, STAGE_LIFT

DT = 1.0 / 30.0


def _episode(stages, obj, ee, grip):
    return {
        "stage": np.asarray(stages),
        "object_pos": np.asarray(obj, float),
        "ee_pos": np.asarray(ee, float),
        "gripper": np.asarray(grip, float),
        "object_vel": np.zeros((len(stages), 3)),
    }


def test_never_closed_then_home_is_never_closed():
    """Went home without a CLOSE: HOME (8) ≥ CLOSE (3) must not count as a close."""
    n = 40
    stages = [STAGE_APPROACH] * 20 + [STAGE_HOME] * 20
    obj = [[0.5, 0.0, 0.03]] * n
    ee = [[0.5, 0.0, 0.20]] * n  # hand 17 cm above when it gave up
    ok, mode = b1_success(_episode(stages, obj, ee, [1.0] * n), dt=DT)
    assert not ok and mode == "never_closed"


def test_clean_grasp_lift_hold_succeeds():
    n_close, n_lift, n_hold = 6, 10, 40
    stages = [STAGE_APPROACH] * 5 + [STAGE_CLOSE] * n_close + [STAGE_LIFT] * n_lift + [STAGE_CARRY] * n_hold
    n = len(stages)
    z = np.r_[np.full(5 + n_close, 0.03), np.linspace(0.03, 0.15, n_lift), np.full(n_hold, 0.15)]
    obj = np.c_[np.full(n, 0.5), np.zeros(n), z]
    ee = obj + [0.0, 0.0, 0.001]
    grip = [1.0] * 5 + [0.0] * (n - 5)
    ok, mode = b1_success(_episode(stages, obj, ee, grip), dt=DT)
    assert ok and mode is None
