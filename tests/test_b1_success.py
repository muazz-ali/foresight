"""Isaac-free: B1 success latch and failure labels."""

from __future__ import annotations

import numpy as np

from eval.b1_success import b1_success
from eval.b1_success import classify_b1
from sim.state_machine import (
    STAGE_APPROACH, STAGE_CARRY, STAGE_CLOSE, STAGE_DONE, STAGE_HOME, STAGE_LIFT, STAGE_OPEN,
)

DT = 1.0 / 30.0


CONT = [0.30, 0.50, 0.03]


def _episode(stages, obj, ee, grip, cont=CONT):
    return {
        "stage": np.asarray(stages),
        "object_pos": np.asarray(obj, float),
        "ee_pos": np.asarray(ee, float),
        "gripper": np.asarray(grip, float),
        "object_vel": np.zeros((len(stages), 3)),
        "container_pos": np.tile(np.asarray(cont, float), (len(stages), 1)),
    }


def test_never_closed_then_home_is_never_closed():
    """Went home without a CLOSE: HOME (8) ≥ CLOSE (3) must not count as a close."""
    n = 40
    stages = [STAGE_APPROACH] * 20 + [STAGE_HOME] * 20
    obj = [[0.5, 0.0, 0.03]] * n
    ee = [[0.5, 0.0, 0.20]] * n  # hand 17 cm above when it gave up
    ok, mode = b1_success(_episode(stages, obj, ee, [1.0] * n), dt=DT)
    assert not ok and mode == "never_closed"


def _grasp_episode(*, place=True, close_dz=0.001, slip_at=None, close_entries=1):
    """APPROACH → CLOSE → LIFT (to 15 cm) → CARRY → (place in box) → DONE."""
    pre = [STAGE_APPROACH] * 5
    if close_entries == 2:
        pre += [STAGE_CLOSE] * 3 + [STAGE_APPROACH] * 5
    n_close, n_lift, n_carry, n_end = 6, 10, 40, 10
    stages = pre + [STAGE_CLOSE] * n_close + [STAGE_LIFT] * n_lift + [STAGE_CARRY] * n_carry + [STAGE_OPEN] * 5 + [STAGE_DONE] * n_end
    n = len(stages)
    k0 = len(pre) + n_close
    z = np.r_[np.full(k0, 0.03), np.linspace(0.03, 0.15, n_lift), np.full(n - k0 - n_lift, 0.15)]
    obj = np.c_[np.full(n, 0.5), np.zeros(n), z]
    ee = obj + [0.0, 0.0, close_dz]
    if place:
        obj[-n_end:] = [CONT[0], CONT[1], 0.04]
        ee[-n_end:] = [CONT[0], CONT[1], 0.15]
    if slip_at is not None:
        obj[slip_at:, 2] -= 0.02
    grip = [1.0] * 5 + [0.0] * (n - 5)
    return _episode(stages, obj, ee, grip), k0


def test_lift_hold_place_succeeds():
    ep, _ = _grasp_episode()
    ok, mode = b1_success(ep, dt=DT)
    assert ok and mode is None
    assert classify_b1(ep, dt=DT)["first_attempt"]


def test_lifted_held_not_placed_fails():
    ep, _ = _grasp_episode(place=False)
    ok, mode = b1_success(ep, dt=DT)
    assert not ok and mode == "not_placed"


def test_closed_above_is_diagnostic_only():
    """Pinch 8 mm above centre at close, but lifted + held + placed → success."""
    ep, _ = _grasp_episode(close_dz=0.008)
    ok, mode = b1_success(ep, dt=DT)
    assert ok and mode is None
    assert classify_b1(ep, dt=DT)["diag_closed_above"]


def test_slip_during_hold_from_lift_fails():
    ep, k0 = _grasp_episode(slip_at=None)
    ep2, k0 = _grasp_episode(slip_at=k0 + 12)  # 2 cm drop inside the 1 s window
    ok, mode = b1_success(ep2, dt=DT)
    assert not ok and mode == "slipped"


def test_retry_success_is_not_first_attempt():
    ep, _ = _grasp_episode(close_entries=2)
    ok, _ = b1_success(ep, dt=DT)
    out = classify_b1(ep, dt=DT)
    assert ok and out["n_close_entries"] == 2 and not out["first_attempt"]
