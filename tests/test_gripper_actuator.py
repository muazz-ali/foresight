"""SPEC-DERIVED Franka Hand close model (Isaac-free)."""

from __future__ import annotations

import pytest

from sim.gripper_actuator import GripperActuator, nominal_t_close_s


def test_nominal_t_close_is_180ms():
    assert nominal_t_close_s() == pytest.approx(0.180, abs=1e-9)


def test_close_from_w_pre_near_180ms():
    """5 mm per finger from w_pre → contact should land near 180 ms."""
    ga = GripperActuator()
    # Gap open = object 44 mm + 2×5 mm = 54 mm; close to 44 mm contact.
    ga.reset(width_m=0.054)
    ga.set_command(0.044)
    dt = 1.0 / 30.0
    for _ in range(30):
        ga.step(dt)
        if ga.measured_close_time_s() is not None:
            break
    t = ga.measured_close_time_s()
    assert t is not None
    # Allow one control step of quantization.
    assert abs(t - 0.180) <= dt + 0.02


def test_max_dv_at_180ms():
    from control.close_guard import max_speed_diff_m_s

    assert max_speed_diff_m_s(close_time_s=0.180) == pytest.approx(0.0025 / 0.180)
