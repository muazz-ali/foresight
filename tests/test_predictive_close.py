"""Predictive close gate: |e + Δv·t_contact| ≤ 2 mm and |Δv| ≤ 1.39 cm/s."""

from __future__ import annotations

import numpy as np
import pytest

from control.close_guard import predictive_close_check
from sim.gripper_actuator import nominal_t_close_s, nominal_t_contact_s


def test_t_contact_is_130ms_nominal():
    assert nominal_t_contact_s() == pytest.approx(0.130)
    assert nominal_t_close_s() == pytest.approx(0.180)


def test_static_matched_uses_current_gap():
    """v=0 → e_pred = e; 1 mm short + matched speed → close."""
    r = predictive_close_check(
        [0.549, 0.0], [0.550, 0.0], [0.0, 0.0], [0.0, 0.0], t_contact_s=0.130
    )
    assert r.ok
    np.testing.assert_allclose(r.e_pred_m, r.e_now_m)


def test_fast_approach_blocks_even_inside_2mm():
    """1 mm short but +5 cm/s approach → e_pred overshoots past 2 mm."""
    r = predictive_close_check(
        [0.549, 0.0],
        [0.550, 0.0],
        [0.05, 0.0],
        [0.0, 0.0],
        t_contact_s=0.130,
    )
    # e_now = -1 mm; e_pred = -1 + 50*0.13 = +5.5 mm → fail XY
    assert not r.ok_xy
    assert not r.ok
    assert abs(r.e_pred_m[0]) > 0.002


def test_predict_alone_cannot_close_outside_2mm_now():
    """Hand 3 mm short with approach that predicts 0 at contact → still wait."""
    # e=-3 mm, v=+23 mm/s → e_pred ≈ 0 at 130 ms, but |e_now| > 2 mm.
    r = predictive_close_check(
        [0.547, 0.0],
        [0.550, 0.0],
        [0.023, 0.0],
        [0.0, 0.0],
        t_contact_s=0.130,
    )
    assert abs(r.e_pred_m[0]) <= 0.002 + 1e-6
    assert not r.ok_xy
