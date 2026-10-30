"""B1 obs-delay buffer + Δ_meas + w_pre (Isaac-free)."""

from __future__ import annotations

import numpy as np
import pytest

from interfaces.state import oracle_from_gt
from sim.obs_delay import (
    DelayedFrame,
    DeltaMeasSampler,
    ObsDelayBuffer,
    finger_axis_width_m,
    w_pre_m,
)


def test_buffer_returns_nearest_capture():
    buf = ObsDelayBuffer(dt=1.0 / 30.0)
    for i in range(20):
        t = i / 30.0
        st = oracle_from_gt([float(i), 0, 0.05], [0.2, 0, 0], t)
        buf.push(
            DelayedFrame(
                t_capture=t,
                state=st,
                wrist_depth_m=0.3,
                wrist_depth_img=None,
                static_depth_img=None,
                static_cam_z=None,
                wrist_cam_z=None,
            )
        )
    now = 19 / 30.0
    delayed = buf.get_delayed(now, 0.200)
    # Nearest to now-0.2 ≈ frame at ~13/30.
    assert abs(delayed.t_capture - (now - 0.200)) <= (1.0 / 30.0) + 1e-9
    assert abs(buf.last_round_dt_s) <= (0.5 / 30.0) + 1e-9


def test_delta_meas_redraws_once_per_replan_and_clamps():
    samp = DeltaMeasSampler(noise_std_s=0.005, replan_period_s=0.33, seed=1)
    a = samp.update(0.0, 0.0)
    eta0 = samp.eta_s
    b = samp.update(0.10, 0.10)
    assert samp.eta_s == eta0  # η held within the cycle
    assert a == b and a >= 0.0  # same age (0) → same Δ_meas, clamped ≥ 0
    samp.update(0.34, 0.34)
    assert samp.eta_s != eta0  # redrawn after 0.33 s
    for k in range(200):  # clamp holds for any draw at age 0
        assert samp.update(0.34 * (k + 2), 0.34 * (k + 2)) >= 0.0


def test_delta_meas_equals_frame_age_when_eta_zero():
    """F1 with η = 0: Δ_meas = t_exec − t_capture, including startup ticks."""
    samp = DeltaMeasSampler(noise_std_s=0.0, seed=3)
    buf = ObsDelayBuffer(dt=1.0 / 30.0)
    delta = 0.300
    for i in range(40):
        now = i / 30.0
        st = oracle_from_gt([0.5, 0, 0.05], [0.3, 0, 0], now)
        buf.push(DelayedFrame(now, st, 0.3, None, None, None, None))
        fr = buf.get_delayed(now, delta)
        d = samp.update(now, fr.t_capture)
        assert d == pytest.approx(now - fr.t_capture, abs=1e-12)
        if i < 9:  # only frame 0 exists: age = now, younger than Δ
            assert fr.t_capture == 0.0 and d < delta
    assert d == pytest.approx(delta, abs=1e-9)


def test_delta_meas_adds_held_eta_to_age():
    samp = DeltaMeasSampler(noise_std_s=0.005, seed=7)
    d0 = samp.update(1.0, 0.8)
    eta = samp.eta_s
    assert d0 == pytest.approx(max(0.0, 0.2 + eta), abs=1e-12)
    d1 = samp.update(1.1, 0.8)  # same η, age grew by 0.1 s
    assert d1 == pytest.approx(max(0.0, 0.3 + eta), abs=1e-12)


def test_w_pre_clamp_at_80mm():
    wp, clamped, raw = w_pre_m([0.090, 0.090, 0.05], max_open_m=0.080, slack_m=0.005)
    assert clamped
    assert wp == pytest.approx(0.080)
    assert raw > 0.080


def test_finger_axis_width_rotates_with_yaw():
    # Long in X: at yaw=0 fingers along Y → width = Y extent.
    w0 = finger_axis_width_m([0.080, 0.040, 0.05], grasp_yaw_rad=0.0)
    w90 = finger_axis_width_m([0.080, 0.040, 0.05], grasp_yaw_rad=np.pi / 2)
    assert w0 == pytest.approx(0.040, abs=1e-9)
    assert w90 == pytest.approx(0.080, abs=1e-9)


def test_leak_audit_counts_outside_judge():
    buf = ObsDelayBuffer(dt=0.033)
    buf.note_live_gt_read("expert")
    assert buf.live_gt_leaks == 1
    buf.allow_live_gt(True)
    buf.note_live_gt_read("judge")
    assert buf.live_gt_leaks == 1
