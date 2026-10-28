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
    samp = DeltaMeasSampler(delta_true_s=0.0, noise_std_s=0.005, replan_period_s=0.33, seed=1)
    a = samp.update(0.0)
    b = samp.update(0.10)
    assert a == b  # held within cycle
    assert a >= 0.0
    c = samp.update(0.34)
    # May differ after replan; still ≥ 0.
    assert c >= 0.0


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
