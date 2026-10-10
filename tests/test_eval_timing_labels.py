"""Isaac-free: timing labels on synthetic hand / object tracks."""

from __future__ import annotations

import numpy as np
import pytest

from eval.timing_labels import NEAR_M, smoothness, timing_label

DT = 0.04
T = 50


def tracks(d_hand, closed_from=None, obj_z=None, open_again=None):
    """Object fixed at the origin (z=0.03); hand on the x axis at distance d_hand[i]."""
    d = np.asarray(d_hand, dtype=float)
    obj = np.zeros((len(d), 3))
    obj[:, 2] = 0.03 if obj_z is None else obj_z
    ee = obj.copy()
    ee[:, 0] += d
    grip = np.ones(len(d))
    if closed_from is not None:
        grip[closed_from:] = 0.0
    if open_again is not None:
        grip[open_again:] = 1.0
    return dict(ee_pos=ee, object_pos=obj, gripper=grip, holding=np.zeros(len(d)), dt=DT)


def label(**kw):
    kw.setdefault("success", False)
    return timing_label(**kw)


def v_shape(lo=0.01, hi=0.20, i_min=25):
    """Hand comes in to `lo` at frame i_min, then leaves."""
    i = np.arange(T)
    return lo + (hi - lo) * np.abs(i - i_min) / i_min


def test_success_wins():
    assert label(**tracks(v_shape(), closed_from=25), success=True)["timing_label"] == "success"


def test_never_closed_and_off_table():
    tr = tracks(np.full(T, 0.3))
    assert label(**tr)["timing_label"] == "never-closed"
    z = np.full(T, 0.03)
    z[30:] = -0.3  # fell off the table at frame 30
    out = label(**tracks(np.full(T, 0.3), obj_z=z))
    assert out["timing_label"] == "off-table" and out["off_table"]
    assert out["off_table_t"] == pytest.approx(30 * DT)


def test_aim_closed_but_never_near():
    out = label(**tracks(v_shape(lo=0.08), closed_from=25))
    assert out["timing_label"] == "aim"
    assert out["min_dist_3d_m"] == pytest.approx(0.08, abs=1e-6)


def test_early_shut_before_getting_near():
    # Closes at frame 10 (hand still 13 cm away), comes in to 1 cm closed.
    out = label(**tracks(v_shape(), closed_from=10))
    assert out["timing_label"] == "early"


def test_late_closed_after_closest_approach():
    # Closest at 25 (1 cm, open); closes at 28 (≈ 3.2 cm, moving away).
    out = label(**tracks(v_shape(), closed_from=28))
    assert out["timing_label"] == "late"
    assert out["close_minus_closest_s"] == pytest.approx(3 * DT)
    assert out["closest_t"] == pytest.approx(25 * DT)


def test_late_when_near_open_and_never_closed():
    out = label(**tracks(v_shape()))
    assert out["timing_label"] == "late"
    assert out["close_t"] is None


def test_short_close_on_time_but_failed():
    out = label(**tracks(v_shape(), closed_from=25))
    assert out["timing_label"] == "short-close"
    assert out["dist_at_close_m"] == pytest.approx(0.01)


def test_one_frame_jitter_is_not_late():
    d = v_shape()
    d[26] = d[25] + 0.004  # closes one frame after the min, only 4 mm farther
    assert label(**tracks(d, closed_from=26))["timing_label"] == "short-close"


def test_reopen_then_close_on_time_is_not_early():
    # Shut early (frame 5), re-opened at 15, then the policy closes on time at 25.
    tr = tracks(v_shape(), closed_from=5, open_again=15)
    tr["gripper"][25:] = 0.0
    assert label(**tr)["timing_label"] == "short-close"


def test_frames_after_fall_do_not_count():
    # Object falls at 20; the hand reaches the (fallen) object later: still off-table.
    z = np.full(T, 0.03)
    z[20:] = -0.3
    tr = tracks(v_shape(i_min=40), obj_z=z, closed_from=40)
    tr["ee_pos"][:, 2] = z + 0.0  # hand follows it down
    assert label(**tr)["timing_label"] == "off-table"


def test_near_constant_and_smoothness():
    assert NEAR_M == 0.05
    p = np.zeros((10, 3))
    p[:, 0] = 0.01 * np.arange(10)  # straight line, constant speed
    s = smoothness(p, DT)
    assert s["path_len_m"] == pytest.approx(0.09)
    assert s["jerk_rms"] == pytest.approx(0.0, abs=1e-6)
    p[5, 1] = 0.01  # one kink → jerk
    assert smoothness(p, DT)["jerk_rms"] > 0
    assert smoothness(p, DT, end=1)["jerk_rms"] is None
