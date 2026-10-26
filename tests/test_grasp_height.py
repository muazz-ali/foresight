"""Isaac-free tests for collect-then-lock grasp height."""

from __future__ import annotations

import numpy as np
import pytest

from interfaces.grasp_height import ZLatch, z_grasp_m


def test_z_grasp_floors_at_table_clear():
    # Short object: table + clear wins over top − insert.
    assert z_grasp_m(0.008, z_table=0.0, insert_m=0.01, clear_m=0.005) == pytest.approx(0.005)
    assert z_grasp_m(0.080, z_table=0.0, insert_m=0.01, clear_m=0.005) == pytest.approx(0.070)


def test_collect_above_hi_lock_below_lo():
    """≥20 cm: keep samples; <10 cm: freeze to median; mid: no new samples."""
    z = ZLatch(insert_m=0.01, clear_m=0.005, latch_lo_m=0.10, latch_hi_m=0.20)
    z_top = 0.08
    # High: collect noisy tops → z_grasp = z_top - 0.01
    for zt in (0.079, 0.081, 0.080, 0.082, 0.078):
        out = z.update(z_top=zt, ee_z=zt + 0.25)
        assert z.latched_z_grasp is None
        assert len(z.samples_z_grasp) >= 1
        assert np.isfinite(out)

    n_hi = len(z.samples_z_grasp)
    assert n_hi == 5

    # Mid band: no new samples, still unlocked.
    mid = z.update(z_top=0.080, ee_z=0.080 + 0.15)
    assert z.latched_z_grasp is None
    assert len(z.samples_z_grasp) == n_hi
    assert mid == pytest.approx(float(np.median(z.samples_z_grasp)))

    # Below 10 cm: lock once.
    locked = z.update(z_top=0.080, ee_z=0.080 + 0.05)
    assert z.latched_z_grasp is not None
    assert locked == pytest.approx(z.latched_z_grasp)
    expect = float(np.median([0.069, 0.071, 0.070, 0.072, 0.068]))
    assert locked == pytest.approx(expect)

    # Later frames stay frozen even if z_top jumps.
    again = z.update(z_top=0.20, ee_z=0.05)
    assert again == pytest.approx(locked)


def test_lock_without_samples_uses_live():
    """Never went above 20 cm: lock to this frame when gap < 10 cm."""
    z = ZLatch()
    z_top = 0.08
    out = z.update(z_top=z_top, ee_z=z_top + 0.05)
    assert z.latched_z_grasp == pytest.approx(z_grasp_m(z_top))
    assert out == pytest.approx(z.latched_z_grasp)


def test_reset_clears_samples_and_lock():
    z = ZLatch()
    z.update(z_top=0.08, ee_z=0.30)
    z.update(z_top=0.08, ee_z=0.05)
    assert z.latched_z_grasp is not None
    z.reset()
    assert z.latched_z_grasp is None
    assert z.samples_z_grasp == []


def test_fuse_z_top_prefers_wrist_when_close():
    from interfaces.grasp_height import fuse_z_top

    assert fuse_z_top(0.08, 0.10, gap_m=0.25) == pytest.approx(0.09)
    assert fuse_z_top(0.08, 0.10, gap_m=0.05) == pytest.approx(0.08)
    assert np.isnan(fuse_z_top(float("nan"), float("nan")))


def test_wrist_z_top_accounts_for_cam_above_tcp():
    """Camera on hand is above TCP — ee_z − depth alone is ~tcp_z too low."""
    from interfaces.grasp_height import z_top_from_wrist_depth

    # Flat depth map: everything 0.30 m from the camera.
    depth = np.full((40, 40), 0.30, dtype=np.float64)
    ee = np.array([0.5, 0.0, 0.40])
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    # Old (broken) mental model: 0.40 − 0.30 = 0.10
    # Correct: cam at ee_z+0.107 → 0.507 − 0.30 = 0.207
    z = z_top_from_wrist_depth(depth, ee, quat, cam_above_tcp_m=0.107)
    assert z == pytest.approx(0.207, abs=1e-6)
    # Live cam pose wins over the tcp fallback.
    z2 = z_top_from_wrist_depth(
        depth, ee, quat, cam_z=0.55, cam_above_tcp_m=0.107
    )
    assert z2 == pytest.approx(0.25, abs=1e-6)
