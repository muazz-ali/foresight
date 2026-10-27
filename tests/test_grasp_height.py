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

    # Flat depth map: everything 0.30 m from the camera (tight cluster → passes spread).
    depth = np.full((40, 40), 0.30, dtype=np.float64)
    ee = np.array([0.5, 0.0, 0.40])
    quat = np.array([0.0, 1.0, 0.0, 0.0])  # hanging down
    # Geometric cam_z ≈ ee_z + pinch_z (0.1034) → 0.5034 − 0.30 = 0.2034
    z = z_top_from_wrist_depth(depth, ee, quat, cam_above_tcp_m=0.1034)
    assert z == pytest.approx(0.2034, abs=1e-6)
    # Live cam pose wins over the EE fallback.
    z2 = z_top_from_wrist_depth(
        depth, ee, quat, cam_z=0.55, cam_above_tcp_m=0.1034
    )
    assert z2 == pytest.approx(0.25, abs=1e-6)
    # A4 bias knob shifts the measured top (fruit latch was ~6.7 mm low).
    z3 = z_top_from_wrist_depth(
        depth, ee, quat, cam_z=0.55, z_bias_m=0.0067
    )
    assert z3 == pytest.approx(0.2567, abs=1e-6)


def test_wrist_z_top_rejects_mixed_near_cluster():
    """Wide near-depth band (spread > 1 cm) → NaN so the latch skips the frame."""
    from interfaces.grasp_height import z_top_from_wrist_depth

    # Full-frame ramp so the near-percentile band spans many centimetres.
    depth = np.linspace(0.25, 0.45, 40 * 40, dtype=np.float64).reshape(40, 40)
    ee = np.array([0.5, 0.0, 0.40])
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    z = z_top_from_wrist_depth(
        depth,
        ee,
        quat,
        max_spread_m=0.010,
        rows=(0.0, 1.0),
        cols=(0.0, 1.0),
    )
    assert np.isnan(z)


def test_wrist_cam_pose_from_ee_hanging_down():
    """Down quat: cam sits ~tcp above the TCP in world Z."""
    from interfaces.grasp_height import wrist_cam_pose_from_ee

    ee = np.array([0.5, 0.0, 0.40])
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    cam, look = wrist_cam_pose_from_ee(ee, quat)
    assert cam[2] == pytest.approx(0.5034, abs=1e-3)
    assert look[2] == pytest.approx(-1.0, abs=1e-2)


def _render_plane_depth(ee, quat, f_px, cx, cy, w, h, z_of_xy):
    """Ray-cast a height field z_of_xy(x, y) into a wrist depth image (image-plane depth)."""
    from interfaces.grasp_height import (
        DEFAULT_TCP_OFFSET_M,
        DEFAULT_WRIST_POS_M,
        DEFAULT_WRIST_QUAT_WXYZ,
        quat_wxyz_to_rotmat,
    )

    R = quat_wxyz_to_rotmat(quat)
    cam = ee - R @ DEFAULT_TCP_OFFSET_M + R @ DEFAULT_WRIST_POS_M
    Rc = R @ quat_wxyz_to_rotmat(DEFAULT_WRIST_QUAT_WXYZ)
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - cx) / f_px, -(vv - cy) / f_px, -np.ones_like(uu, float)], -1) @ Rc.T
    depth = np.full((h, w), np.inf)
    for d in np.linspace(0.05, 1.0, 400):  # march in image-plane depth
        p = cam + dirs * d
        hit = np.isinf(depth) & (p[..., 2] <= z_of_xy(p[..., 0], p[..., 1]))
        depth[hit] = d
    return depth


def test_wrist_cloud_finds_box_top_with_tilted_camera():
    """A4: whole-image lift reads a 50 mm tall box from 30 cm away with a tilted hand."""
    from interfaces.grasp_height import z_top_from_wrist_cloud
    from scipy.spatial.transform import Rotation

    w, h, f_px = 160, 120, 80.0
    obj = np.array([0.60, 0.05])
    top = 0.050
    box = lambda x, y: np.where((np.abs(x - obj[0]) < 0.02) & (np.abs(y - obj[1]) < 0.02), top, 0.0)
    down = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing down (wxyz)
    # Tilted 35° so the wrist cam looks forward/down (+x), like early in a real approach.
    tilt = Rotation.from_euler("y", -35, degrees=True) * Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
    q_xyzw = tilt.as_quat()
    tilted = np.array([q_xyzw[3], *q_xyzw[:3]])
    for quat, ee in ((down, np.array([0.62, 0.05, 0.30])), (tilted, np.array([0.31, 0.05, 0.35]))):
        depth = _render_plane_depth(ee, quat, f_px, w / 2, h / 2, w, h, box)
        z = z_top_from_wrist_cloud(depth, ee, quat, obj, f_px=f_px, cx=w / 2, cy=h / 2, stride=1, min_points=5)
        assert z == pytest.approx(top, abs=0.004), (quat, z)
    # Object not in view → NaN, not the table.
    far = z_top_from_wrist_cloud(depth, ee, tilted, np.array([2.0, 2.0]), f_px=f_px, cx=w / 2, cy=h / 2)
    assert np.isnan(far)
