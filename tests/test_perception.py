"""Isaac-free tests for the Phase 2 perception stack.

Run from the repo root:  pytest tests/test_perception.py -v

Synthetic frames and tracks only — the real-data numbers come from
``scripts/g2_perception_report.py``. Synthetic jitter is 0.5 mm per frame; real
static-cam fixes jitter ~0.3 mm frame to frame (on top of a slow ~1.4 mm bias).
Speed / position bars are B's training noise (3 cm/s, 1.5 cm), not tighter.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from interfaces.config import load_yaml
from interfaces.state import conditioning_vector
from perception.camera import PinholeCamera
from perception.estimator import StaticCamEstimator
from perception.kalman import CVKalman
from perception.track import BlobTracker

CFG = load_yaml(Path(__file__).resolve().parents[1] / "sim" / "phase0_cfg.yaml")
DT = 0.04
JITTER = 0.0005  # m per frame
VEL_BAR = 0.03  # B's training noise on v̂ (m/s)


@pytest.fixture(scope="module")
def cam() -> PinholeCamera:
    return PinholeCamera.from_cfg(CFG)


def _frame(uv, *, colour=(230, 180, 20), radius=9, grid=False):
    img = np.full((360, 480, 3), 40, dtype=np.uint8)  # dark grey table
    if grid:
        for x in range(20, 480, 60):
            img[:, x : x + 5] = 200  # white lines ~5 px wide, like the sim grid
        for y in range(30, 360, 60):
            img[y : y + 5, :] = 200
    # Sub-pixel centre (shift=4 → 1/16 px) so the synthetic blob moves smoothly.
    centre = (int(round(uv[0] * 16)), int(round(uv[1] * 16)))
    cv2.circle(img, centre, radius * 16, colour, -1, lineType=cv2.LINE_AA, shift=4)
    return img


def test_camera_matches_measured_intrinsics(cam):
    # a15_camera_geometry.json: fx = fy = 240, principal point at the image centre.
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (240.0, 240.0, 240.0, 180.0)


def test_project_lift_round_trip(cam):
    for p in ([0.50, -0.20, 0.03], [0.72, 0.10, 0.045], [0.60, -0.05, 0.025]):
        uv = cam.project(p)
        assert 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height
        np.testing.assert_allclose(cam.lift(uv, p[2]), p[:2], atol=1e-9)


def test_wrong_plane_height_biases_xy(cam):
    p = np.array([0.60, -0.05, 0.03])
    err = np.linalg.norm(cam.lift(cam.project(p), 0.0) - p[:2])
    assert err > 0.015  # why plane_z must be the object centre height


def test_tracker_hist_follows_nearest_blob(cam):
    tr = BlobTracker()
    uv = cam.project([0.60, -0.05, 0.03])
    img = _frame(uv)
    cv2.circle(img, (int(uv[0]) + 80, int(uv[1])), 20, (200, 40, 40), -1)  # bigger decoy
    assert tr.start(img, uv) and tr.mode == "hist"
    np.testing.assert_allclose(tr.step(img, uv + 5), uv, atol=1.0)


def test_tracker_bright_mode_finds_egg_on_grid(cam):
    tr = BlobTracker()
    uv = cam.project([0.60, -0.05, 0.03])
    img = _frame(uv, colour=(205, 205, 205), radius=9, grid=True)
    assert tr.start(img, uv) and tr.mode == "bright"
    np.testing.assert_allclose(tr.step(img, uv + 5), uv, atol=1.5)
    # Grid lines alone (egg gone) must not be picked up.
    assert tr.step(_frame(uv, colour=(40, 40, 40), grid=True), uv) is None


def _vel_rms(xs, ts, v_final, *, seeds=50, **kw):
    """RMS speed error at the last frame over many noise draws (σ-level, not max)."""
    errs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        kf = CVKalman(**kw)
        for x, t in zip(xs, ts):
            kf.step(t, x + rng.normal(0, JITTER, 2))
        errs.append(np.linalg.norm(kf.state(0.03).velocity[:2] - v_final))
    return float(np.sqrt(np.mean(np.square(errs))))


def test_kalman_straight_line_converges():
    v = np.array([0.14, -0.14])
    ts = np.arange(30) * DT
    xs = np.array([0.6, 0.0]) + ts[:, None] * v
    assert _vel_rms(xs, ts, v) < VEL_BAR
    kf = CVKalman()
    for x, t in zip(xs, ts):
        kf.step(t, x)
    s = kf.state(0.03)
    assert np.linalg.norm(s.position[:2] - xs[-1]) < 1e-4  # clean track: exact
    assert s.valid and kf.n_bounces == 0


def _bounce_track():
    ts = np.arange(40) * DT
    x = 0.60 + 0.2 * ts
    x = np.where(x > 0.72, 2 * 0.72 - x, x)  # wall at x = 0.72
    return np.c_[x, np.zeros_like(x)], ts


def test_kalman_recovers_after_wall_bounce():
    xs, ts = _bounce_track()
    assert _vel_rms(xs, ts, np.array([-0.2, 0.0])) < VEL_BAR


def test_kalman_bounce_reset_fires_when_filter_is_slow():
    # A sluggish filter cannot follow a reversal on its own; the NIS gate must fire.
    xs, ts = _bounce_track()
    kf = CVKalman(accel_std=0.3)
    for x, t in zip(xs, ts):
        kf.step(t, x)
    assert kf.n_bounces >= 1
    assert abs(kf.state(0.03).velocity[0] - (-0.2)) < VEL_BAR


def test_kalman_coast_turns_invalid():
    kf = CVKalman(max_coast_s=0.5)
    kf.step(0.0, np.array([0.6, 0.0]))
    kf.step(0.04, np.array([0.604, 0.0]))
    kf.step(0.40, None)
    assert kf.state(0.03).valid
    kf.step(0.60, None)
    assert not kf.state(0.03).valid


def test_estimator_pack_matches_oracle_on_synthetic_frames(cam):
    v = np.array([-0.10, 0.12, 0.0])
    p0 = np.array([0.66, -0.12, 0.03])
    est = StaticCamEstimator(cam)
    assert est.start(_frame(cam.project(p0)), p0)
    state = None
    for k in range(25):
        p = p0 + v * k * DT
        state = est.step(_frame(cam.project(p)), k * DT)
    from interfaces.state import oracle_from_gt

    oracle = conditioning_vector(oracle_from_gt(p, v, 24 * DT), 0.25)
    mine = conditioning_vector(state, 0.25)
    assert np.linalg.norm(mine[:2] - oracle[:2]) < 0.01  # p̂ within 1 cm
    assert np.linalg.norm(mine[2:] - oracle[2:]) < VEL_BAR  # v̂ inside training noise


def test_kalman_holds_still_after_long_blind_spell():
    kf = CVKalman(max_coast_s=0.5)
    for k in range(10):
        kf.step(k * DT, np.array([0.6 + 0.2 * k * DT, 0.0]))
    kf.step(10 * DT + 0.6, None)  # blind > 0.5 s
    held = kf.state(0.03)
    assert not held.valid and np.allclose(held.velocity, 0.0)
    kf.step(10 * DT + 2.0, None)
    assert np.allclose(kf.state(0.03).position, held.position)  # no drift off the table


def test_estimator_redetects_after_hand_hides_object(cam):
    est = StaticCamEstimator.from_cfg(CFG)
    p0 = np.array([0.55, -0.10, 0.03])
    assert est.start(_frame(cam.project(p0)), p0)
    for k in range(5):
        est.step(_frame(cam.project(p0)), k * DT)
    hidden = np.full((360, 480, 3), 40, dtype=np.uint8)  # object behind the hand
    for k in range(5, 25):
        est.step(hidden, k * DT)
    assert not est.kf.state(0.03).valid
    p1 = np.array([0.70, 0.05, 0.03])  # reappears far from the stale guess (~60 px)
    for k in range(25, 30):
        state = est.step(_frame(cam.project(p1)), k * DT)
    assert est.n_redetects == 1 and state.valid
    assert np.linalg.norm(state.position[:2] - p1[:2]) < 0.005


def _wood(seed=0):
    """Orange-brown table with darker grain stripes and noise (saturated, like house wood)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 480, 3), dtype=np.float64)
    img[:] = (170, 115, 60)
    rows = np.sin(np.arange(360) / 3.0 + rng.uniform(0, 6)) * 18  # grain
    img += rows[:, None, None] + rng.normal(0, 6, (360, 480, 1))
    img[:150] = (215, 215, 215)  # pale wall above the table
    return np.clip(img, 0, 255).astype(np.uint8)


def _on_wood(uv, *, colour=(225, 200, 30), radius=7, seed=0):
    img = _wood(seed)
    centre = (int(round(uv[0] * 16)), int(round(uv[1] * 16)))
    cv2.circle(img, centre, radius * 16, colour, -1, lineType=cv2.LINE_AA, shift=4)
    return img


def test_old_colour_look_drowns_on_wood(cam):
    # Why rung 2 exists: on wood "any saturated pixel" is the whole table.
    uv = cam.project([0.60, -0.05, 0.03])
    hsv = cv2.cvtColor(_on_wood(uv), cv2.COLOR_RGB2HSV)
    assert ((hsv[..., 1] > 90) & (hsv[..., 2] > 50))[200:].mean() > 0.5
    tr = BlobTracker()
    assert tr.start(_on_wood(uv), uv) and tr.mode == "hist"


def test_estimator_tracks_lemon_on_wood(cam):
    # Yellow fruit on orange-brown wood, moving 20 cm/s; fresh table noise each frame.
    v = np.array([-0.20, 0.05, 0.0])
    p0 = np.array([0.66, -0.08, 0.03])
    est = StaticCamEstimator.from_cfg(CFG)
    assert est.start(_on_wood(cam.project(p0)), p0)
    fixes = 0
    for k in range(25):
        p = p0 + v * k * DT
        state = est.step(_on_wood(cam.project(p), seed=k), k * DT)
        fixes += est.last_uv is not None
    assert fixes == 25
    from interfaces.state import oracle_from_gt

    oracle = conditioning_vector(oracle_from_gt(p, v, 24 * DT), 0.25)
    mine = conditioning_vector(state, 0.25)
    assert np.linalg.norm(mine[:2] - oracle[:2]) < 0.01
    assert np.linalg.norm(mine[2:] - oracle[2:]) < VEL_BAR


def test_estimator_tracks_upright_bottle(cam):
    # Tall dark bottle standing on wood: centre height 12 cm, ~3 cm wide.
    p0 = np.array([0.58, 0.02, 0.12])
    v = np.array([-0.15, 0.0, 0.0])

    def frame(p, k):
        img = _wood(k)
        top, bot = cam.project(p + [0, 0, 0.12]), cam.project(p - [0, 0, 0.12])
        u = int(round(top[0]))
        cv2.rectangle(img, (u - 5, int(top[1])), (u + 5, int(bot[1])), (60, 25, 15), -1)
        return img

    est = StaticCamEstimator.from_cfg(CFG)
    assert est.start(frame(p0, 0), p0)
    for k in range(20):
        p = p0 + v * k * DT
        st = est.step(frame(p, k), k * DT)
        assert est.last_uv is not None
    # A straight-sided tall box's pixel middle is not the projection of its centre
    # (perspective): ~1.1 cm here. Grade the tracker against the drawn middle, and
    # the total against the 2 cm G2a RMSE bar.
    top, bot = cam.project(p + [0, 0, 0.12]), cam.project(p - [0, 0, 0.12])
    drawn = cam.lift((top + bot) / 2, 0.12)
    assert np.linalg.norm(st.position[:2] - drawn) < 0.005
    assert np.linalg.norm(st.position[:2] - p[:2]) < 0.02


def test_play_area_check_ignores_wall_pixels(cam):
    # House frames show walls above the horizon; re-detect must skip them, not crash.
    est = StaticCamEstimator.from_cfg(CFG)
    est.plane_z = 0.03
    assert est._in_play_area(np.array([240.0, 0.0])) is False
