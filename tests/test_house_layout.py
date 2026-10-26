"""House-layout unit tests. Pure numpy + pxr; no Isaac AppLauncher.

Run (sim env has pxr):
    conda run -n dynamicVLA_isaac python -m pytest tests/test_house_layout.py -q
Pure-numpy tests also run in any env with numpy + scipy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.house_layout import (
    YUP_TO_ZUP_WXYZ,
    _sample_throw_full,
    compute_play_layout,
    find_tables,
    pick_house,
    sample_light,
    sample_throw,
)

SCENE_DIR = Path("/home/gpuadmin/Desktop/muazzam/scenes")
# One camera, foresight yaml style: {name, position, rotation_deg}.
CAMERAS = [
    {"name": "static_cam", "position": [1.2, 0.0, 0.80], "rotation_deg": [0.0, 48.0, 90.0]}
]

_has_pxr = True
try:  # pxr only lives in the sim env; skip USD tests elsewhere.
    import pxr  # noqa: F401
except Exception:  # pragma: no cover
    _has_pxr = False

_scene_ready = _has_pxr and SCENE_DIR.is_dir()


# --------------------------------------------------------------------------
# (1) find_tables returns >=1 table on a real scene
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _scene_ready, reason="no pxr or scene dir missing")
def test_find_tables_on_real_scenes():
    # Walk a few scenes; at least one should expose a usable table.
    usds = sorted(SCENE_DIR.glob("*.usd"))[:12]
    assert usds, "no .usd scenes to test"
    found_any = False
    for usd in usds:
        tables = find_tables(usd, CAMERAS)
        for t in tables:
            assert t.height > 0.0
            assert t.bbox_max[2] == pytest.approx(t.height)  # max_z forced to height
            assert any(a.side == "long" for a in t.anchors)  # long-side kept
        if tables:
            found_any = True
            break
    assert found_any, "no table with a long-side anchor in the first 12 scenes"


# --------------------------------------------------------------------------
# (2) pick_house puts the anchor on the origin; play layout is sane
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _scene_ready, reason="no pxr or scene dir missing")
def test_pick_house_puts_anchor_at_origin_and_layout_valid():
    rng = np.random.default_rng(0)
    # Same retry as sim/scene.py: skip tables too small for play+bowl.
    choice = None
    layout = None
    for _ in range(40):
        try:
            cand = pick_house(SCENE_DIR, rng, CAMERAS, max_tries=1)
            layout = compute_play_layout(cand.table_bbox_min_w, cand.table_bbox_max_w)
            choice = cand
            break
        except Exception:
            continue
    assert choice is not None and layout is not None

    # Transform the chosen anchor by the spawn transform: Z-up point ->
    # p_w = R_a_inv @ p + t_spawn. It must land on the origin.
    from scipy.spatial.transform import Rotation

    q = choice.anchor.quat_wxyz
    R_a = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    p_w = R_a.T @ choice.anchor.pos + choice.spawn_pos  # R_a.T == inv(R_a)
    assert np.linalg.norm(p_w) < 1e-5

    # Table top sits at z=0 after the transform.
    assert choice.table_bbox_max_w[2] == pytest.approx(0.0, abs=1e-6)
    assert choice.anchor.side == "long"

    (xlo, xhi), (ylo, yhi) = layout.object_xy_bounds
    assert xlo < xhi and ylo < yhi  # positive area
    assert (xhi - xlo) >= 0.35 - 1e-9  # min runway for ≤50 cm/s chase
    assert layout.table_z == 0.0

    # container is clear of the play band by >= 3 cm
    cy = float(layout.container_xy[1])
    if cy > 0:
        assert cy - 0.04 >= yhi + 0.03 - 1e-9
    else:
        assert cy + 0.04 <= ylo - 0.03 + 1e-9


# --------------------------------------------------------------------------
# pure-numpy layout check on a synthetic symmetric table (no pxr needed)
# --------------------------------------------------------------------------
def _synthetic_layout():
    # 1.4 x 1.4 m table, top at z=0, centered so inner half straddles y=0.
    bmin = np.array([-0.10, -0.70, -0.75])
    bmax = np.array([1.30, 0.70, 0.0])
    return compute_play_layout(bmin, bmax)


def test_compute_play_layout_positive_area_and_container_outside_band():
    layout = _synthetic_layout()
    (xlo, xhi), (ylo, yhi) = layout.object_xy_bounds
    assert xlo < xhi and ylo < yhi
    assert xlo >= 0.30 - 1e-9
    # Far X reaches near the real table rim (bmax=1.30 − 2 cm), not mid-table.
    assert xhi >= 1.20 - 1e-6
    cx, cy = float(layout.container_xy[0]), float(layout.container_xy[1])
    # Toward +Y rim, clear of play, but reach-clamped (not raw rim 0.64).
    assert cy >= 0.28 - 1e-6
    assert cy <= 0.64 + 1e-6
    assert cy - 0.04 >= yhi + 0.03 - 1e-9
    # Default seed 0.43 − 3 cm toward −X.
    assert cx == pytest.approx(0.40, abs=1e-6)
    # clip box never overlaps the play band in y
    (_, _), (clo, chi) = layout.container_xy_clip
    assert clo >= yhi - 1e-9
    assert chi > clo


def test_throw_spawns_on_far_x_edge_not_mid_table():
    """Spawn X must sit on the play-band +X rim (policy needs a long track runway)."""
    bounds = ((0.40, 0.80), (-0.14, 0.14))
    for s in range(50):
        out = _sample_throw_full(
            np.random.default_rng(s), bounds, speed=0.5, throw_randomness=1.0
        )
        assert out["spawn"][0] == pytest.approx(0.80, abs=1e-9)
        assert out["spawn"][0] > out["target"][0] - 1e-9  # heads toward robot


def test_compute_play_layout_raises_on_empty():
    # A table entirely off to the side -> no overlap with the play rect.
    bmin = np.array([2.0, 2.0, -0.75])
    bmax = np.array([3.0, 3.0, 0.0])
    with pytest.raises(ValueError):
        compute_play_layout(bmin, bmax)


# --------------------------------------------------------------------------
# (3) sample_throw: velocity aims at a point inside the catch band.
#
# Faithful to DynamicVLA: the object is thrown from an edge toward a catch
# point in the band, so the velocity points AT that catch point. (For a front
# throw that means it travels back toward the robot, i.e. -X -- so a literal
# "dot(vel, +X) > 0" is the wrong check. We assert the honest invariant: the
# aim target lies in the catch band and the velocity points toward it.)
# --------------------------------------------------------------------------
def test_sample_throw_uses_180_deg_fan_toward_robot():
    """r sets the fan around 90° (straight at robot); velocity stays in the half-plane."""
    rng = np.random.default_rng(7)
    bounds = ((0.40, 0.66), (-0.14, 0.14))
    for r, lo, hi in ((0.0, 90.0, 90.0), (0.5, 45.0, 135.0), (1.0, 0.0, 180.0)):
        angles = []
        for _ in range(200):
            out = _sample_throw_full(rng, bounds, speed=0.3, throw_randomness=r)
            ang = float(out["throw_angle_deg"])
            vel = out["vel_xy"]
            angles.append(ang)
            assert np.linalg.norm(vel) == pytest.approx(0.3, rel=1e-6)
            assert vel[0] <= 1e-9  # never away from the robot (+X)
            assert lo - 1e-6 <= ang <= hi + 1e-6
        if r > 0.0:
            assert max(angles) - min(angles) > 20.0 * r


def test_sample_throw_ignores_bowl_and_goes_toward_robot():
    """Bowl XY must not steer the throw; heading stays in the 180° robot-facing fan."""
    rng = np.random.default_rng(21)
    bounds = ((0.40, 0.66), (-0.14, 0.14))
    bowl = np.array([0.48, 0.33])  # +Y pocket — must not be the aim
    for _ in range(200):
        out = _sample_throw_full(
            rng, bounds, speed=0.2, container_xy=bowl, throw_randomness=1.0
        )
        vel = out["vel_xy"]
        assert np.linalg.norm(vel) == pytest.approx(0.2, rel=1e-6)
        assert vel[0] <= 1e-9
        assert 0.0 - 1e-6 <= float(out["throw_angle_deg"]) <= 180.0 + 1e-6
        # Catch stays in the play band, not pulled into the +Y bowl pocket.
        assert abs(out["target"][1]) <= 0.14 + 1e-6


def test_sample_throw_public_shapes_and_padded_bounds():
    rng = np.random.default_rng(11)
    bounds = ((0.40, 0.66), (-0.14, 0.14))
    for _ in range(200):
        pos, vel = sample_throw(rng, bounds, speed=0.3)
        assert pos.shape == (3,) and vel.shape == (3,)
        assert pos[2] == 0.0 and vel[2] == 0.0
        # Spawn on the far +X rim; Y stays inside the padded band.
        assert pos[0] == pytest.approx(0.66, abs=1e-9)
        assert -0.14 + 0.03 - 1e-9 <= pos[1] <= 0.14 - 0.03 + 1e-9
        assert np.linalg.norm(vel) == pytest.approx(0.3, rel=1e-6)


def test_sample_throw_zero_speed_is_still():
    rng = np.random.default_rng(1)
    bounds = ((0.40, 0.66), (-0.14, 0.14))
    pos, vel = sample_throw(rng, bounds, speed=0.0)
    assert np.allclose(vel, 0.0)
    assert pos.shape == (3,) and pos[2] == 0.0
    # still-object spawn also respects the padded bounds
    assert 0.40 + 0.03 - 1e-9 <= pos[0] <= 0.66 - 0.03 + 1e-9


def test_throw_randomness_zero_is_fixed_90_deg_center_line():
    """r=0: only 90° — straight at the robot (−X), reproducible."""
    from sim.house_layout import _sample_throw_full

    bounds = ((0.40, 0.66), (-0.14, 0.14))
    a = _sample_throw_full(
        np.random.default_rng(0), bounds, speed=0.2, throw_randomness=0.0
    )
    b = _sample_throw_full(
        np.random.default_rng(99), bounds, speed=0.2, throw_randomness=0.0
    )
    assert a["side"] == b["side"] == "front"
    assert a["throw_angle_deg"] == pytest.approx(90.0)
    np.testing.assert_allclose(a["spawn"], b["spawn"], atol=1e-9)
    np.testing.assert_allclose(a["vel_xy"], b["vel_xy"], atol=1e-9)
    np.testing.assert_allclose(a["vel_xy"], [-0.2, 0.0], atol=1e-9)


def test_throw_randomness_keeps_catchable_diverse_fan():
    """r=1: diverse headings, but every coast stays in the play band to catch_x.

    Grazing ~0°/180° throws from the far edge leave the table before the hand
    can meet them — those are rejected (smoke off-table fix).
    """
    from sim.house_layout import _coast_reaches_catch, _heading_from_edge_deg, _sample_throw_full

    bounds = ((0.40, 0.66), (-0.14, 0.14))
    (xmin, xmax), (ymin, ymax) = bounds
    angles = []
    for s in range(400):
        out = _sample_throw_full(
            np.random.default_rng(s),
            bounds,
            speed=0.2,
            throw_randomness=1.0,
        )
        ang = float(out["throw_angle_deg"])
        angles.append(ang)
        assert out["vel_xy"][0] <= 1e-9
        heading = _heading_from_edge_deg(ang)
        assert _coast_reaches_catch(
            out["spawn"],
            heading,
            0.2,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            until_x=0.55,
        )
    # Still a wide fan (not only 90°), but not the uncatchable extremes.
    assert max(angles) - min(angles) > 40.0
    assert min(angles) > 15.0
    assert max(angles) < 165.0


def test_grazing_heading_is_rejected_by_coast_check():
    """Pure ±Y from the far rim never reaches the catch zone on the band."""
    from sim.house_layout import _coast_reaches_catch, _heading_from_edge_deg

    spawn = np.array([0.66, -0.10])
    for theta in (0.0, 5.0, 175.0, 180.0):
        ok = _coast_reaches_catch(
            spawn,
            _heading_from_edge_deg(theta),
            0.5,
            xmin=0.40,
            xmax=0.66,
            ymin=-0.14,
            ymax=0.14,
            until_x=0.55,
        )
        assert not ok, theta


def test_tip_on_side_quat_yaws_with_travel():
    """Tip ±90°; yaw follows velocity so the long axis is across travel."""
    from scipy.spatial.transform import Rotation

    from sim.house_layout import sit_half_height_m, tip_on_side_quat_wxyz

    rng = np.random.default_rng(3)
    vel = np.array([0.2, 0.0])
    q = tip_on_side_quat_wxyz(vel, rng)
    assert q.shape == (4,)
    # wxyz → xyzw for scipy
    eul = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
    assert abs(abs(eul[0]) - 0.5 * np.pi) < 1e-6
    assert abs(eul[2]) < 1e-6  # travel along +X → yaw ~ 0
    # Tipped sit height uses diameter, not upright height.
    size = np.array([0.06, 0.06, 0.12])
    assert sit_half_height_m(size, tipped=True) == pytest.approx(0.03)
    assert sit_half_height_m(size, tipped=False) == pytest.approx(0.06)


# --------------------------------------------------------------------------
# sample_light ranges + YUP constant sanity
# --------------------------------------------------------------------------
def test_look_at_quat_is_right_handed():
    from scipy.spatial.transform import Rotation

    from sim.house_layout import look_at_quat_wxyz

    rng = np.random.default_rng(0)
    for _ in range(100):
        pos = [
            float(rng.uniform(-50, 50)),
            float(rng.uniform(-50, 50)),
            float(rng.uniform(10, 20)),
        ]
        q = look_at_quat_wxyz(pos, (0.0, 0.0, 0.0))
        mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        assert np.linalg.det(mat) > 0.99


def test_look_at_quat_emits_toward_target():
    """USD DistantLight shines along local -Z; that axis must hit the table."""
    from scipy.spatial.transform import Rotation

    from sim.house_layout import look_at_quat_wxyz

    # Behind static_cam (cam ≈ +X): light further out, aimed at origin.
    cases = [
        (5.0, 0.0, 3.0),
        (8.0, 1.5, 10.0),
        (1.0, 0.0, 12.0),  # mostly above
        (-4.0, 2.0, 6.0),
    ]
    target = np.array([0.0, 0.0, 0.0])
    for pos in cases:
        q = look_at_quat_wxyz(pos, target)
        mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        emit = -mat[:, 2]  # USD DistantLight
        want = target - np.asarray(pos, dtype=float)
        want = want / np.linalg.norm(want)
        assert float(np.dot(emit, want)) > 0.99, (pos, emit, want)


def test_sample_light_ranges():
    rng = np.random.default_rng(3)
    cam = (1.0, 0.0, 0.60)
    for _ in range(50):
        light = sample_light(rng, cam)
        assert 4000 <= light["temperature"] <= 8000
        assert 150.0 <= light["intensity"] <= 750.0
        x, y, z = light["position"]
        # Behind / above the camera: further out in +X than the lens, or higher.
        assert light["style"] in ("behind", "above", "behind_above")
        assert z >= cam[2] - 1e-6
        # Never between camera and table (x < cam.x with no height).
        assert x >= cam[0] - 1e-6 or z >= cam[2] + 1.5


def test_sample_light_stays_camera_side():
    rng = np.random.default_rng(9)
    cam = np.array([1.0, 0.0, 0.60])
    for _ in range(100):
        light = sample_light(rng, cam)
        pos = np.asarray(light["position"], dtype=float)
        # Vector from cam to light should not point toward the table (-X).
        delta = pos - cam
        assert float(delta[0]) >= -1e-6 or float(delta[2]) > 1.0
        # Aim quaternion still right-handed.
        from scipy.spatial.transform import Rotation

        from sim.house_layout import look_at_quat_wxyz

        q = look_at_quat_wxyz(pos, (0.0, 0.0, 0.0))
        mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        assert np.linalg.det(mat) > 0.99


def test_yup_constant():
    assert YUP_TO_ZUP_WXYZ == (0.7071068, 0.7071068, 0.0, 0.0)
