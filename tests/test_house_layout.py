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
    {"name": "static_cam", "position": [1.0, 0.0, 0.60], "rotation_deg": [0.0, 60.0, 90.0]}
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
    assert (xhi - xlo) >= 0.12 - 1e-9
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
    assert xlo >= 0.32 - 1e-9 and xhi <= 0.75 - 0.08 + 1e-9
    cy = float(layout.container_xy[1])
    # pocket must clear the play band with a >=3cm gap
    assert cy - 0.04 >= yhi + 0.03 - 1e-9 or cy + 0.04 <= ylo - 0.03 + 1e-9
    # clip box never overlaps the play band in y
    (_, _), (clo, chi) = layout.container_xy_clip
    assert clo >= yhi + 0.03 - 1e-9 or chi <= ylo - 0.03 + 1e-9


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
def test_sample_throw_velocity_points_toward_catch_band():
    rng = np.random.default_rng(7)
    bounds = ((0.40, 0.66), (-0.14, 0.14))  # symmetric -> reach_dir ~ +X
    catch_band = (0.38, 0.55)
    lateral = 0.18
    for _ in range(300):
        out = _sample_throw_full(
            rng, bounds, speed=0.3, catch_band=catch_band, lateral=lateral
        )
        spawn, vel, target = out["spawn"], out["vel_xy"], out["target"]
        reach_dir = out["reach_dir"]

        # velocity magnitude is the requested speed
        assert np.linalg.norm(vel) == pytest.approx(0.3, rel=1e-6)
        # velocity points toward the catch point (into the band)
        d = target - spawn
        if np.linalg.norm(d) > 1e-6:
            assert float(np.dot(vel, d)) > 0.0

        # the catch point really sits inside the band
        along = float(np.dot(target, reach_dir))
        across = float(np.dot(target, [-reach_dir[1], reach_dir[0]]))
        assert catch_band[0] - 1e-9 <= along <= catch_band[1] + 1e-9
        assert abs(across) <= lateral + 1e-9


def test_sample_throw_aims_toward_bowl_with_side_variety():
    """With a bowl XY, throws head toward it (straight / diagonal / side)."""
    rng = np.random.default_rng(21)
    bounds = ((0.40, 0.66), (-0.14, 0.14))
    bowl = np.array([0.50, 0.30])  # +Y pocket, same as layout default
    sides = set()
    toward_bowl = 0
    for _ in range(400):
        out = _sample_throw_full(
            rng, bounds, speed=0.2, container_xy=bowl
        )
        spawn, vel, target = out["spawn"], out["vel_xy"], out["target"]
        sides.add(out["side"])
        assert np.linalg.norm(vel) == pytest.approx(0.2, rel=1e-6)
        # Catch stays inside the padded play rect.
        assert 0.40 + 0.03 - 1e-9 <= target[0] <= 0.66 - 0.03 + 1e-9
        assert -0.14 + 0.03 - 1e-9 <= target[1] <= 0.14 - 0.03 + 1e-9
        # Velocity points at the catch point.
        d = target - spawn
        if np.linalg.norm(d) > 1e-6:
            assert float(np.dot(vel, d)) > 0.0
        # Path from spawn through catch is bowl-ward (catch between spawn and bowl).
        if float(np.dot(bowl - spawn, d)) > 1e-6:
            toward_bowl += 1
    assert toward_bowl >= 300  # most throws are bowl-facing
    # Need more than one spawn style (far / diag / side).
    assert len(sides) >= 3


def test_sample_throw_public_shapes_and_padded_bounds():
    rng = np.random.default_rng(11)
    bounds = ((0.40, 0.66), (-0.14, 0.14))
    for _ in range(200):
        pos, vel = sample_throw(rng, bounds, speed=0.3)
        assert pos.shape == (3,) and vel.shape == (3,)
        assert pos[2] == 0.0 and vel[2] == 0.0
        # spawn stays inside the padded bounds
        assert 0.40 + 0.03 - 1e-9 <= pos[0] <= 0.66 - 0.03 + 1e-9
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
