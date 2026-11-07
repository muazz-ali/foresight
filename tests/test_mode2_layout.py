"""Isaac-free checks for Mode 2 cube layout + language keys."""

from __future__ import annotations

import numpy as np

from policy.features import default_language, object_key
from sim.mode2_layout import CUBE_COLORS, randomize


def test_mode2_layout_places_all_four():
    """Every color gets a pose; target is one of them; gaps hold."""
    out = randomize(
        seed=41,
        xy_bounds=((0.40, 0.80), (-0.25, 0.25)),
        table_z=0.0,
        half_height=0.0235,
        spawn_pad=0.03,
        container_xy=np.array([0.35, 0.42]),
        min_separation_m=0.12,
        container_clearance_m=0.15,
    )
    assert out["target_color"] in CUBE_COLORS
    assert set(out["poses"]) == set(CUBE_COLORS)
    poses = list(out["poses"].values())
    for i, a in enumerate(poses):
        assert a.shape == (3,)
        for b in poses[i + 1 :]:
            assert float(np.linalg.norm(a[:2] - b[:2])) >= 0.12 - 1e-9


def test_mode2_layout_repeatable():
    """Same seed → same target and poses."""
    kw = dict(
        xy_bounds=((0.40, 0.80), (-0.25, 0.25)),
        table_z=0.0,
        half_height=0.0235,
        spawn_pad=0.03,
        container_xy=np.array([0.35, 0.42]),
    )
    a = randomize(seed=7, **kw)
    b = randomize(seed=7, **kw)
    assert a["target_color"] == b["target_color"]
    for c in CUBE_COLORS:
        assert np.allclose(a["poses"][c], b["poses"][c])


def test_cube_language_keys():
    """Category and sentence both map to cube_* templates."""
    assert object_key("cube_red") == "cube_red"
    assert object_key("Grasp the red cube and put it in the bowl.") == "cube_red"
    assert object_key("cube_blue") == "cube_blue"
    text = default_language("cube_yellow", "bowl", seed=0)
    assert "yellow" in text.lower()
    assert "cube" in text.lower()
