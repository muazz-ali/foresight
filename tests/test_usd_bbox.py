"""Isaac-free: collision vs visual USD AABB helpers."""

from pathlib import Path

import numpy as np
import pytest

from sim.usd_bbox import usd_collision_size, usd_visual_collision_boxes

OBJ = Path("/home/gpuadmin/Desktop/muazzam/objects")


@pytest.mark.skipif(not (OBJ / "egg" / "egg00.usd").is_file(), reason="no egg USD")
def test_egg_collision_matches_visual():
    boxes = usd_visual_collision_boxes(OBJ / "egg" / "egg00.usd")
    assert boxes["used"] == "collision"
    assert boxes["collision_size"] is not None
    assert np.allclose(boxes["collision_size"], boxes["visual_size"], atol=1e-4)


@pytest.mark.skipif(not (OBJ / "avocado" / "avocado00.usd").is_file(), reason="no avocado")
def test_avocado00_collision_narrower_than_visual():
    """Render mesh is ~83 mm; collider capsule ~57 mm — gate must use collision."""
    boxes = usd_visual_collision_boxes(OBJ / "avocado" / "avocado00.usd")
    assert boxes["collision_size"] is not None
    assert boxes["visual_size"] is not None
    assert float(max(boxes["collision_size"][:2])) < float(max(boxes["visual_size"][:2])) - 0.01
    w = float(max(usd_collision_size(OBJ / "avocado" / "avocado00.usd")[:2]))
    assert w < 0.070  # passes 70 mm collision gate
