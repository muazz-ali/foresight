"""Size gates for pickable objects and placeable containers.

Pure numpy / pathlib. No Isaac. Used when building the USD catalog so too-wide
fruit and flat plates never enter a run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def horizontal_width_m(size_m: np.ndarray) -> float:
    """Widest horizontal side of an axis-aligned bbox (metres)."""
    s = np.asarray(size_m, dtype=np.float64).reshape(-1)
    if s.size < 2:
        return float(s[0]) if s.size else 0.0
    return float(max(s[0], s[1]))


def object_fits_gripper(size_m: np.ndarray, max_width_m: float) -> bool:
    """True if the object is narrow enough for the open gripper."""
    return horizontal_width_m(size_m) <= float(max_width_m)


def container_can_hold(
    object_size_m: np.ndarray,
    container_size_m: np.ndarray,
    *,
    min_clearance_m: float = 0.01,
    min_xy_margin_m: float = 0.01,
    open_top: bool = False,
    finger_clearance_m: float = 0.0,
) -> bool:
    """True if the container can take this object.

    Closed boxes need height + XY margins. Open bowls only need XY room — the
    fruit may sit above the rim. Plates stay out via ``exclude_usd_containers``.

    ``finger_clearance_m``: the closed fingers must also fit. They open along
    world Y (down-facing Franka hand), so the container's Y side must exceed the
    object's widest side by this much (two finger pads + place error). Without
    it a 66 mm potato in a 122 mm box parks a finger on the rim.
    """
    obj = np.asarray(object_size_m, dtype=np.float64).reshape(-1)
    ctr = np.asarray(container_size_m, dtype=np.float64).reshape(-1)
    if obj.size < 3 or ctr.size < 3:
        return False
    if not open_top:
        if float(ctr[2]) < float(obj[2]) + float(min_clearance_m):
            return False
    if float(ctr[0]) < float(obj[0]) + float(min_xy_margin_m):
        return False
    if float(ctr[1]) < float(obj[1]) + float(min_xy_margin_m):
        return False
    if float(ctr[1]) < float(max(obj[0], obj[1])) + float(finger_clearance_m):
        return False
    return True


def stem_or_category_banned(path: Path, ban: set[str]) -> bool:
    """Match exact stem, category folder, or stem prefix (tomato → tomato03)."""
    stem = path.stem.lower()
    cat = path.parent.name.lower()
    if stem in ban or cat in ban:
        return True
    for b in ban:
        if b and stem.startswith(b):
            return True
    return False


def merge_exclude_names(*groups: list | set | None) -> set[str]:
    """Flatten yaml exclude lists into lower-case stems / category names."""
    out: set[str] = set()
    for g in groups:
        if not g:
            continue
        for x in g:
            out.add(str(x).lower().removesuffix(".usd"))
    return out
