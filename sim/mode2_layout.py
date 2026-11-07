"""Mode 2 table layout: four static cubes, one language target.

Call ``randomize`` once per episode reset. Pure numpy — no Isaac.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Category folder names under object_dir (must match mode2_cfg).
CUBE_COLORS: tuple[str, ...] = (
    "cube_blue",
    "cube_red",
    "cube_green",
    "cube_yellow",
)


def randomize(
    *,
    seed: int,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]],
    table_z: float,
    half_height: float,
    spawn_pad: float,
    container_xy: np.ndarray | None,
    colors: tuple[str, ...] | list[str] = CUBE_COLORS,
    min_separation_m: float = 0.12,
    container_clearance_m: float = 0.15,
    max_tries: int = 80,
) -> dict[str, Any]:
    """Pick a target color and XYZ for every cube on the table.

    All cubes share the same sit height ``table_z + half_height``. Positions
    stay inside the play band (inset by ``spawn_pad``), at least
    ``min_separation_m`` apart, and away from the container.

    Returns
    -------
    dict with ``target_color`` (str) and ``poses`` (color → (3,) xyz).
    """
    rng = np.random.default_rng(int(seed))
    cols = tuple(str(c) for c in colors)
    if not cols:
        raise ValueError("mode2_layout.randomize needs at least one color")
    target = str(cols[int(rng.integers(len(cols)))])

    (xmin, xmax), (ymin, ymax) = xy_bounds
    lo_x = float(xmin) + float(spawn_pad)
    hi_x = float(xmax) - float(spawn_pad)
    lo_y = float(ymin) + float(spawn_pad)
    hi_y = float(ymax) - float(spawn_pad)
    if hi_x <= lo_x or hi_y <= lo_y:
        raise RuntimeError(
            f"mode2 play band empty after pad: x[{lo_x},{hi_x}] y[{lo_y},{hi_y}]"
        )

    z = float(table_z) + float(half_height)
    ctr = None
    if container_xy is not None:
        ctr = np.asarray(container_xy, dtype=np.float64).reshape(-1)[:2]

    poses: dict[str, np.ndarray] = {}
    for color in cols:
        placed = False
        for _ in range(int(max_tries)):
            xy = np.array(
                [float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y))],
                dtype=np.float64,
            )
            if ctr is not None and float(np.linalg.norm(xy - ctr)) < float(
                container_clearance_m
            ):
                continue
            if any(
                float(np.linalg.norm(xy - p[:2])) < float(min_separation_m)
                for p in poses.values()
            ):
                continue
            poses[color] = np.array([xy[0], xy[1], z], dtype=np.float64)
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"mode2_layout failed to place {color} after {max_tries} tries "
                f"(seed={seed}). Widen play band or lower min_separation."
            )

    return {"target_color": target, "poses": poses}
