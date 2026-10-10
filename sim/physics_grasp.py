"""Small helpers for the real-physics grasp (no weld) and the wrist depth gate.

Numpy-only except ``bind_grip_material`` (Isaac, import after AppLauncher).
Must not: decide when to grasp. The expert (sim/state_machine.py) does that.
"""

from __future__ import annotations

import numpy as np


def xy_outside_box(xy, box_min_xy, box_max_xy) -> bool:
    """True if the XY point is past the table footprint (the object would fall)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    return bool(np.any(xy < np.asarray(box_min_xy)) or np.any(xy > np.asarray(box_max_xy)))


def gripper_depth_m(
    depth: np.ndarray | None,
    rows: tuple[float, float] = (0.70, 1.0),
    cols: tuple[float, float] = (0.40, 0.60),
) -> float:
    """Nearest valid depth (m) in the window between the fingertips of the wrist view.

    The wrist cam looks down the fingers: the object shows up at the BOTTOM centre
    of the image, between the two finger tips (image centre is just table).
    ``depth`` is (H, W) or (H, W, 1). Returns NaN if there is no image or no
    finite reading (never 0.0, so "no data" cannot pass for "touching").
    """
    if depth is None:
        return float("nan")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    assert d.ndim == 2, d.shape
    h, w = d.shape
    crop = d[int(h * rows[0]) : int(h * rows[1]), int(w * cols[0]) : int(w * cols[1])]
    ok = crop[np.isfinite(crop) & (crop > 0.0)]
    return float(ok.min()) if ok.size else float("nan")


def bind_grip_material(prim_path: str, friction: float) -> None:
    """Give the object a sticky material. ``max`` combine so it beats the finger pad.

    USD objects spawn without a physics material (Isaac Lab ``UsdFileCfg`` does not
    bind one), so we create one under the object and bind it to its colliders.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(friction),
        dynamic_friction=float(friction),
        friction_combine_mode="max",
    )
    mat_path = f"{prim_path}/gripMaterial"
    mat.func(mat_path, mat)
    bind_physics_material(prim_path, mat_path)
