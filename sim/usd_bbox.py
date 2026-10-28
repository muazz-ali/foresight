"""USD visual / collision AABB helpers (pxr only — no Isaac).

Use collision size for w_pre, z_grasp half-height, and the gripper width gate.
Visual size is for audit / debug only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_GEOM = None  # lazy


def _geom_types():
    from pxr import UsdGeom

    return (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
    )


def _finite_box(rng) -> tuple[np.ndarray, np.ndarray] | None:
    mn = np.array(rng.GetMin(), dtype=np.float64)
    mx = np.array(rng.GetMax(), dtype=np.float64)
    if not (np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
        return None
    if np.any(mx <= mn + 1e-12):
        return None
    # Reject the empty-mesh sentinel (±1e38).
    if float(np.max(np.abs(mn))) > 1e6 or float(np.max(np.abs(mx))) > 1e6:
        return None
    return mn, mx


def _union(boxes: list[tuple[np.ndarray, np.ndarray]]):
    if not boxes:
        return None, None, None
    mn = np.min(np.stack([b[0] for b in boxes], axis=0), axis=0)
    mx = np.max(np.stack([b[1] for b in boxes], axis=0), axis=0)
    return mn, mx, mx - mn


def usd_visual_collision_boxes(usd_path: Path) -> dict:
    """Return visual and collision AABB size/centre in the USD root frame.

    Raises ``RuntimeError`` if neither box can be read.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD open failed: {usd_path}")
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"USD has no default prim: {usd_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    vis, col = [], []
    for prim in stage.Traverse():
        if not any(prim.IsA(t) for t in _geom_types()):
            continue
        box = _finite_box(cache.ComputeWorldBound(prim).ComputeAlignedBox())
        if box is None:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col.append(box)
        else:
            vis.append(box)

    vm, vx, vs = _union(vis)
    cm, cx, cs = _union(col)
    if cs is None and vs is None:
        raise RuntimeError(f"USD bbox failed: no finite geom in {usd_path}")
    # Prefer collision; fall back to visual if the mesh has no CollisionAPI.
    size = cs if cs is not None else vs
    centre = (0.5 * (cm + cx)) if cs is not None else (0.5 * (vm + vx))
    return {
        "visual_size": vs,
        "visual_centre": None if vs is None else 0.5 * (vm + vx),
        "collision_size": cs,
        "collision_centre": None if cs is None else 0.5 * (cm + cx),
        "size": np.asarray(size, dtype=np.float64),
        "centre": np.asarray(centre, dtype=np.float64),
        "used": "collision" if cs is not None else "visual",
    }


def usd_collision_size(usd_path: Path) -> np.ndarray:
    """Collision AABB size (m). Falls back to visual if no collider prims."""
    return usd_visual_collision_boxes(usd_path)["size"]


def usd_collision_centre(usd_path: Path) -> np.ndarray:
    """Collision AABB centre in USD root frame (m). Used to recenter spawn."""
    return usd_visual_collision_boxes(usd_path)["centre"]
