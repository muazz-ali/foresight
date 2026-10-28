"""House / table / throw layout for Foresight.

Plain idea: DynamicVLA has nice tricks for picking a table in a random house
and throwing a fast object across it. We copy the *geometry* of those tricks
here, but we do NOT import DynamicVLA. The robot stays at the world origin.
We move the whole house so a chosen table edge (an "anchor") lands exactly at
(0, 0, 0) with no rotation. The table top then sits at z = 0.

Only pxr + numpy + scipy are used. No Isaac AppLauncher, no isaaclab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Quaternion (w, x, y, z) that turns a Y-up house into a Z-up house.
YUP_TO_ZUP_WXYZ = (0.7071068, 0.7071068, 0.0, 0.0)

# Pad used when checking if an anchor bumps into furniture / walls / cameras.
_COLLISION_PAD = 0.25


# --- small quaternion helpers (wxyz is our convention; scipy wants xyzw) ---
def _wxyz_to_xyzw(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def _rot_from_wxyz(q) -> Rotation:
    return Rotation.from_quat(_wxyz_to_xyzw(q))


# ---------------------------------- data ----------------------------------
@dataclass
class TableAnchor:
    pos: np.ndarray  # (3,) in Z-up house frame
    quat_wxyz: np.ndarray  # (4,)
    side: str  # "long" | "short"


@dataclass
class TableInfo:
    name: str
    height: float
    bbox_min: np.ndarray  # (3,) Z-up house frame (max_z forced to height)
    bbox_max: np.ndarray
    anchors: list[TableAnchor]  # uncollided only


@dataclass
class HouseChoice:
    usd_path: Path
    table: TableInfo
    anchor: TableAnchor
    spawn_pos: np.ndarray  # (3,) Isaac spawn translation
    spawn_quat_wxyz: np.ndarray  # (4,) Isaac spawn orientation
    table_bbox_min_w: np.ndarray  # after spawn transform, world
    table_bbox_max_w: np.ndarray


@dataclass
class PlayLayout:
    table_z: float  # 0.0 after our transform
    object_xy_bounds: tuple[tuple[float, float], tuple[float, float]]
    container_xy: np.ndarray  # (2,)
    container_xy_clip: tuple[tuple[float, float], tuple[float, float]]


# ------------------------------- scene files -------------------------------
def list_scene_usds(scene_dir: Path) -> list[Path]:
    """All USD files in a folder, sorted so runs are reproducible."""
    scene_dir = Path(scene_dir)
    out: list[Path] = []
    for ext in ("*.usd", "*.usda", "*.usdc"):
        out.extend(scene_dir.glob(ext))
    return sorted(out)


# ------------------------------- find tables -------------------------------
def _bbox_to_np(bbox) -> tuple[np.ndarray, np.ndarray]:
    bmin = np.array([bbox.min[0], bbox.min[1], bbox.min[2]], dtype=float)
    bmax = np.array([bbox.max[0], bbox.max[1], bbox.max[2]], dtype=float)
    return bmin, bmax


def _make_anchors(bmin, bmax, height: float) -> list[dict]:
    # Corner / anchor layout (copied from DynamicVLA scene_cfg):
    #   0   A   1
    #   +-------+
    # D |       | B
    #   +-------+
    #   3   C   2
    size = bmax - bmin
    long_x = size[0] > size[1]
    mid = lambda a, b: [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
    c0 = (bmin[0], bmin[1])
    c1 = (bmax[0], bmin[1])
    c2 = (bmax[0], bmax[1])
    c3 = (bmin[0], bmax[1])
    return [
        {  # A
            "pos": np.array(mid(c0, c1) + [height]),
            "quat": np.array([0.7071068, 0, 0, 0.7071068]),
            "side": "long" if long_x else "short",
            "collision": False,
        },
        {  # B
            "pos": np.array(mid(c1, c2) + [height]),
            "quat": np.array([0, 0, 0, 1]),
            "side": "short" if long_x else "long",
            "collision": False,
        },
        {  # C
            "pos": np.array(mid(c2, c3) + [height]),
            "quat": np.array([-0.7071068, 0, 0, 0.7071068]),
            "side": "long" if long_x else "short",
            "collision": False,
        },
        {  # D
            "pos": np.array(mid(c3, c0) + [height]),
            "quat": np.array([1, 0, 0, 0]),
            "side": "short" if long_x else "long",
            "collision": False,
        },
    ]


def _camera_world_pos(anchor: dict, cam: dict) -> np.ndarray:
    # Where this camera would sit if mounted on this anchor's frame.
    r = _rot_from_wxyz(anchor["quat"])
    return r.apply(np.asarray(cam["position"], dtype=float)) + anchor["pos"]


def _ranges_overlap(amin, amax, bmin, bmax) -> bool:
    return (
        amin[0] < bmax[0]
        and amax[0] > bmin[0]
        and amin[1] < bmax[1]
        and amax[1] > bmin[1]
        and amin[2] < bmax[2]
        and amax[2] > bmin[2]
    )


def _point_box_collides(prim_min, prim_max, point, pad: float = _COLLISION_PAD) -> bool:
    # A pad-sized box sitting on top of the anchor point (z from point up).
    amin = np.array([point[0] - pad, point[1] - pad, point[2]])
    amax = np.array([point[0] + pad, point[1] + pad, point[2] + pad])
    return _ranges_overlap(prim_min, prim_max, amin, amax)


def find_tables(usd_path: Path, cameras: list[dict]) -> list[TableInfo]:
    """Open a house USD and return tables we could grasp around.

    Steps (match DynamicVLA, but pure pxr + in-memory, never saved):
      - open stage, add a Y-up->Z-up orient on the default prim (in memory),
      - look under /house/furniture for prims whose name has 'Table' and that
        carry a 'height' attribute,
      - build 4 edge-midpoint anchors, drop any that would collide (0.25 m pad)
        with other furniture, walls, or a camera mounted on that anchor,
      - keep the table only if at least one long-side anchor survives,
      - force the bbox top to the table 'height'.

    `cameras` items are ``[{name, position}, ...]`` in metres (from
    ``static_camera.position``); only ``position`` is read here.
    """
    from pxr import Gf, Usd, UsdGeom  # local import: pxr is only needed here

    usd_path = Path(usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return []
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        return []

    # Turn the house Z-up, in memory only.
    xform = UsdGeom.Xformable(default_prim)
    xform.AddOrientOp().Set(Gf.Quatf(0.7071068, 0.7071068, 0, 0))

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    furniture_root = stage.GetPrimAtPath("/house/furniture")
    structure_root = stage.GetPrimAtPath("/house/structure")
    if not furniture_root or not furniture_root.IsValid():
        return []
    furniture = list(furniture_root.GetChildren())
    structure = (
        list(structure_root.GetChildren())
        if structure_root and structure_root.IsValid()
        else []
    )

    tables: list[TableInfo] = []
    for prim in furniture:
        if "Table" not in prim.GetName() or not prim.HasAttribute("height"):
            continue
        height = prim.GetAttribute("height").Get()
        if height is None:
            continue
        height = float(height)
        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bmin, bmax = _bbox_to_np(bbox)

        anchors = _make_anchors(bmin, bmax, height)

        # Drop anchors that collide with any other furniture / wall, or whose
        # mounted camera would collide.
        others = [p for p in furniture if p != prim] + structure
        for other in others:
            o_box = bbox_cache.ComputeWorldBound(other).ComputeAlignedBox()
            o_min, o_max = _bbox_to_np(o_box)
            for a in anchors:
                if a["collision"]:
                    continue
                if _point_box_collides(o_min, o_max, a["pos"]):
                    a["collision"] = True
                    continue
                for cam in cameras:
                    if _point_box_collides(o_min, o_max, _camera_world_pos(a, cam)):
                        a["collision"] = True
                        break

        kept = [a for a in anchors if not a["collision"]]
        if not any(a["side"] == "long" for a in kept):
            continue

        bmax = bmax.copy()
        bmax[2] = height  # table top is the height, not whatever sits on it
        tables.append(
            TableInfo(
                name=prim.GetName(),
                height=height,
                bbox_min=bmin,
                bbox_max=bmax,
                anchors=[
                    TableAnchor(pos=a["pos"], quat_wxyz=a["quat"], side=a["side"])
                    for a in kept
                ],
            )
        )
    return tables


# ------------------------------- pick a house ------------------------------
def _transform_bbox_corners(bmin, bmax, R_a_inv, t) -> tuple[np.ndarray, np.ndarray]:
    # 8 corners of the Z-up table box -> world, then re-fit an axis box.
    xs = [bmin[0], bmax[0]]
    ys = [bmin[1], bmax[1]]
    zs = [bmin[2], bmax[2]]
    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z])
                pts.append(R_a_inv @ p + t)
    pts = np.stack(pts, axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def pick_house(
    scene_dir,
    asset_rng: np.random.Generator,
    cameras,
    *,
    max_tries: int = 40,
) -> HouseChoice:
    """Pick a random house + table + long-side anchor, and work out the spawn
    transform that drops that anchor onto the origin with identity rotation.

    Math (anchor is in the Z-up house frame):
      R_yz    = rotation of YUP_TO_ZUP_WXYZ
      R_spawn = inv(R_anchor) @ R_yz          # house spawn orientation
      t_spawn = -inv(R_anchor) @ p_anchor     # house spawn translation
      p_world = inv(R_anchor) @ p_zup + t_spawn   # any Z-up table point -> world
    """
    usds = list_scene_usds(scene_dir)
    if not usds:
        raise FileNotFoundError(f"No USD scenes under {scene_dir}")

    R_yz = _rot_from_wxyz(YUP_TO_ZUP_WXYZ)

    for _ in range(int(max_tries)):
        usd_path = usds[int(asset_rng.integers(len(usds)))]
        tables = find_tables(usd_path, cameras)
        if not tables:
            continue
        table = tables[int(asset_rng.integers(len(tables)))]
        long_anchors = [a for a in table.anchors if a.side == "long"]
        if not long_anchors:
            continue
        anchor = long_anchors[int(asset_rng.integers(len(long_anchors)))]

        R_anchor = _rot_from_wxyz(anchor.quat_wxyz)
        R_a_inv = R_anchor.inv()
        R_a_inv_m = R_a_inv.as_matrix()
        t_spawn = -(R_a_inv_m @ anchor.pos)

        R_spawn = R_a_inv * R_yz
        spawn_quat_wxyz = _xyzw_to_wxyz(R_spawn.as_quat())

        bbox_min_w, bbox_max_w = _transform_bbox_corners(
            table.bbox_min, table.bbox_max, R_a_inv_m, t_spawn
        )
        return HouseChoice(
            usd_path=usd_path,
            table=table,
            anchor=anchor,
            spawn_pos=t_spawn,
            spawn_quat_wxyz=spawn_quat_wxyz,
            table_bbox_min_w=bbox_min_w,
            table_bbox_max_w=bbox_max_w,
        )

    raise RuntimeError(f"No table found in {max_tries} random scenes under {scene_dir}")


# ----------------------------- play layout box -----------------------------
def _inner_half(a0: float, a1: float) -> tuple[float, float]:
    # Middle 50% of a span (25% .. 75%).
    lo = a0 + 0.25 * (a1 - a0)
    hi = a0 + 0.75 * (a1 - a0)
    return (min(lo, hi), max(lo, hi))


def compute_play_layout(
    table_bbox_min_w,
    table_bbox_max_w,
    *,
    max_reach_dist: float = 0.85,
    play_x: tuple[float, float] = (0.40, 0.80),
    play_y_half: float = 0.14,
    # Bowl X seed (m). Y is always the +Y inner-half edge (not this |y|).
    # Scene may pass cfg scene.container_xy; layout overwrites Y.
    container_y: float = 0.35,
    container_x: float = 0.43,
    container_y_half_width: float = 0.04,
    min_play_y_half: float = 0.08,
    container_gap: float = 0.03,
    table_edge_pad: float = 0.02,
) -> PlayLayout:
    """Where objects may spawn (play rect) and where the drop container sits.

    Robot at origin facing +X. Near X stays in the reachable catch zone; **far
    X** is the real table +X rim so throws get a long track runway. Bowl sits
    on the **real +Y table rim** (not inner-half), X = seed − 3 cm toward the
    robot, clear of the throw path.
    """
    bmin = np.asarray(table_bbox_min_w, dtype=float)
    bmax = np.asarray(table_bbox_max_w, dtype=float)

    ix0, ix1 = _inner_half(bmin[0], bmax[0])
    iy0, iy1 = _inner_half(bmin[1], bmax[1])

    # Near X: catch zone (inner half ∩ reach floor). Far X: **real table +X
    # rim** — do NOT clip by arm reach (object runs into the catch zone).
    # max_reach_dist still clamps the bowl place pose below.
    table_far = float(bmax[0]) - float(table_edge_pad)
    px0 = max(play_x[0], ix0, 0.32)
    px1 = float(table_far)
    # Need enough runway to chase under 50 cm/s. Smoke i16 had only ~15 cm of
    # play and closed once then slipped off the near edge.
    min_play_x = 0.35
    if px1 - px0 < min_play_x:
        # Prefer keeping the far rim; slide the near edge in if needed.
        px0 = min(px0, px1 - min_play_x)
        px0 = max(px0, 0.30)
    if px1 - px0 < min_play_x:
        raise ValueError(
            f"play rect too thin in X: x=({px0:.3f},{px1:.3f}) "
            f"(need ≥{min_play_x:.2f} m); table_far={table_far:.3f}"
        )

    hw = container_y_half_width
    # Seed X nudged 3 cm toward the robot (−X). Smoke H5 had the bowl at
    # x≈0.35 mid-reach; −3 cm keeps place closer to the catch zone.
    cx = float(np.clip(float(container_x) - 0.03, ix0 + 0.06, ix1 - 0.06))

    # Toward the **real +Y table rim**, but clamped to arm reach so place does
    # not DiffIK-swing the hand off the table (rim-only put cy≈0.74 → timeouts
    # / ee-off-table). Inner-half alone sat at ~0.36 (looked mid-table).
    _ = container_y  # API compat; Y is forced toward the +Y rim, not this seed.
    table_y_far = float(bmax[1]) - float(table_edge_pad)
    cy_rim = float(table_y_far - hw)
    # ~72% of max reach keeps Franka place inside the table footprint.
    reach_xy = 0.72 * float(max_reach_dist)
    cy_reach = float(np.sqrt(max(reach_xy * reach_xy - cx * cx, 0.05)))
    cy_min = float(play_y_half + container_gap + hw)  # clear of the throw band
    cy = float(np.clip(min(cy_rim, cy_reach), cy_min, cy_rim))
    if cy <= hw:
        raise ValueError(
            f"table too narrow for +Y bowl pocket: cy={cy:.3f} (need >{hw})"
        )
    # Play band stays on the robot side of the bowl (below +Y pocket).
    room = cy - hw - container_gap  # play ymax must be ≤ this
    py_half = min(play_y_half, max(0.0, room), abs(iy0), abs(iy1))
    if py_half < min_play_y_half:
        raise ValueError(
            "table too narrow for play+bowl: "
            f"py_half={py_half:.3f} (need ≥{min_play_y_half})"
        )
    py0 = max(-py_half, iy0)
    py1 = min(py_half, iy1, room)
    if not (py0 < py1):
        raise ValueError(
            f"play rect has no Y span: y=({py0:.3f},{py1:.3f})"
        )
    object_xy_bounds = ((float(px0), float(px1)), (float(py0), float(py1)))
    container_xy = np.array([cx, cy], dtype=float)

    # Small pocket box around the container, never crossing into the play band.
    clip_x0 = max(cx - 0.06, ix0)
    clip_x1 = min(cx + 0.06, ix1)
    clip_y0 = max(cy - hw, py1, iy0)  # start at/above play ymax
    clip_y1 = min(cy + hw, table_y_far)
    if not (clip_x0 < clip_x1 and clip_y0 < clip_y1):
        raise ValueError("container clip box empty after play shrink")
    container_xy_clip = (
        (float(clip_x0), float(clip_x1)),
        (float(clip_y0), float(clip_y1)),
    )

    return PlayLayout(
        table_z=0.0,
        object_xy_bounds=object_xy_bounds,
        container_xy=container_xy,
        container_xy_clip=container_xy_clip,
    )


# ------------------------------- throw sampler -----------------------------
# Categories laid on their side so cylinders can roll (DynamicVLA-style tip).
ROLL_TIP_CATEGORIES = frozenset({"can", "bottle", "cup", "beer", "fcan"})


def tip_on_side_quat_wxyz(vel_xy, rng: np.random.Generator) -> np.ndarray:
    """Tip a tall object onto its side; long axis across travel (for rolling).

    Same idea as DynamicVLA ``get_object_init_quat``: ±90° tip about X, yaw =
    travel heading. Returns quaternion wxyz. Call only for roll categories.
    """
    v = np.asarray(vel_xy, dtype=float).reshape(2)
    n = float(np.linalg.norm(v))
    yaw = float(np.arctan2(v[1], v[0])) if n > 1e-9 else 0.0
    tip = 0.5 * np.pi * float(rng.choice([-1.0, 1.0]))
    q_xyzw = Rotation.from_euler("xyz", [tip, 0.0, yaw]).as_quat()
    return _xyzw_to_wxyz(q_xyzw)


def sit_half_height_m(size_xyz, *, tipped: bool) -> float:
    """Table sit height (half extent along world Z) for upright or tipped mesh."""
    size = np.asarray(size_xyz, dtype=float).reshape(3)
    if tipped:
        # Upright Z was height; on its side the diameter (XY) is vertical.
        return 0.5 * float(min(abs(size[0]), abs(size[1])))
    return 0.5 * float(abs(size[2]))


def _padded_play(object_xy_bounds, spawn_pad: float):
    (xmin, xmax), (ymin, ymax) = object_xy_bounds
    pxmin, pxmax = xmin + spawn_pad, xmax - spawn_pad
    pymin, pymax = ymin + spawn_pad, ymax - spawn_pad
    if pxmin > pxmax:
        pxmin = pxmax = 0.5 * (xmin + xmax)
    if pymin > pymax:
        pymin = pymax = 0.5 * (ymin + ymax)
    return (pxmin, pxmax), (pymin, pymax), (xmin, xmax, ymin, ymax)


def _finish_throw(spawn, target, speed, reach_dir, side, pad_box) -> dict:
    """Clip spawn/target, build unit velocity * speed."""
    (pxmin, pxmax), (pymin, pymax) = pad_box
    spawn = np.asarray(spawn, dtype=float).copy()
    target = np.asarray(target, dtype=float).copy()
    spawn[0] = float(np.clip(spawn[0], pxmin, pxmax))
    spawn[1] = float(np.clip(spawn[1], pymin, pymax))
    target[0] = float(np.clip(target[0], pxmin, pxmax))
    target[1] = float(np.clip(target[1], pymin, pymax))
    d = target - spawn
    n = float(np.linalg.norm(d))
    if n < 1e-3:
        push = -np.asarray(reach_dir, dtype=float).reshape(2)
        pn = float(np.linalg.norm(push))
        push = (push / pn) if pn > 1e-6 else np.array([-1.0, 0.0])
        spawn = np.clip(target + push * 0.15, [pxmin, pymin], [pxmax, pymax])
        d = target - spawn
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            d = np.array([1.0, 0.0])
            n = 1.0
    vel_xy = d / n * float(speed)
    return {
        "spawn": spawn,
        "vel_xy": vel_xy,
        "target": target,
        "reach_dir": np.asarray(reach_dir, dtype=float).reshape(2),
        "side": side,
    }


def _heading_from_edge_deg(theta_deg: float) -> np.ndarray:
    """Unit velocity from a 180° fan at the far (+X) table edge.

    Standing on the far edge looking at the robot:
      0°   → along +Y (table edge left)
      90°  → straight at the robot (−X)
      180° → along −Y (table edge right)

    ``vx = -sin(θ)``, ``vy = cos(θ)`` so the open half-plane always has vx ≤ 0.
    """
    t = np.deg2rad(float(theta_deg))
    return np.array([-np.sin(t), np.cos(t)], dtype=float)


def _coast_reaches_catch(
    spawn,
    heading,
    speed: float,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    until_x: float,
    margin: float = 0.01,
    dt: float = 0.02,
    tmax: float = 3.0,
) -> bool:
    """True if a scripted coast stays in the play box until ``x <= until_x``.

    Why: a full 180° fan from the far edge includes grazing throws that leave
    the table before the hand can meet them (smoke off-table at ~±80° from −X).
    Callers only keep headings that pass this GT check.
    """
    if float(speed) <= 1e-9:
        return True
    p = np.asarray(spawn, dtype=float).reshape(2).copy()
    v = np.asarray(heading, dtype=float).reshape(2) * float(speed)
    if float(v[0]) >= -1e-9:
        return False  # not heading toward the robot
    # Near/side walls get a pad; far +X is the spawn rim — do not inset xmax
    # or the first step from the rim looks "already off" and every throw fails.
    lo_x = float(xmin) + float(margin)
    hi_x = float(xmax) + 1e-6
    lo_y = float(ymin) + float(margin)
    hi_y = float(ymax) - float(margin)
    goal_x = float(until_x)
    t = 0.0
    while t <= float(tmax) + 1e-12:
        if float(p[0]) <= goal_x:
            return True
        p = p + v * float(dt)
        t += float(dt)
        if not (lo_x <= float(p[0]) <= hi_x and lo_y <= float(p[1]) <= hi_y):
            return False
    return False


def _throw_toward_robot(
    rng: np.random.Generator,
    object_xy_bounds,
    *,
    speed: float,
    spawn_pad: float,
    catch_band: tuple[float, float],
    lateral: float,
    jitter: float,
) -> dict:
    """Spawn on the far +X edge; aim with a catchable fan toward the robot.

    ``jitter`` (= ``throw_randomness``) in [0, 1] sets the fan width around 90°:
      0 → only 90° (straight −X center line)
      1 → try uniform on [0°, 180°], keep only paths that stay in the play
          band until the catch zone (rejects grazing off-table throws)
      r → same with fan [90° − 90°r, 90° + 90°r]

    ``lateral`` unused (API compat). Never aims at the bowl.
    """
    del lateral
    (xmin, xmax), (ymin, ymax) = object_xy_bounds
    # Pad only Y so the object stays on the band; keep spawn on the true +X rim
    # (do not inset xmax — that made throws look mid-table on short play bands).
    pymin = float(ymin + spawn_pad)
    pymax = float(ymax - spawn_pad)
    if pymin > pymax:
        pymin = pymax = 0.5 * (ymin + ymax)
    pxmin = float(xmin + spawn_pad)
    pxmax = float(xmax)  # far table/play rim
    j = float(np.clip(jitter, 0.0, 1.0))

    # Catch zone far edge: path must stay on the band at least this far in.
    until_x = float(np.clip(max(catch_band), pxmin + 0.05, pxmax - 0.10))

    # Spawn on the far +X edge. r=0 pins Y to center; else anywhere on the edge.
    spawn_x = float(pxmax)
    mid_y = 0.5 * (pymin + pymax)
    spawn_y = mid_y if j <= 0.0 else float(rng.uniform(pymin, pymax))
    spawn = np.array([spawn_x, spawn_y], dtype=float)

    half_deg = 90.0 * j
    theta_lo = 90.0 - half_deg
    theta_hi = 90.0 + half_deg

    def _pack(theta_deg: float) -> dict:
        heading = _heading_from_edge_deg(theta_deg)
        vel_xy = heading * float(speed)
        depth = 0.35 if j <= 0.0 else float(rng.uniform(0.25, 0.50))
        target = spawn + heading * depth
        target[0] = float(np.clip(target[0], pxmin, pxmax))
        target[1] = float(np.clip(target[1], pymin, pymax))
        if j <= 0.0:
            side = "front"
        else:
            side = "angle_left" if theta_deg < 90.0 else "angle_right"
        return {
            "spawn": spawn,
            "vel_xy": vel_xy,
            "target": target,
            "reach_dir": np.array([1.0, 0.0], dtype=float),
            "side": side,
            "throw_angle_deg": float(theta_deg),
        }

    if j <= 0.0:
        return _pack(90.0)

    # Rejection sample: keep diversity, drop headings that leave the band
    # before the catch zone (those were the off-table smoke fails).
    for _ in range(48):
        theta_deg = float(rng.uniform(theta_lo, theta_hi))
        heading = _heading_from_edge_deg(theta_deg)
        if _coast_reaches_catch(
            spawn,
            heading,
            speed,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            until_x=until_x,
        ):
            return _pack(theta_deg)

    # Fallback: straight at the robot (always catchable from the far rim).
    return _pack(90.0)


# Old name kept so call sites / tests that still import it keep working.
_throw_edge_toward_robot = _throw_toward_robot


def _sample_throw_full(
    rng: np.random.Generator,
    object_xy_bounds,
    *,
    speed: float,
    spawn_pad: float = 0.03,
    catch_band: tuple[float, float] = (0.38, 0.55),
    lateral: float = 0.18,
    container_xy=None,
    throw_randomness: float = 1.0,
) -> dict:
    """Core throw math: far +X edge, 180° fan toward the robot.

    ``throw_randomness`` r ∈ [0, 1] = fan width around the 90° center line:
      0 → straight at robot; 1 → uniform over the full 180° half-circle.
    ``container_xy`` is ignored (kept for call-site compatibility).
    """
    del container_xy  # throws must not aim at the bowl
    (pxmin, pxmax), (pymin, pymax), _ = _padded_play(object_xy_bounds, spawn_pad)
    U = lambda a, b: float(rng.uniform(a, b))
    r = float(np.clip(throw_randomness, 0.0, 1.0))

    if speed <= 1e-6:
        spawn = np.array([U(pxmin, pxmax), U(pymin, pymax)])
        return {
            "spawn": spawn,
            "vel_xy": np.zeros(2),
            "target": spawn.copy(),
            "reach_dir": np.array([1.0, 0.0]),
            "side": "still",
            "throw_angle_deg": 90.0,
        }

    return _throw_toward_robot(
        rng,
        object_xy_bounds,
        speed=speed,
        spawn_pad=spawn_pad,
        catch_band=catch_band,
        lateral=lateral,
        jitter=r,
    )


def sample_throw(
    rng: np.random.Generator,
    object_xy_bounds,
    *,
    speed: float,
    spawn_pad: float = 0.03,
    catch_band: tuple[float, float] = (0.38, 0.55),
    lateral: float = 0.18,
    container_xy=None,
    throw_randomness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a spawn point and a velocity for a thrown object.

    Returns (pos (3,) with z=0, lin_vel (3,) with z=0). Spawns on the far
    (+X) edge. ``throw_randomness`` sets the 180° fan width: 0 = 90° straight
    at the robot, 1 = full half-circle randomness.
    """
    out = _sample_throw_full(
        rng,
        object_xy_bounds,
        speed=speed,
        spawn_pad=spawn_pad,
        catch_band=catch_band,
        lateral=lateral,
        container_xy=container_xy,
        throw_randomness=throw_randomness,
    )
    s = out["spawn"]
    v = out["vel_xy"]
    pos = np.array([s[0], s[1], 0.0])
    lin_vel = np.array([v[0], v[1], 0.0])
    return pos, lin_vel


# --------------------------------- lighting --------------------------------
def look_at_quat_wxyz(from_pos, to_pos=(0.0, 0.0, 0.0)) -> list[float]:
    """Quaternion (wxyz) that aims a USD DistantLight from ``from_pos`` at ``to_pos``.

    USD DistantLight / Camera emit and look along local **-Z**. So the rotation
    must put local -Z on the vector toward the target (not local +X).
    DynamicVLA's ``get_quat_from_look_at`` stacks [fwd, right, up] as columns —
    that aims +X at the target and leaves sun rays almost sideways, which is
    why robot shadows looked lit from the wrong side of the third-person cam.
    """
    fwd = np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float)
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    fwd = fwd / n
    # Local +Z points away from the target so local -Z = emit/view direction.
    z_axis = -fwd
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(world_up, z_axis)
    xn = float(np.linalg.norm(x_axis))
    if xn < 1e-9:
        # Looking straight up/down — pick any side axis.
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        x_axis = x_axis / xn
    y_axis = np.cross(z_axis, x_axis)
    mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def sample_light(
    rng: np.random.Generator,
    camera_pos=(1.2, 0.0, 0.70),
    *,
    lighting: dict | None = None,
) -> dict:
    """Distant-light knobs in DynamicVLA style, keyed to our third-person cam.

    DynamicVLA randomizes temp / intensity the same way. Their sun sits far away
    and aims at the origin. We keep that, but park the light **behind or above**
    ``static_cam`` (never between the camera and the table), so the main view
    is lit from the camera side — matches a real third-person rig.
    """
    lighting = lighting or {}
    temp_lo, temp_hi = lighting.get("temperature", [4000, 8000])
    inten_lo, inten_hi = lighting.get("intensity", [150, 750])
    back_lo, back_hi = lighting.get("behind_m", [3.0, 12.0])
    up_lo, up_hi = lighting.get("above_m", [2.0, 15.0])
    side_lo, side_hi = lighting.get("side_m", [-2.0, 2.0])

    cam = np.asarray(camera_pos, dtype=float).reshape(3)
    # Horizontal "away from table through the camera" axis (+X for our rig).
    radial = np.array([cam[0], cam[1], 0.0], dtype=float)
    rn = float(np.linalg.norm(radial))
    radial = (
        np.array([1.0, 0.0, 0.0], dtype=float)
        if rn < 1e-6
        else radial / rn
    )
    perp = np.array([-radial[1], radial[0], 0.0], dtype=float)

    style = str(rng.choice(["behind", "above", "behind_above"]))
    back = float(rng.uniform(back_lo, back_hi))
    up = float(rng.uniform(up_lo, up_hi))
    side = float(rng.uniform(side_lo, side_hi))

    if style == "behind":
        # Past the camera, slightly raised so it is not in the lens.
        pos = cam + radial * back + perp * side + np.array([0.0, 0.0, float(rng.uniform(0.5, 3.0))])
    elif style == "above":
        # Mostly over the camera, tiny back offset so it stays camera-side.
        pos = cam + radial * float(rng.uniform(0.0, 2.0)) + perp * side + np.array([0.0, 0.0, up])
    else:
        # Behind and high (sun over the operator's shoulder).
        pos = cam + radial * back + perp * side + np.array([0.0, 0.0, up])

    return {
        "temperature": int(rng.integers(int(temp_lo), int(temp_hi) + 1)),
        "intensity": float(rng.uniform(float(inten_lo), float(inten_hi))),
        "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        "style": style,
    }
