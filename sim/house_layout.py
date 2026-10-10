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
    max_reach_dist: float = 0.75,
    play_x: tuple[float, float] = (0.40, 0.68),
    play_y_half: float = 0.14,
    container_y: float = 0.30,
    container_x: float = 0.50,
    container_y_half_width: float = 0.04,
    min_play_y_half: float = 0.08,
    container_gap: float = 0.03,
) -> PlayLayout:
    """Where objects may spawn (play rect) and where the drop container sits.

    Robot is at the origin facing +X into the table. We keep the play rect
    inside the table's middle 50% and inside the arm's reach. The container
    goes to one side (prefer +Y), kept on the table and clear of the play band.

    On a narrow table we shrink the play band in Y so the bowl still has a
    clear side pocket. If that would make the play band too thin, we raise.
    """
    bmin = np.asarray(table_bbox_min_w, dtype=float)
    bmax = np.asarray(table_bbox_max_w, dtype=float)

    ix0, ix1 = _inner_half(bmin[0], bmax[0])
    iy0, iy1 = _inner_half(bmin[1], bmax[1])

    # Play rect X = desired band ∩ table inner half ∩ reach limits.
    px0 = max(play_x[0], ix0, 0.32)
    px1 = min(play_x[1], ix1, max_reach_dist - 0.08)
    if px1 - px0 < 0.12:
        raise ValueError(
            f"play rect too thin in X: x=({px0:.3f},{px1:.3f}) (need ≥0.12 m)"
        )

    hw = container_y_half_width
    cx = float(np.clip(container_x, ix0 + 0.06, ix1 - 0.06))

    # Park the bowl near the preferred |y|, clipped to the table inner half.
    # Prefer +Y; fall back to -Y if that side has more room.
    def side_room(sign: float) -> tuple[float, float]:
        cy = float(np.clip(sign * container_y, iy0 + hw, iy1 - hw))
        # How far we can push the play edge before we hit the bowl pocket.
        if sign > 0:
            room = (cy - hw - container_gap)  # play ymax must be ≤ this
        else:
            room = -(cy + hw + container_gap)  # |play ymin| must be ≤ this
        return cy, room

    cy_p, room_p = side_room(1.0)
    cy_m, room_m = side_room(-1.0)
    if room_p >= room_m:
        sign, cy, room = 1.0, cy_p, room_p
    else:
        sign, cy, room = -1.0, cy_m, room_m

    # Shrink play Y so the bowl keeps a clear gap; never grow past play_y_half.
    py_half = min(play_y_half, max(0.0, room), abs(iy0), abs(iy1))
    if py_half < min_play_y_half:
        raise ValueError(
            "table too narrow for play+bowl: "
            f"py_half={py_half:.3f} (need ≥{min_play_y_half})"
        )
    py0 = max(-py_half, iy0)
    py1 = min(py_half, iy1)
    if not (py0 < py1):
        raise ValueError(
            f"play rect has no Y span: y=({py0:.3f},{py1:.3f})"
        )
    object_xy_bounds = ((float(px0), float(px1)), (float(py0), float(py1)))
    container_xy = np.array([cx, cy], dtype=float)

    # Small pocket box around the container, never crossing into the play band.
    clip_x0 = max(cx - 0.06, ix0)
    clip_x1 = min(cx + 0.06, ix1)
    clip_y0 = max(cy - hw, iy0)
    clip_y1 = min(cy + hw, iy1)
    if sign > 0:
        clip_y0 = max(clip_y0, py1)
    else:
        clip_y1 = min(clip_y1, py0)
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
def _sample_throw_full(
    rng: np.random.Generator,
    object_xy_bounds,
    *,
    speed: float,
    spawn_pad: float = 0.03,
    catch_band: tuple[float, float] = (0.38, 0.55),
    lateral: float = 0.18,
    container_xy=None,
) -> dict:
    """Core throw math. Returns spawn/vel plus the catch target it aims at.

    Aim a catch point in front of the robot (reachable band), then spawn from a
    side or the far edge so the straight path crosses that point.

    When ``container_xy`` (the bowl) is set, the catch sits on the way toward
    the bowl and spawns come from the opposite side / diagonals — so a miss
    heads at the bowl (fail-fast pass-through) instead of random table edges.
    """
    (xmin, xmax), (ymin, ymax) = object_xy_bounds

    # Padded bounds: inset from the walls so objects don't spawn on an edge.
    pxmin, pxmax = xmin + spawn_pad, xmax - spawn_pad
    pymin, pymax = ymin + spawn_pad, ymax - spawn_pad
    if pxmin > pxmax:
        pxmin = pxmax = 0.5 * (xmin + xmax)
    if pymin > pymax:
        pymin = pymax = 0.5 * (ymin + ymax)

    U = lambda a, b: float(rng.uniform(a, b))

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    center = np.array([cx, cy], dtype=float)

    # Still object: random spawn, no velocity, no aim.
    if speed <= 1e-6:
        spawn = np.array([U(pxmin, pxmax), U(pymin, pymax)])
        return {
            "spawn": spawn,
            "vel_xy": np.zeros(2),
            "target": spawn.copy(),
            "reach_dir": np.array([1.0, 0.0]),
            "side": "still",
        }

    bowl = None if container_xy is None else np.asarray(container_xy, dtype=float).reshape(2)

    if bowl is not None:
        # Straight / diagonal throws toward the bowl, catch still inside play.
        toward = bowl - center
        tn = float(np.linalg.norm(toward))
        if tn < 1e-6:
            toward = np.array([0.0, 1.0 if bowl[1] >= 0.0 else -1.0], dtype=float)
            tn = 1.0
        toward = toward / tn
        # Across-table axis (left/right relative to bowl direction).
        across = np.array([-toward[1], toward[0]], dtype=float)

        # Catch: in the play rect, biased onto the bowl-ward half (+ small jitter).
        # Keep it reachable in X (catch_band) when the bowl is mostly off to +Y/-Y.
        along_x = U(*catch_band)
        # Blend robot-forward catch with a step toward the bowl (not into the pocket).
        bowl_pull = U(0.35, 0.75)  # how far from centre toward the bowl edge of play
        # Play half-extent along `toward` from centre.
        # Clamp catch into padded play bounds below.
        target = center + toward * (bowl_pull * min(abs(ymax - cy), abs(cy - ymin), 0.12))
        target[0] = float(np.clip(along_x + U(-0.04, 0.04), pxmin, pxmax))
        target[1] = float(np.clip(target[1] + U(-0.03, 0.03), pymin, pymax))

        # Spawn: far side, left diagonal, or right diagonal — path aims at bowl.
        side = str(rng.choice(["far", "diag_left", "diag_right", "side_left", "side_right"]))
        if side == "far":
            # Opposite the bowl, still in play.
            spawn = target - toward * U(0.18, 0.40) + across * U(-0.06, 0.06)
        elif side.startswith("diag"):
            s = 1.0 if side.endswith("left") else -1.0
            spawn = target - toward * U(0.10, 0.30) + s * across * U(0.12, 0.28)
        else:
            s = 1.0 if side.endswith("left") else -1.0
            spawn = target + s * across * U(0.16, 0.32) - toward * U(0.0, 0.12)

        reach_dir = toward.copy()
    else:
        # Legacy: catch band in front of the robot (no bowl hint).
        cnorm = float(np.linalg.norm(center))
        if cx > 0 and cnorm > 1e-6:
            reach_dir = center / cnorm
        else:
            reach_dir = np.array([1.0, 0.0], dtype=float)
        perp = np.array([-reach_dir[1], reach_dir[0]], dtype=float)
        target = reach_dir * U(*catch_band) + perp * U(-lateral, lateral)

        side = str(rng.choice(["left", "right", "front"]))
        if side == "front":
            spawn = target + reach_dir * U(0.25, 0.45)
        else:
            s = 1.0 if side == "left" else -1.0
            spawn = target + s * perp * U(0.30, 0.50) + reach_dir * U(0.0, 0.25)

    spawn = np.asarray(spawn, dtype=float).copy()
    spawn[0] = float(np.clip(spawn[0], pxmin, pxmax))
    spawn[1] = float(np.clip(spawn[1], pymin, pymax))
    target = np.asarray(target, dtype=float).copy()
    target[0] = float(np.clip(target[0], pxmin, pxmax))
    target[1] = float(np.clip(target[1], pymin, pymax))

    # If spawn landed on the catch point, push it away from the bowl / +X.
    d = target - spawn
    n = float(np.linalg.norm(d))
    if n < 1e-3:
        push = -reach_dir if bowl is None else -(bowl - center)
        pn = float(np.linalg.norm(push[:2]))
        push = (push[:2] / pn) if pn > 1e-6 else np.array([-1.0, 0.0])
        spawn = np.clip(target + push * 0.15, [pxmin, pymin], [pxmax, pymax])
        d = target - spawn
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            d = np.array([1.0, 0.0])
            n = 1.0
    vel_xy = d / n * speed

    # With a bowl: prefer paths where the catch sits between spawn and bowl.
    if bowl is not None:
        to_bowl = bowl - spawn
        if float(np.dot(to_bowl, d)) < 0.0:
            # Flip catch past spawn toward the bowl.
            target = spawn + to_bowl / max(float(np.linalg.norm(to_bowl)), 1e-6) * min(
                0.20, 0.5 * float(np.linalg.norm(to_bowl))
            )
            target[0] = float(np.clip(target[0], pxmin, pxmax))
            target[1] = float(np.clip(target[1], pymin, pymax))
            d = target - spawn
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                d = to_bowl
                n = float(np.linalg.norm(d)) or 1.0
            vel_xy = d / n * speed

    return {
        "spawn": spawn,
        "vel_xy": vel_xy,
        "target": target,
        "reach_dir": reach_dir,
        "side": side,
    }


def sample_throw(
    rng: np.random.Generator,
    object_xy_bounds,
    *,
    speed: float,
    spawn_pad: float = 0.03,
    catch_band: tuple[float, float] = (0.38, 0.55),
    lateral: float = 0.18,
    container_xy=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a spawn point and a velocity for a thrown object.

    Returns (pos (3,) with z=0, lin_vel (3,) with z=0). The caller adds
    table_z + half object height to z. Pass ``container_xy`` so throws aim
    toward the bowl (straight / diagonal / side). See `_sample_throw_full`.
    """
    out = _sample_throw_full(
        rng,
        object_xy_bounds,
        speed=speed,
        spawn_pad=spawn_pad,
        catch_band=catch_band,
        lateral=lateral,
        container_xy=container_xy,
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
    camera_pos=(1.0, 0.0, 0.60),
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
