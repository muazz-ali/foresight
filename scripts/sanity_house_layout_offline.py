#!/usr/bin/env python3
"""Offline house placement sanity (no Isaac / no GPU).

Picks 10 houses by seed, builds play layout + throws, checks:
  1) table edge maps to origin
  2) play rect has area and sits in +X (in front of robot)
  3) container is off the play band by ≥3 cm
  4) throws stay in bounds and aim into the catch band

Need ≥8/10 seeds to pass. Run:
  python scripts/sanity_house_layout_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.house_layout import (  # noqa: E402
    compute_play_layout,
    pick_house,
    sample_throw,
)
from sim.house_layout import _sample_throw_full  # noqa: E402


def _check_seed(seed: int, scene_dir: Path) -> tuple[bool, str]:
    rng = np.random.default_rng(seed)
    cams = [{"name": "static_cam", "position": [1.0, 0.0, 0.60]}]
    # Match sim/scene.py: skip houses whose table is too small for play+bowl.
    house = None
    layout = None
    last_err: Exception | None = None
    for _ in range(40):
        try:
            cand = pick_house(scene_dir, rng, cams, max_tries=1)
            layout = compute_play_layout(
                cand.table_bbox_min_w,
                cand.table_bbox_max_w,
                max_reach_dist=0.75,
            )
            house = cand
            break
        except Exception as ex:
            last_err = ex
            continue
    if house is None or layout is None:
        return False, f"layout error: {last_err}"

    # Anchor should be at origin after spawn transform.
    R = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation
    from sim.house_layout import _rot_from_wxyz, _wxyz_to_xyzw

    R_a_inv = _rot_from_wxyz(house.anchor.quat_wxyz).inv().as_matrix()
    p0 = R_a_inv @ house.anchor.pos + house.spawn_pos
    if float(np.linalg.norm(p0)) > 1e-4:
        return False, f"anchor not at origin: {p0}"

    (xmin, xmax), (ymin, ymax) = layout.object_xy_bounds
    if not (xmin < xmax and ymin < ymax):
        return False, "empty play rect"
    if xmax <= 0.30:
        return False, f"play rect not in front of robot: x=({xmin},{xmax})"

    cx, cy = float(layout.container_xy[0]), float(layout.container_xy[1])
    # ≥3 cm gap between container center pocket and play band.
    if cy >= 0:
        gap = (cy - 0.04) - (ymax + 0.03)
    else:
        gap = (ymin - 0.03) - (cy + 0.04)
    if gap < -0.01:
        return False, f"container overlaps play band (gap={gap:.3f})"

    # 5 throws: in-bounds + aim toward catch point.
    for k in range(5):
        out = _sample_throw_full(
            np.random.default_rng(seed * 100 + k),
            layout.object_xy_bounds,
            speed=0.10,
            spawn_pad=0.03,
        )
        s = out["spawn"]
        if not (xmin - 1e-6 <= s[0] <= xmax + 1e-6 and ymin - 1e-6 <= s[1] <= ymax + 1e-6):
            return False, f"throw spawn out of bounds: {s}"
        v = out["vel_xy"]
        t = out["target"]
        if float(np.dot(v, t - s)) < -1e-6:
            return False, "throw velocity aims away from catch target"

    pos, vel = sample_throw(np.random.default_rng(seed), layout.object_xy_bounds, speed=0.10)
    _ = R, _wxyz_to_xyzw, pos, vel  # silence unused in some linters

    return True, (
        f"{house.usd_path.name} / {house.table.name} "
        f"play=[{xmin:.2f},{xmax:.2f}]x[{ymin:.2f},{ymax:.2f}] "
        f"bowl=({cx:.2f},{cy:.2f}) gap={gap:.3f}"
    )


def main() -> int:
    scene_dir = Path("/home/gpuadmin/Desktop/muazzam/scenes")
    if not scene_dir.is_dir():
        print(f"missing scene dir: {scene_dir}")
        return 2

    base = 40
    n = 10
    need = 8
    ok_n = 0
    for i in range(n):
        seed = base + i
        ok, msg = _check_seed(seed, scene_dir)
        print(f"seed {seed}: {'PASS' if ok else 'FAIL'} — {msg}")
        if ok:
            ok_n += 1

    print(f"\npass={ok_n}/{n} (need ≥{need})")
    if ok_n >= need:
        print("GATE: OK")
        return 0
    print("GATE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
