#!/usr/bin/env python3
"""Ride contact log: finger/palm vs object and table before CLOSE.

Runs the sm on scout seeds. From first ride tick to CLOSE, logs:
  - geometric clearance (finger tip vs object collision radius / table)
  - net contact-force magnitude on finger+hand bodies (PhysX link forces)

Any penetration or force above threshold before CLOSE = scene clearance bug
(fix aim Z / open wider / collider), not a gate tweak.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python -u tools/diag/ride_contact_log.py --headless --enable_cameras --device cuda:0 \\
    --seeds 0,1,2,3,4 -o data/eval/probes/ride_contact_log.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FORESIGHT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORESIGHT))

# Same order as eval/launch_c30_all_objects.sh so seed blocks match the scout.
SCOUT_OBJECTS = (
    "apple",
    "avocado",
    "egg",
    "kiwi",
    "lemon",
    "lime",
    "onion",
    "orange",
    "peach",
    "tangerine",
)
# Probe these (orange/tangerine often empty under width gate).
OBJECTS = tuple(o for o in SCOUT_OBJECTS if o not in ("orange", "tangerine"))
FORCE_THRESH_N = 0.05  # ignore sensor noise
# Finger pad half-thickness for geometric contact (m). Keep small: a 70 mm
# object in an 80 mm stroke only has 5 mm/side before pads — 8 mm pad always
# false-alarms. Real scrape ≈ gap < 0 with a thin pad.
FINGER_HALF_M = 0.003
# Tip below table + this → table scrape.
TABLE_CLEAR_M = 0.002


def _seed_for(obj: str, local_i: int, seed_start: int = 1000) -> int:
    """Match launch_c30_all_objects: apple 1000–1004, … peach 1040–1044."""
    idx = SCOUT_OBJECTS.index(obj) if obj in SCOUT_OBJECTS else 0
    return int(seed_start) + 5 * idx + int(local_i)


def _body_indices(robot, names: tuple[str, ...]) -> list[int]:
    """Indices of named rigid bodies on the articulation (skip missing)."""
    all_names = list(robot.data.body_names)
    out = []
    for n in names:
        if n in all_names:
            out.append(all_names.index(n))
    return out


def _net_force_N(robot, idxs: list[int]) -> float:
    """Sum of ‖net contact force‖ on the given bodies (N). 0 if unavailable."""
    if not idxs:
        return 0.0
    # Prefer PhysX net contact forces when the view exists.
    view = getattr(robot, "root_physx_view", None)
    if view is not None and hasattr(view, "get_link_incoming_joint_force"):
        # Fallback path below — joint force ≠ contact; use body_incoming_wrench.
        pass
    data = robot.data
    wrench = getattr(data, "body_incoming_wrench", None)
    if wrench is None:
        # Older IsaacLab: no per-body contact wrench on ArticulationData.
        return 0.0
    # wrench: (num_envs, n_bodies, 6) — force then torque.
    import torch

    w = wrench[0, idxs, :3]
    return float(torch.linalg.norm(w, dim=-1).sum().item())


def _finger_tips_world(scene) -> np.ndarray:
    """(2,3) world positions of left/right fingertips (or finger bodies)."""
    names = list(scene.robot.data.body_names)
    tips = []
    for cand in (
        ("panda_leftfingertip", "panda_rightfingertip"),
        ("panda_leftfinger", "panda_rightfinger"),
    ):
        if cand[0] in names and cand[1] in names:
            pos = scene.robot.data.body_pos_w[0].detach().cpu().numpy()
            tips.append(pos[names.index(cand[0])])
            tips.append(pos[names.index(cand[1])])
            break
    if tips:
        return np.asarray(tips, dtype=np.float64)
    # Last resort: TCP ± half opening along Y in hand frame is wrong — use TCP.
    ee, _ = scene.get_ee_pose()
    return np.stack([ee, ee], axis=0)


def _clearances(
    tips: np.ndarray,
    obj_xy: np.ndarray,
    obj_r: float,
    table_z: float,
) -> tuple[float, float]:
    """Min horizontal gap to object surface (m) and min tip-Z − table (m).

    Negative gap = geometric overlap with the collision cylinder.
    """
    gaps = []
    tip_zs = []
    for t in tips:
        gaps.append(float(np.linalg.norm(t[:2] - obj_xy[:2]) - obj_r - FINGER_HALF_M))
        tip_zs.append(float(t[2] - table_z))
    return float(min(gaps)), float(min(tip_zs))


def _write_report(out: Path, objs: list[str], all_rows: list[dict]) -> None:
    """Write markdown + JSON summary for the contact probe."""
    lines = [
        "# Ride contact log",
        "",
        f"**Force threshold:** {FORCE_THRESH_N} N (when body wrench available).  ",
        f"**Geom object contact:** finger tip inside collision radius + {1000*FINGER_HALF_M:.0f} mm pad.  ",
        f"**Geom table contact:** tip Z < table + {1000*TABLE_CLEAR_M:.0f} mm.  ",
        "**Any contact before CLOSE = scene clearance bug** (lower aim / open wider / fix collider), not a gate tweak.",
        "",
        "| object | seed | ride | min gap→obj mm | tip−table mm | max F N | "
        "geom-obj | geom-tbl | force | contact? | end |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    by_obj: dict[str, list] = {}
    for r in all_rows:
        if "error" in r:
            lines.append(
                f"| {r.get('object','?')} | {r.get('seed','—')} | — | — | — | — | "
                f"— | — | — | ERR | {r['error'][:40]} |"
            )
            continue
        by_obj.setdefault(r["object"], []).append(r)
        g = r["min_gap_obj_m"]
        tz = r["min_tip_z_above_table_m"]
        lines.append(
            f"| {r['object']} | {r['seed']} | {r['n_ride']} | "
            f"{'—' if g is None else f'{1000*g:.1f}'} | "
            f"{'—' if tz is None else f'{1000*tz:.1f}'} | "
            f"{r['max_force_N']:.3f} | "
            f"{r['ticks_geom_obj']} | {r['ticks_geom_table']} | {r['ticks_force']} | "
            f"{'YES' if r['contact_before_close'] else 'no'} | {r['mode']} |"
        )

    lines += ["", "## Per-object summary", ""]
    lines += [
        "| object | n | any contact | mean min-gap mm | mean tip−table mm | mean max F |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for obj in objs:
        rs = by_obj.get(obj, [])
        if not rs:
            lines.append(f"| {obj} | 0 | — | — | — | — |")
            continue
        n_c = sum(1 for r in rs if r["contact_before_close"])
        gaps = [r["min_gap_obj_m"] for r in rs if r["min_gap_obj_m"] is not None]
        tzs = [
            r["min_tip_z_above_table_m"]
            for r in rs
            if r["min_tip_z_above_table_m"] is not None
        ]
        lines.append(
            f"| {obj} | {len(rs)} | {n_c}/{len(rs)} | "
            f"{'—' if not gaps else f'{1000*np.mean(gaps):.1f}'} | "
            f"{'—' if not tzs else f'{1000*np.mean(tzs):.1f}'} | "
            f"{np.mean([r['max_force_N'] for r in rs]):.3f} |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(all_rows, indent=2))
    print(f"wrote {out}", flush=True)


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=Path, default=FORESIGHT / "sim" / "phase0_cfg.yaml")
    p.add_argument("--objects", default=",".join(OBJECTS))
    p.add_argument("--seeds", default="0,1,2,3,4", help="Local indices 0..4 → seed ids")
    p.add_argument("--cell-delta-ms", type=float, default=0.0, help="C30=0, C31=100, C33=300")
    p.add_argument("--speed-cms", type=float, default=30.0)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument(
        "--one-object",
        action="store_true",
        help="Internal: run only --objects (single). Parent forks one Kit per object.",
    )
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])
    objs = [x.strip() for x in args.objects.split(",") if x.strip()]

    # Phase0Scene cannot rebuild GroundPlane — one Kit process per object.
    if len(objs) > 1 and not args.one_object:
        import subprocess

        all_rows: list[dict] = []
        for obj in objs:
            part = args.out.with_name(args.out.stem + f"_{obj}.json")
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--cfg",
                str(args.cfg),
                "--objects",
                obj,
                "--seeds",
                args.seeds,
                "--cell-delta-ms",
                str(args.cell_delta_ms),
                "--speed-cms",
                str(args.speed_cms),
                "-o",
                str(args.out.with_name(args.out.stem + f"_{obj}.md")),
                "--one-object",
                "--headless",
                "--enable_cameras",
                "--device",
                str(getattr(args, "device", "cuda:0") or "cuda:0"),
            ]
            print(f"spawn {obj}…", flush=True)
            env = dict(**{k: v for k, v in __import__("os").environ.items()})
            rc = subprocess.call(cmd, env=env)
            if part.exists():
                all_rows.extend(json.loads(part.read_text()))
            elif args.out.with_name(args.out.stem + f"_{obj}.json").exists():
                all_rows.extend(
                    json.loads(args.out.with_name(args.out.stem + f"_{obj}.json").read_text())
                )
            else:
                all_rows.append({"object": obj, "error": f"subprocess rc={rc}"})
        _write_report(args.out, objs, all_rows)
        return

    AppLauncher(args)

    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import STAGE_APPROACH, STAGE_CLOSE, PickPlaceStateMachine

    cfg = load_cfg(args.cfg)
    cfg.setdefault("claim_timing", {})["enabled_noise_on_replan"] = True
    cfg["claim_timing"]["eval_delta_ms"] = float(args.cell_delta_ms)
    # Catalog = this category only (same as run_c30_object_scout).
    cfg.setdefault("scene", {})["object_categories"] = objs
    local_seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    speed = float(args.speed_cms) * 0.01

    all_rows = []
    finger_bodies = (
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_hand",
        "panda_leftfingertip",
        "panda_rightfingertip",
    )

    obj = objs[0]
    print(f"=== object={obj} ===", flush=True)
    try:
        scene = Phase0Scene(
            cfg, enable_cameras=False, object_usd=obj, asset_seed=1000
        )
    except Exception as ex:  # noqa: BLE001
        print(f"  scene fail: {ex}", flush=True)
        all_rows.append({"object": obj, "error": str(ex)})
        _write_report(args.out, objs, all_rows)
        return

    idxs = _body_indices(scene.robot, finger_bodies)
    print(f"  finger/hand body idxs={idxs}", flush=True)

    for li in local_seeds:
        seed = _seed_for(obj, li)
        try:
            meta = scene.reset_episode(seed=seed, speed=speed)
        except Exception as ex:  # noqa: BLE001
            all_rows.append({"object": obj, "seed": seed, "error": f"reset: {ex}"})
            print(f"  s{seed} reset fail: {ex}", flush=True)
            continue

        sm = PickPlaceStateMachine(cfg, dt=float(scene.dt))
        sm.set_box_poses(
            np.asarray(meta.get("carry_pose", [0.5, 0.3, 0.12])),
            drop_pos=np.asarray(meta.get("drop_pose", [0.5, 0.3, 0.06])),
        )
        if meta.get("object_half_height") is not None:
            sm.set_object_half_height(
                float(meta["object_half_height"]),
                z_table=float(meta.get("table_z", 0.0)),
            )

        # Collision XY radius ≈ half of max horizontal size.
        half_h = float(meta.get("object_half_height") or 0.03)
        raw_size = getattr(scene, "_object_size", None)
        if raw_size is None:
            size = np.array([2 * half_h, 2 * half_h, 2 * half_h], dtype=np.float64)
        else:
            size = np.asarray(raw_size, dtype=np.float64).reshape(-1)
        obj_r = 0.5 * float(max(float(size[0]), float(size[1])))
        table_z = float(meta.get("table_z", 0.0))

        ride_started = False
        max_f = 0.0
        min_gap = 1e9
        min_tip_z = 1e9
        n_force_hit = 0
        n_geom_obj = 0
        n_geom_tbl = 0
        n_ride = 0
        closed = False
        mode = "timeout"

        for _ in range(300):
            st = scene.get_object_state()
            ee, eq = scene.get_ee_pose()
            ee_vel = scene.get_ee_lin_vel()
            # Live GT is fine for a clearance probe (not a delay-accuracy test).
            cmd = sm.step(st, ee, eq, ee_vel=ee_vel)
            scene.set_ee_target(
                cmd.position,
                cmd.quat_wxyz,
                cmd.gripper_open,
                vel_w=cmd.velocity,
            )
            scene.step(
                kinematic_object=bool(cmd.kinematic_object),
                attach_object=bool(cmd.attach_object),
                freeze_object=bool(cmd.freeze_object),
            )

            if int(cmd.stage) == STAGE_APPROACH:
                gap_xy = float(np.linalg.norm(ee[:2] - st.position[:2]))
                if gap_xy <= 0.05:
                    ride_started = True
                if ride_started:
                    n_ride += 1
                    tips = _finger_tips_world(scene)
                    g_obj, tip_z = _clearances(tips, st.position, obj_r, table_z)
                    min_gap = min(min_gap, g_obj)
                    min_tip_z = min(min_tip_z, tip_z)
                    if g_obj < 0.0:
                        n_geom_obj += 1
                    if tip_z < TABLE_CLEAR_M:
                        n_geom_tbl += 1
                    f = _net_force_N(scene.robot, idxs)
                    max_f = max(max_f, f)
                    if f > FORCE_THRESH_N:
                        n_force_hit += 1

            if int(cmd.stage) == STAGE_CLOSE:
                closed = True
                mode = "closed"
                break
            if int(cmd.stage) >= 8:
                mode = "home_abort"
                break

        contact = bool(n_geom_obj > 0 or n_geom_tbl > 0 or n_force_hit > 0)
        row = {
            "object": obj,
            "seed": seed,
            "mode": mode,
            "closed": closed,
            "n_ride": n_ride,
            "min_gap_obj_m": None if min_gap > 1e8 else min_gap,
            "min_tip_z_above_table_m": None if min_tip_z > 1e8 else min_tip_z,
            "max_force_N": max_f,
            "ticks_geom_obj": n_geom_obj,
            "ticks_geom_table": n_geom_tbl,
            "ticks_force": n_force_hit,
            "contact_before_close": contact,
            "obj_r_m": obj_r,
            "half_h_m": half_h,
        }
        all_rows.append(row)
        print(
            f"  s{seed}: ride={n_ride} gap={min_gap*1000:.1f}mm "
            f"tipZ={min_tip_z*1000:.1f}mm F={max_f:.3f}N "
            f"hits={n_geom_obj}/{n_geom_tbl}/{n_force_hit} {mode}",
            flush=True,
        )

    _write_report(args.out, objs, all_rows)


if __name__ == "__main__":
    main()
