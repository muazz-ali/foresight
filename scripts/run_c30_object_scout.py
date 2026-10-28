#!/usr/bin/env python3
"""C30/C31/C33 scout for one object category (any fruit in phase0_cfg).

Why a separate script: ``run_b1_grid.py`` hardcodes the USD catalog to
egg/lemon/onion, so apple/kiwi/… never enter the pool. This scout keeps the
same expert / cells / dump layout, but builds the catalog from the requested
category only (no control or cfg edits).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python scripts/run_c30_object_scout.py --headless --enable_cameras \\
      --object apple --seeds 1000,1001,1002,1003,1004 \\
      --save-mp4 -o data/eval/C1_c30_all_objects/apple
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

from scripts.run_b1_grid import DEFAULT_CFG, freeze_config, is_seen, load_frozen  # noqa: E402

CELLS = (
    ("C30", 30, 0),
    ("C31", 30, 100),
    ("C33", 30, 300),
)


def _parse_seeds(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--object", required=True, help="Category under objects/ (e.g. apple)")
    p.add_argument(
        "--seeds",
        default="1000,1001,1002,1003,1004",
        help="Comma-separated episode seeds",
    )
    p.add_argument("--cells", default="C30,C31,C33")
    p.add_argument("--sim_cfg_file", type=Path, default=DEFAULT_CFG)
    p.add_argument("--t-close-ms", type=float, default=180.0)
    p.add_argument("--save-mp4", action="store_true", help="Dump h5+json+mp4")
    p.add_argument(
        "--save-h5",
        action="store_true",
        help="Dump h5+json only (no mp4; faster scout re-runs)",
    )
    # ``device`` comes from AppLauncher (do not add it here — name clash).
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("c30_scout")
    seeds = _parse_seeds(args.seeds)
    want_cells = {c.strip() for c in str(args.cells).split(",") if c.strip()}
    cells = [c for c in CELLS if c[0] in want_cells]
    obj_cat = str(args.object).strip()

    if not (out / "phase0_cfg_snapshot.yaml").exists():
        freeze_config(out, Path(args.sim_cfg_file))
    # Drop the B1 egg/lemon/onion catalog override so this fruit can load.
    ov_path = out / "b1_overrides.json"
    if ov_path.exists():
        ov = json.loads(ov_path.read_text())
        ov["scene.object_categories"] = [obj_cat]
        ov_path.write_text(json.dumps(ov, indent=2) + "\n")

    AppLauncher(args)
    exit_code = 0
    try:
        from eval.b1_success import classify_b1
        from sim.collect import episode_for_json, run_episode, write_episode_h5
        from sim.scene import Phase0Scene
        from sim.state_machine import PickPlaceStateMachine
        from sim.video import gates_from_expert, write_debug_mp4

        cfg = load_frozen(out, Path(args.sim_cfg_file))
        cfg.setdefault("robot", {})["gripper_close_time_s"] = float(args.t_close_ms) * 1e-3
        cfg.setdefault("claim_timing", {})["obs_delay_buffer"] = True
        cfg["claim_timing"]["enforce_apply_at"] = False
        # Catalog = this category only (phase0_cfg may list many; pick_usd filters).
        cfg.setdefault("scene", {})["object_categories"] = [obj_cat]
        device = str(getattr(args, "device", None) or "cuda:0")
        cfg.setdefault("sim", {})["device"] = device

        log.info("building scene object=%s device=%s seeds=%s", obj_cat, device, seeds)
        scene = Phase0Scene(
            cfg,
            enable_cameras=True,
            object_usd=obj_cat,
            asset_seed=int(seeds[0]),
        )
        expert = PickPlaceStateMachine(cfg)
        want_mp4 = bool(args.save_mp4)
        want_h5 = bool(args.save_h5 or args.save_mp4)
        # RGB only needed when stitching mp4.
        want_images = want_mp4

        for cell, v_cms, d_ms in cells:
            cell_dir = out / "cells" / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                expert.reset()
                t0 = time.time()
                ep, meta = run_episode(
                    scene,
                    expert,
                    seed=int(seed),
                    speed=float(v_cms) / 100.0,
                    save_images=want_images,
                    obs_delay_s=float(d_ms) / 1000.0,
                    forecast_off=False,
                    depth_noise=True,
                )
                dt = float(cfg.get("sim", {}).get("dt", 1.0 / 30.0))
                scored = classify_b1(
                    ep, meta, dt=dt, z_table=float(meta.get("table_z", 0.0))
                )
                stem = f"{cell}_v{v_cms:02d}_d{d_ms:03d}_s{int(seed)}_{obj_cat}"
                h5_path = None
                video_path = None
                if want_h5 and ep is not None:
                    h5_path = cell_dir / f"{stem}.h5"
                    write_episode_h5(
                        h5_path,
                        ep,
                        extra={
                            "speed_m_s": np.asarray(
                                [float(v_cms) / 100.0], dtype=np.float32
                            ),
                            "seed": np.asarray([int(seed)], dtype=np.int64),
                            "success": np.asarray(
                                [int(bool(scored["success"]))], dtype=np.int8
                            ),
                            "delta_s": np.asarray(
                                [float(d_ms) / 1000.0], dtype=np.float32
                            ),
                            "delta_ms": np.asarray([int(d_ms)], dtype=np.int32),
                        },
                    )
                    payload = {
                        "cell": cell,
                        "seed": int(seed),
                        "object": str(meta.get("category") or obj_cat),
                        "v_cms": int(v_cms),
                        "delta_ms": int(d_ms),
                        "success": bool(scored["success"]),
                        "failure_mode": scored["failure_mode"],
                        "first_attempt": bool(scored.get("first_attempt", False)),
                        "n_close_entries": int(scored.get("n_close_entries", 0)),
                        **meta,
                        **episode_for_json(ep),
                    }
                    with (cell_dir / f"{stem}.json").open("w") as fp:
                        json.dump(payload, fp, indent=2, default=str)
                    if want_mp4:
                        fps = int(cfg.get("static_camera", {}).get("fps", 25))
                        video_path = cell_dir / f"{stem}.mp4"
                        write_debug_mp4(
                            ep, video_path, fps=fps, gates=gates_from_expert(expert)
                        )
                row = {
                    "cell": cell,
                    "seed": int(seed),
                    "object": str(meta.get("category") or obj_cat),
                    "v_cms": int(v_cms),
                    "delta_nom_ms": int(d_ms),
                    "success": bool(scored["success"]),
                    "failure_mode": scored["failure_mode"],
                    "first_attempt": bool(scored.get("first_attempt", False)),
                    "n_close_entries": int(scored.get("n_close_entries", 0)),
                    "spawn_redraws": int(meta.get("spawn_redraws", 0) or 0),
                    "sim_steps": int(meta.get("n_frames", 0) or 0),
                    "wall_time_s": float(time.time() - t0),
                    "seen": bool(is_seen(v_cms, d_ms)),
                    "h5": None if h5_path is None else str(h5_path),
                    "video": None if video_path is None else str(video_path),
                }
                with (cell_dir / "episodes.jsonl").open("a") as fp:
                    fp.write(json.dumps(row) + "\n")
                log.info(
                    "%s s%d succ=%s mode=%s",
                    cell,
                    seed,
                    row["success"],
                    row["failure_mode"],
                )
            # Per-object cell summary (egg/lemon style).
            rows = [
                json.loads(l)
                for l in (cell_dir / "episodes.jsonl").read_text().splitlines()
                if l.strip()
            ]
            k = sum(1 for r in rows if r.get("success"))
            summary = {
                "cell": cell,
                "object": obj_cat,
                "n": len(rows),
                "successes": k,
                "rate": k / len(rows) if rows else float("nan"),
            }
            (cell_dir / f"summary_{obj_cat}.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
        log.info("done object=%s", obj_cat)
    except Exception:
        logging.exception("c30 object scout failed")
        exit_code = 1
    finally:
        log.info("exiting code=%d (skip app.close hang)", exit_code)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
