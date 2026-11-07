#!/usr/bin/env python3
"""Foresight Mode 2 — static colored cubes + language (no throw).

All four cubes sit on the table. Language names the grasp target. Speed is
always 0. Claim throw path stays on ``scripts/run_phase0.py``.

  conda activate dynamicVLA_isaac
  python scripts/run_mode2.py -n 5 --seed 41 --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "mode2_cfg.yaml"
DEFAULT_OUT = FORESIGHT_ROOT / "data" / "mode2_smoke"

sys.path.insert(0, str(FORESIGHT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    """CLI for Mode 2 smoke / collect (no speed grid)."""
    p = argparse.ArgumentParser(description="Foresight Mode 2 static cubes + language")
    p.add_argument("-n", "--n_simulations", type=int, default=5)
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("-o", "--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--no-save", action="store_false", dest="save")
    p.add_argument("--debug", action="store_true", default=True)
    p.add_argument("--save-images", action="store_true", default=False)
    p.add_argument(
        "--delta",
        type=float,
        default=None,
        help="conditioning look-ahead (s); default from yaml",
    )
    return p


def _setup_logging(output_dir: str | Path, *, debug: bool = False) -> Path:
    """Console + file logger under output_dir (mode2.log)."""
    from sim.state_machine_logs import HumanFormatter

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "mode2.log"
    level = logging.DEBUG if debug else logging.INFO
    fmt = HumanFormatter(
        "%(asctime)s  %(levelname)s  %(filename)s:%(lineno)d  %(message)s"
        if debug
        else "%(asctime)s  %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, mode="a")):
        h.setLevel(level)
        h.setFormatter(fmt)
        root.addHandler(h)
    if not debug:
        for name in ("omni", "pxr", "carb", "isaaclab", "isaacsim", "h5py"):
            logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


def run(args, scene, sm) -> dict:
    """Run ``n`` static Mode-2 episodes. Returns a small summary dict."""
    from sim.collect import episode_for_json, run_episode, write_episode_h5
    from sim.state_machine_logs import log_blank, log_event, log_result, log_section
    from sim.success import classify_failure, episode_diagnostics, episode_success

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    n = int(args.n_simulations)
    seed0 = int(args.seed)
    attempts = successes = 0
    t0 = time.time()
    log = logging.getLogger()
    log_section(log, "Mode 2  (static cubes + language)")

    while attempts < n:
        seed = seed0 + attempts
        log_blank(log)
        log_event(log, f"--- Episode {attempts + 1} of {n}  (seed {seed}) ---")
        episode = None
        meta: dict = {}
        err = None
        success = False
        try:
            episode, meta = run_episode(
                scene,
                sm,
                seed=seed,
                speed=0.0,
                save_images=bool(args.save_images or args.debug),
                delta=(
                    float(args.delta)
                    if args.delta is not None
                    else float(
                        getattr(sm, "policy_look_ahead_s", sm.max_aim_ahead_s)
                    )
                ),
            )
            success = episode_success(
                episode,
                z_lift_min=float(sm.lift_height_m),
                container_xy_tol=float(sm.place_xy_tol),
                place_stage=int(sm.stage_schema.done),
            )
        except Exception as ex:
            logging.exception("Episode crashed (seed=%s): %s", seed, ex)
            err = str(ex)
            success = False

        attempts += 1
        if success:
            successes += 1

        diag = episode_diagnostics(episode, z_lift_min=float(sm.lift_height_m))
        failure = classify_failure(
            success=success,
            max_stage=diag["max_stage"],
            object_z_min=diag["z_min"],
            object_z_max=diag["z_max"],
            object_z_end=diag["z_end"],
            z_lift_min=float(sm.lift_height_m),
            schema=sm.stage_schema,
            off_table=bool(meta.get("off_table", False)),
            abort_reason=meta.get("abort_reason"),
        )
        if err:
            failure = "safety-abort"

        record = {
            "mode": 2,
            "speed_m_s": 0.0,
            "seed": seed,
            "attempt": attempts,
            "success": success,
            "failure": failure,
            "error": err,
            "language": meta.get("language"),
            "category": meta.get("category"),
            "target_color": meta.get("target_color"),
            "cubes": meta.get("cubes"),
            "n_frames": int(len(episode["stage"])) if episode else 0,
            "max_stage": diag["max_stage"],
            "z_max": diag["z_max"],
            "z_end": diag["z_end"],
            "z_min": diag["z_min"],
            "init_vel": meta.get("init_vel"),
        }

        video_path = None
        if episode is not None and args.save and (success or args.debug):
            ep_name = f"m2_{meta.get('category', 'cube')}_{seed}_{uuid.uuid4().hex[:4]}"
            payload = {
                "seed": seed,
                "speed_m_s": 0.0,
                "robot": "franka",
                "success": success,
                "failure": failure,
                **meta,
                **{k: record[k] for k in ("max_stage", "z_max", "z_end", "z_min")},
                **episode_for_json(episode),
            }
            with open(out_dir / f"{ep_name}.json", "w") as fp:
                json.dump(payload, fp, indent=2, default=str)
            write_episode_h5(
                out_dir / f"{ep_name}.h5",
                episode,
                extra={
                    "speed_m_s": np.asarray([0.0], dtype=np.float32),
                    "seed": np.asarray([int(seed)], dtype=np.int64),
                    "success": np.asarray([int(bool(success))], dtype=np.int8),
                },
            )
            if args.debug:
                from sim.video import gates_from_state_machine, write_debug_mp4

                fps = int(scene.cfg["static_camera"]["fps"])
                video_path = out_dir / f"{ep_name}.mp4"
                write_debug_mp4(
                    episode, video_path, fps=fps, gates=gates_from_state_machine(sm)
                )

        log_result(
            log,
            failure=failure,
            max_stage=diag["max_stage"],
            z_max=diag["z_max"],
            z_end=diag["z_end"],
            n_frames=record["n_frames"],
            video=video_path,
            names=sm.stage_schema.names,
        )
        log_event(
            log,
            f"language={record.get('language')!r}  target={record.get('target_color')}",
        )
        with open(metrics_path, "a") as fp:
            fp.write(json.dumps(record, default=str) + "\n")

    elapsed = time.time() - t0
    rate = successes / max(attempts, 1)
    summary = {
        "mode": 2,
        "attempts": attempts,
        "successes": successes,
        "success_rate": rate,
        "elapsed_s": elapsed,
        "seed0": seed0,
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    log_event(log, f"Mode 2 done: {successes}/{attempts} = {100.0 * rate:.0f}%")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Start Isaac, build Mode-2 scene + sm, run, exit."""
    print("[foresight] Mode 2 starting AppLauncher…", flush=True)
    from isaaclab.app import AppLauncher

    sys.argv = [a for a in sys.argv if a != ""]
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[foresight] Ignoring unknown args: {unknown}", flush=True)

    app_launcher = AppLauncher(args)
    log_path = _setup_logging(args.output_dir, debug=bool(args.debug))
    args.enable_cameras = bool(getattr(app_launcher, "_enable_cameras", False))

    from sim.state_machine_logs import log_event, log_section

    log = logging.getLogger()
    log_section(log, "Mode 2  (static cubes + language)")
    log_event(log, f"Log file: {log_path}")
    log_event(log, f"Cfg: {args.sim_cfg_file}")

    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine as StateMachineClass

    cfg = load_cfg(args.sim_cfg_file)
    if int(cfg.get("scene", {}).get("mode", 1)) != 2:
        raise RuntimeError(
            f"Expected scene.mode=2 in {args.sim_cfg_file}; got "
            f"{cfg.get('scene', {}).get('mode')}"
        )
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    exit_code = 0
    try:
        scene = Phase0Scene(
            cfg,
            enable_cameras=bool(args.enable_cameras),
            asset_seed=int(args.seed),
        )
        if int(getattr(scene, "_mode", 1)) != 2:
            raise RuntimeError("Phase0Scene did not enter mode 2")
        sm = StateMachineClass(cfg, dt=scene.dt)
        log_event(log, f"Cubes: {list(scene._mode2_cubes)}")
        run(args, scene, sm)
    except Exception:
        logging.exception("Mode 2 run failed")
        exit_code = 1
    finally:
        log_event(log, f"Exiting code={exit_code} (skip app.close hang)")
        os._exit(exit_code)


if __name__ == "__main__":
    main()
