#!/usr/bin/env python3
"""Record Phase 1 demos with the scripted expert (for policy training).

Always saves camera images + a text instruction. Speeds vary 0-20 cm/s.
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
DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
DEFAULT_OUT = FORESIGHT_ROOT / "data" / "p1-retrain"

sys.path.insert(0, str(FORESIGHT_ROOT))
from eval.gate_g1 import G1_SPEEDS, HIGH_M_S, MID_M_S  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Foresight Phase 1 oracle collection")
    p.add_argument("--target-successes", type=int, default=20)
    p.add_argument("--max-attempts", type=int, default=None, help="Safety cap (default 3x target)")
    p.add_argument(
        "--speed-mode",
        choices=["stratified", "continuous", "fixed"],
        default="stratified",
        help="stratified: equal mass in 0/10/15/20 cm/s bins; continuous: U[0,0.2]; fixed: --speed",
    )
    p.add_argument("--speed", type=float, default=MID_M_S, help="Used when --speed-mode fixed")
    p.add_argument("--speed-max", type=float, default=HIGH_M_S)
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("-o", "--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--object-usd", default=None)
    p.add_argument("--container-usd", default=None)
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--no-save", action="store_false", dest="save")
    p.add_argument("--save-failures", action="store_true", default=False)
    p.add_argument("--debug", action="store_true", default=True)
    p.add_argument("--no-debug", action="store_false", dest="debug")
    p.add_argument(
        "--delta",
        type=float,
        default=None,
        help=(
            "conditioning look-ahead (s); default is expert.conditioning_delta_s "
            "(else lookahead_s)"
        ),
    )
    p.add_argument("--robot", default="franka", choices=["franka"])
    return p


def _setup_logging(output_dir: Path, *, debug: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phase1_collect.log"
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(name)s %(filename)s:%(lineno)d %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, mode="a")):
        handler.setLevel(level)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    if not debug:
        for name in ("omni", "pxr", "carb", "isaaclab", "isaacsim"):
            logging.getLogger(name).setLevel(logging.WARNING)
    logging.info("Logging to %s", log_path)


def _sample_speed(mode: str, rng: np.random.Generator, *, fixed: float, speed_max: float) -> float:
    """Draw object speed: fixed, uniform, or G1 bins with small jitter."""
    if mode == "fixed":
        return float(fixed)
    if mode == "continuous":
        return float(rng.uniform(0.0, speed_max))
    # Stratified bins matching plan §8 (0 / 10 / 15 / 20 cm/s).
    bins = [b for b in G1_SPEEDS if b <= speed_max + 1e-9]
    base = float(rng.choice(bins))
    if base <= 1e-9:
        return 0.0
    # Small jitter inside the 5 cm/s bin, clipped to speed_max.
    jitter = float(rng.uniform(-0.02, 0.02))
    return float(np.clip(base + jitter, 0.0, speed_max))


def run_collection(args, scene, expert) -> dict:
    from sim.collect import episode_for_json, run_episode, write_episode_h5
    from sim.success import classify_failure, episode_diagnostics, episode_success

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "metrics.jsonl"

    target = int(args.target_successes)
    max_attempts = int(args.max_attempts or max(3 * target, target + 50))
    rng = np.random.default_rng(int(args.seed))
    seed = int(args.seed)

    successes = attempts = 0
    t0 = time.time()
    # Always save images for policy training.
    save_images = True

    while successes < target and attempts < max_attempts:
        speed = _sample_speed(
            args.speed_mode, rng, fixed=float(args.speed), speed_max=float(args.speed_max)
        )
        logging.info(
            "P1 collect attempt=%d success=%d/%d seed=%d speed=%.3f",
            attempts + 1,
            successes,
            target,
            seed,
            speed,
        )
        episode = None
        meta: dict = {}
        err = None
        success = False
        try:
            episode, meta = run_episode(
                scene,
                expert,
                seed=seed,
                speed=speed,
                save_images=save_images,
                delta=(
                    float(args.delta)
                    if args.delta is not None
                    else float(
                        getattr(expert, "conditioning_delta_s", expert.lookahead_s)
                    )
                ),
            )
            success = episode_success(
                episode,
                z_lift_min=float(expert.lift_height),
                container_xy_tol=float(expert.place_xy_tol),
                place_stage=int(expert.stage_schema.done),
            )
        except Exception as ex:
            logging.exception("episode failed seed=%s: %s", seed, ex)
            err = str(ex)
            success = False

        attempts += 1
        if success:
            successes += 1

        diag = episode_diagnostics(
            episode, z_lift_min=float(expert.lift_height)
        )
        failure = classify_failure(
            success=success,
            max_stage=diag["max_stage"],
            object_z_min=diag["z_min"],
            object_z_max=diag["z_max"],
            object_z_end=diag["z_end"],
            z_lift_min=float(expert.lift_height),
            schema=expert.stage_schema,
        )
        if err:
            failure = "safety-abort"

        record = {
            "speed_m_s": speed,
            "seed": seed,
            "attempt": attempts,
            "success": success,
            "failure": failure,
            "error": err,
            "n_frames": int(len(episode["stage"])) if episode else 0,
            "max_stage": diag["max_stage"],
            "z_max": diag["z_max"],
            "z_end": diag["z_end"],
            "z_min": diag["z_min"],
            "language": meta.get("language"),
            "category": meta.get("category"),
            "container_category": meta.get("container_category"),
        }
        with open(metrics_path, "a") as fp:
            fp.write(json.dumps(record) + "\n")

        if episode is not None and args.save and (success or args.save_failures or args.debug):
            speed_dir = out_root / f"speed_{speed:.2f}"
            speed_dir.mkdir(parents=True, exist_ok=True)
            ep_name = f"p1_franka_{speed:.2f}_{seed}_{uuid.uuid4().hex[:4]}"
            payload = {
                "seed": seed,
                "speed_m_s": speed,
                "robot": "franka",
                "success": success,
                "failure": failure,
                **meta,
                **{k: record[k] for k in ("max_stage", "z_max", "z_end", "z_min")},
                **episode_for_json(episode),
            }
            with open(speed_dir / f"{ep_name}.json", "w") as fp:
                json.dump(payload, fp, indent=2, default=str)
            write_episode_h5(
                speed_dir / f"{ep_name}.h5",
                episode,
                extra={
                    "speed_m_s": np.asarray([float(speed)], dtype=np.float32),
                    "seed": np.asarray([int(seed)], dtype=np.int64),
                    "success": np.asarray([int(bool(success))], dtype=np.int8),
                    "delta": np.asarray(
                        [
                            float(
                                meta.get(
                                    "delta",
                                    getattr(
                                        expert,
                                        "conditioning_delta_s",
                                        expert.lookahead_s,
                                    ),
                                )
                            )
                        ],
                        dtype=np.float32,
                    ),
                },
            )
            if args.debug:
                from sim.video import write_debug_mp4

                fps = int(scene.cfg.get("camera", {}).get("fps", 25))
                write_debug_mp4(episode, speed_dir / f"{ep_name}.mp4", fps=fps)

        seed += 1

    elapsed = time.time() - t0
    report = {
        "phase": "P1_collect",
        "target_successes": target,
        "successes": successes,
        "attempts": attempts,
        "success_rate": successes / attempts if attempts else 0.0,
        "elapsed_s": elapsed,
        "episodes_per_hour": attempts / elapsed * 3600 if elapsed > 0 else 0.0,
        "output_dir": str(out_root),
        "speed_mode": args.speed_mode,
        "seed_start": int(args.seed),
        "seed_end": seed - 1,
    }
    report_path = out_root / "collect_report.json"
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    logging.info("Wrote %s", report_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    print("[foresight] Phase1 collect — starting AppLauncher…", flush=True)
    from isaaclab.app import AppLauncher

    sys.argv = [a for a in sys.argv if a != ""]
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[foresight] Ignoring unknown args: {unknown}", flush=True)

    # Force cameras for policy RGB.
    if not getattr(args, "enable_cameras", False):
        print("[foresight] enabling cameras for Phase1 collection", flush=True)
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    _setup_logging(Path(args.output_dir), debug=bool(args.debug))
    args.enable_cameras = True

    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    scene = Phase0Scene(
        cfg,
        enable_cameras=True,
        object_usd=args.object_usd,
        container_usd=args.container_usd,
        asset_seed=int(args.seed),
    )
    expert = PickPlaceStateMachine(cfg, dt=scene.dt)
    logging.info(
        "P1 scene ready object=%s container=%s",
        getattr(scene, "_active_usd", None),
        getattr(scene, "_container_usd", None),
    )
    try:
        run_collection(args, scene, expert)
    except Exception:
        logging.exception("Phase1 collect failed")
        raise
    finally:
        logging.info("exiting (skip app.close hang)")
        os._exit(0)


if __name__ == "__main__":
    main()
