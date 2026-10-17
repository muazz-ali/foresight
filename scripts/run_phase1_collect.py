#!/usr/bin/env python3
"""Record Phase 1 demos with the scripted expert (for policy training).

Always saves camera images + a text instruction. Speeds vary 0-20 cm/s.

What it does: runs our expert over many episodes, keeps the ones that succeed,
and saves them (images, hand moves, the 4 policy numbers) for training.
It runs by hand, in the sim conda env, before the training step.
It must NOT use DynamicVLA code, and it does not train. Training lives in a
separate env. Speed bins match G1 so train and eval see the same speeds.
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
DEFAULT_OUT = FORESIGHT_ROOT / "data" / "g3-smoke"

# So `import sim...` / `import eval...` work when run as a script.
sys.path.insert(0, str(FORESIGHT_ROOT))
# Reuse the G1 speed bins so collected demos cover the same speeds the gate tests.
from eval.gate_g1 import G1_SPEEDS, HIGH_M_S, MID_M_S  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """Command-line flags (Isaac's own flags are added later in main)."""
    p = argparse.ArgumentParser(description="Foresight Phase 1 oracle collection")
    # We stop at N successful demos, not N tries. Failures are skipped for training.
    p.add_argument("--target-successes", type=int, default=20, help="clean wins to collect (sim/success.clean_reject_reason)")
    p.add_argument("--max-attempts", type=int, default=None, help="Safety cap (default 3x target)")
    p.add_argument(
        "--speed-mode",
        choices=["stratified", "continuous", "fixed", "list"],
        default="stratified",
        help="stratified: G1 bins; continuous: U[0,speed-max]; fixed: --speed; list: --speed-list",
    )
    p.add_argument("--speed", type=float, default=MID_M_S, help="Used when --speed-mode fixed")
    p.add_argument(
        "--speed-list",
        type=float,
        nargs="+",
        default=None,
        help="Exact speeds (m/s) for successive successes; implies --speed-mode list",
    )
    p.add_argument("--speed-max", type=float, default=HIGH_M_S)
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("-o", "--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--object-usd", default=None)
    p.add_argument("--container-usd", default=None)
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--no-save", action="store_false", dest="save")
    p.add_argument("--save-failures", action="store_true", default=False)
    # Debug is ON by default here: it also saves failed runs and an MP4 per run.
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
    """Log to the screen and to ``phase1_collect.log`` (appended). Quiet Isaac unless debug."""
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


def _sample_speed(
    mode: str,
    rng: np.random.Generator,
    *,
    fixed: float,
    speed_max: float,
    speed_list: list[float] | None = None,
    success_idx: int = 0,
) -> float:
    """Draw object speed: fixed, uniform, G1 bins, or exact list entry.

    Why: the policy must see many speeds in training. Default "stratified" picks
    a G1 bin at random, so each bin gets about the same share of demos.
    ``rng`` is seeded by the caller, so the speed list is repeatable.
    """
    if mode == "list":
        if not speed_list:
            raise ValueError("speed-mode list requires --speed-list")
        # Walk the list by number of successes so far; wrap around at the end.
        return float(speed_list[int(success_idx) % len(speed_list)])
    if mode == "fixed":
        return float(fixed)
    if mode == "continuous":
        return float(rng.uniform(0.0, speed_max))
    # Stratified bins matching plan §8 (0 / 10 / 15 / 20 cm/s).
    bins = [b for b in G1_SPEEDS if b <= speed_max + 1e-9]
    base = float(rng.choice(bins))
    # The 0 bin stays exactly 0 (no jitter): "object sits still" must be truly still.
    if base <= 1e-9:
        return 0.0
    # Small jitter inside the 5 cm/s bin, clipped to speed_max.
    jitter = float(rng.uniform(-0.02, 0.02))
    return float(np.clip(base + jitter, 0.0, speed_max))


def run_collection(args, scene, expert) -> dict:
    """Keep running episodes until enough succeed (or the try cap hits). Returns a report.

    Each try is judged by OUR success rule (sim/success.py) and labelled with a
    failure code. Every try gets a line in metrics.jsonl, so the real success
    rate is on record even though training only uses the wins.
    """
    from sim.collect import episode_for_json, run_episode, write_episode_h5
    from sim.success import (
        classify_failure,
        clean_reject_reason,
        episode_diagnostics,
        episode_success,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "metrics.jsonl"

    speed_mode = str(args.speed_mode)
    speed_list = list(args.speed_list) if args.speed_list else None
    if speed_list is not None:
        speed_mode = "list"

    target = int(args.target_successes)
    # In list mode, make sure every listed speed gets at least one success.
    if speed_mode == "list" and speed_list is not None:
        target = max(target, len(speed_list))
    # Safety cap so a broken expert cannot loop forever. Default is the larger
    # of 3x target and target+50 (small targets still get room to retry).
    max_attempts = int(args.max_attempts or max(3 * target, target + 50))
    # Seeded: the same --seed gives the same speed draws.
    rng = np.random.default_rng(int(args.seed))
    seed = int(args.seed)

    # ``target`` counts CLEAN wins (first-try grasp, real open, no release drag),
    # the same rule as convert --clean-only, so N here means N usable demos.
    successes = attempts = clean = 0
    clean_rejects: dict[str, int] = {}
    t0 = time.time()
    # Always save images for policy training.
    save_images = True

    # Loop on successes, not attempts: a failed try does not count toward the target.
    while clean < target and attempts < max_attempts:
        speed = _sample_speed(
            speed_mode,
            rng,
            fixed=float(args.speed),
            speed_max=float(args.speed_max),
            speed_list=speed_list,
            success_idx=clean,
        )
        logging.info(
            "P1 collect attempt=%d clean=%d/%d (wins %d) seed=%d speed=%.3f",
            attempts + 1,
            clean,
            target,
            successes,
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
            # OUR success rule (sim/success.py), not DynamicVLA's object_picked.
            success = episode_success(
                episode,
                z_lift_min=float(expert.lift_height),
                container_xy_tol=float(expert.place_xy_tol),
                place_stage=int(expert.stage_schema.done),
            )
        except Exception as ex:
            # Log the full error and score this try as a failure (not a skip).
            logging.exception("episode failed seed=%s: %s", seed, ex)
            err = str(ex)
            success = False

        attempts += 1
        clean_reject = None
        if success:
            successes += 1
            clean_reject = clean_reject_reason(episode, meta.get("retries"))
            if clean_reject is None:
                clean += 1
            else:
                clean_rejects[clean_reject] = clean_rejects.get(clean_reject, 0) + 1

        # Facts (farthest stage, object height low/high/end) kept so a failure
        # label can be double-checked later.
        diag = episode_diagnostics(
            episode, z_lift_min=float(expert.lift_height)
        )
        # Why it failed: never-engaged | too-late | early-close | ... or "success".
        failure = classify_failure(
            success=success,
            max_stage=diag["max_stage"],
            object_z_min=diag["z_min"],
            object_z_max=diag["z_max"],
            object_z_end=diag["z_end"],
            z_lift_min=float(expert.lift_height),
            schema=expert.stage_schema,
            off_table=bool(meta.get("off_table", False)),
            abort_reason=meta.get("abort_reason"),
        )
        # A crash overrides any other label: no trustworthy data to classify.
        if err:
            failure = "safety-abort"

        # language/category tell us which instruction text + objects each demo used.
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
            "grasp_err_xy": meta.get("grasp_err_xy"),
            "grasp_err_z": meta.get("grasp_err_z"),
            "grasp_speed_m_s": meta.get("grasp_speed_m_s"),
            "retries": meta.get("retries"),
            "clean": bool(success and clean_reject is None),
            "clean_reject": clean_reject,
            "language": meta.get("language"),
            "category": meta.get("category"),
            "container_category": meta.get("container_category"),
        }
        with open(metrics_path, "a") as fp:
            fp.write(json.dumps(record) + "\n")

        # Wins are always saved. Failures only with --save-failures or --debug
        # (so debug-on-by-default means failures are kept unless --no-debug).
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
                "clean": bool(success and clean_reject is None),
                "clean_reject": clean_reject,
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
                from sim.video import gates_from_expert, write_debug_mp4

                fps = int(scene.cfg.get("static_camera", {}).get("fps", 25))
                write_debug_mp4(
                    episode,
                    speed_dir / f"{ep_name}.mp4",
                    fps=fps,
                    gates=gates_from_expert(expert),
                )

        # New seed each try: different start, still repeatable. seed_start/seed_end
        # in the report tell you exactly which seeds this run used.
        seed += 1

    elapsed = time.time() - t0
    report = {
        "phase": "P1_collect",
        "target_successes": target,
        "successes": successes,
        "clean_successes": clean,
        "clean_rejects": clean_rejects,
        "attempts": attempts,
        "success_rate": successes / attempts if attempts else 0.0,
        "elapsed_s": elapsed,
        "episodes_per_hour": attempts / elapsed * 3600 if elapsed > 0 else 0.0,
        "output_dir": str(out_root),
        "speed_mode": speed_mode,
        "speed_list": speed_list,
        "seed_start": int(args.seed),
        "seed_end": seed - 1,
    }
    # One report per process: the launcher runs many chunks into one folder.
    report_path = out_root / f"collect_report_seed{int(args.seed)}.json"
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    with open(out_root / "collect_report.json", "w") as fp:  # latest, for old readers
        json.dump(report, fp, indent=2)
    logging.info("Wrote %s", report_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Start Isaac (cameras forced on), build scene + expert, collect, then exit.

    Isaac's AppLauncher must start BEFORE any isaaclab / omni import, so the
    sim imports sit inside this function.
    """
    print("[foresight] Phase1 collect — starting AppLauncher…", flush=True)
    from isaaclab.app import AppLauncher

    sys.argv = [a for a in sys.argv if a != ""]
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[foresight] Ignoring unknown args: {unknown}", flush=True)

    # Force cameras for policy RGB. Without --enable_cameras the frames are black.
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

    exit_code = 0
    try:
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
        run_collection(args, scene, expert)
    except Exception:
        logging.exception("Phase1 collect failed")
        exit_code = 1
    finally:
        # Hard exit: Isaac's normal close often hangs when headless. Safe here
        # because metrics and the report are already written to disk.
        # Must pass the real status — a bare os._exit(0) made the launcher
        # think every crash was a success.
        logging.info("exiting code=%d (skip app.close hang)", exit_code)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
