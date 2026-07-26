#!/usr/bin/env python3
"""Foresight Phase 0 / Gate G0 — in-repo Franka expert (no DynamicVLA).

Launch (conda: dynamicVLA_isaac):

  python scripts/run_phase0.py --speed 0.0 -n 1 --headless --enable_cameras
  python scripts/run_phase0.py --g0 --headless --enable_cameras
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
DEFAULT_OUT = FORESIGHT_ROOT / "data" / "g0"

G0_SPEEDS_M_S = (0.0, 0.1, 0.2, 0.3, 0.4)
G0_TRIALS_PER_SPEED = 20
G0_SUCCESS_THRESHOLD = 0.70

sys.path.insert(0, str(FORESIGHT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foresight Phase 0 / Gate G0 (Franka)")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--g0", action="store_true")
    parser.add_argument("-n", "--n_simulations", type=int, default=5)
    parser.add_argument("--trials-per-speed", type=int, default=G0_TRIALS_PER_SPEED)
    parser.add_argument("--robot", default="franka", choices=["franka"])
    parser.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    parser.add_argument("-o", "--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", action="store_false", dest="save")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--save-images", action="store_true", default=False)
    parser.add_argument("--delta", type=float, default=0.25)
    return parser


def _count_taxonomy(metrics_path: Path, speed: float) -> dict:
    counts: dict[str, int] = {}
    if not metrics_path.exists():
        return counts
    with open(metrics_path) as fp:
        for line in fp:
            rec = json.loads(line)
            if abs(rec["speed_m_s"] - speed) > 1e-9:
                continue
            counts[rec["taxonomy"]] = counts.get(rec["taxonomy"], 0) + 1
    return counts


def run_speed_bin(args, scene, expert, speed: float, n_trials: int, metrics_path: Path) -> dict:
    import h5py

    from sim.collect import run_episode
    from sim.success import classify_failure, episode_diagnostics, episode_success

    out_dir = Path(args.output_dir) / f"speed_{speed:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    attempts = successes = 0
    t0 = time.time()
    seed = int(args.seed) + int(round(speed * 1000))

    while attempts < n_trials:
        logging.info(
            "Phase0 speed=%.2f m/s attempt %d/%d seed=%d",
            speed,
            attempts + 1,
            n_trials,
            seed,
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
                save_images=bool(args.save_images),
                delta=float(args.delta),
            )
            success = episode_success(episode)
        except Exception as ex:
            logging.exception("episode failed seed=%s: %s", seed, ex)
            err = str(ex)
            success = False

        attempts += 1
        if success:
            successes += 1

        diag = episode_diagnostics(episode)
        taxonomy = classify_failure(
            success=success,
            max_sm_state=diag["max_sm"],
            object_z_min=diag["z_min"],
            object_z_max=diag["z_max"],
            object_z_end=diag["z_end"],
        )
        if err:
            taxonomy = "safety-abort"

        record = {
            "speed_m_s": speed,
            "seed": seed,
            "attempt": attempts,
            "success": success,
            "taxonomy": taxonomy,
            "error": err,
            "n_frames": int(len(episode["sm_state"])) if episode else 0,
            "max_sm": diag["max_sm"],
            "z_max": diag["z_max"],
            "z_end": diag["z_end"],
            "z_min": diag["z_min"],
        }

        if episode is not None and args.save and (success or args.debug):
            ep_name = f"p0_franka_{speed:.2f}_{seed}_{uuid.uuid4().hex[:4]}"
            payload = {
                "seed": seed,
                "speed_m_s": speed,
                "robot": "franka",
                "success": success,
                "taxonomy": taxonomy,
                **meta,
                **{k: record[k] for k in ("max_sm", "z_max", "z_end", "z_min")},
            }
            with open(out_dir / f"{ep_name}.json", "w") as fp:
                json.dump(payload, fp, indent=2, default=str)
            with h5py.File(out_dir / f"{ep_name}.h5", "w") as fp:
                for k, v in episode.items():
                    fp.create_dataset(k, data=np.asarray(v), compression="gzip")

        with open(metrics_path, "a") as fp:
            fp.write(json.dumps(record) + "\n")

        seed += 1

    elapsed = time.time() - t0
    rate = successes / attempts if attempts else 0.0
    eps_per_hour = attempts / elapsed * 3600 if elapsed > 0 else 0.0
    summary = {
        "speed_m_s": speed,
        "attempts": attempts,
        "successes": successes,
        "success_rate": rate,
        "elapsed_s": elapsed,
        "episodes_per_hour": eps_per_hour,
        "gate_g0_bin_pass": rate >= G0_SUCCESS_THRESHOLD,
        "taxonomy_counts": _count_taxonomy(metrics_path, speed),
    }
    logging.info(
        "speed=%.2f success=%d/%d (%.0f%%) throughput=%.0f eps/h",
        speed,
        successes,
        attempts,
        100 * rate,
        eps_per_hour,
    )
    return summary


def run(args, scene, expert) -> dict:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    if args.g0:
        speeds = list(G0_SPEEDS_M_S)
        n_trials = args.trials_per_speed
    elif args.speed is not None:
        speeds = [float(args.speed)]
        n_trials = args.n_simulations
    else:
        speeds = [0.0]
        n_trials = args.n_simulations

    summaries = []
    wall0 = time.time()
    for speed in speeds:
        summaries.append(run_speed_bin(args, scene, expert, speed, n_trials, metrics_path))

    wall = time.time() - wall0
    overall_attempts = sum(s["attempts"] for s in summaries)
    overall_success = sum(s["successes"] for s in summaries)
    overall_rate = overall_success / overall_attempts if overall_attempts else 0.0
    throughput = overall_attempts / wall * 3600 if wall > 0 else 0.0
    bins_pass = all(s["gate_g0_bin_pass"] for s in summaries)
    throughput_pass = throughput >= 100.0

    report = {
        "gate": "G0",
        "robot": "franka",
        "criterion": {
            "per_speed_success_ge": G0_SUCCESS_THRESHOLD,
            "speeds_m_s": speeds,
            "throughput_ge_per_hour": 100,
        },
        "summaries": summaries,
        "overall_success_rate": overall_rate,
        "overall_throughput_eps_per_hour": throughput,
        "all_bins_pass": bins_pass,
        "throughput_pass": throughput_pass,
        "gate_g0_pass": bool(bins_pass and throughput_pass) if args.g0 else bool(bins_pass),
        "wall_time_s": wall,
    }
    report_path = Path(args.output_dir) / "gate_g0_report.json"
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    logging.info("Wrote %s", report_path)
    logging.info(
        "Gate G0: %s (bins_pass=%s throughput_pass=%s rate=%.0f%% @ %.0f eps/h)",
        "PASS" if report["gate_g0_pass"] else "FAIL",
        bins_pass,
        throughput_pass if args.g0 else "n/a",
        100 * overall_rate,
        throughput,
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    print("[foresight] starting AppLauncher…", flush=True)

    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning("Ignoring unknown args: %s", unknown)

    app_launcher = AppLauncher(args)
    args.enable_cameras = bool(getattr(app_launcher, "_enable_cameras", False))
    print(
        f"[foresight] app up. enable_cameras={args.enable_cameras} "
        f"device={getattr(args, 'device', None)}",
        flush=True,
    )

    # Isaac / foresight imports only after AppLauncher.
    from sim.expert import PickExpert
    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    scene = Phase0Scene(cfg, enable_cameras=bool(args.enable_cameras))
    expert = PickExpert(cfg)
    print("[foresight] Franka Phase0 scene ready", flush=True)

    try:
        run(args, scene, expert)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Kit `app.close()` often hangs headless; report is already on disk.
        print("[foresight] exiting (skip app.close hang)", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
