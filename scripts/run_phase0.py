#!/usr/bin/env python3
"""Foresight Phase 0 / Gate G0 — in-repo Franka expert.

Launch (conda: dynamicVLA_isaac):

  python scripts/run_phase0.py --speed 0.0 -n 1 --headless --enable_cameras
  python scripts/run_phase0.py --g0 --headless --enable_cameras

Object/container USD meshes are fixed for the process (2A). Diversify with
distinct --seed across workers, or pass --object-usd / --container-usd.
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

G0_SPEEDS_M_S = (0.0, 0.10, 0.15, 0.20)  # same bins as G1 today; plan G0 also allowed 40 cm/s
G0_TRIALS_PER_SPEED = 20
G0_SUCCESS_THRESHOLD = 0.70

sys.path.insert(0, str(FORESIGHT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foresight Phase 0 / Gate G0 (Franka)")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument(
        "--speeds",
        type=str,
        default=None,
        help="comma-separated speeds in m/s (e.g. 0.0,0.05). Overrides --speed.",
    )
    parser.add_argument("--g0", action="store_true")
    parser.add_argument("-n", "--n_simulations", type=int, default=5)
    parser.add_argument("--trials-per-speed", type=int, default=G0_TRIALS_PER_SPEED)
    parser.add_argument("--robot", default="franka", choices=["franka"])
    parser.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    parser.add_argument("-o", "--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument(
        "--object-usd",
        default=None,
        help=(
            "Grasp object: .usd path, category (e.g. can), or stem (e.g. can00). "
            "Default: seed-pick from object_categories"
        ),
    )
    parser.add_argument(
        "--container-usd",
        default=None,
        help=(
            "Container: .usd path, category (e.g. bowl), or stem (e.g. bowl00). "
            "Default: seed-pick from container_categories"
        ),
    )
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", action="store_false", dest="save")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--save-images", action="store_true", default=False)
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help=(
            "conditioning look-ahead (s) for the future-XY pack; "
            "default is expert.conditioning_delta_s (else lookahead_s)"
        ),
    )
    return parser


def _setup_logging(output_dir: str | Path, *, debug: bool = False) -> Path:
    """Console + file logger under output_dir (phase0.log). --debug → DEBUG."""
    from sim.state_machine_logs import HumanFormatter

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "phase0.log"
    level = logging.DEBUG if debug else logging.INFO
    if debug:
        fmt = HumanFormatter("%(asctime)s  %(levelname)s  %(filename)s:%(lineno)d  %(message)s")
    else:
        fmt = HumanFormatter("%(asctime)s  %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_h = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_h.setLevel(level)
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    # Isaac/Kit spam; keep foresight traces readable unless --debug.
    if not debug:
        for name in ("omni", "pxr", "carb", "isaaclab", "isaacsim", "h5py"):
            logging.getLogger(name).setLevel(logging.WARNING)

    return log_path


def _count_failure(metrics_path: Path, speed: float) -> dict:
    counts: dict[str, int] = {}
    if not metrics_path.exists():
        return counts
    with open(metrics_path) as fp:
        for line in fp:
            rec = json.loads(line)
            if abs(rec["speed_m_s"] - speed) > 1e-9:
                continue
            label = rec["failure"]
            counts[label] = counts.get(label, 0) + 1
    return counts


def run_speed_bin(args, scene, expert, speed: float, n_trials: int, metrics_path: Path) -> dict:
    from sim.collect import episode_for_json, run_episode, write_episode_h5
    from sim.state_machine_logs import log_blank, log_event, log_result, log_section, speed_title
    from sim.success import classify_failure, episode_diagnostics, episode_success

    out_dir = Path(args.output_dir) / f"speed_{speed:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    attempts = successes = 0
    t0 = time.time()
    seed = int(args.seed) + int(round(speed * 1000))

    log = logging.getLogger()
    log_section(log, speed_title(speed))

    while attempts < n_trials:
        log_blank(log)
        log_event(
            log,
            f"--- Episode {attempts + 1} of {n_trials}  (seed {seed}) ---",
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
                # --debug needs camera buffers for the stitched MP4.
                save_images=bool(args.save_images or args.debug),
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
            logging.exception("Episode crashed (seed=%s): %s", seed, ex)
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
            "grasp_err_xy": meta.get("grasp_err_xy"),
            "grasp_err_z": meta.get("grasp_err_z"),
            "grasp_speed_m_s": meta.get("grasp_speed_m_s"),
            "retries": meta.get("retries"),
        }

        video_path = None
        if episode is not None and args.save and (success or args.debug):
            ep_name = f"p0_franka_{speed:.2f}_{seed}_{uuid.uuid4().hex[:4]}"
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
            with open(out_dir / f"{ep_name}.json", "w") as fp:
                json.dump(payload, fp, indent=2, default=str)
            write_episode_h5(
                out_dir / f"{ep_name}.h5",
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

                fps = int(scene.cfg["camera"]["fps"])
                video_path = out_dir / f"{ep_name}.mp4"
                write_debug_mp4(episode, video_path, fps=fps)

        log_result(
            log,
            failure=failure,
            max_stage=diag["max_stage"],
            z_max=diag["z_max"],
            z_end=diag["z_end"],
            n_frames=record["n_frames"],
            video=video_path,
            names=expert.stage_schema.names,
        )

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
        "failure_counts": _count_failure(metrics_path, speed),
    }
    cms = speed * 100.0
    speed_txt = "0" if cms < 0.5 else f"{cms:.0f}"
    log_blank(log)
    log_event(
        log,
        f"SPEED {speed_txt} cm/s summary: {successes} of {attempts} success "
        f"({100 * rate:.0f}%), about {eps_per_hour:.0f} episodes per hour",
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
    elif args.speeds:
        speeds = [float(x) for x in str(args.speeds).split(",") if x.strip()]
        n_trials = args.n_simulations
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
    log = logging.getLogger()
    from sim.state_machine_logs import log_blank, log_event, log_section

    log_section(log, "Run summary")
    log_event(log, f"Wrote {report_path}")
    passed = bool(report["gate_g0_pass"])
    if args.g0:
        log_event(log, f"First checkpoint (G0): {'PASS' if passed else 'FAIL'}")
        log_event(
            log,
            f"  Each speed at least 70% success: {'yes' if bins_pass else 'no'}",
        )
        log_event(
            log,
            f"  Also need about 100 episodes per hour: "
            f"{'yes' if throughput_pass else 'no'} ({throughput:.0f} / hour)",
        )
    else:
        log_event(
            log,
            f"This is a smoke check, not Gate G0. "
            f"Each speed at least 70% success: {'yes' if bins_pass else 'no'}.",
        )
    log_event(
        log,
        f"  Overall {100 * overall_rate:.0f}% success, "
        f"about {throughput:.0f} episodes per hour",
    )
    log_blank(log)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    print("[foresight] starting AppLauncher…", flush=True)

    from isaaclab.app import AppLauncher

    # VS Code launch inputs use "" for disabled flags; strip so argparse stays clean.
    sys.argv = [a for a in sys.argv if a != ""]

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        # Logger not ready yet; stderr is fine for this one-liner.
        print(f"[foresight] Ignoring unknown args: {unknown}", flush=True)

    app_launcher = AppLauncher(args)
    # Kit may reconfigure logging on startup — install our handlers after.
    log_path = _setup_logging(args.output_dir, debug=bool(args.debug))
    args.enable_cameras = bool(getattr(app_launcher, "_enable_cameras", False))

    from sim.state_machine_logs import log_blank, log_event, log_how_to_read, log_section

    log = logging.getLogger()
    log_section(log, "Phase 0  (scripted pick-and-place)")
    log_event(log, f"Log file: {log_path}")
    log_event(
        log,
        f"Cameras {'on' if args.enable_cameras else 'off'}. "
        f"Device {getattr(args, 'device', None)}.",
    )

    # Isaac / foresight imports only after AppLauncher.
    from sim.scene import Phase0Scene, load_cfg

    from sim.state_machine import PickPlaceStateMachine as ExpertClass

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    scene = Phase0Scene(
        cfg,
        enable_cameras=bool(args.enable_cameras),
        object_usd=args.object_usd,
        container_usd=args.container_usd,
        asset_seed=int(args.seed),
    )
    expert = ExpertClass(cfg, dt=scene.dt)
    log_event(log, f"Expert intercept-servo. Step time {expert.dt:.3f} s. Look-ahead {expert.lookahead_s:.2f} s.")
    log_event(
        log,
        f"Servo lag {expert.servo_lag_s:.3f} s (tau {expert.tau:.3f} s), "
        f"hand speed cap {expert.ee_speed_max:.2f} m/s, "
        f"bounce guard {expert.bounce_guard_s:.2f} s.",
    )
    log_how_to_read(
        log,
        lookahead_s=float(expert.lookahead_s),
        names=expert.stage_schema.names,
        why=expert.stage_schema.why,
    )
    log_blank(log)
    # Try block is responsible for running the expert and scene.
    try:
        run(args, scene, expert)
    except Exception:
        logging.exception("Phase 0 run failed")
        raise
    finally:
        # Kit `app.close()` often hangs headless; report is already on disk.
        log_event(log, "Exiting (skip app.close hang)")
        os._exit(0)


if __name__ == "__main__":
    main()
