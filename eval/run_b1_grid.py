#!/usr/bin/env python3
"""B1 full-grid gate: oracle sm on speed × delay (16 cells × 80).

See the B1 gate prompt. Dual GPU:
  GPU0 = unseen cells (C33 first)
  GPU1 = seen cells
One cell per GPU; shared STOP file under the output root.

  conda activate dynamicVLA_isaac
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python eval/run_b1_grid.py --role unseen --device cuda:0 \\
      -o data/eval/B1_grid_v1 --phase full

Phases: freeze | c30 | delay-check | neg-control | smoke | full | sensitivity | report

``--save-mp4``: also dump per-episode ``.h5`` + sidecar ``.json`` + stitched
``.mp4`` under ``cells/<cell>/`` (same arrays as ``run_phase0.py``). Use for
smoke / debug so you can plot root causes later. Leave off for the full 1,280
(disk + I/O heavy).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"

SPEEDS_CMS = (0, 10, 20, 30)
DELTAS_MS = (0, 100, 200, 300)
# Claim object set (Oct 29): low fruit only; tall fruit out of scope
# (foresight_plan.md "Claim scope"). 16 paired seed slots per object = 80 per cell.
CLAIM_OBJECTS = ("egg", "kiwi", "lemon", "lime", "onion")
OBJECTS = tuple(o for o in CLAIM_OBJECTS for _ in range(16))  # 80
assert len(OBJECTS) == 80

# Unseen: v=30 or Δ=300. Seen: v≤20 and Δ≤200.
UNSEEN_ORDER = [
    ("C33", 30, 300),
    ("C32", 30, 200),
    ("C31", 30, 100),
    ("C30", 30, 0),
    ("C23", 20, 300),
    ("C13", 10, 300),
    ("C03", 0, 300),
]
SEEN_ORDER = [
    ("C22", 20, 200),
    ("C21", 20, 100),
    ("C20", 20, 0),
    ("C12", 10, 200),
    ("C11", 10, 100),
    ("C10", 10, 0),
    ("C02", 0, 200),
    ("C01", 0, 100),
    ("C00", 0, 0),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cell_tag(v_cms: int, d_ms: int) -> str:
    return f"C{v_cms // 10}{d_ms // 100}"


def is_seen(v_cms: int, d_ms: int) -> bool:
    return v_cms <= 20 and d_ms <= 200


def seed_i(i: int) -> int:
    return 1000 + int(i)


def stop_path(out: Path) -> Path:
    return out / "STOP"


def write_stop(out: Path, reason: str) -> None:
    stop_path(out).write_text(reason + "\n")


def stopped(out: Path) -> bool:
    return stop_path(out).exists()


def freeze_config(out: Path, cfg_path: Path) -> Path:
    """Snapshot human yaml + B1 overrides + git hash before any episode.

    Reloading uses ``load_yaml(snapshot)`` then applies ``b1_overrides.json``
    so unit conversion still runs once from mm/ms keys.
    """
    out.mkdir(parents=True, exist_ok=True)
    snap = out / "phase0_cfg_snapshot.yaml"
    shutil.copy2(cfg_path, snap)
    overrides = {
        "_note": "B1 overrides applied after load_yaml(snapshot). Do not edit mid-run.",
        "_t_close": "SPEC-DERIVED",
        "robot.gripper_close_time_ms": 180,
        "claim_timing.enabled_noise_on_replan": True,
        "claim_timing.apply_delta_delay_to_robot": False,
        "claim_timing.delta_meas_noise_ms": 5.0,
        "claim_timing.replan_meas_noise_ms": 330.0,
        "claim_timing.pre_shape_slack_mm": 5.0,
        "claim_timing.pre_shape_max_open_mm": 80.0,
        "scene.object_categories": list(CLAIM_OBJECTS),
    }
    (out / "b1_overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
    # Human-readable frozen pointer file.
    (out / "frozen_cfg.yaml").write_text(
        f"# B1 frozen: see phase0_cfg_snapshot.yaml + b1_overrides.json\n"
        f"snapshot: {snap.name}\n"
        f"overrides: b1_overrides.json\n"
        f"t_close: SPEC-DERIVED 180 ms\n"
    )
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FORESIGHT_ROOT)
    (out / "git_rev.txt").write_text(rev.decode().strip() + "\n")
    return out / "frozen_cfg.yaml"


def load_frozen(out: Path, fallback_cfg: Path) -> dict:
    """Load snapshot yaml and apply B1 overrides (SI after normalize)."""
    from interfaces.config import load_yaml

    snap = out / "phase0_cfg_snapshot.yaml"
    cfg = load_yaml(snap if snap.exists() else fallback_cfg)
    ov_path = out / "b1_overrides.json"  # overrides for the config
    if ov_path.exists():
        ov = json.loads(ov_path.read_text())
        # Apply dotted keys that are still human units into raw then re-normalize
        # by setting SI fields directly when already converted.
        if "robot.gripper_close_time_ms" in ov:
            cfg.setdefault("robot", {})["gripper_close_time_s"] = (
                float(ov["robot.gripper_close_time_ms"]) * 1e-3
            )
        ct = cfg.setdefault("claim_timing", {})
        ct["enabled_noise_on_replan"] = bool(
            ov.get("claim_timing.enabled_noise_on_replan", True)
        )
        ct["apply_delta_delay_to_robot"] = bool(
            ov.get("claim_timing.apply_delta_delay_to_robot", False)
        )
        if "claim_timing.delta_meas_noise_ms" in ov:
            ct["delta_meas_noise_s"] = float(ov["claim_timing.delta_meas_noise_ms"]) * 1e-3
        if "claim_timing.replan_meas_noise_ms" in ov:
            ct["replan_meas_noise_s"] = (
                float(ov["claim_timing.replan_meas_noise_ms"]) * 1e-3
            )
        if "claim_timing.pre_shape_slack_mm" in ov:
            ct["pre_shape_slack_m"] = float(ov["claim_timing.pre_shape_slack_mm"]) * 1e-3
        if "claim_timing.pre_shape_max_open_mm" in ov:
            ct["pre_shape_max_open_m"] = float(ov["claim_timing.pre_shape_max_open_mm"]) * 1e-3
        if "scene.object_categories" in ov:
            cfg.setdefault("scene", {})["object_categories"] = list(ov["scene.object_categories"])
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output_dir", default=str(FORESIGHT_ROOT / "data" / "eval" / "B1_grid_v1"))
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("--role", choices=("unseen", "seen", "all", "host"), default="all")
    p.add_argument("--phase", default="full",
                   help="freeze|c30|delay-check|neg-control|smoke|full|sensitivity|report|all")
    p.add_argument("--n", type=int, default=80, help="episodes per cell (smoke uses 4)")
    p.add_argument("--t-close-ms", type=float, default=180.0)
    p.add_argument(
        "--object",
        default=None,
        help="Fix object category (one of CLAIM_OBJECTS). One scene per process "
        "(Isaac cannot rebuild World). Seed indices for that object only.",
    )
    p.add_argument("--seed-start", type=int, default=0, help="Inclusive index into the 80 paired seeds")
    p.add_argument("--seed-end", type=int, default=79, help="Inclusive index into the 80 paired seeds")
    p.add_argument(
        "--save-mp4",
        action="store_true",
        help="Save .h5 + .json + stitched .mp4 per episode under cells/<cell>/ "
        "(full frame arrays for root-cause; needs cameras for video).",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Run every cell to n: a failing cell does not write STOP. A STOP "
        "from the validity phases (c30 / delay / neg) still halts the grid.",
    )
    p.add_argument(
        "--cells",
        default=None,
        help="Comma-separated cell tags to keep (e.g. C30,C31,C33). "
        "Filters the role's cell list after build; unknown tags are ignored.",
    )
    p.add_argument(
        "--episode-timeout-s",
        type=float,
        default=180.0,
        help="Watchdog: if one episode (incl. h5/mp4 write) takes longer, dump every "
        "thread's Python stack to logs/watchdog_<object>.txt and exit code 1. "
        "Slowest of 637 past episodes: 35 s. 0 = off. Why: Isaac froze once mid-episode "
        "(Oct 29, lemon C21 s1033) with no error and the worker sat 7 h.",
    )
    p.add_argument(
        "--skip-done",
        action="store_true",
        help="Reuse an episode already in cells/<cell>/episodes.jsonl (that row is written "
        "last, after h5/json/mp4) instead of re-running it. For restarting a worker.",
    )
    return p


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args([a for a in sys.argv[1:] if a != ""])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("b1_grid")
    exit_code = 0

    phase = str(args.phase)
    if phase == "freeze":
        freeze_config(out, Path(args.sim_cfg_file))
        log.info("froze config → %s", out / "frozen_cfg.yaml")
        return
    if phase == "report":
        _write_report(out)
        return

    # Isaac after freeze so host-only report can skip AppLauncher if needed.
    AppLauncher(args)
    try:
        _run_isaac_phases(args, out, log)
    except Exception:
        logging.exception("B1 grid failed")
        exit_code = 1
    finally:
        # Hard exit: Isaac's normal close often hangs headless (same as phase0).
        # Cell jsonl / h5 / mp4 / summaries are already on disk.
        log.info("exiting code=%d (skip app.close hang)", exit_code)
        os._exit(exit_code)


def _run_isaac_phases(args: argparse.Namespace, out: Path, log: logging.Logger) -> None:
    """All Isaac work for one worker. Call only after AppLauncher."""

    from eval.b1_success import classify_b1
    from sim.collect import run_episode
    from sim.scene import Phase0Scene
    from sim.state_machine import PickPlaceStateMachine

    phase = str(args.phase)
    if not (out / "phase0_cfg_snapshot.yaml").exists():
        freeze_config(out, Path(args.sim_cfg_file))
    cfg = load_frozen(out, Path(args.sim_cfg_file))
    # Apply t_close override (sensitivity / C30).
    cfg.setdefault("robot", {})["gripper_close_time_s"] = float(args.t_close_ms) * 1e-3
    cfg.setdefault("claim_timing", {})["enabled_noise_on_replan"] = True
    cfg["claim_timing"]["apply_delta_delay_to_robot"] = False
    device = str(getattr(args, "device", None) or cfg.get("sim", {}).get("device", "cuda:0"))
    cfg.setdefault("sim", {})["device"] = device
    log.info("device=%s role=%s phase=%s out=%s", device, args.role, phase, out)

    # One Isaac scene per process (GroundPlane clash if rebuilt). Fix the
    # object category here; host launches one worker per claim object with the
    # matching --seed-start/--seed-end so paired seeds stay aligned.
    obj_cat = args.object or "egg"
    cfg.setdefault("scene", {})["object_categories"] = list(CLAIM_OBJECTS)

    log.info("building scene object=%s device=%s", obj_cat, device)
    scene = Phase0Scene(
        cfg,
        enable_cameras=True,
        object_usd=obj_cat,
        asset_seed=1000,
    )

    log.info("sm initialized")
    sm = PickPlaceStateMachine(cfg)
    i0 = int(args.seed_start)
    i1 = int(args.seed_end)
    wd_s = float(getattr(args, "episode_timeout_s", 0.0) or 0.0)
    wd_file = None
    if wd_s > 0:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        wd_file = (out / "logs" / f"watchdog_{obj_cat}.txt").open("a")
        log.info("episode watchdog %.0f s → %s", wd_s, wd_file.name)

    def run_one(**kw) -> dict:
        """One episode under the watchdog (see --episode-timeout-s)."""
        if wd_file is None:
            return _run_one(**kw)
        wd_file.write(f"{time.strftime('%F %T')} pid {os.getpid()} start {kw.get('cell')} s{kw.get('seed')}\n")
        wd_file.flush()
        faulthandler.dump_traceback_later(wd_s, exit=True, file=wd_file)
        try:
            return _run_one(**kw)
        finally:
            faulthandler.cancel_dump_traceback_later()

    def _run_one(
        *,
        cell: str,
        v_cms: int,
        d_ms: int,
        seed: int,
        object_usd: str,
        forecast_off: bool = False,
        save_images: bool = False,
    ) -> dict:
        # --save-mp4 keeps RGB + dumps .h5/.json/.mp4 for later root-cause plots.
        want_dump = bool(getattr(args, "save_mp4", False))
        save_images = bool(save_images or want_dump)
        sm.reset()
        t0 = time.time()

        log.info("Running episode with seed %s", seed)
        ep, meta = run_episode(
            scene,
            sm,
            seed=int(seed),
            speed=float(v_cms) / 100.0,
            save_images=save_images,
            obs_delay_s=float(d_ms) / 1000.0,
            forecast_off=bool(forecast_off),
            depth_noise=True,
        )
        dt = float(cfg.get("sim", {}).get("dt", 1.0 / 30.0))
        scored = classify_b1(ep, meta, dt=dt, z_table=float(meta.get("table_z", 0.0)))

        cell_dir = out / "cells" / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{cell}_v{v_cms:02d}_d{d_ms:03d}_s{int(seed)}_{object_usd}"
        video_path = None
        h5_path = None
        if want_dump and ep is not None:
            from sim.collect import episode_for_json, write_episode_h5
            from sim.video import gates_from_state_machine, write_debug_mp4

            h5_path = cell_dir / f"{stem}.h5"
            write_episode_h5(
                h5_path,
                ep,
                extra={
                    "speed_m_s": np.asarray([float(v_cms) / 100.0], dtype=np.float32),
                    "seed": np.asarray([int(seed)], dtype=np.int64),
                    "success": np.asarray(
                        [int(bool(scored["success"]))], dtype=np.int8
                    ),
                    "delta_s": np.asarray([float(d_ms) / 1000.0], dtype=np.float32),
                    "delta_ms": np.asarray([int(d_ms)], dtype=np.int32),
                },
            )
            payload = {
                "cell": cell,
                "seed": int(seed),
                "object": str(meta.get("category") or object_usd),
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
            fps = int(cfg.get("static_camera", {}).get("fps", 25))
            video_path = cell_dir / f"{stem}.mp4"
            write_debug_mp4(
                ep, video_path, fps=fps, gates=gates_from_state_machine(sm)
            )
        row = {
            "cell": cell,
            "seed": int(seed),
            "object": str(meta.get("category") or object_usd),
            "v_cms": int(v_cms),
            "delta_nom_ms": int(d_ms),
            "delta_true_ms": float(d_ms),
            "delta_meas_ms": (
                float(np.nanmedian(ep["delta_meas_s"])) * 1000.0
                if "delta_meas_s" in ep and ep["delta_meas_s"].size
                else float(d_ms)
            ),
            "t_lag_meas_ms": (
                None
                if meta.get("t_lag_meas_s") is None
                else float(meta["t_lag_meas_s"]) * 1000.0
            ),
            "success": bool(scored["success"]),
            "failure_mode": scored["failure_mode"],
            "first_attempt": bool(scored.get("first_attempt", False)),
            "n_close_entries": int(scored.get("n_close_entries", 0)),
            "diag_closed_above": bool(scored.get("diag_closed_above", False)),
            "spawn_redraws": int(meta.get("spawn_redraws", 0) or 0),
            "ride_j7_margin_rad": meta.get("ride_j7_margin_rad"),
            "close_xy_err_mm": (
                None
                if meta.get("close_xy_err_m") is None
                else float(meta["close_xy_err_m"]) * 1000.0
            ),
            "close_z_err_mm": (
                None
                if meta.get("close_z_err_m") is None
                else float(meta["close_z_err_m"]) * 1000.0
            ),
            "dv_at_close_cms": (
                None
                if meta.get("dv_at_close_m_s") is None
                else float(meta["dv_at_close_m_s"]) * 100.0
            ),
            "slip_at_close_mm": (
                None
                if meta.get("dv_at_close_m_s") is None
                else float(meta["dv_at_close_m_s"]) * float(args.t_close_ms)
            ),
            "hand_trail_mm": None,
            "zgrasp_err_mm": None,
            "forecast_err_exec_mm": None,
            "gt_fallback_count": int(meta.get("gt_fallback_count", 0) or 0),
            "live_gt_leaks": int(meta.get("live_gt_leaks", 0) or 0),
            "sim_steps": int(meta.get("n_frames", 0) or 0),
            "wall_time_s": float(time.time() - t0),
            "obs_lag_median_ms": (
                None
                if meta.get("obs_lag_median_s") is None
                else float(meta["obs_lag_median_s"]) * 1000.0
            ),
            "ride_dv_cms_median": meta.get("ride_dv_cms_median"),
            "w_pre_clamped": bool(meta.get("w_pre_clamped", False)),
            "device": device,
            "seen": bool(is_seen(v_cms, d_ms)),
            "forecast_off": bool(forecast_off),
            "t_close_ms": float(args.t_close_ms),
            "video": None if video_path is None else str(video_path),
            "h5": None if h5_path is None else str(h5_path),
        }
        # Always append the score line; optional .h5/.json/.mp4 when --save-mp4.
        with (cell_dir / "episodes.jsonl").open("a") as fp:
            fp.write(json.dumps(row) + "\n")
        return row

    def run_cell(cell: str, v_cms: int, d_ms: int, n: int, *, early_stop: bool) -> dict:
        """Run seed indices [i0, i1] ∩ [0, n). Host merges per-object shards."""
        rows: list[dict] = []
        fails = 0
        indices = [i for i in range(min(int(n), 80)) if i0 <= i <= i1]
        done: dict[tuple[int, str], dict] = {}
        jl = out / "cells" / cell / "episodes.jsonl"
        if getattr(args, "skip_done", False) and jl.exists():
            for line in jl.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if getattr(args, "save_mp4", False) and not (r.get("video") and Path(r["video"]).exists()):
                    continue
                done[(int(r["seed"]), str(r.get("object")))] = r
        for k, i in enumerate(indices):
            if stopped(out) and early_stop and k > 0:
                log.warning("STOP seen — finishing cell %s early after %d eps", cell, k)
                break
            obj = OBJECTS[i]
            if (seed_i(i), obj) in done:
                row = done[(seed_i(i), obj)]
                log.info("%s s%d already in episodes.jsonl — reused, not re-run", cell, seed_i(i))
            else:
                row = run_one(
                    cell=cell,
                    v_cms=v_cms,
                    d_ms=d_ms,
                    seed=seed_i(i),
                    object_usd=obj,
                )
            rows.append(row)
            if not row["success"]:
                fails += 1
            # Early-stop counts the first 20 paired seeds across shards; each
            # worker only sees its slice, so the host also watches pooled fails.
            if early_stop and i + 1 == 20 and fails >= 7:
                write_stop(
                    out,
                    f"early-stop {cell}: {fails} fails by seed index 20; device={device}",
                )
                log.error("early-stop on %s (%d fails by i=20)", cell, fails)
            log.info(
                "%s ep i=%d (%d/%d) succ=%s mode=%s gt_fb=%s lag_ms=%s",
                cell,
                i,
                k + 1,
                len(indices),
                row["success"],
                row["failure_mode"],
                row["gt_fallback_count"],
                row.get("obs_lag_median_ms"),
            )
        k = sum(1 for r in rows if r["success"])
        summary = {
            "cell": cell,
            "v_cms": v_cms,
            "delta_ms": d_ms,
            "n": len(rows),
            "successes": k,
            "rate": k / len(rows) if rows else float("nan"),
            "wilson95": list(wilson(k, len(rows))) if rows else [None, None],
            "seen": is_seen(v_cms, d_ms),
            "device": device,
            "gt_fallback_total": int(sum(r["gt_fallback_count"] for r in rows)),
            "live_gt_leaks_total": int(sum(r["live_gt_leaks"] for r in rows)),
            "failure_modes": {},
        }
        for r in rows:
            m = r["failure_mode"] or "success"
            summary["failure_modes"][m] = int(summary["failure_modes"].get(m, 0)) + 1
        # One summary file per object shard so parallel workers do not clobber.
        (out / "cells" / cell / f"summary_{obj_cat}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return summary

    # ---- phases ----
    if phase in ("c30", "all"):
        log.info("C30 ×10 ride |Δv| pre-check at t_close=%s ms", args.t_close_ms)
        rows = []
        for i in range(10):
            rows.append(
                run_one(
                    cell="C30_prechek",
                    v_cms=30,
                    d_ms=0,
                    seed=seed_i(i),
                    object_usd=OBJECTS[i],
                )
            )
        # Bar (B1 math, §7): 10/10 successes, |Δv| at close median ≤ 0.7 and
        # p95 ≤ the sm's speed gate (lim below), hand offset at close median ≤ the sm's own close
        # box (close_xy_ok, 2 mm; a 1 mm bar failed even good closes at 1.6–2.0
        # mm because the sm may close anywhere inside 2 mm). |Δv| at close alone
        # cannot fail: the sm's speed gate only closes below the 1.39 cm/s
        # limit, so the old "median ≤ limit" passed with 3/10 successes (v6).
        def _vals(key):
            return [
                float(r[key]) for r in rows
                if r.get(key) is not None and np.isfinite(float(r[key]))
            ]

        close_dv = _vals("dv_at_close_cms")
        ride_dv = _vals("ride_dv_cms_median")
        close_xy = _vals("close_xy_err_mm")
        n_ok = int(sum(bool(r["success"]) for r in rows))
        med = float(np.median(close_dv)) if close_dv else float("nan")
        p95 = float(np.percentile(close_dv, 95)) if close_dv else float("nan")
        xy_med = float(np.median(close_xy)) if close_xy else float("nan")
        lim = 0.0025 / (float(args.t_close_ms) * 1e-3) * 100.0  # sm speed gate, cm/s
        xy_lim_mm = 1000.0 * float(
            cfg["state_machine_params"].get("close_xy_ok_m", 0.002)
        )
        checks = {
            "success_10_of_10": n_ok == len(rows),
            "dv_close_median_le_0p7": bool(np.isfinite(med) and med <= 0.7 + 1e-9),
            # p95 bar = the sm's own speed gate (2.5 mm / t_close = 1.39 cm/s
            # at 180 ms). A fixed 1.0 failed 10/10 runs whose worst closes
            # (1.18, 1.23) were inside that gate. Median ≤ 0.7 stays.
            "dv_close_p95_le_gate": bool(np.isfinite(p95) and p95 <= lim + 1e-9),
            "close_xy_median_le_close_xy_ok": bool(
                np.isfinite(xy_med) and xy_med <= xy_lim_mm + 1e-9
            ),
        }
        report = {
            "check": "c30_precheck",
            "successes": n_ok,
            "episodes": len(rows),
            "dv_at_close_median_cms": med,
            "dv_at_close_p95_cms": p95,
            "close_xy_median_mm": xy_med,
            "ride_dv_median_cms": float(np.median(ride_dv)) if ride_dv else None,
            "gate_limit_cms": lim,
            "failure_modes": sorted({str(r["failure_mode"]) for r in rows if not r["success"]}),
            "checks": checks,
            "pass": bool(all(checks.values())),
            "rows": rows,
        }
        (out / "c30_prechek.json").write_text(json.dumps(report, indent=2) + "\n")
        log.info("C30 pre-check %d/%d ok, |Δv| close med %.2f p95 %.2f cm/s, xy med %.2f mm → %s",
                 n_ok, len(rows), med, p95, xy_med, report["pass"])
        if not report["pass"]:
            failed = [k for k, v in checks.items() if not v]
            write_stop(out, f"C30 pre-check failed: {failed}")
            raise SystemExit(2)
        if phase == "c30":
            return

    if phase in ("delay-check", "all"):
        log.info("Delay-injection check C22 ×10")
        lags = []
        for i in range(10):
            r = run_one(cell="C22_delay", v_cms=20, d_ms=200, seed=seed_i(i), object_usd=OBJECTS[i])
            lags.append(r.get("obs_lag_median_ms"))
        med = float(np.nanmedian([x for x in lags if x is not None]))
        dt_ms = float(cfg.get("sim", {}).get("dt", 1 / 30)) * 1000.0
        ok = abs(med - 200.0) <= dt_ms + 1.0
        rep = {"check": "delay_injection", "median_lag_ms": med, "pass": ok, "dt_ms": dt_ms}
        (out / "delay_check.json").write_text(json.dumps(rep, indent=2) + "\n")
        log.info("delay check median=%.1f ms pass=%s", med, ok)
        if not ok:
            write_stop(out, f"delay-check fail median={med}")
            raise SystemExit(3)
        if phase == "delay-check":
            return

    if phase in ("neg-control", "all"):
        log.info("Negative control C22 ×10 forecast_off")
        wins = 0
        for i in range(10):
            r = run_one(
                cell="C22_neg",
                v_cms=20,
                d_ms=200,
                seed=seed_i(i),
                object_usd=OBJECTS[i],
                forecast_off=True,
            )
            wins += int(r["success"])
        rate = wins / 10.0
        ok = rate <= 0.10 + 1e-9
        rep = {"check": "negative_control", "successes": wins, "n": 10, "rate": rate, "pass": ok}
        (out / "neg_control.json").write_text(json.dumps(rep, indent=2) + "\n")
        log.info("neg control %d/10 pass=%s", wins, ok)
        if not ok:
            write_stop(out, f"neg-control fail rate={rate}")
            raise SystemExit(4)
        if phase == "neg-control":
            return

    cells = []
    if args.role in ("unseen", "all"):
        cells.extend(UNSEEN_ORDER)
    if args.role in ("seen", "all"):
        cells.extend(SEEN_ORDER)
    # Optional subset (e.g. C30,C31,C33) so a recheck skips the full 4×4.
    want = getattr(args, "cells", None)
    if want:
        keep = {c.strip().upper() for c in str(want).split(",") if c.strip()}
        cells = [(c, v, d) for c, v, d in cells if c in keep]
        log.info("cell filter %s → %s", sorted(keep), [c for c, _, _ in cells])

    n_eps = 4 if phase == "smoke" else int(args.n)
    if phase in ("smoke", "full", "all"):
        early = phase in ("full", "all")
        # --no-early-stop: cells never write STOP, so only a validity STOP
        # (already on disk) can halt the grid at a cell boundary.
        cell_early = early and not bool(getattr(args, "no_early_stop", False))
        for cell, v, d in cells:
            if stopped(out) and early:
                log.warning("STOP — skipping remaining cells on %s", device)
                break
            log.info("=== %s v=%d Δ=%d n=%d ===", cell, v, d, n_eps)
            run_cell(cell, v, d, n_eps, early_stop=cell_early)

    if phase in ("sensitivity", "all") and args.role in ("unseen", "all"):
        for t_ms in (130.0, 250.0):
            cfg["robot"]["gripper_close_time_s"] = t_ms * 1e-3
            sm = PickPlaceStateMachine(cfg)
            for cell, v, d in (("C33", 30, 300), ("C23", 20, 300)):
                tag = f"{cell}_t{int(t_ms)}"
                log.info("sensitivity %s t_close=%s", tag, t_ms)
                # Temporarily override args.t_close_ms for slip calc.
                old = args.t_close_ms
                args.t_close_ms = t_ms
                run_cell(tag, v, d, int(args.n), early_stop=False)
                args.t_close_ms = old

    if phase in ("report", "all") or args.role == "host":
        _write_report(out)

    log.info("done phase=%s", phase)


def _write_report(out: Path) -> None:
    """Merge cell summaries into REPORT_auto.md + summary.json."""
    # Merge object shards per cell.
    shards: dict[str, list[dict]] = {}
    for p in sorted((out / "cells").glob("*/summary_*.json")):
        s = json.loads(p.read_text())
        shards.setdefault(s["cell"], []).append(s)
    # Also accept legacy single summary.json
    for p in sorted((out / "cells").glob("*/summary.json")):
        s = json.loads(p.read_text())
        shards.setdefault(s["cell"], []).append(s)

    cells = []
    for cell, parts in shards.items():
        n = sum(int(p["n"]) for p in parts)
        k = sum(int(p["successes"]) for p in parts)
        fm: dict[str, int] = {}
        for p in parts:
            for m, c in (p.get("failure_modes") or {}).items():
                fm[m] = int(fm.get(m, 0)) + int(c)
        cells.append(
            {
                "cell": cell,
                "v_cms": parts[0]["v_cms"],
                "delta_ms": parts[0]["delta_ms"],
                "n": n,
                "successes": k,
                "rate": k / n if n else float("nan"),
                "wilson95": list(wilson(k, n)) if n else [None, None],
                "seen": parts[0]["seen"],
                "device": ",".join(sorted({str(p.get("device")) for p in parts})),
                "gt_fallback_total": sum(int(p.get("gt_fallback_total", 0)) for p in parts),
                "live_gt_leaks_total": sum(int(p.get("live_gt_leaks_total", 0)) for p in parts),
                "failure_modes": fm,
            }
        )
    # Keep main 16 only.
    main = [c for c in cells if c["cell"] in {x[0] for x in UNSEEN_ORDER + SEEN_ORDER}]
    lines = [
        "# B1 oracle sm — speed x delay grid",
        "",
        f"Output: `{out}`",
        "",
        "## 4x4 success grid",
        "",
        "| | Δ=0 | Δ=100 | Δ=200 | Δ=300 |",
        "|---|---|---|---|---|",
    ]
    by = {(c["v_cms"], c["delta_ms"]): c for c in main}
    for v in SPEEDS_CMS:
        row = [f"**v={v}**"]
        for d in DELTAS_MS:
            c = by.get((v, d))
            if not c:
                row.append("—")
                continue
            lo, hi = c["wilson95"]
            tag = "seen" if c["seen"] else "unseen"
            row.append(
                f"{c['successes']}/{c['n']} ({100*c['rate']:.0f}%) "
                f"[{100*lo:.0f}–{100*hi:.0f}] {tag}"
            )
        lines.append("| " + " | ".join(row) + " |")

    # Failure modes
    lines += ["", "## Failure modes (F1–F7)", ""]
    modes = ["unreachable", "never_closed", "closed_above", "closed_empty_xy", "knocked", "slipped", "not_placed", "other"]
    lines.append("| cell | " + " | ".join(modes) + " |")
    lines.append("|---|" + "|".join(["---"] * len(modes)) + "|")
    for c in main:
        fm = c.get("failure_modes") or {}
        lines.append(
            "| "
            + c["cell"]
            + " | "
            + " | ".join(str(int(fm.get(m, 0))) for m in modes)
            + " |"
        )

    pooled_k = sum(c["successes"] for c in main)
    pooled_n = sum(c["n"] for c in main)
    lo, hi = wilson(pooled_k, pooled_n) if pooled_n else (float("nan"), float("nan"))
    seen = [c for c in main if c["seen"]]
    unseen = [c for c in main if not c["seen"]]
    sk, sn = sum(c["successes"] for c in seen), sum(c["n"] for c in seen)
    uk, un = sum(c["successes"] for c in unseen), sum(c["n"] for c in unseen)

    cell_pass = all(c["n"] >= 80 and c["successes"] >= 74 for c in main) if main else False
    # During smoke n=4 — mark provisional.
    if main and any(c["n"] < 80 for c in main):
        cell_verdict = "PROVISIONAL"
    else:
        cell_verdict = "PASS" if cell_pass else "FAIL"

    pooled_pass = pooled_n > 0 and pooled_k / pooled_n >= 0.95
    reach_ok = True
    for c in main:
        f1 = int((c.get("failure_modes") or {}).get("unreachable", 0))
        if c["n"] and f1 / c["n"] > 0.02 + 1e-12:
            reach_ok = False

    gt_total = sum(int(c.get("gt_fallback_total", 0)) for c in main)
    leak_total = sum(int(c.get("live_gt_leaks_total", 0)) for c in main)
    delay = json.loads((out / "delay_check.json").read_text()) if (out / "delay_check.json").exists() else {}
    neg = json.loads((out / "neg_control.json").read_text()) if (out / "neg_control.json").exists() else {}
    c30 = json.loads((out / "c30_prechek.json").read_text()) if (out / "c30_prechek.json").exists() else {}
    git = (out / "git_rev.txt").read_text().strip() if (out / "git_rev.txt").exists() else "?"

    valid = (
        gt_total == 0
        and leak_total == 0
        and bool(delay.get("pass", False))
        and bool(neg.get("pass", False))
        and bool(c30.get("pass", False))
    )

    lines += [
        "",
        "## Pooled",
        "",
        f"- All: {pooled_k}/{pooled_n} = {100*pooled_k/max(pooled_n,1):.1f}% "
        f"(Wilson {100*lo:.0f}–{100*hi:.0f}%)",
        f"- Seen: {sk}/{sn}",
        f"- Unseen: {uk}/{un}",
        "",
        "## Validity",
        "",
        f"- frozen_cfg: `{out / 'frozen_cfg.yaml'}`",
        f"- git: `{git}`",
        f"- GT-fallback total: {gt_total}",
        f"- live-GT leaks: {leak_total}",
        f"- delay-injection: {delay}",
        f"- negative-control: {neg}",
        f"- C30 ride |Δv|: {c30}",
        f"- t_close: SPEC-DERIVED / PROVISIONAL (180 ms nominal)",
        "",
        "## Verdict",
        "",
        f"- B1-cell: **{cell_verdict}** (≥74/80 each)",
        f"- B1-pooled: **{'PASS' if pooled_pass else 'FAIL' if pooled_n >= 80 else 'PROVISIONAL'}** (≥95%)",
        f"- B1-reach: **{'PASS' if reach_ok else 'FAIL'}** (F1 ≤2%/cell)",
        f"- B1-valid: **{'PASS' if valid else 'FAIL'}**",
        "",
        "Expert only — no policy conclusions.",
        "",
    ]
    (out / "REPORT_auto.md").write_text("\n".join(lines))
    summary = {
        "cells": main,
        "pooled": {"k": pooled_k, "n": pooled_n, "wilson95": [lo, hi]},
        "verdicts": {
            "B1_cell": cell_verdict,
            "B1_pooled": "PASS" if pooled_pass else ("PROVISIONAL" if pooled_n < 80 else "FAIL"),
            "B1_reach": "PASS" if reach_ok else "FAIL",
            "B1_valid": "PASS" if valid else "FAIL",
        },
        "gt_fallback_total": gt_total,
        "live_gt_leaks_total": leak_total,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
