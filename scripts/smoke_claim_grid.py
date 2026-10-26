#!/usr/bin/env python3
"""Claim smoke collect: 10 successes × speed × delay (random seeds).

Speeds: 0 / 0.10 / 0.15 / 0.20 m/s
Delays (applied): 0 / 0.100 / 0.200 s — pins claim_timing train min=max.

One Isaac scene for the whole grid (do not rebuild World — GroundPlane clash).

  conda activate dynamicVLA_isaac
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python scripts/smoke_claim_grid.py --headless --enable_cameras --device cuda:0 \\
      --per-cell 10 --object-usd tangerine -o data/claim_smoke_tangerine
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
SPEEDS = (0.0, 0.10, 0.15, 0.20)
DELAYS_S = (0.0, 0.100, 0.200)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Claim smoke collect grid")
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--max-attempts-per-cell", type=int, default=40)
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("-o", "--output_dir", default=str(FORESIGHT_ROOT / "data" / "claim_smoke"))
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--object-usd", default=None, help="optional category/stem, e.g. tangerine")
    return p


def _pin_delay_on_cfg(cfg: dict, delay_s: float) -> None:
    """In-place: fix applied Δ for this cell (SI keys on live cfg)."""
    ct = cfg.setdefault("claim_timing", {})
    ct["enabled"] = True
    ct["train_delta_min_s"] = float(delay_s)
    ct["train_delta_max_s"] = float(delay_s)
    ct["eval_delta_s"] = float(delay_s)
    ct.setdefault("delta_jitter_s", 0.010)
    ct.setdefault("enforce_apply_at", False)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args([a for a in sys.argv[1:] if a != ""])
    AppLauncher(args)

    from sim.collect import episode_for_json, run_episode, write_episode_h5
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine
    from sim.success import clean_reject_reason, episode_success

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("smoke_claim")

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device
    _pin_delay_on_cfg(cfg, DELAYS_S[0])

    scene = Phase0Scene(
        cfg,
        enable_cameras=True,
        object_usd=args.object_usd,
        asset_seed=int(args.seed),
    )
    expert = PickPlaceStateMachine(cfg)

    report: dict = {
        "speeds": list(SPEEDS),
        "delays_s": list(DELAYS_S),
        "per_cell": int(args.per_cell),
        "object_usd": args.object_usd,
        "cells": [],
        "started": time.time(),
    }

    for delay_s in DELAYS_S:
        _pin_delay_on_cfg(cfg, delay_s)
        # Expert reads cfg at init for SM params; delay is re-read in collect each episode.
        expert = PickPlaceStateMachine(cfg)

        for speed in SPEEDS:
            wins = 0
            attempts = 0
            cell_dir = (
                out_root
                / f"spd_{int(round(speed * 100)):02d}_del_{int(round(delay_s * 1000)):03d}"
            )
            cell_dir.mkdir(parents=True, exist_ok=True)
            seeds_used: list[int] = []
            while wins < int(args.per_cell) and attempts < int(args.max_attempts_per_cell):
                attempts += 1
                seed = int(args.seed) + int(delay_s * 1e4) + int(speed * 1e3) + attempts
                ep, meta = run_episode(
                    scene,
                    expert,
                    seed=seed,
                    speed=float(speed),
                    save_images=True,
                )
                ok = episode_success(
                    ep,
                    z_lift_min=float(expert.lift_height),
                    container_xy_tol=float(expert.place_xy_tol),
                    place_stage=int(expert.stage_schema.done),
                )
                reason = None if ok else clean_reject_reason(ep, meta.get("retries"))
                if ok:
                    wins += 1
                    seeds_used.append(seed)
                    stem = f"ep{wins:02d}_seed{seed}"
                    meta = dict(meta)
                    meta["smoke_speed"] = float(speed)
                    meta["smoke_delay_s"] = float(delay_s)
                    write_episode_h5(cell_dir / f"{stem}.h5", ep, meta)
                    (cell_dir / f"{stem}.json").write_text(
                        json.dumps(episode_for_json(ep, meta), indent=2)
                    )
                    log.info(
                        "WIN speed=%.2f delay=%.3f %d/%d %s",
                        speed,
                        delay_s,
                        wins,
                        args.per_cell,
                        stem,
                    )
                else:
                    log.info(
                        "miss speed=%.2f delay=%.3f try=%d reason=%s",
                        speed,
                        delay_s,
                        attempts,
                        reason,
                    )

            report["cells"].append(
                {
                    "speed": speed,
                    "delay_s": delay_s,
                    "wins": wins,
                    "attempts": attempts,
                    "seeds": seeds_used,
                    "dir": str(cell_dir),
                }
            )
            (out_root / "smoke_collect_report.json").write_text(json.dumps(report, indent=2))

    report["finished"] = time.time()
    report["elapsed_s"] = report["finished"] - report["started"]
    (out_root / "smoke_collect_report.json").write_text(json.dumps(report, indent=2))
    log.info("done elapsed=%.1f min → %s", report["elapsed_s"] / 60.0, out_root)
    os._exit(0)


if __name__ == "__main__":
    main()
