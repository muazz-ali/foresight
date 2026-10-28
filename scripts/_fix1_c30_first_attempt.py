#!/usr/bin/env python3
"""C30 first-attempt: 30 cm/s, Δ=0, one object category, listed seeds.

Writes ``c30_first_attempt_<object>.json`` plus per-seed ``.h5`` / ``.json`` /
``.mp4`` under ``cells/C30/``.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python scripts/_fix1_c30_first_attempt.py --headless --enable_cameras \\
      --device cuda:0 --object egg --seeds 1000,1001,1003
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEEDS = (1000, 1001, 1003)


def _parse_seeds(raw: str | None) -> tuple[int, ...]:
    """Parse ``1000,1001,1003`` → ints. Empty / None → default egg seeds."""
    if raw is None or not str(raw).strip():
        return DEFAULT_SEEDS
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit("--seeds must list at least one int")
    return tuple(out)


def main() -> None:
    from isaaclab.app import AppLauncher
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output_dir",
        default=str(ROOT / "data/eval/fix1_static_offset/c30_seeds"),
    )
    p.add_argument("-c", "--sim_cfg_file", default=str(ROOT / "sim/phase0_cfg.yaml"))
    p.add_argument(
        "--object",
        default="egg",
        help="Object category (egg|lemon|apple|…). One scene per process.",
    )
    p.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated episode seeds (throw path). Default: 1000,1001,1003.",
    )
    p.add_argument(
        "--asset-seed",
        type=int,
        default=None,
        help="Mesh/house pick seed (default: first episode seed).",
    )
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    AppLauncher(args)

    from eval.b1_success import b1_success
    from sim.collect import episode_for_json, run_episode, write_episode_h5
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine
    from sim.video import gates_from_expert, write_debug_mp4

    obj = str(args.object).strip().lower()
    seeds = _parse_seeds(args.seeds)
    asset_seed = int(args.asset_seed) if args.asset_seed is not None else int(seeds[0])

    cfg = load_cfg(Path(args.sim_cfg_file))
    cfg.setdefault("claim_timing", {})["obs_delay_buffer"] = True
    cfg["claim_timing"]["enforce_apply_at"] = False
    # Δ=0 first-attempt: freeze η so aim lead and gate are repeatable.
    cfg["claim_timing"]["delta_meas_noise_ms"] = 0.0
    cfg.setdefault("sim", {})["device"] = str(getattr(args, "device", None) or "cuda:0")
    out = Path(args.output_dir)
    cell_dir = out / "cells" / "C30"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # One Isaac scene per process (cannot rebuild World for a new category).
    scene = Phase0Scene(
        cfg, enable_cameras=True, object_usd=obj, asset_seed=asset_seed
    )
    rows = []
    for seed in seeds:
        expert = PickPlaceStateMachine(cfg)
        log, meta = run_episode(
            scene,
            expert,
            seed=int(seed),
            speed=0.30,
            save_images=True,
            obs_delay_s=0.0,
        )
        ok, mode = b1_success(log, meta, z_table=float(getattr(scene, "_table_z", 0.0)))
        cat = str(meta.get("category") or obj)
        stem = f"C30_v30_d000_s{int(seed)}_{cat}"
        h5_path = cell_dir / f"{stem}.h5"
        write_episode_h5(
            h5_path,
            log,
            extra={
                "speed_m_s": np.asarray([0.30], dtype=np.float32),
                "seed": np.asarray([int(seed)], dtype=np.int64),
                "success": np.asarray([int(bool(ok))], dtype=np.int8),
                "delta_s": np.asarray([0.0], dtype=np.float32),
                "delta_ms": np.asarray([0], dtype=np.int32),
            },
        )
        payload = {
            "cell": "C30",
            "seed": int(seed),
            "object": cat,
            "v_cms": 30,
            "delta_ms": 0,
            "success": bool(ok),
            "failure_mode": mode,
            "first_attempt": True,
            **meta,
            **episode_for_json(log),
        }
        with (cell_dir / f"{stem}.json").open("w") as fp:
            json.dump(payload, fp, indent=2, default=str)
        fps = int(cfg.get("static_camera", {}).get("fps", 25))
        mp4_path = cell_dir / f"{stem}.mp4"
        write_debug_mp4(log, mp4_path, fps=fps, gates=gates_from_expert(expert))

        row = {
            "seed": int(seed),
            "success": bool(ok),
            "failure_mode": mode,
            "first_attempt": True,
            "category": cat,
            "close_xy_err_m": getattr(expert, "close_xy_err_m", None),
            "dv_at_close_m_s": getattr(expert, "dv_at_close_m_s", None),
            "h5": str(h5_path),
            "mp4": str(mp4_path),
        }
        rows.append(row)
        xy = row["close_xy_err_m"]
        print(
            f"C30 object={cat} seed={seed} success={ok} mode={mode} "
            f"xy_err_mm={(1000 * xy) if xy is not None else None}"
        )

    summary = {
        "note": f"First attempt only; {obj}; v=30 cm/s; Δ=0. DiffIK velocity FF on.",
        "object": obj,
        "n_ok": int(sum(1 for r in rows if r["success"])),
        "n": len(rows),
        "rows": rows,
    }
    summary_path = out / f"c30_first_attempt_{obj}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    # Keep legacy name when running the original egg trio.
    if obj == "egg" and seeds == DEFAULT_SEEDS:
        (out / "c30_first_attempt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
