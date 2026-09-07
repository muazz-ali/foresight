#!/usr/bin/env python3
"""Measure the two arm numbers the intercept-servo expert depends on: ``servo_lag_s`` and ``ee_speed_max``.

Both are currently *guesses* in ``sim/phase0_cfg.yaml``. Every 40 / 80 cm/s
claim rests on them, so measure before tuning anything else.

  conda activate dynamicVLA_isaac
  python tools/diag/calibrate_servo_lag.py --headless

What it does
------------
1. **Step response.** Command a 15 cm XY step from the init pose and fit a
   first-order lag: ``servo_lag_s`` is the time to cover 63.2% of the step.
2. **Ramp tracking.** Command a target sliding at 0.8 m/s and read the
   steady-state trail distance. ``lag = trail / speed`` cross-checks (1); the
   two should agree within ~20%.
3. **Speed cap.** Peak TCP speed reached during a long step is ``ee_speed_max``.

Writes ``tools/diag/out/servo_lag.json`` and prints the yaml lines to paste
into ``state_machine_params``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORESIGHT_ROOT))

DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
OUT = Path(__file__).resolve().parent / "out" / "servo_lag.json"


def _hold(scene, target: np.ndarray, quat: np.ndarray, n: int) -> np.ndarray:
    """Command ``target`` for ``n`` steps; return the TCP track (n, 3)."""
    track = []
    for _ in range(n):
        scene.set_ee_target(target, quat, 1.0)
        scene.step(kinematic_object=False)
        track.append(scene.get_ee_pose()[0].copy())
    return np.asarray(track)


def step_response(scene, quat: np.ndarray, *, step_m: float, n: int) -> dict:
    """First-order lag from a pure XY step, plus the peak TCP speed it reached."""
    start = scene.get_ee_pose()[0].copy()
    target = start.copy()
    target[1] += step_m
    track = _hold(scene, target, quat, n)

    travelled = np.linalg.norm(track - start, axis=1)
    total = float(travelled[-1])
    speeds = np.linalg.norm(np.diff(track, axis=0), axis=1) / scene.dt

    lag = float("nan")
    if total > 0.5 * step_m:
        hit = np.argmax(travelled >= 0.632 * total)
        if travelled[hit] >= 0.632 * total:
            lag = float((hit + 1) * scene.dt)
    return {
        "commanded_step_m": float(step_m),
        "settled_m": total,
        "steady_state_error_m": float(step_m - total),
        "servo_lag_s": lag,
        "peak_tcp_speed_m_s": float(speeds.max()) if len(speeds) else 0.0,
    }


def ramp_response(scene, quat: np.ndarray, *, speed: float, n: int) -> dict:
    """Steady-state trail behind a target sliding at ``speed``; lag = trail / speed."""
    start = scene.get_ee_pose()[0].copy()
    trail = []
    cmd = start.copy()
    for i in range(n):
        cmd = start.copy()
        cmd[1] += speed * (i + 1) * scene.dt
        scene.set_ee_target(cmd, quat, 1.0)
        scene.step(kinematic_object=False)
        ee = scene.get_ee_pose()[0]
        trail.append(float(np.linalg.norm(cmd[:2] - ee[:2])))
    tail = np.asarray(trail[-max(5, n // 4):])
    trail_m = float(tail.mean())
    return {
        "ramp_speed_m_s": float(speed),
        "steady_trail_m": trail_m,
        "servo_lag_s": trail_m / speed if speed > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure servo lag + TCP speed cap")
    parser.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    parser.add_argument("--step-m", type=float, default=0.15)
    parser.add_argument("--ramp-speed", type=float, default=0.80)
    parser.add_argument("--steps", type=int, default=60)

    from isaaclab.app import AppLauncher

    sys.argv = [a for a in sys.argv if a != ""]
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    app_launcher = AppLauncher(args)

    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device
    scene = Phase0Scene(cfg, enable_cameras=False)
    scene.reset_episode(seed=0, speed=0.0)
    quat = np.asarray(cfg["robot"]["init_pose"][3:], dtype=np.float64)

    _hold(scene, np.asarray(cfg["robot"]["init_pose"][:3]), quat, 25)  # settle at home
    step = step_response(scene, quat, step_m=args.step_m, n=args.steps)

    scene.reset_episode(seed=0, speed=0.0)
    _hold(scene, np.asarray(cfg["robot"]["init_pose"][:3]), quat, 25)
    ramp = ramp_response(scene, quat, speed=args.ramp_speed, n=args.steps)

    lags = [x for x in (step["servo_lag_s"], ramp["servo_lag_s"]) if np.isfinite(x)]
    recommend = {
        "servo_lag_s": round(float(max(lags)), 3) if lags else None,
        "ee_speed_max": round(0.9 * step["peak_tcp_speed_m_s"], 2),
    }
    report = {
        "sim_dt": scene.dt,
        "step_response": step,
        "ramp_response": ramp,
        "recommend_state_machine_params": recommend,
        "agreement_ok": bool(
            len(lags) == 2 and abs(lags[0] - lags[1]) <= 0.2 * max(lags)
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nPaste into sim/phase0_cfg.yaml under state_machine_params:")
    print(f"  servo_lag_s: {recommend['servo_lag_s']}")
    print(f"  ee_speed_max: {recommend['ee_speed_max']}")
    if not report["agreement_ok"]:
        print("\nWARNING: step and ramp disagree by >20%. The arm is not a clean")
        print("first-order lag — report both numbers before trusting either.")
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
