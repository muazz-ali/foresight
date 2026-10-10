#!/usr/bin/env python3
"""Sanity-check collected Phase-1 episodes: Δ + gripper vs yaml.

Logged gripper is open∈[0,1] (1=open, 0=closed). Yaml ``gripper_open`` /
``gripper_close`` are finger widths (m) used only when the scene maps 0/1 → mm.

Why it exists: bad recordings waste a long training run. This is a quick
check after collection. For each .h5 it asks:
  1. Is Δ saved, and is it the Δ in the yaml?
  2. Does the saved 4-number pack match one rebuilt from the saved truth?
  3. Does the gripper stay in [0,1] and both open and close at least once?
Exit code 0 = all good, 1 = something is off (see "issues" in the JSON).

Usage:
  python scripts/check_collect_recordings.py --root data/p1_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

from interfaces.config import conditioning_delta_s, gripper_limits, load_yaml  # noqa: E402
from interfaces.state import conditioning_vector, oracle_from_gt  # noqa: E402


def _check_h5(path: Path, *, expect_delta: float) -> dict:
    """Check one episode file. Returns a dict; ``issues`` lists what failed."""
    out: dict = {"path": str(path), "ok": True, "issues": []}
    with h5py.File(path, "r") as f:
        delta = float(np.asarray(f["delta"]).reshape(-1)[0]) if "delta" in f else None
        cond = np.asarray(f["conditioning"], dtype=np.float64)
        pos = np.asarray(f["object_pos"], dtype=np.float64)
        vel = np.asarray(f["object_vel"], dtype=np.float64)
        grip = np.asarray(f["gripper"], dtype=np.float64).reshape(-1)
        action = np.asarray(f["action"], dtype=np.float64) if "action" in f else None

    out["delta"] = delta
    out["n_frames"] = int(cond.shape[0])
    out["cond_dim"] = int(cond.shape[1]) if cond.ndim == 2 else None

    # Check 1: Δ in the file must equal Δ in the yaml (train and eval use the same one).
    if delta is None:
        out["ok"] = False
        out["issues"].append("missing delta dataset")
    elif abs(delta - expect_delta) > 1e-6:
        out["ok"] = False
        out["issues"].append(f"delta={delta} != yaml {expect_delta}")

    # Check 2: rebuild the pack from saved truth with the shared helper.
    # It should match the saved pack almost exactly. A gap means the pack is
    # not consistent with the saved object truth. (0.04 s = frame step used
    # here for the timestamp; it does not change the pack.)
    recon = []
    for t in range(pos.shape[0]):
        st = oracle_from_gt(pos[t], vel[t], timestamp=float(t) * 0.04)
        recon.append(conditioning_vector(st, delta=expect_delta))
    recon_a = np.asarray(recon, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((cond - recon_a) ** 2)))
    out["cond_vs_cv_rmse"] = rmse
    if rmse > 1e-5:
        out["ok"] = False
        out["issues"].append(f"conditioning RMSE vs CV@{expect_delta}s = {rmse:.3e}")

    # Check 3: gripper stays in range, and the hand really opened AND closed
    # (an episode that never closes cannot have grasped anything).
    g_min, g_max = float(np.min(grip)), float(np.max(grip))
    out["gripper_min"] = g_min
    out["gripper_max"] = g_max
    # Logged command is open∈[0,1], not yaml finger meters.
    if g_min < -1e-3 or g_max > 1.0 + 1e-3:
        out["ok"] = False
        out["issues"].append(f"gripper [{g_min:.4f},{g_max:.4f}] outside [0,1]")
    near_open = bool(np.any(grip > 0.9))
    near_close = bool(np.any(grip < 0.1))
    out["saw_open"] = near_open
    out["saw_close"] = near_close
    if not (near_open and near_close):
        out["ok"] = False
        out["issues"].append("gripper never both open (>0.9) and closed (<0.1)")

    # Info only (no pass/fail): gripper column of the saved action.
    if action is not None and action.ndim == 2 and action.shape[1] >= 8:
        a_grip = action[:, 7]
        out["action_gripper_min"] = float(np.min(a_grip))
        out["action_gripper_max"] = float(np.max(a_grip))

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Check Δ + gripper in collected episodes")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("-c", "--sim_cfg_file", default=str(FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"))
    args = p.parse_args()

    cfg = load_yaml(args.sim_cfg_file)
    expect_delta = conditioning_delta_s(cfg)
    grip_open_m, grip_close_m = gripper_limits(cfg)
    root = Path(args.root)
    h5s = sorted(root.rglob("*.h5"))
    if not h5s:
        print(json.dumps({"ok": False, "error": f"no .h5 under {root}"}, indent=2))
        raise SystemExit(1)

    rows = [_check_h5(h, expect_delta=expect_delta) for h in h5s]
    mp4s = sorted(root.rglob("*.mp4"))
    # Speed list is read from folder names like speed_0.15 (shows which bins were collected).
    speeds = sorted(
        {float(p.parent.name.split("_")[-1]) for p in h5s if "speed_" in p.parent.name}
    )
    report = {
        "ok": all(r["ok"] for r in rows),
        "n_h5": len(h5s),
        "n_mp4": len(mp4s),
        "speed_dirs_m_s": speeds,
        "expect_delta_s": expect_delta,
        "yaml_gripper_open_m": grip_open_m,
        "yaml_gripper_close_m": grip_close_m,
        "logged_gripper_note": "open∈[0,1]; scene maps to yaml finger widths",
        "episodes": rows,
    }
    # Save the report next to the data, print it, and set the exit code.
    out_path = root / "recording_check.json"
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
