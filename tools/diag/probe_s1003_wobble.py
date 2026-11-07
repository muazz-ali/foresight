#!/usr/bin/env python3
"""s1003 ride wobble probe: aim vs hand along-track (signed).

Runs one egg episode at 30 cm/s, Δ=0, seed 1003. Logs every tick:
aim (command) XY, hand XY, object XY, stage. Writes JSONL + PNG of
signed along-track errors (aim−object, hand−object) over the ride.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=1 \\
    python tools/diag/probe_s1003_wobble.py --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "out" / "s1003_wobble"
STAGE_NAME = {
    0: "START",
    1: "INTERCEPT",
    2: "DESCEND",
    3: "CLOSE",
    4: "LIFT",
    5: "CARRY",
    6: "LOWER",
    7: "OPEN",
    8: "HOME",
    9: "DONE",
}


def _along_cross(err_xy: np.ndarray, uh: np.ndarray) -> tuple[float, float]:
    nh = np.array([-uh[1], uh[0]], dtype=np.float64)
    return float(err_xy @ uh), float(err_xy @ nh)


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--sim_cfg_file", default=str(ROOT / "sim" / "phase0_cfg.yaml"))
    p.add_argument("-o", "--output_dir", default=str(OUT))
    p.add_argument("--seed", type=int, default=1003)
    p.add_argument("--speed", type=float, default=0.30)
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args([a for a in sys.argv[1:] if a != ""])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    AppLauncher(args)

    from eval.b1_success import b1_success
    from sim.collect import run_episode
    from sim.scene import Phase0Scene, load_cfg
    from sim.state_machine import PickPlaceStateMachine

    cfg = load_cfg(Path(args.sim_cfg_file))
    cfg.setdefault("claim_timing", {})["enabled_noise_on_replan"] = True
    cfg["claim_timing"]["apply_delta_delay_to_robot"] = False
    cfg.setdefault("sim", {})["device"] = str(getattr(args, "device", None) or "cuda:0")

    scene = Phase0Scene(cfg, enable_cameras=False, object_usd="egg", asset_seed=1000)
    sm = PickPlaceStateMachine(cfg)
    log, meta = run_episode(
        scene,
        sm,
        seed=int(args.seed),
        speed=float(args.speed),
        save_images=False,
        obs_delay_s=0.0,
    )
    ok, mode = b1_success(log, meta, z_table=float(getattr(scene, "_table_z", 0.0)))

    stage = np.asarray(log["stage"], dtype=np.int64)
    ee = np.asarray(log["ee_pos"], dtype=np.float64)
    act = np.asarray(log["action_pos"], dtype=np.float64)
    obj = np.asarray(log["object_pos"], dtype=np.float64)
    vel = np.asarray(log["object_vel"], dtype=np.float64)
    t = np.asarray(log["timestamp"], dtype=np.float64)

    # Ride = approach + close (and any early lift). Use object vel at first moving tick.
    ride = np.isin(stage, [1, 3])
    if not bool(ride.any()):
        ride = stage == 1
    idx = np.where(ride)[0]
    # Unit along object motion (coast).
    speeds = np.linalg.norm(vel[idx, :2], axis=1)
    i_u = int(idx[int(np.argmax(speeds > 0.05))]) if bool((speeds > 0.05).any()) else int(idx[0])
    uh = vel[i_u, :2].copy()
    nrm = float(np.linalg.norm(uh))
    uh = uh / nrm if nrm > 1e-9 else np.array([1.0, 0.0])

    rows = []
    for i in range(len(stage)):
        e_aim = act[i, :2] - obj[i, :2]
        e_hand = ee[i, :2] - obj[i, :2]
        a_along, a_cross = _along_cross(e_aim, uh)
        h_along, h_cross = _along_cross(e_hand, uh)
        st = int(stage[i])
        rows.append(
            {
                "i": int(i),
                "t_s": float(t[i]),
                "stage": st,
                "stage_name": STAGE_NAME.get(st, str(st)),
                "aim_xy_m": act[i, :2].tolist(),
                "hand_xy_m": ee[i, :2].tolist(),
                "obj_xy_m": obj[i, :2].tolist(),
                "aim_along_mm": 1000.0 * a_along,
                "hand_along_mm": 1000.0 * h_along,
                "aim_cross_mm": 1000.0 * a_cross,
                "hand_cross_mm": 1000.0 * h_cross,
                "on_ride": bool(ride[i]),
            }
        )

    jsonl = out / f"s{int(args.seed)}_ticks.jsonl"
    with jsonl.open("w") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")

    # Peak period of signed hand_along on ride (local extrema).
    ride_rows = [r for r in rows if r["on_ride"]]
    hand_al = np.array([r["hand_along_mm"] for r in ride_rows], dtype=np.float64)
    aim_al = np.array([r["aim_along_mm"] for r in ride_rows], dtype=np.float64)
    tt = np.array([r["t_s"] for r in ride_rows], dtype=np.float64)
    tt0 = float(tt[0]) if len(tt) else 0.0
    tt_rel = tt - tt0

    def _signed_peak_period(x: np.ndarray, t_rel: np.ndarray) -> dict:
        """Period from consecutive local extrema of a signed series."""
        if len(x) < 5:
            return {"n_peaks": 0, "periods_s": [], "mean_period_s": None}
        peaks = []
        for i in range(2, len(x) - 2):
            w = x[i - 2 : i + 3]
            if x[i] == np.max(w) or x[i] == np.min(w):
                if abs(x[i]) >= 0.5:  # ignore sub-mm noise
                    peaks.append(i)
        # de-dupe neighbours
        keep = []
        for i in peaks:
            if not keep or i - keep[-1] >= 2:
                keep.append(i)
        ts = t_rel[keep]
        periods = np.diff(ts).tolist() if len(ts) >= 2 else []
        return {
            "n_peaks": len(keep),
            "peak_t_s": ts.tolist(),
            "periods_s": periods,
            "mean_period_s": float(np.mean(periods)) if periods else None,
        }

    # Aim wobble: std / peak-to-peak of aim_along on ride after first 0.25 s settle.
    if len(tt_rel):
        steady = tt_rel >= 0.25
    else:
        steady = np.array([], dtype=bool)
    aim_steady = aim_al[steady] if steady.size else aim_al
    hand_steady = hand_al[steady] if steady.size else hand_al

    # Stage flips on ride.
    st_ride = [r["stage"] for r in ride_rows]
    flips = sum(1 for a, b in zip(st_ride, st_ride[1:]) if a != b)
    stage_seq = []
    for s in st_ride:
        if not stage_seq or stage_seq[-1] != s:
            stage_seq.append(s)

    report = {
        "seed": int(args.seed),
        "speed_m_s": float(args.speed),
        "success": bool(ok),
        "failure_mode": mode,
        "n_ride": len(ride_rows),
        "uh": uh.tolist(),
        "aim_along_steady_mm": {
            "mean": float(np.mean(aim_steady)) if len(aim_steady) else None,
            "std": float(np.std(aim_steady)) if len(aim_steady) else None,
            "ptp": float(np.ptp(aim_steady)) if len(aim_steady) else None,
        },
        "hand_along_steady_mm": {
            "mean": float(np.mean(hand_steady)) if len(hand_steady) else None,
            "std": float(np.std(hand_steady)) if len(hand_steady) else None,
            "ptp": float(np.ptp(hand_steady)) if len(hand_steady) else None,
        },
        "aim_peak_period": _signed_peak_period(aim_al, tt_rel),
        "hand_peak_period": _signed_peak_period(hand_al, tt_rel),
        "stage_flips_on_ride": int(flips),
        "stage_sequence_on_ride": [STAGE_NAME.get(s, str(s)) for s in stage_seq],
        # Verdict helpers
        "aim_wobbles": bool(
            len(aim_steady) and float(np.std(aim_steady)) > 1.0  # >1 mm std
        ),
        "hand_wobbles_on_smooth_aim": bool(
            len(aim_steady)
            and float(np.std(aim_steady)) <= 1.0
            and float(np.std(hand_steady)) > 1.0
        ),
    }
    (out / f"s{int(args.seed)}_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0, ax1 = axes
    ax0.axhline(0.0, color="k", lw=0.6, ls="--")
    ax0.plot(tt_rel, aim_al, "C0-", label="aim − object (along)")
    ax0.plot(tt_rel, hand_al, "C1-", label="hand − object (along)")
    ax0.set_ylabel("along-track error (mm)")
    ax0.set_title(
        f"s{args.seed} ride  success={ok} fail={mode}  "
        f"aim_std={report['aim_along_steady_mm']['std']:.2f} "
        f"hand_std={report['hand_along_steady_mm']['std']:.2f} mm"
    )
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Stage as step
    st_y = np.array(st_ride, dtype=float)
    ax1.step(tt_rel, st_y, where="post", color="C2")
    ax1.set_yticks([1, 3, 8])
    ax1.set_yticklabels(["INTERCEPT", "CLOSE", "HOME"])
    ax1.set_xlabel("time on ride (s)")
    ax1.set_ylabel("stage")
    ax1.set_title(f"stage flips on ride = {flips}  seq={report['stage_sequence_on_ride']}")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out / f"s{int(args.seed)}_along_track.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"wrote {jsonl}")
    print(f"wrote {png}")
    print(f"wrote {out / f's{int(args.seed)}_report.json'}")
    os._exit(0)


if __name__ == "__main__":
    main()
