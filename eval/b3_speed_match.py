"""B3: speed matching at close.  Pass: |v_obj − v_hand| · t_close ≤ 2.5 mm.

The rule lives in control/close_guard.py (2.5 mm slack over robot.gripper_close_time_s → ≈3.1 cm/s).
This script checks it on recorded episodes at the first close command (gripper < 0.5):

  instant   |v_obj − v_hand| · t_close at the close command, XY (the slip that matters is across
            the pads) and 3D (what close_speed_check computes when given 3D vectors)
  window    the slide that actually happens while the fingers shut: change of (p_obj − p_hand) in XY
            from the close command to t_close later

Velocities: central differences of logged positions (30 Hz). Offline; reads recordings made with the
current setup (default data/eval/A5_recordings). Stops if the close time is not configured or the
recordings lack the needed fields; episodes with no close are counted, not scored.

Usage: python eval/b3_speed_match.py [--root data/eval/A5_recordings]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.close_guard import DEFAULT_SLACK_M, close_speed_check  # noqa: E402
from interfaces.config import load_yaml  # noqa: E402

NEED = ("timestamp", "ee_pos", "object_pos", "gripper")


def stop(msg):
    sys.exit(f"B3 stop: {msg}")


def vel(p, ts, i):
    lo, hi = max(i - 1, 0), min(i + 1, len(ts) - 1)
    return (p[hi] - p[lo]) / max(ts[hi] - ts[lo], 1e-9)


def one(path, t_close):
    with h5py.File(path) as h:
        for k in NEED:
            if k not in h:
                stop(f"{path} lacks '{k}'.")
        ts = h["timestamp"][:]
        ee = h["ee_pos"][:].astype(float)
        ob = h["object_pos"][:].astype(float)
        g = h["gripper"][:]
    idx = np.flatnonzero(g < 0.5)
    if not len(idx):
        return {"path": path, "closed": False}
    c = int(idx[0])
    v_h, v_o = vel(ee, ts, c), vel(ob, ts, c)
    chk_xy = close_speed_check(v_h[:2], v_o[:2], slack_m=DEFAULT_SLACK_M, close_time_s=t_close)
    chk_3d = close_speed_check(v_h, v_o, slack_m=DEFAULT_SLACK_M, close_time_s=t_close)
    j = int(np.searchsorted(ts, ts[c] + t_close))
    j = min(j, len(ts) - 1)
    rel0, rel1 = ob[c, :2] - ee[c, :2], ob[j, :2] - ee[j, :2]
    return {
        "path": path, "closed": True, "t_close_cmd": float(ts[c]),
        "dv_xy": chk_xy.speed_diff_m_s, "dv_3d": chk_3d.speed_diff_m_s,
        "slip_xy_mm": chk_xy.speed_diff_m_s * t_close * 1000, "slip_3d_mm": chk_3d.speed_diff_m_s * t_close * 1000,
        "window_slide_mm": float(np.linalg.norm(rel1 - rel0) * 1000),
        "v_obj": float(np.linalg.norm(v_o[:2])), "v_hand": float(np.linalg.norm(v_h[:2])),
        "hand_z_speed": float(abs(v_h[2])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "eval" / "A5_recordings"))
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "B3_speed_match"))
    a = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    t_close = (cfg.get("robot") or {}).get("gripper_close_time_s")
    if not t_close:
        stop("robot.gripper_close_time_s (gripper_close_time_ms) is not set in sim/phase0_cfg.yaml.")
    t_close = float(t_close)
    sm = cfg.get("state_machine_params") or {}
    slack_mm = float(sm.get("close_speed_slack_m", DEFAULT_SLACK_M)) * 1000
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, allres = [], []
    for vdir in sorted(Path(a.root).glob("v*")):
        files = sorted(glob.glob(str(vdir / "**" / "*.h5"), recursive=True))
        if not files:
            continue
        res = [one(f, t_close) for f in files]
        allres += res
        R = [r for r in res if r["closed"]]
        if not R:
            rows.append({"speed": float(vdir.name[1:]), "episodes": len(res), "closed": 0})
            continue
        s = lambda k: np.array([r[k] for r in R])
        rows.append({
            "speed": float(vdir.name[1:]), "episodes": len(res), "closed": len(R),
            "pass_xy": int(np.sum(s("slip_xy_mm") <= slack_mm)), "pass_3d": int(np.sum(s("slip_3d_mm") <= slack_mm)),
            "slip_xy_median": float(np.median(s("slip_xy_mm"))), "slip_xy_p95": float(np.percentile(s("slip_xy_mm"), 95)),
            "slip_3d_median": float(np.median(s("slip_3d_mm"))),
            "window_median": float(np.median(s("window_slide_mm"))), "window_p95": float(np.percentile(s("window_slide_mm"), 95)),
            "pass_window": int(np.sum(s("window_slide_mm") <= slack_mm)),
            "v_obj": float(np.median(s("v_obj"))), "v_hand": float(np.median(s("v_hand"))), "hand_z": float(np.median(s("hand_z_speed"))),
        })
    if not rows:
        stop(f"no recordings under {a.root}.")
    (out / "b3_report.json").write_text(json.dumps({"rows": rows, "episodes": allres, "t_close_s": t_close,
                                                   "close_speed_slack_mm": slack_mm}, indent=1))
    L = ["# B3: speed matching at close (pass: |v_obj − v_hand| · t_close ≤ 2.5 mm)", "",
         f"t_close = {t_close*1000:.0f} ms (robot.gripper_close_time_s) → limit {slack_mm/t_close/10:.2f} cm/s. "
         f"close_speed_slack_mm = {slack_mm:.1f}. Recordings: {a.root}", "",
         "| speed | closed / episodes | pass, instant XY | instant XY mm, median / p95 | pass, instant 3D | 3D median mm | "
         "actual slide in window mm, median / p95 | pass, window | obj / hand XY speed cm/s | hand Z speed cm/s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["closed"]:
            L.append(f"| {r['speed']:.2f} | 0/{r['episodes']} | – | – | – | – | – | – | – | – |"); continue
        L.append(f"| {r['speed']:.2f} | {r['closed']}/{r['episodes']} | {r['pass_xy']}/{r['closed']} | "
                 f"{r['slip_xy_median']:.2f} / {r['slip_xy_p95']:.2f} | {r['pass_3d']}/{r['closed']} | {r['slip_3d_median']:.2f} | "
                 f"{r['window_median']:.2f} / {r['window_p95']:.2f} | {r['pass_window']}/{r['closed']} | "
                 f"{r['v_obj']*100:.1f} / {r['v_hand']*100:.1f} | {r['hand_z']*100:.1f} |")
    tot = sum(r.get("closed", 0) for r in rows)
    ok = sum(r.get("pass_xy", 0) for r in rows)
    L += ["", f"**Instant XY: {ok}/{tot} closes within 2.5 mm → {'PASS' if ok == tot and tot else 'FAIL'}**"]
    (out / "REPORT_auto.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
