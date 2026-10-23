"""Offline check: how well does the wrist RGB-D locate the object near the grasp?

For each expert frame with the TCP 0-12 cm above the object (before the close):
  1. Mask the object by colour (HSV) in the wrist RGB, keep pixels with valid depth.
  2. Back-project them with the wrist depth (distance to image plane) through the
     wrist camera model (sim/phase0_cfg.yaml: 240 px focal, mount 65 mm along the
     hand x, OpenGL frame) into the world.
  3. Fit the known object shape (axis-aligned ellipsoid, the visual mesh extents)
     to those surface points -> object centre.
  4. Error vs sim truth, in the gripper frame (pad axis, closing axis, height).

Then the measured errors feed the close-rule replay (the P_s rule and the distance
rule fire on wrist-estimated offsets; success is scored with the true P_s).

Usage: python eval/wrist_estimate.py --mesh tangerine06 -o data/eval/wrist_estimate
Conda: dynamicVLA_isaac. Reads data/p3_throw (expert demos keep wrist RGB + depth).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

FX, CX, CY = 240.0, 240.0, 180.0  # 2.3 mm / 4.6 mm aperture * 480 px
CAM_IN_HAND = np.array([0.065, 0.0, 0.0])
R_CAM_IN_HAND = Rot.from_quat([0.7071068, 0.7071068, 0.0, 0.0])  # cfg wxyz (0, .707, .707, 0)
TCP_IN_HAND = 0.107


def cam_pose(tcp, q_wxyz):
    rh = Rot.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]])
    hand = np.asarray(tcp) - rh.apply([0, 0, TCP_IN_HAND])
    return hand + rh.apply(CAM_IN_HAND), rh * R_CAM_IN_HAND


def orange_mask(rgb):
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    # orange: red highest, green between, blue low, saturated
    return (r > g) & (g > b) & (sat > 0.45) & (mx > 0.25) & ((r - g) > 0.15)


def estimate(rgb, depth, tcp, q, axes, max_pts=1500, rng=None):
    """Object centre (world) from one RGB-D frame, or None. Also returns pixel count."""
    m = orange_mask(rgb) & np.isfinite(depth) & (depth > 0.01) & (depth < 0.6)
    vs, us = np.nonzero(m)
    if len(us) < 40:
        return None, len(us)
    if len(us) > max_pts:
        idx = (rng or np.random.default_rng(0)).choice(len(us), max_pts, replace=False)
        us, vs = us[idx], vs[idx]
    z = depth[vs, us].astype(np.float64)
    pc = np.stack([(us - CX) * z / FX, -(vs - CY) * z / FX, -z], 1)  # OpenGL camera frame
    p_cam, r_cam = cam_pose(tcp, q)
    pw = p_cam + r_cam.apply(pc)
    c0 = pw.mean(0) + (pw.mean(0) - p_cam) / np.linalg.norm(pw.mean(0) - p_cam) * axes.min()

    def res(c):
        return np.linalg.norm((pw - c) / axes, axis=1) - 1.0

    sol = least_squares(res, c0, loss="soft_l1", f_scale=0.05)
    return sol.x, len(us)


def gripper_frame(d, q):
    return Rot.from_quat(np.asarray(q)[[1, 2, 3, 0]]).inv().apply(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default="tangerine06")
    ap.add_argument("--axes-cm", type=float, nargs=3, default=[3.49, 3.50, 2.55],
                    help="visual mesh half-extents (USD extent x scale)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("data/eval/wrist_estimate"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    axes = np.array(a.axes_cm) / 100.0
    files = [j for j in sorted(glob.glob("data/p3_throw/gpu*/speed_*/*.json"))
             if json.load(open(j))["usd"].endswith(f"/{a.mesh}.usd")]
    if a.max_episodes:
        files = files[: a.max_episodes]
    rng = np.random.default_rng(0)
    rows = []
    for j in files:
        meta = json.load(open(j))
        with h5py.File(j[:-5] + ".h5", "r") as f:
            g, ee, q, op = f["gripper"][:], f["ee_pos"][:], f["ee_quat"][:], f["object_pos"][:]
            i = int(np.argmax(g < 0.5)) if (g < 0.5).any() else len(g)
            # gripper local z points down, so its z component of (object - TCP) = TCP height above the object
            ks = [k for k in range(i) if 0.0 <= gripper_frame(op[k] - ee[k], q[k])[2] <= 0.12]
            for k in ks:
                c, npx = estimate(f["wrist_cam_rgb"][k], f["wrist_cam_depth"][k].astype(np.float32), ee[k], q[k], axes, rng=rng)
                e_true = gripper_frame(op[k] - ee[k], q[k])
                h_true = e_true[2]
                row = {"ep": Path(j).stem, "speed": float(meta["speed_m_s"]), "k": k, "frames_to_close": i - k,
                       "h_cm": h_true * 100, "npx": npx}
                if c is not None:
                    err = gripper_frame(c - op[k], q[k])
                    row.update(err_pad=err[0] * 100, err_close=err[1] * 100, err_h=err[2] * 100,
                               err_xy=float(np.linalg.norm(c[:2] - op[k][:2]) * 100))
                rows.append(row)
    (a.out / "frames.json").write_text(json.dumps(rows))
    ok = [r for r in rows if "err_xy" in r]
    print(f"{a.mesh}: {len(files)} expert episodes, {len(rows)} frames 0-12 cm above, estimate found on {len(ok)} ({len(ok)/max(len(rows),1):.0%})")
    print("TCP height above object | n | found | |err| pad / closing / height (cm): median, p90 | xy median, p90")
    out = {}
    for lo, hi in [(8, 12), (5, 8), (3, 5), (1.5, 3), (0, 1.5)]:
        sel = [r for r in rows if lo <= r["h_cm"] < hi]
        so = [r for r in sel if "err_xy" in r]
        if not sel:
            continue
        def st(key):
            v = np.abs([r[key] for r in so])
            return (round(float(np.median(v)), 2), round(float(np.percentile(v, 90)), 2)) if len(v) else (None, None)
        out[f"{lo}-{hi}"] = {"n": len(sel), "found": len(so), **{k: st(k) for k in ("err_pad", "err_close", "err_h", "err_xy")}}
        print(f"  {lo:>4}-{hi:<4} cm | {len(sel):5d} | {len(so)/len(sel):.0%} | pad {st('err_pad')} closing {st('err_close')} height {st('err_h')} | xy {st('err_xy')}")
    (a.out / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
