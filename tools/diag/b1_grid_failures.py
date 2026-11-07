"""Per-episode failure analysis for a B1 grid folder (reads cells/*/episodes.jsonl + .h5).

Usage: PYTHONPATH=. python tools/diag/b1_grid_failures.py data/eval/C1_grid_v4 out.json
"""
import json, glob, os, sys
from multiprocessing import Pool
import numpy as np, h5py

D = sys.argv[1]
OUT = sys.argv[2]
LO4 = -3.0718
XY_OK, T_CONTACT, DV_LIM = 0.002, 0.130, 0.0025 / 0.180
R_ZONE = 0.20

def path_to_zone(p, v, r=R_ZONE):
    s = np.linalg.norm(v)
    if s < 1e-6: return np.inf
    u = v / s; b = p @ u; c = p @ p - r * r
    if c <= 0: return 0.0
    disc = b * b - c
    if b >= 0 or disc < 0: return np.inf
    return -b - np.sqrt(disc)

def one(row):
    p = row["h5"]
    out = dict(cell=row["cell"], seed=row["seed"], object=row["object"], v=row["v_cms"],
               d=row["delta_nom_ms"], ok=bool(row["success"]), mode=row["failure_mode"] or "ok",
               steps=row["sim_steps"])
    try:
        f = h5py.File(p, "r")
    except Exception as e:
        out["err"] = str(e); return out
    st = f["stage"][:]; ee = f["ee_pos"][:]; ob = f["object_pos"][:]; vo = f["object_vel"][:]
    act = f["action_pos"][:]; q = f["proprio"][:, :7]; grip = f["gripper"][:]
    hold = f["holding"][:] if "holding" in f else None
    n = len(st); dt = 1 / 30
    v0 = vo[min(5, n - 1), :2]; spd = np.linalg.norm(v0)
    out["heading_deg"] = float(np.degrees(np.arctan2(v0[1], v0[0]))) if spd > 1e-3 else None
    out["start_xy"] = ob[0, :2].round(3).tolist()
    out["timeout"] = bool(n >= 300)
    gap = np.abs(ee[:, :2] - ob[:, :2]).max(1)
    out["min_gap_mm"] = float(1000 * gap[st <= 1].min()) if (st <= 1).any() else None
    ap = np.where(st == 1)[0]
    near = [t for t in ap if gap[t] <= 0.010]
    out["t_catch_s"] = float(near[0] * dt) if near else None
    out["r_obj_catch"] = float(np.linalg.norm(ob[near[0], :2])) if near else None
    out["ptz_catch_cm"] = float(100 * path_to_zone(ob[near[0], :2], vo[near[0], :2])) if near else None
    # ride: stage-1 rows with hand within 10 mm; hand vel from consecutive rows
    r = np.array([t for t in near if t >= 1])
    if len(r) >= 3 and spd > 1e-3:
        u = vo[r, :2] / np.maximum(np.linalg.norm(vo[r, :2], axis=1, keepdims=True), 1e-9)
        nn = np.stack([-u[:, 1], u[:, 0]], 1)
        e = ee[r, :2] - ob[r, :2]
        vh = (ee[r, :2] - ee[r - 1, :2]) / dt
        dv = vh - vo[r, :2]
        ep = e + dv * T_CONTACT
        okxy = (np.abs(e).max(1) <= XY_OK) & (np.abs(ep).max(1) <= XY_OK)
        okv = np.linalg.norm(dv, axis=1) <= DV_LIM
        aim = act[r, :2] - ob[r, :2]
        out.update(
            ride_n=int(len(r)),
            e_along_mm=float(1000 * np.median((e * u).sum(1))),
            e_cross_mm=float(1000 * np.median((e * nn).sum(1))),
            gap_med_mm=float(1000 * np.median(np.abs(e).max(1))),
            dv_along_cms=float(100 * np.median((dv * u).sum(1))),
            dv_cross_cms=float(100 * np.median((dv * nn).sum(1))),
            dv_med_cms=float(100 * np.median(np.linalg.norm(dv, axis=1))),
            tick_dv_cms=float(100 * np.median(np.linalg.norm(np.diff(vh, axis=0), axis=1))) if len(r) > 3 else None,
            hand_speed_cms=float(100 * np.median(np.linalg.norm(vh, axis=1))),
            frac_xy=float(okxy.mean()), frac_v=float(okv.mean()), frac_both=float((okxy & okv).mean()),
            aim_along_mm=float(1000 * np.median((aim * u).sum(1))),
            aim_cross_mm=float(1000 * np.median((aim * nn).sum(1))),
            hand_z_err_mm=float(1000 * np.median(ee[r, 2] - ob[r, 2])),
        )
    # stop / end of chase
    home = np.where(st == 7)[0]
    closes = np.where(st == 2)[0]
    if len(closes):
        c = closes[0]
        out["close_t_s"] = float(c * dt)
        out["close_r_obj"] = float(np.linalg.norm(ob[c, :2]))
        out["close_dz_mm"] = float(1000 * (ee[c, 2] - ob[c, 2]))
        out["close_xy_mm"] = float(1000 * np.linalg.norm(ee[c, :2] - ob[c, :2]))
        # after close: object–hand distance (slip) and object z
        post = np.arange(c, n)
        d_oh = np.linalg.norm(ob[post] - ee[post], axis=1)
        out["max_obj_hand_after_close_mm"] = float(1000 * d_oh.max())
        lift = np.where(st == 3)[0]
        out["obj_z_gain_mm"] = float(1000 * (ob[post, 2].max() - ob[c, 2]))
        out["stages_after_close"] = sorted(set(st[c:].tolist()))
    elif len(home):
        h = home[0]
        out["stop_t_s"] = float(h * dt)
        out["stop_r_obj"] = float(np.linalg.norm(ob[h, :2]))
        out["stop_gap_mm"] = float(1000 * np.linalg.norm(ee[h, :2] - ob[h, :2]))
        out["stop_ptz_cm"] = float(100 * path_to_zone(ob[h, :2], vo[h, :2]))
        out["stop_obj_xy"] = ob[h, :2].round(3).tolist()
    m4 = np.degrees(q[:, 3] - LO4)
    sel = (st >= 1) & (st <= 4)
    out["j4_min_deg"] = float(m4[sel].min()) if sel.any() else None
    out["hand_r_min"] = float(np.linalg.norm(ee[sel, :2], axis=1).min()) if sel.any() else None
    # delayed obs in this file
    for k in ("delta_meas_s", "applied_delta_s"):
        if k in f:
            out[k + "_med"] = float(np.median(f[k][:]))
    f.close()
    return out

rows = []
for p in sorted(glob.glob(f"{D}/cells/*/episodes.jsonl")):
    rows += [json.loads(l) for l in open(p)]
rows = [r for r in rows if r.get("h5") and os.path.exists(r["h5"])]
with Pool(24) as pool:
    res = pool.map(one, rows)
json.dump(res, open(OUT, "w"))
print(len(res), "episodes analysed →", OUT)
