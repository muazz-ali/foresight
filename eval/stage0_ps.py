"""Stage 0 (offline): does the grasp-capture probability P_s separate grasps from misses?

Reads existing eval logs only (no sim, no training). For each gripper close:

  1. Predict the hand-object offset at the moment the fingers finish closing,
     t* = t_decide + dt + t_g, in the gripper frame (pad axis x, closing axis y,
     height z = TCP above object centre).
  2. P_s = p_max * prod_axes Pr[e(t*) inside the capture interval], Gaussian belief.
     Interval sizes come from the collider geometry (USD capsule), finger opening,
     pad width and friction (foresight_docs/RESEARCH.md §3.2).
  3. Label = object reached z >= 12 cm (the eval's lift rule) before the next re-open.

Reports AUC (bootstrap 95% CI) of P_s against simple rules (distance now, predicted
distance, time-to-contact, relative speed) and a cross-validated logistic model.
Variants: a-priori P_s (constants from the sim config, no fitting) and a calibrated
P_s (7 physical parameters, 5-fold CV on the primary object, then transferred to a
second object with no refit). Also: close timing vs the P_s peak, P_s at expert
closes, belief-based P_s on filter runs, filter error vs occlusion.

Usage:
  python eval/stage0_ps.py --primary tangerine06 --transfer egg07 egg04 -o data/eval/stage0
Conda: dynamicVLA_isaac (needs pxr for the collider sizes).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

LIFT_Z_M = 0.12  # eval_policy: held and z_max >= 12 cm counts as lifted
T_G_S = 0.08  # robot.gripper_close_time_ms
HALF_OPEN_M = 0.040  # robot.gripper_open_mm (per finger)
MU_SIM = 2.0  # scene.object_friction, "max" combine
OBJECTS = "/home/gpuadmin/Desktop/muazzam/objects"
TIMING_WINDOW = 12  # frames before the close scanned for the P_s peak

# ---------------------------------------------------------------- geometry


def collider_dims(mesh: str) -> tuple[float, float]:
    """(horizontal radius, half-length of the straight part) of the USD capsule, metres."""
    from pxr import Usd

    cat = "".join(c for c in mesh if not c.isdigit())
    st = Usd.Stage.Open(f"{OBJECTS}/{cat}/{mesh}.usd")
    scale = float(st.GetPrimAtPath("/Object").GetAttribute("xformOp:scale").Get()[2])
    col = st.GetPrimAtPath("/Object/collider")
    if col.GetTypeName() != "Capsule":
        raise RuntimeError(f"{mesh}: collider is {col.GetTypeName()}, expected Capsule")
    r = float(col.GetAttribute("radius").Get()) * scale
    h = float(col.GetAttribute("height").Get()) * scale
    return r, h / 2.0


# ---------------------------------------------------------------- events


def _pred_offset(ee, q, op, ov, k, tau, zrest, obj_p=None, obj_v=None):
    """Hand-object offset at decision frame k: now and predicted tau ahead, gripper frame."""
    dt = 0.04
    vh = (ee[k] - ee[k - 1]) / dt
    po = op[k].copy() if obj_p is None else obj_p.copy()
    vo = ov[k].copy() if obj_v is None else obj_v.copy()
    po[2] = zrest
    vo[2] = 0.0
    rot = Rotation.from_quat(q[k][[1, 2, 3, 0]]).inv()  # wxyz -> xyzw
    e_now = rot.apply(po - ee[k])
    e_star = rot.apply((po + vo * tau) - (ee[k] + vh * tau))
    # gripper local z points down, so local z = TCP height above the object centre
    return e_now, e_star, vo[:2] - vh[:2]


def load_events(mesh: str, roots: list[str], all_closes: bool = False, expert: bool = False,
                horizon: float | None = None, z_pop: float | None = None) -> list[dict]:
    """One dict per close event (first close per episode unless all_closes).

    Before the first close the object is pose-scripted at the visual half-height, which
    for some meshes sits below where the collider rests. ``z_pop`` (from rest_offset)
    lifts it to the rest height. It is one constant per mesh, measured on closes far
    from the hand, so the label cannot leak into the height.
    """
    rows = []
    if expert:
        for j in glob.glob("data/p3_throw/gpu*/speed_*/*.json"):
            d = json.load(open(j))
            if d["usd"].endswith(f"/{mesh}.usd") and d.get("success"):
                rows.append({"h5": j[:-5] + ".h5", "speed_m_s": d["speed_m_s"], "_dir": "expert",
                             "failure": "success", "cond_source": "expert", "seed": d["seed"]})
    else:
        for root in roots:
            for f in sorted(glob.glob(f"{root}/**/metrics.jsonl", recursive=True)):
                for line in open(f):
                    if line.strip():
                        r = json.loads(line)
                        if r.get("usd", "").endswith(f"/{mesh}.usd"):
                            r["_dir"] = os.path.relpath(os.path.dirname(f), "data/eval")
                            rows.append(r)
    out = []
    for r in rows:
        with h5py.File(r["h5"], "r") as f:
            g, ee, q = f["gripper"][:], f["ee_pos"][:], f["ee_quat"][:]
            op, ov, t = f["object_pos"][:], f["object_vel"][:], f["timestamp"][:]
            fp = f["filter_pack"][:] if "filter_pack" in f else None
            tp = f["truth_pack"][:] if "truth_pack" in f else None
            delta = float(f["delta"][0]) if "delta" in f else 0.25
        closed = g < 0.5
        starts = [i for i in range(1, len(g)) if closed[i] and not closed[i - 1]]
        if closed[0]:
            starts = [0] + starts
        for n_close, i in enumerate(starts):
            if i < 3:
                continue
            j = i + int(np.argmax(~closed[i:])) if (~closed[i:]).any() else len(g)
            k = i - 1  # last frame seen before the close is applied
            zrest = op[k, 2] + (z_pop or 0.0) if n_close == 0 else op[k, 2]
            tau = (t[i] - t[k]) + T_G_S if horizon is None else horizon
            e_now, e_star, vrel = _pred_offset(ee, q, op, ov, k, tau, zrest)
            # time to contact: distance now over closing speed (frames k-1 -> k)
            e_prev, _, _ = _pred_offset(ee, q, op, ov, k - 1, tau, zrest)
            d1, d0 = np.linalg.norm(e_now), np.linalg.norm(e_prev)
            ttc = d1 / ((d0 - d1) / 0.04) if d0 > d1 else np.inf
            ev = dict(
                dir=r["_dir"], seed=r.get("seed"), speed=round(float(r["speed_m_s"]), 2),
                cond=r.get("cond_source", "?"), failure=r["failure"], n_close=n_close,
                lifted=bool(op[i:j, 2].max() >= LIFT_Z_M) if j > i else False,
                # the policy actually raised the hand >= 5 cm before re-opening
                attempted=bool(ee[i:j, 2].max() - ee[k, 2] >= 0.05) if j > i else False,
                e_now=e_now, e_star=e_star, vrel=vrel, ttc=ttc, tau=tau, zrest=zrest, zpre=float(op[k, 2]),
            )
            # P_s inputs over the frames before the close (timing analysis)
            ks = list(range(max(2, k - TIMING_WINDOW + 1), k + 1))
            ev["timing_e"] = np.array([_pred_offset(ee, q, op, ov, kk, tau, zrest)[1] for kk in ks])
            # Belief-based offset (filter runs): object from the tracker, not the truth
            if fp is not None and r.get("cond_source") == "filter" and np.isfinite(fp[k]).all():
                p_hat = np.r_[fp[k, :2] - fp[k, 2:] * delta, zrest]
                v_hat = np.r_[fp[k, 2:], 0.0]
                ev["e_star_belief"] = _pred_offset(ee, q, op, ov, k, tau, zrest, p_hat, v_hat)[1]
            out.append(ev)
            if not all_closes:
                break
    return out


def rest_offset(mesh: str, roots: list[str]) -> tuple[float, int]:
    """Median z jump at release on first closes with the hand > 8 cm away (no finger contact)."""
    d = []
    for root in roots:
        for f in glob.glob(f"{root}/**/metrics.jsonl", recursive=True):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if not r.get("usd", "").endswith(f"/{mesh}.usd"):
                        continue
                    with h5py.File(r["h5"], "r") as h:
                        g, ee, op = h["gripper"][:], h["ee_pos"][:], h["object_pos"][:]
                    if not (g < 0.5).any():
                        continue
                    i = int(np.argmax(g < 0.5))
                    if 1 <= i < len(g) - 2 and np.linalg.norm(ee[i - 1] - op[i - 1]) > 0.08:
                        d.append(np.median(op[i:i + 3, 2]) - op[i - 1, 2])
    return (float(np.median(d)) if d else 0.0), len(d)


# ---------------------------------------------------------------- capture model

PARAM_NAMES = ["c_t", "c_n", "m_n", "a_t", "mu", "sigma", "p_max"]
APRIORI_SIM = dict(c_t=0.0, c_n=0.0, m_n=0.0, a_t=0.010, mu=MU_SIM, sigma=0.005, p_max=1.0)
APRIORI_DOC = dict(APRIORI_SIM, mu=0.5)  # formulation's real-gripper friction assumption
BOUNDS = [(-0.03, 0.03), (-0.02, 0.02), (-0.02, 0.02), (0.002, 0.05), (0.05, 5.0), (0.001, 0.03), (0.3, 1.0)]


def _interval(mu, lo, hi, s):
    return np.clip(ndtr((hi - mu) / s) - ndtr((lo - mu) / s), 0.0, 1.0)


def p_success(e_star: np.ndarray, geom: tuple[float, float, float], p: dict) -> np.ndarray:
    """P_s for offsets e_star (N, 3) = (pad axis, closing axis, height) in metres."""
    r, hc, zrest = geom
    e = np.atleast_2d(e_star)
    a_n = max(HALF_OPEN_M - r - p["m_n"], 1e-4)  # 1. fits between the fingers
    a_t = p["a_t"]  # 2. inside the pad
    h_hi = hc + r * np.sin(np.arctan(p["mu"]))  # 3. contact inside the friction cone
    h_lo = 0.005 - zrest  # fingertips stay above the table
    s = p["sigma"]
    pt = _interval(e[:, 0] - p["c_t"], -a_t, a_t, s)
    pn = _interval(e[:, 1] - p["c_n"], -a_n, a_n, s)
    ph = _interval(e[:, 2], h_lo, h_hi, s)
    return p["p_max"] * pt * pn * ph


def fit_params(E: np.ndarray, y: np.ndarray, geoms: np.ndarray) -> dict:
    """Max-likelihood fit of the 7 physical parameters (per-event geometry)."""

    def nll(x):
        p = dict(zip(PARAM_NAMES, x))
        ps = np.concatenate([p_success(E[m], tuple(g), p) for g, m in _groups(geoms)])
        yy = np.concatenate([y[m] for _, m in _groups(geoms)])
        ps = np.clip(ps, 1e-4, 1 - 1e-4)
        return -np.mean(yy * np.log(ps) + (1 - yy) * np.log(1 - ps))

    x0 = [APRIORI_SIM[k] for k in PARAM_NAMES]
    best = None
    for mu0 in (0.5, 2.0):
        for sig0 in (0.003, 0.008):
            x0[4], x0[5] = mu0, sig0
            res = minimize(nll, x0, method="L-BFGS-B", bounds=BOUNDS)
            if best is None or res.fun < best.fun:
                best = res
    return dict(zip(PARAM_NAMES, best.x))


def _groups(geoms):
    keys = [tuple(g) for g in geoms]
    for g in sorted(set(keys)):
        yield g, np.array([k == g for k in keys])


def ps_events(evs, geoms, p, key="e_star"):
    E = np.array([e[key] for e in evs])
    out = np.empty(len(evs))
    for g, m in _groups(geoms):
        out[m] = p_success(E[m], g, p)
    return out


# ---------------------------------------------------------------- stats


def auc_ci(y, s, n_boot=1000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2:
        return None
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return round(float(a), 3), round(float(lo), 3), round(float(hi), 3)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    rel = []
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
            rel.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return round(float(tot), 3), rel


def features(evs):
    E = np.array([e["e_star"] for e in evs])
    N = np.array([e["e_now"] for e in evs])
    V = np.array([e["vrel"] for e in evs])
    return np.c_[np.abs(E[:, 0]), np.abs(E[:, 1]), E[:, 2], E[:, 2] ** 2, np.linalg.norm(V, axis=1), np.linalg.norm(N, axis=1)]


def baselines(evs):
    N = np.array([e["e_now"] for e in evs])
    E = np.array([e["e_star"] for e in evs])
    V = np.array([e["vrel"] for e in evs])
    ttc = np.array([e["ttc"] for e in evs])
    return {
        "distance now (-|e_now|)": -np.linalg.norm(N, axis=1),
        "xy distance now": -np.linalg.norm(N[:, :2], axis=1),
        "predicted distance (-|e*|)": -np.linalg.norm(E, axis=1),
        "time-to-contact rule (-|TTC-0.15 s|)": -np.abs(np.nan_to_num(ttc, posinf=99.0) - 0.15),
        "relative speed (-|v_rel|)": -np.linalg.norm(V, axis=1),
    }


# ---------------------------------------------------------------- occlusion proxy


def occlusion_rows(roots, mesh, r_obj, z_pop):
    """Filter error vs a geometric visible-fraction proxy from the static camera."""
    import yaml

    from perception.camera import PinholeCamera

    cam = PinholeCamera.from_cfg({"static_camera": _cam_cfg(yaml.safe_load(open("sim/phase0_cfg.yaml")))})
    rng = np.random.default_rng(0)
    disc = rng.uniform(-1, 1, (400, 2))
    disc = disc[np.linalg.norm(disc, axis=1) <= 1][:200]
    rows = []
    for root in roots:
        for f in glob.glob(f"{root}/**/metrics.jsonl", recursive=True):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("usd", "").endswith(f"/{mesh}.usd") or r.get("cond_source") != "filter":
                    continue
                with h5py.File(r["h5"], "r") as h:
                    g, ee, q, op = h["gripper"][:], h["ee_pos"][:], h["ee_quat"][:], h["object_pos"][:]
                    fp, tp = h["filter_pack"][:], h["truth_pack"][:]
                stop = int(np.argmax(g < 0.5)) if (g < 0.5).any() else len(g)
                for k in range(3, stop):
                    if not (np.isfinite(fp[k]).all() and np.isfinite(tp[k]).all()):
                        continue
                    pc = op[k] + np.array([0, 0, z_pop])  # rest centre (scripted z sits low by z_pop)
                    rho = _visible(cam, pc, r_obj, ee[k], q[k], disc)
                    rows.append((rho, np.linalg.norm(ee[k, :2] - op[k, :2]),
                                 float(np.linalg.norm(fp[k, :2] - tp[k, :2])), float(r["speed_m_s"])))
    return np.array(rows)


def _cam_cfg(cfg):
    c = dict(cfg["static_camera"])
    c["position"] = [x / 1000.0 for x in c["position_mm"]]
    return c


def _visible(cam, pc, r_obj, tcp, q, disc):
    """Fraction of the object's image disc not covered by hand/arm spheres in front of it."""
    up = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0, 0, -1.0])  # TCP -> flange
    occ = [(tcp + up * s, 0.045) for s in (0.0, 0.04, 0.08, 0.12)]
    flange = tcp + up * 0.107
    base = np.array([0.0, 0.0, 0.33])
    occ += [(flange + (base - flange) * s, 0.06) for s in (0.15, 0.3, 0.45)]
    Rcw = cam.R_wc.T

    def to_img(p):
        x, y, z = Rcw @ (p - cam.position)
        return np.array([x / -z, y / -z]), -z

    c0, d0 = to_img(pc)
    r0 = r_obj / d0
    pts = c0 + disc * r0
    hidden = np.zeros(len(pts), bool)
    for p, rad in occ:
        c1, d1 = to_img(p)
        if d1 < d0:
            hidden |= np.linalg.norm(pts - c1, axis=1) <= rad / d1
    return 1.0 - hidden.mean()


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", default="tangerine06")
    ap.add_argument("--transfer", nargs="*", default=["egg07", "egg04"])
    ap.add_argument("--roots", nargs="*", default=["data/eval"])
    ap.add_argument("-o", "--out", type=Path, default=Path("data/eval/stage0"))
    ap.add_argument("--horizon", type=float, default=None,
                    help="look-ahead (s) for the offset; default = one frame + t_g (formulation)")
    ap.add_argument("--skip-occlusion", action="store_true")
    ap.add_argument("--population", choices=["attempted", "all"], default="attempted",
                    help="attempted = only closes after which the hand rose >= 5 cm (a lift was tried)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"primary": a.primary, "transfer": a.transfer, "t_g_s": T_G_S, "lift_z_m": LIFT_Z_M,
                 "horizon_s": a.horizon if a.horizon is not None else "dt + t_g"}

    geom = {m: collider_dims(m) for m in [a.primary] + a.transfer}
    res["collider_m"] = {m: {"radius": g[0], "half_straight": g[1]} for m, g in geom.items()}

    def geoms_of(evs, mesh):
        return np.array([(geom[mesh][0], geom[mesh][1], e["zrest"]) for e in evs])

    zp = {}
    for m in [a.primary] + a.transfer:
        zp[m] = rest_offset(m, a.roots)
    res["release_z_jump_cm"] = {m: {"cm": round(v[0] * 100, 2), "n_far_closes": v[1]} for m, v in zp.items()}
    keep = (lambda evs: [e for e in evs if e["attempted"]]) if a.population == "attempted" else (lambda evs: evs)
    P_all = load_events(a.primary, a.roots, horizon=a.horizon, z_pop=zp[a.primary][0])
    res["population"] = a.population
    res["n_events_before_filter"] = {"first_closes": len(P_all), "attempted_lift": sum(e["attempted"] for e in P_all)}
    P = keep(P_all)
    yP = np.array([e["lifted"] for e in P], int)
    gP = geoms_of(P, a.primary)
    res["n_events"] = {"primary_first_close": len(P), "lifted": int(yP.sum())}
    res["label_vs_eval_failure"] = {f"{k[0]}|{k[1]}": v for k, v in Counter((e["failure"], e["lifted"]) for e in P).items()}

    # 5-fold CV (one event per episode, so folds are by episode)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(P))
    ps_cv = np.empty(len(P))
    lr_cv = np.empty(len(P))
    X = features(P)
    fold_params = []
    for k in range(5):
        tr, te = fold != k, fold == k
        pk = fit_params(np.array([e["e_star"] for e in P])[tr], yP[tr], gP[tr])
        fold_params.append(pk)
        ps_cv[te] = ps_events([P[i] for i in np.where(te)[0]], gP[te], pk)
        lr = LogisticRegression(max_iter=2000).fit(X[tr], yP[tr])
        lr_cv[te] = lr.predict_proba(X[te])[:, 1]
    p_all = fit_params(np.array([e["e_star"] for e in P]), yP, gP)
    res["calibrated_params_all"] = {k: round(float(v), 4) for k, v in p_all.items()}
    res["calibrated_params_folds_sd"] = {k: round(float(np.std([fp[k] for fp in fold_params])), 4) for k in PARAM_NAMES}

    scores = {
        "P_s a-priori (sim mu=2.0)": ps_events(P, gP, APRIORI_SIM),
        "P_s a-priori (doc mu=0.5)": ps_events(P, gP, APRIORI_DOC),
        "P_s calibrated (5-fold CV)": ps_cv,
        "logistic on same features (5-fold CV)": lr_cv,
        **baselines(P),
    }
    near = np.array([np.linalg.norm(e["e_now"][:2]) < 0.03 and abs(e["e_now"][2]) < 0.03 for e in P])
    subsets = {"all first closes": np.ones(len(P), bool), "near (hand within 3 cm)": near}
    for sp in sorted({e["speed"] for e in P}):
        subsets[f"speed {sp:.2f}"] = np.array([e["speed"] == sp for e in P])
    for c in ("oracle", "filter"):
        subsets[f"cond {c}"] = np.array([e["cond"] == c for e in P])
    res["auc"] = {}
    for sname, m in subsets.items():
        res["auc"][sname] = {"n": int(m.sum()), "lifted": int(yP[m].sum()),
                             **{k: auc_ci(yP[m], v[m]) for k, v in scores.items()}}
    res["ece_calibrated_cv"], res["reliability_cv"] = ece(yP, ps_cv)
    res["ece_apriori_sim"], _ = ece(yP, scores["P_s a-priori (sim mu=2.0)"])

    # Belief-based P_s on filter runs (tracker estimate instead of the truth)
    fb = [i for i, e in enumerate(P) if "e_star_belief" in e]
    if fb:
        Eb = np.array([P[i]["e_star_belief"] for i in fb])
        ps_b = np.concatenate([p_success(Eb[m], g, p_all) for g, m in _groups(gP[fb])])
        order = np.concatenate([np.array(fb)[m] for _, m in _groups(gP[fb])])
        yb = yP[order]
        ps_t = ps_events([P[i] for i in order], gP[order], p_all)
        res["belief_vs_truth_filter_runs"] = {"n": len(fb), "truth P_s": auc_ci(yb, ps_t), "belief P_s": auc_ci(yb, ps_b),
                                              "median |e*_belief - e*_truth| cm": round(float(np.median(np.linalg.norm(Eb - np.array([P[i]['e_star'] for i in fb]), axis=1)) * 100), 2)}

    # Timing: where was the P_s peak relative to the close?
    tim = Counter()
    for i, e in enumerate(P):
        ps_t = p_success(e["timing_e"], tuple(gP[i]), fold_params[fold[i]])
        if len(ps_t) < 4:
            continue
        lag = len(ps_t) - 1 - int(np.argmax(ps_t))  # frames between peak and close
        rising = ps_t[-1] > ps_t[-2] + 1e-3 and lag == 0
        cls = "closed while P_s still rising" if rising else ("closed at peak (<=1 frame)" if lag <= 1 else "closed late (peak >=2 frames earlier)")
        good = ps_t.max() >= 0.5
        tim[(cls, "lifted" if e["lifted"] else "missed", "peak>=0.5" if good else "peak<0.5")] += 1
    res["timing"] = {" | ".join(k): v for k, v in sorted(tim.items())}

    # Expert closes (all successes): where does P_s sit?
    X_ = load_events(a.primary, [], expert=True, horizon=a.horizon, z_pop=zp[a.primary][0])
    if X_:
        ps_x = ps_events(X_, geoms_of(X_, a.primary), p_all)
        res["expert"] = {"n": len(X_), "P_s median": round(float(np.median(ps_x)), 3),
                         "frac P_s>=0.5": round(float((ps_x >= 0.5).mean()), 3),
                         "TTC median s": round(float(np.median([e["ttc"] for e in X_ if np.isfinite(e["ttc"])])), 3),
                         "e* median cm (pad, closing, height)": np.round(np.median([e["e_star"] for e in X_], axis=0) * 100, 2).tolist()}

    # Transfer: same parameters, new object geometry, no refit
    res["transfer"] = {}
    for mesh in a.transfer:
        T = keep(load_events(mesh, a.roots, horizon=a.horizon, z_pop=zp[mesh][0]))
        if not T:
            continue
        yT = np.array([e["lifted"] for e in T], int)
        gT = geoms_of(T, mesh)
        sc = {"P_s calibrated on primary (no refit)": ps_events(T, gT, p_all),
              "P_s a-priori (sim mu=2.0)": ps_events(T, gT, APRIORI_SIM),
              "logistic trained on primary": LogisticRegression(max_iter=2000).fit(X, yP).predict_proba(features(T))[:, 1],
              **baselines(T)}
        nearT = np.array([np.linalg.norm(e["e_now"][:2]) < 0.03 and abs(e["e_now"][2]) < 0.03 for e in T])
        res["transfer"][mesh] = {
            "n": len(T), "lifted": int(yT.sum()),
            "auc_all": {k: auc_ci(yT, v) for k, v in sc.items()},
            "auc_near": {"n": int(nearT.sum()), **{k: auc_ci(yT[nearT], v[nearT]) for k, v in sc.items()}},
            "ece_calibrated": ece(yT, sc["P_s calibrated on primary (no refit)"])[0],
        }

    # All closes (re-grasps included), primary object
    A = keep(load_events(a.primary, a.roots, all_closes=True, horizon=a.horizon, z_pop=zp[a.primary][0]))
    yA = np.array([e["lifted"] for e in A], int)
    res["all_closes"] = {"n": len(A), "lifted": int(yA.sum()),
                         "P_s calibrated": auc_ci(yA, ps_events(A, geoms_of(A, a.primary), p_all)),
                         "distance now": auc_ci(yA, -np.linalg.norm([e["e_now"] for e in A], axis=1))}

    # Occlusion: filter error vs visible fraction
    z_pop = zp[a.primary][0]
    occ = np.empty((0, 4)) if a.skip_occlusion else occlusion_rows(a.roots, a.primary, geom[a.primary][0], z_pop)
    if len(occ):
        bins = [0, 0.25, 0.5, 0.75, 0.95, 1.01]
        tab = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (occ[:, 0] >= lo) & (occ[:, 0] < hi)
            if m.any():
                tab.append({"rho": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                            "err_median_cm": round(float(np.median(occ[m, 2]) * 100), 2),
                            "err_p90_cm": round(float(np.percentile(occ[m, 2], 90) * 100), 2)})
        # fit err^2 ~ s0^2 / rho^2 + b^2 (1 - rho)^2 on per-bin mean squares (rho >= 0.05)
        m = occ[:, 0] >= 0.05
        A_fit = np.c_[1 / occ[m, 0] ** 2, (1 - occ[m, 0]) ** 2]
        coef, *_ = np.linalg.lstsq(A_fit, occ[m, 2] ** 2, rcond=None)
        pred = A_fit @ coef
        r2 = 1 - np.sum((occ[m, 2] ** 2 - pred) ** 2) / np.sum((occ[m, 2] ** 2 - np.mean(occ[m, 2] ** 2)) ** 2)
        res["occlusion"] = {"n_frames": len(occ), "by_visible_fraction": tab,
                            "fit_sigma0_cm": round(float(np.sqrt(max(coef[0], 0)) * 100), 2),
                            "fit_b_cm": round(float(np.sqrt(max(coef[1], 0)) * 100), 2), "fit_R2_on_err2": round(float(r2), 3),
                            "spearman_rho_vs_err": round(float(_spearman(occ[:, 0], occ[:, 2])), 3)}

    _figures(a.out, P, yP, scores, ps_cv, gP, p_all, occ if len(occ) else None)
    (a.out / "stage0.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


def _spearman(x, y):
    from scipy.stats import spearmanr

    return spearmanr(x, y).statistic


def _figures(out, P, y, scores, ps_cv, gP, p_all, occ):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = np.array([e["e_star"] for e in P]) * 100
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for k in ["P_s calibrated (5-fold CV)", "P_s a-priori (sim mu=2.0)", "logistic on same features (5-fold CV)",
              "distance now (-|e_now|)", "time-to-contact rule (-|TTC-0.15 s|)", "relative speed (-|v_rel|)"]:
        f, t, _ = roc_curve(y, scores[k])
        ax[0].plot(f, t, label=f"{k} ({roc_auc_score(y, scores[k]):.2f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC, first closes")
    ax[0].legend(fontsize=7)
    r, hc, _ = gP[0]
    a_n = HALF_OPEN_M - r - p_all["m_n"]
    h_hi = hc + r * np.sin(np.arctan(p_all["mu"]))
    for j, (ix, iy, lx, ly) in enumerate([(0, 2, "pad axis e_t (cm)", "height h (cm)"), (1, 2, "closing axis e_n (cm)", "height h (cm)")]):
        A = ax[j + 1]
        A.scatter(E[y == 0, ix], E[y == 0, iy], s=4, c="tab:red", alpha=0.4, label="not lifted")
        A.scatter(E[y == 1, ix], E[y == 1, iy], s=4, c="tab:green", alpha=0.4, label="lifted")
        half = (p_all["a_t"] if ix == 0 else a_n) * 100
        c = (p_all["c_t"] if ix == 0 else p_all["c_n"]) * 100
        A.add_patch(plt.Rectangle((c - half, -4), 2 * half, 4 + h_hi * 100, fill=False, ec="k", lw=1.5))
        A.set(xlim=(-6, 6), ylim=(-5, 8), xlabel=lx, ylabel=ly, title="predicted offset at t* and fitted capture box")
        A.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "stage0_roc_capture.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(ps_cv, bins) - 1, 0, 9)
    xs = [ps_cv[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[0].plot([0, 1], [0, 1], "k:")
    ax[0].plot(xs, ys, "o-")
    ax[0].set(xlabel="predicted P_s (CV)", ylabel="observed lift rate", title="calibration")
    if occ is not None:
        ax[1].scatter(occ[:, 0], occ[:, 2] * 100, s=2, alpha=0.2)
        ax[1].set(xlabel="visible fraction rho (geometric proxy)", ylabel="filter error |p_hat - p| (cm)",
                  title="tracker error vs occlusion", ylim=(0, 15))
    fig.tight_layout()
    fig.savefig(out / "stage0_calibration_occlusion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
