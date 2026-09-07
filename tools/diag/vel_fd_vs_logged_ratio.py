from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="|Δp/Δt| vs logged |object_vel| ratio diagnostic")
parser.add_argument(
    "--root",
    type=Path,
    default=Path("data/p1/reeval_n8_n91_p5/eval_b_oracle"),
    help="Directory of *_cond.h5 episodes",
)
parser.add_argument(
    "--out",
    type=Path,
    default=Path("tools/diag/out"),
    help="Output directory for json/png",
)
parser.add_argument(
    "--expect-n",
    type=int,
    default=0,
    help="If >0, assert exactly this many moving episodes (legacy: 364)",
)
parser.add_argument(
    "--accept-ratio-tol",
    type=float,
    default=0.1,
    help="Fail if any bin median frame ratio is outside 1±tol (0 disables)",
)
args = parser.parse_args()

ROOT = args.root
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
SIM_DT = 0.04  # sim/phase0_cfg.yaml
EPS = 1e-6

def _load_episode(fp: Path) -> dict:
    with h5py.File(fp, "r") as f:
        return {k: np.asarray(f[k]) for k in f.keys()}


files = []
for fp in sorted(ROOT.glob("*_cond.h5")):
    z = _load_episode(fp)
    if float(np.asarray(z["speed_m_s"]).reshape(-1)[0]) >= 0.1:
        files.append(fp)
if int(args.expect_n) > 0:
    assert len(files) == int(args.expect_n), len(files)
assert len(files) > 0, f"no *_cond.h5 with speed>=0.1 under {ROOT}"

# Per-frame samples (free motion only). holding==0 means not snapped to the hand.
speed_cmd = []
v_fd_norm = []
v_log_norm = []
ratio = []
ratio_xy = []  # |Δp_xy|/Δt / |v_xy|
v_fd_vec = []
v_log_vec = []
# Per-episode medians
ep_med_ratio = []
ep_med_ratio_xy = []
ep_speed = []
ep_n = []
n_skip_low_v = 0
n_pairs = 0

for fp in files:
    z = _load_episode(fp)
    pos = z["object_pos"].astype(np.float64)
    vel = z["object_vel"].astype(np.float64)
    if "holding" in z:
        free = np.asarray(z["holding"]).reshape(-1) < 0.5
    else:
        free = np.ones(pos.shape[0], dtype=bool)
    spd = float(np.asarray(z["speed_m_s"]).reshape(-1)[0])
    T = pos.shape[0]
    r_ep = []
    rxy_ep = []
    for t in range(T - 1):
        if not (free[t] and free[t + 1]):
            continue
        dp = pos[t + 1] - pos[t]
        v_fd = dp / SIM_DT
        v_log = vel[t]
        n_fd = float(np.linalg.norm(v_fd))
        n_log = float(np.linalg.norm(v_log))
        if n_log < EPS:
            n_skip_low_v += 1
            continue
        r = n_fd / n_log
        n_fd_xy = float(np.linalg.norm(v_fd[:2]))
        n_log_xy = float(np.linalg.norm(v_log[:2]))
        rxy = n_fd_xy / n_log_xy if n_log_xy >= EPS else np.nan

        speed_cmd.append(spd)
        v_fd_norm.append(n_fd)
        v_log_norm.append(n_log)
        ratio.append(r)
        ratio_xy.append(rxy)
        v_fd_vec.append(v_fd)
        v_log_vec.append(v_log)
        r_ep.append(r)
        if np.isfinite(rxy):
            rxy_ep.append(rxy)
        n_pairs += 1

    if r_ep:
        ep_med_ratio.append(float(np.median(r_ep)))
        ep_med_ratio_xy.append(float(np.median(rxy_ep)) if rxy_ep else float("nan"))
        ep_speed.append(spd)
        ep_n.append(len(r_ep))

ratio = np.asarray(ratio)
ratio_xy = np.asarray(ratio_xy)
v_fd_norm = np.asarray(v_fd_norm)
v_log_norm = np.asarray(v_log_norm)
speed_cmd = np.asarray(speed_cmd)
ep_med_ratio = np.asarray(ep_med_ratio)
ep_med_ratio_xy = np.asarray(ep_med_ratio_xy)
ep_speed = np.asarray(ep_speed)
v_fd_arr = np.asarray(v_fd_vec)
v_log_arr = np.asarray(v_log_vec)

# Component-wise: mean |v_fd_i / v_log_i| where |v_log| large
comp_ratio = []
for i, ax in enumerate("xyz"):
    mask = np.abs(v_log_arr[:, i]) >= 0.05  # m/s
    if mask.any():
        cr = v_fd_arr[mask, i] / v_log_arr[mask, i]
        comp_ratio.append(
            {
                "axis": ax,
                "n": int(mask.sum()),
                "median": float(np.median(cr)),
                "mean": float(np.mean(cr)),
                "p05": float(np.percentile(cr, 5)),
                "p95": float(np.percentile(cr, 95)),
            }
        )

summary = {
    "pool": {
        "n_episodes": len(files),
        "root": str(ROOT),
        "filter": f"speed_m_s >= 0.1 from {ROOT} *_cond.h5",
        "frame_filter": "free→free (holding == 0 at t and t+1)",
        "dt_s": SIM_DT,
        "n_pairs": int(n_pairs),
        "n_skip_low_logged_v": int(n_skip_low_v),
        "n_episodes_with_pairs": int(len(ep_med_ratio)),
    },
    "ratio_speed": {
        "definition": "|Δp/Δt| / |object_vel|  (= logged post-step FD / observed vel)",
        "median": float(np.median(ratio)),
        "mean": float(np.mean(ratio)),
        "p05": float(np.percentile(ratio, 5)),
        "p25": float(np.percentile(ratio, 25)),
        "p75": float(np.percentile(ratio, 75)),
        "p95": float(np.percentile(ratio, 95)),
        "frac_near_1_pm_10pct": float(np.mean(np.abs(ratio - 1.0) <= 0.1)),
        "frac_lt_0.5": float(np.mean(ratio < 0.5)),
        "frac_gt_2": float(np.mean(ratio > 2.0)),
    },
    "ratio_xy_speed": {
        "definition": "|Δp_xy/Δt| / |v_xy|",
        "median": float(np.nanmedian(ratio_xy)),
        "mean": float(np.nanmean(ratio_xy)),
        "p05": float(np.nanpercentile(ratio_xy, 5)),
        "p95": float(np.nanpercentile(ratio_xy, 95)),
    },
    "per_episode_median_ratio": {
        "median": float(np.median(ep_med_ratio)),
        "mean": float(np.mean(ep_med_ratio)),
        "p05": float(np.percentile(ep_med_ratio, 5)),
        "p95": float(np.percentile(ep_med_ratio, 95)),
    },
    "component_signed_ratio_vfd_over_vlog": comp_ratio,
    "scatter": {
        "pearson_|v_fd|_vs_|v_log|": float(
            np.corrcoef(v_fd_norm, v_log_norm)[0, 1]
        ),
        "mean_|v_fd|": float(v_fd_norm.mean()),
        "mean_|v_log|": float(v_log_norm.mean()),
    },
}

# by commanded speed
by_spd = {}
for s in sorted(set(np.round(ep_speed, 2))):
    m = np.isclose(ep_speed, s)
    by_spd[f"{s:.2f}"] = {
        "n_eps": int(m.sum()),
        "median_ep_ratio": float(np.median(ep_med_ratio[m])),
        "median_frame_ratio": float(np.median(ratio[np.isclose(speed_cmd, s)])),
    }
summary["by_commanded_speed"] = by_spd

(OUT / "vel_fd_vs_logged_ratio.json").write_text(json.dumps(summary, indent=2))

# ---- plot ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    r"$|\Delta p/\Delta t|$ vs logged $|v|$ (root_lin_vel_w / kinematic)"
    f"\n{len(files)} moving episodes (v≥0.1), free→free pairs, Δt={SIM_DT}s"
    f"  ·  n_pairs={n_pairs:,}",
    fontsize=11,
)

# 1) histogram of ratio
ax = axes[0, 0]
clip = np.clip(ratio, 0, 3)
ax.hist(clip, bins=60, color="#2c5f7c", edgecolor="white", linewidth=0.3)
ax.axvline(1.0, color="#c0392b", lw=1.5, label="ratio=1 (match)")
ax.axvline(np.median(ratio), color="#e67e22", ls="--", lw=1.5, label=f"median={np.median(ratio):.3f}")
ax.set_xlabel(r"ratio  $|\Delta p/\Delta t| \;/\; |v_{\mathrm{logged}}|$  (clipped to [0,3])")
ax.set_ylabel("frame count")
ax.legend(fontsize=8)
ax.set_title("Frame-wise speed ratio")

# 2) scatter |v_fd| vs |v_log|
ax = axes[0, 1]
# subsample for clarity
rng = np.random.default_rng(0)
idx = rng.choice(len(ratio), size=min(8000, len(ratio)), replace=False)
ax.scatter(v_log_norm[idx], v_fd_norm[idx], s=4, alpha=0.25, c="#2c5f7c", rasterized=True)
lim = max(v_log_norm[idx].max(), v_fd_norm[idx].max()) * 1.05
ax.plot([0, lim], [0, lim], "r-", lw=1.2, label="y=x")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel(r"$|v_{\mathrm{logged}}|$  (m/s)")
ax.set_ylabel(r"$|\Delta p/\Delta t|$  (m/s)")
ax.set_title("Finite-diff speed vs logged speed")
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=8)

# 3) per-episode median ratio vs commanded speed
ax = axes[1, 0]
ax.scatter(ep_speed, ep_med_ratio, s=18, alpha=0.55, c="#2c5f7c")
ax.axhline(1.0, color="#c0392b", lw=1.2)
# box-ish: median per speed bin
for s, info in by_spd.items():
    ax.scatter([float(s)], [info["median_ep_ratio"]], s=60, c="#e67e22", zorder=3, marker="D")
ax.set_xlabel("commanded speed (m/s)")
ax.set_ylabel("per-episode median ratio")
ax.set_title("Episode median ratio vs commanded speed")
ax.set_ylim(0, max(2.5, float(np.percentile(ep_med_ratio, 99)) * 1.1))

# 4) ratio distribution by speed (violin-ish via hist overlay / box)
ax = axes[1, 1]
speeds_u = sorted(set(np.round(speed_cmd, 2)))
data = [ratio[np.isclose(speed_cmd, s)] for s in speeds_u]
bp = ax.boxplot(
    data,
    positions=speeds_u,
    widths=0.03,
    showfliers=False,
    patch_artist=True,
    manage_ticks=False,
)
for patch in bp["boxes"]:
    patch.set_facecolor("#a8c5d4")
ax.axhline(1.0, color="#c0392b", lw=1.2)
ax.set_xlabel("commanded speed (m/s)")
ax.set_ylabel("frame ratio")
ax.set_title("Ratio by speed (no outliers)")
ax.set_xlim(min(speeds_u) - 0.05, max(speeds_u) + 0.05)
ax.set_ylim(0, 2.5)

fig.tight_layout()
png = OUT / "vel_fd_vs_logged_ratio.png"
fig.savefig(png, dpi=140)
plt.close(fig)

print(json.dumps(summary["pool"], indent=2))
print(json.dumps(summary["ratio_speed"], indent=2))
print(json.dumps(summary["ratio_xy_speed"], indent=2))
print(json.dumps(summary["per_episode_median_ratio"], indent=2))
print(json.dumps(summary["by_commanded_speed"], indent=2))
print(json.dumps(summary["component_signed_ratio_vfd_over_vlog"], indent=2))
print("wrote", png)
print("wrote", OUT / "vel_fd_vs_logged_ratio.json")

tol = float(args.accept_ratio_tol)
if tol > 0:
    bad = []
    for k, v in summary["by_commanded_speed"].items():
        r = float(v["median_frame_ratio"])
        if abs(r - 1.0) > tol:
            bad.append((k, r))
    if bad:
        raise SystemExit(
            f"ACCEPT FAIL: bin median frame ratio outside 1±{tol}: {bad}"
        )
    print(f"ACCEPT PASS: all bins median_frame_ratio within 1±{tol}")