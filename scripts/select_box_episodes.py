#!/usr/bin/env python3
"""Pick the old "bounce box" samples that are safe to mix into the throw data.

Plain words:
  The old dataset (data/p1-retrain, LeRobot copy in data/lerobot/p1-retrain)
  was recorded in a scene where the object bounced off invisible walls. The new
  task (throw across the table, catch before it falls off) never bounces. A
  50-step action chunk that "sees" a bounce teaches the policy a path that will
  never happen again. So we keep only:
    * every still episode (speed 0.00 -- nothing moves, nothing bounces), and
    * frames of moving episodes whose next 50 steps (the action chunk) have no
      wall hit  -> frame-level list, and
    * moving episodes with no wall hit before the grasp -> episode-level list.

  Wall hit = a velocity component (oracle_state[:, 3:5]) flips sign compared
  with the last frame where that component was clearly moving (|v| > 0.02 m/s).
  Same rule as `_wall_hits` in scripts/g2_perception_report.py.

Mapping raw .h5 -> LeRobot episode_index:
  policy/convert_h5.py walks `sorted(input_dir.rglob("*.h5"))`, keeps the
  successful ones that have cameras + conditioning, and numbers them 0, 1, 2...
  We rebuild that same order, then CHECK it per episode: frame count vs
  meta/episodes.jsonl, task text vs the episode json, and oracle_state from the
  parquet vs the .h5 (max abs diff). Any mismatch is reported; the index is
  marked unreliable if one fails.

Output: data/lerobot/p1-retrain-nobounce-index.json (counts + per-episode safe
frame runs as [start, end) half-open ranges of LeRobot frame_index).
Read-only on the datasets. CPU only. Env: dynamicVLA_isaac (numpy/h5py/pyarrow).

Usage:
  python scripts/select_box_episodes.py
  python scripts/select_box_episodes.py --chunk 50 --skip-parquet-check
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V_MIN = 0.02  # m/s, same threshold as _wall_hits


def wall_hits(vel_xy: np.ndarray) -> np.ndarray:
    """Frame indices where vx or vy flips sign (copy of g2_perception_report._wall_hits)."""
    hits = set()
    for axis in (0, 1):
        idx = np.where(np.abs(vel_xy[:, axis]) > V_MIN)[0]
        s = np.sign(vel_xy[idx, axis])
        hits.update((idx[:-1][s[1:] != s[:-1]] + 1).tolist())
    return np.array(sorted(hits), dtype=int)


def runs(mask: np.ndarray) -> list[list[int]]:
    """Boolean mask -> list of [start, end) runs of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return [[int(a), int(b)] for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def converter_order(input_dir: Path) -> list[Path]:
    """Same filter + order as policy.convert_h5.list_success_episodes (not imported: no lerobot needed)."""
    out = []
    for h5_path in sorted(input_dir.rglob("*.h5")):
        js = h5_path.with_suffix(".json")
        if not js.exists() or not json.loads(js.read_text()).get("success", False):
            continue
        with h5py.File(h5_path, "r") as f:
            if not {"static_cam_rgb", "wrist_cam_rgb", "conditioning", "oracle_state"} <= set(f):
                continue
        out.append(h5_path)
    return out


def analyse(args: tuple) -> dict:
    ep_idx, h5_path, lr_root, chunk, chunks_size, check_parquet = args
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        orc = f["oracle_state"][:]
        hold = f["holding"][:]
    meta = json.loads(h5_path.with_suffix(".json").read_text())
    T = len(orc)
    speed = float(meta.get("speed_m_s", h5_path.parent.name.split("_")[-1]))
    still = speed == 0.0
    g = int(np.argmax(hold > 0.5)) if hold.max() > 0.5 else T  # first held frame
    hits = wall_hits(orc[:, 3:5])
    # Frame t is safe if no hit lands in [t, t+chunk): the chunk it trains on.
    bad = np.zeros(T, dtype=bool)
    for h in hits:
        bad[max(0, h - chunk + 1): h + 1] = True
    safe = ~bad
    row = {
        "episode_index": ep_idx,
        "h5": str(h5_path.relative_to(ROOT)),
        "speed_m_s": speed,
        "still": still,
        "length": T,
        "grasp_frame": g,
        "wall_hits": hits.tolist(),
        "clean_before_grasp": bool(np.all(hits >= g)),
        "n_safe_frames": int(safe.sum()),
        "safe_runs": runs(safe),
        "task": meta.get("language"),
    }
    if check_parquet:
        import pyarrow.parquet as pq

        pq_path = Path(lr_root) / f"data/chunk-{ep_idx // chunks_size:03d}/episode_{ep_idx:06d}.parquet"
        t = pq.read_table(pq_path, columns=["observation.oracle_state", "frame_index", "episode_index"])
        po = np.asarray(t.column("observation.oracle_state").to_pylist(), dtype=np.float32)
        row["pq_len"] = len(po)
        row["pq_oracle_maxdiff"] = (
            float(np.max(np.abs(po - orc.astype(np.float32)))) if po.shape == orc.shape else None
        )
        row["pq_episode_index_ok"] = bool(np.all(np.asarray(t.column("episode_index")) == ep_idx))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, default=ROOT / "data/p1-retrain")
    p.add_argument("--lerobot", type=Path, default=ROOT / "data/lerobot/p1-retrain")
    p.add_argument("--out", type=Path, default=ROOT / "data/lerobot/p1-retrain-nobounce-index.json")
    p.add_argument("--chunk", type=int, default=50, help="action chunk length (steps)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-parquet-check", action="store_true")
    a = p.parse_args()

    info = json.loads((a.lerobot / "meta/info.json").read_text())
    lr_eps = [json.loads(l) for l in (a.lerobot / "meta/episodes.jsonl").read_text().splitlines() if l]
    order = converter_order(a.raw)
    print(f"raw convertible episodes: {len(order)}; LeRobot episodes: {len(lr_eps)}")
    jobs = [
        (i, str(h), str(a.lerobot), a.chunk, int(info["chunks_size"]), not a.skip_parquet_check)
        for i, h in enumerate(order)
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(analyse, jobs, chunksize=16))

    # --- mapping checks ---
    problems = []
    if len(order) != len(lr_eps):
        problems.append(f"count mismatch raw={len(order)} lerobot={len(lr_eps)}")
    for r, e in zip(rows, lr_eps):
        i = r["episode_index"]
        if e["episode_index"] != i or e["length"] != r["length"]:
            problems.append(f"ep {i}: length h5={r['length']} lerobot={e['length']}")
        if r["task"] not in e["tasks"]:
            problems.append(f"ep {i}: task {r['task']!r} vs {e['tasks']}")
        if "pq_len" in r:
            if r["pq_len"] != r["length"] or not r["pq_episode_index_ok"]:
                problems.append(f"ep {i}: parquet len/index mismatch")
            elif r["pq_oracle_maxdiff"] is None or r["pq_oracle_maxdiff"] > 1e-5:
                problems.append(f"ep {i}: oracle diff {r['pq_oracle_maxdiff']}")
    max_diff = max((r.get("pq_oracle_maxdiff") or 0.0) for r in rows)

    # --- counts ---
    still = [r for r in rows if r["still"]]
    moving = [r for r in rows if not r["still"]]
    mov_frames = sum(r["length"] for r in moving)
    mov_safe = sum(r["n_safe_frames"] for r in moving)
    pre_frames = sum(r["grasp_frame"] for r in moving)
    pre_safe = sum(
        sum(max(0, min(e, r["grasp_frame"]) - s) for s, e in r["safe_runs"]) for r in moving
    )
    clean_eps = [r for r in moving if r["clean_before_grasp"]]
    counts = {
        "episodes_total": len(rows),
        "frames_total": int(sum(r["length"] for r in rows)),
        "still_episodes": len(still),
        "still_frames": int(sum(r["length"] for r in still)),
        "still_episodes_with_hits": sum(bool(r["wall_hits"]) for r in still),
        "moving_episodes": len(moving),
        "moving_frames": int(mov_frames),
        "moving_safe_frames": int(mov_safe),
        "moving_safe_frame_frac": round(mov_safe / max(mov_frames, 1), 4),
        "moving_pre_grasp_frames": int(pre_frames),
        "moving_pre_grasp_safe_frames": int(pre_safe),
        "moving_pre_grasp_safe_frac": round(pre_safe / max(pre_frames, 1), 4),
        "moving_episodes_with_any_hit": sum(bool(r["wall_hits"]) for r in moving),
        "moving_episodes_clean_before_grasp": len(clean_eps),
        "moving_frames_in_clean_episodes": int(sum(r["length"] for r in clean_eps)),
        "frame_level_keep_frames": int(sum(r["length"] for r in still) + mov_safe),
        "episode_level_keep_episodes": len(still) + len(clean_eps),
        "episode_level_keep_frames": int(
            sum(r["length"] for r in still) + sum(r["length"] for r in clean_eps)
        ),
    }
    out = {
        "what": "Old bounce-box samples safe to mix into the throw/no-bounce training set",
        "lerobot_root": str(a.lerobot.relative_to(ROOT)),
        "raw_root": str(a.raw.relative_to(ROOT)),
        "rule": {
            "wall_hit": "sign flip of oracle_state vx or vy vs last frame with |v|>0.02 m/s",
            "frame_safe": f"no wall hit at frame indices [t, t+{a.chunk})",
            "episode_clean": "no wall hit before first holding>0.5 frame",
            "frame_ranges": "safe_runs are [start, end) LeRobot frame_index; global index = "
            "dataset_from_index[episode] + frame_index",
        },
        "mapping": {
            "method": "sorted(rglob('*.h5')) + success/camera filter, same as policy/convert_h5.py",
            "reliable": not problems,
            "checked": "length vs episodes.jsonl, task text, parquet episode_index, "
            + ("oracle_state values" if not a.skip_parquet_check else "(parquet check skipped)"),
            "oracle_max_abs_diff": max_diff,
            "problems": problems[:50],
            "n_problems": len(problems),
        },
        "counts": counts,
        "episode_level_indices": sorted(r["episode_index"] for r in still + clean_eps),
        "episodes": [
            {k: r[k] for k in ("episode_index", "h5", "speed_m_s", "length", "grasp_frame",
                               "wall_hits", "clean_before_grasp", "safe_runs")}
            for r in rows
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps({"mapping_reliable": not problems, "n_problems": len(problems),
                      "oracle_max_abs_diff": max_diff, **counts}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
