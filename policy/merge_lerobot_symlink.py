"""Disk-cheap LeRobot merge: symlink main parquet, remap+write tail only.

Images stay inside parquet bytes. We do NOT copy the PNG trees.
Tail must be rewritten because episode_index / global index / task_index
inside those files start at 0 and the task vocabulary order differs from main.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_tasks(path: Path) -> dict[int, str]:
    return {int(r["task_index"]): r["task"] for r in _load_jsonl(path)}


def _task_remap(main_tasks: dict[int, str], tail_tasks: dict[int, str]) -> dict[int, int]:
    """Map tail task_index -> main task_index (same text, different order)."""
    text_to_main = {text: idx for idx, text in main_tasks.items()}
    missing = set(tail_tasks.values()) - set(text_to_main)
    if missing:
        raise ValueError(f"tail tasks not in main: {sorted(missing)[:5]}")
    return {t_idx: text_to_main[text] for t_idx, text in tail_tasks.items()}


def _chunk_dir(root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    d = root / "data" / f"chunk-{chunk:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _symlink_main(main_root: Path, out_root: Path, n_main: int, chunks_size: int) -> None:
    """Point merged episode_000000..episode_{n_main-1} at main parquet files."""
    for ep in range(n_main):
        src = main_root / "data" / f"chunk-{ep // chunks_size:03d}" / f"episode_{ep:06d}.parquet"
        if not src.is_file():
            raise FileNotFoundError(f"missing main parquet: {src}")
        dst = _chunk_dir(out_root, ep, chunks_size) / f"episode_{ep:06d}.parquet"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def _remap_tail_parquet(
    src: Path,
    dst: Path,
    *,
    new_episode_index: int,
    global_index_offset: int,
    task_map: dict[int, int],
) -> int:
    """Rewrite one tail episode with new episode_index / index / task_index. Returns n_frames.

    Source ``index`` is already global within the tail set (0..n_tail_frames-1),
    so we only add ``global_index_offset`` (= main total_frames) once — do not
    also add a running per-episode offset.
    """
    table = pq.read_table(src)
    n = table.num_rows
    old_task = table.column("task_index").to_numpy()
    new_task = np.array([task_map[int(t)] for t in old_task], dtype=np.int64)
    old_index = table.column("index").to_numpy().astype(np.int64)
    new_index = old_index + int(global_index_offset)
    new_ep = np.full(n, int(new_episode_index), dtype=np.int64)

    names = table.column_names
    cols = []
    for name in names:
        if name == "episode_index":
            cols.append(pa.array(new_ep))
        elif name == "index":
            cols.append(pa.array(new_index))
        elif name == "task_index":
            cols.append(pa.array(new_task))
        else:
            cols.append(table.column(name))
    out = pa.Table.from_arrays(cols, names=names)
    # Keep HF feature metadata so LeRobot still recognizes image structs.
    if table.schema.metadata:
        out = out.replace_schema_metadata(table.schema.metadata)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    pq.write_table(out, dst)
    return n


def merge_lerobot_symlink(
    main_root: Path,
    tail_root: Path,
    out_root: Path,
    *,
    overwrite: bool = False,
) -> dict:
    main_root = main_root.resolve()
    tail_root = tail_root.resolve()
    out_root = out_root.resolve()

    main_info = json.loads((main_root / "meta" / "info.json").read_text())
    tail_info = json.loads((tail_root / "meta" / "info.json").read_text())
    if main_info.get("features") != tail_info.get("features"):
        raise ValueError("main/tail features differ — refuse merge")
    chunks_size = int(main_info["chunks_size"])
    n_main = int(main_info["total_episodes"])
    n_tail = int(tail_info["total_episodes"])
    frames_main = int(main_info["total_frames"])
    frames_tail = int(tail_info["total_frames"])

    main_tasks = _load_tasks(main_root / "meta" / "tasks.jsonl")
    tail_tasks = _load_tasks(tail_root / "meta" / "tasks.jsonl")
    task_map = _task_remap(main_tasks, tail_tasks)

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists (pass --overwrite)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    (out_root / "meta").mkdir()

    logger.info("symlinking %d main episodes → %s", n_main, out_root)
    _symlink_main(main_root, out_root, n_main, chunks_size)

    logger.info("remapping %d tail episodes (parquet copy, indices + task_index)", n_tail)
    remapped_frames = 0
    for i in tqdm(range(n_tail), desc="tail remap"):
        src = tail_root / "data" / f"chunk-{i // chunks_size:03d}" / f"episode_{i:06d}.parquet"
        if not src.is_file():
            raise FileNotFoundError(f"missing tail parquet: {src}")
        new_ep = n_main + i
        dst = _chunk_dir(out_root, new_ep, chunks_size) / f"episode_{new_ep:06d}.parquet"
        n = _remap_tail_parquet(
            src,
            dst,
            new_episode_index=new_ep,
            global_index_offset=frames_main,
            task_map=task_map,
        )
        remapped_frames += n
    if remapped_frames != frames_tail:
        raise RuntimeError(f"tail frame count mismatch: wrote {remapped_frames}, info says {frames_tail}")

    # Meta: keep main task ids; shift tail episode rows.
    tasks_rows = [{"task_index": i, "task": main_tasks[i]} for i in sorted(main_tasks)]
    _write_jsonl(out_root / "meta" / "tasks.jsonl", tasks_rows)

    main_eps = _load_jsonl(main_root / "meta" / "episodes.jsonl")
    tail_eps = _load_jsonl(tail_root / "meta" / "episodes.jsonl")
    merged_eps = list(main_eps)
    for row in tail_eps:
        r = dict(row)
        r["episode_index"] = int(r["episode_index"]) + n_main
        merged_eps.append(r)
    _write_jsonl(out_root / "meta" / "episodes.jsonl", merged_eps)

    main_stats = _load_jsonl(main_root / "meta" / "episodes_stats.jsonl")
    tail_stats = _load_jsonl(tail_root / "meta" / "episodes_stats.jsonl")
    merged_stats = list(main_stats)
    for row in tail_stats:
        r = dict(row)
        r["episode_index"] = int(r["episode_index"]) + n_main
        merged_stats.append(r)
    _write_jsonl(out_root / "meta" / "episodes_stats.jsonl", merged_stats)

    total_episodes = n_main + n_tail
    total_frames = frames_main + frames_tail
    total_chunks = (total_episodes + chunks_size - 1) // chunks_size
    info = dict(main_info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(main_tasks)
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    summary = {
        "out_root": str(out_root),
        "n_main": n_main,
        "n_tail": n_tail,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_chunks": total_chunks,
        "task_remap_size": len(task_map),
    }
    logger.info("merge done: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main", type=Path, default=Path("data/lerobot/p3_throw"))
    p.add_argument("--tail", type=Path, default=Path("data/lerobot/p3_throw_tail"))
    p.add_argument("--out", type=Path, default=Path("data/lerobot/p3_throw_all"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    merge_lerobot_symlink(args.main, args.tail, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
