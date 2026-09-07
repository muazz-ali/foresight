"""Turn Foresight demo H5 files into a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from policy.features import (
    ACTION_DIM,
    CONDITIONING_DIM,
    DEFAULT_FPS,
    ORACLE_DIM,
    PROPRIO_DIM,
    default_language,
    make_features,
)

logger = logging.getLogger(__name__)


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Isaac RGB may be float [0,1] or uint8; LeRobot wants HWC uint8."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr)) if arr.size else 1.0
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if mx <= 1.5 else np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _gripper_from_open(open_cmd: float, open_val: float = 0.04, close_val: float = 0.0) -> float:
    """Map expert open∈{0,1} or continuous to a scalar gripper action in [0,1] open."""
    v = float(open_cmd)
    if v > 0.5:
        return 1.0
    if v < 0.0:
        return 0.0
    # Continuous meters → normalize by open width.
    span = max(open_val - close_val, 1e-6)
    return float(np.clip((v - close_val) / span, 0.0, 1.0))


def pack_proprio(ee_pos: np.ndarray, ee_quat: np.ndarray, gripper_open: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(ee_pos, dtype=np.float32).reshape(3),
            np.asarray(ee_quat, dtype=np.float32).reshape(4),
            [np.float32(_gripper_from_open(gripper_open))],
        ]
    ).astype(np.float32)


def pack_action(action_pos: np.ndarray, action_quat: np.ndarray, gripper_open: float) -> np.ndarray:
    """Same 8-D layout as pack_proprio (command, not measured state)."""
    return pack_proprio(action_pos, action_quat, gripper_open)


def list_success_episodes(input_dir: Path, *, success_only: bool = True) -> list[tuple[Path, Path]]:
    """Return (h5, json) pairs under input_dir (recursive)."""
    pairs: list[tuple[Path, Path]] = []
    for h5_path in sorted(input_dir.rglob("*.h5")):
        js = h5_path.with_suffix(".json")
        if not js.exists():
            logger.warning("skip %s (no json)", h5_path)
            continue
        meta = json.loads(js.read_text())
        if success_only and not meta.get("success", False):
            continue
        # Need cameras for policy.
        with h5py.File(h5_path, "r") as f:
            if "static_cam_rgb" not in f or "wrist_cam_rgb" not in f:
                logger.warning("skip %s (missing camera datasets)", h5_path.name)
                continue
            if "conditioning" not in f or "oracle_state" not in f:
                logger.warning("skip %s (missing conditioning/oracle)", h5_path.name)
                continue
        pairs.append((h5_path, js))
    return pairs


def episode_language(meta: dict) -> str:
    if meta.get("language"):
        return str(meta["language"])
    return default_language(meta.get("category"), meta.get("container_category"))


def convert_episodes(
    pairs: list[tuple[Path, Path]],
    *,
    repo_id: str,
    root: Path,
    fps: int = DEFAULT_FPS,
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite to replace")
        shutil.rmtree(root)

    if max_episodes is not None:
        pairs = pairs[: int(max_episodes)]
    if not pairs:
        raise RuntimeError("No convertible episodes found")

    features = make_features(include_aux=True)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=root,
        robot_type="franka",
        use_videos=False,
        image_writer_threads=4,
    )

    for h5_path, js_path in tqdm(pairs, desc="convert"):
        meta = json.loads(js_path.read_text())
        task = episode_language(meta)
        with h5py.File(h5_path, "r") as f:
            T = int(f["object_pos"].shape[0])
            assert f["conditioning"].shape == (T, CONDITIONING_DIM)
            assert f["oracle_state"].shape == (T, ORACLE_DIM)
            for t in range(T):
                state = pack_proprio(f["ee_pos"][t], f["ee_quat"][t], float(f["gripper"][t]))
                action = pack_action(
                    f["action_pos"][t], f["action_quat"][t], float(f["gripper"][t])
                )
                assert state.shape == (PROPRIO_DIM,)
                assert action.shape == (ACTION_DIM,)
                frame = {
                    "observation.state": state,
                    "action": action,
                    "observation.conditioning": np.asarray(f["conditioning"][t], dtype=np.float32),
                    "observation.oracle_state": np.asarray(f["oracle_state"][t], dtype=np.float32),
                    "observation.images.static_cam": _as_uint8_rgb(f["static_cam_rgb"][t]),
                    "observation.images.wrist_cam": _as_uint8_rgb(f["wrist_cam_rgb"][t]),
                }
                ds.add_frame(frame, task=task)
        ds.save_episode()
        logger.info("saved %s frames=%d task=%s", h5_path.name, T, task)

    logger.info(
        "done repo=%s root=%s episodes=%d frames=%d",
        repo_id,
        root,
        ds.meta.total_episodes,
        ds.meta.total_frames,
    )
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Convert Foresight H5 episodes to LeRobot")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--repo-id", type=str, default="foresight/p1")
    p.add_argument("--root", type=Path, default=None, help="LeRobot dataset root")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--include-failures", action="store_true")
    args = p.parse_args()

    root = args.root or (Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
    pairs = list_success_episodes(args.input_dir, success_only=not args.include_failures)
    logger.info("found %d episodes under %s", len(pairs), args.input_dir)
    convert_episodes(
        pairs,
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
