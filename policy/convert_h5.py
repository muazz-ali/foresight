"""Turn Foresight demo H5 files into a LeRobot dataset.

Step 1 of 3 (convert -> train -> infer). Sim writes one H5 + one JSON per
episode. Training wants a LeRobot dataset. This file copies frames over.

Layout rule: the slots here come from policy/features.py. 
We store the 8 robot numbers as "state", and keep the 4 object numbers ("conditioning")
and the perfect object pose ("oracle_state") on the side. Training glues
them into the 12-number state. Do not glue here, or noise cannot be added later.
"""

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
    # 1 = open, 0 = closed. Keeps the gripper on one simple scale.
    v = float(open_cmd)
    if v > 0.5:
        return 1.0
    if v < 0.0:
        return 0.0
    # Continuous meters → normalize by open width.
    span = max(open_val - close_val, 1e-6)
    return float(np.clip((v - close_val) / span, 0.0, 1.0))


def pack_proprio(ee_pos: np.ndarray, ee_quat: np.ndarray, gripper_open: float) -> np.ndarray:
    # 8 robot numbers: hand XYZ (3) + rotation (4) + gripper (1). Order = PROPRIO_NAMES.
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


def clean_reject_reason(meta: dict, f) -> str | None:
    """Why a success is not a clean demo (None = clean). Rule: sim/success.py."""
    from sim.success import clean_reject_reason as _rule

    if "clean_reject" in meta:  # recorded at collect time
        return meta["clean_reject"]
    episode = {k: np.asarray(f[k][:]) for k in ("stage", "object_pos", "ee_pos")}
    return _rule(episode, meta.get("retries"))


def list_success_episodes(
    input_dir: Path, *, success_only: bool = True, clean_only: bool = False
) -> list[tuple[Path, Path]]:
    """Return (h5, json) pairs under input_dir (recursive).

    By default keep only episodes the expert finished (success). Learning from
    failed demos would teach the policy to fail. Skip files that lack cameras
    or the object numbers, since training needs both. ``clean_only`` also drops
    successes that retried, skipped the open stage, or dragged the object.
    """
    pairs: list[tuple[Path, Path]] = []
    rejected: dict[str, int] = {}
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
            if clean_only:
                why = clean_reject_reason(meta, f)
                if why is not None:
                    rejected[why] = rejected.get(why, 0) + 1
                    continue
        pairs.append((h5_path, js))
    if clean_only:
        logger.info("clean filter kept %d, dropped %s", len(pairs), rejected or "none")
    return pairs


def episode_language(meta: dict) -> str:
    # The text instruction the policy reads. Use the saved one, else make one.
    if meta.get("language"):
        return str(meta["language"])
    return default_language(meta.get("category"), meta.get("container_category"))


def episodes_done_in_root(root: Path) -> int:
    """How many episodes a prior convert already wrote (from meta/info.json)."""
    info = root / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"no LeRobot info at {info}")
    n = int(json.loads(info.read_text())["total_episodes"])
    if n < 0:
        raise ValueError(f"bad total_episodes in {info}: {n}")
    return n


def prefer_fast_pairs(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """Fast speeds first so a mid-run OOM still keeps 15–20 cm/s demos."""

    def speed_key(pair: tuple[Path, Path]) -> float:
        meta = json.loads(pair[1].read_text())
        return float(meta.get("speed_m_s", 0.0))

    return sorted(pairs, key=speed_key, reverse=True)


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

    # include_aux=True keeps the side data (4 object numbers + oracle pose).
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
    # Sidecar for later skip/merge: one line per saved episode → source H5.
    source_log = root / "meta" / "source_h5.jsonl"
    source_log.parent.mkdir(parents=True, exist_ok=True)

    for ep_i, (h5_path, js_path) in enumerate(tqdm(pairs, desc="convert")):
        meta = json.loads(js_path.read_text())
        task = episode_language(meta)
        with h5py.File(h5_path, "r") as f:
            T = int(f["object_pos"].shape[0])  # number of frames in this episode
            # Shape checks: catch a wrong layout now, not after hours of training.
            assert f["conditioning"].shape == (T, CONDITIONING_DIM)
            assert f["oracle_state"].shape == (T, ORACLE_DIM)
            for t in range(T):
                state = pack_proprio(f["ee_pos"][t], f["ee_quat"][t], float(f["gripper"][t]))
                # Action = the pose the expert commanded at this frame.
                action = pack_action(
                    f["action_pos"][t], f["action_quat"][t], float(f["gripper"][t])
                )
                assert state.shape == (PROPRIO_DIM,)
                assert action.shape == (ACTION_DIM,)
                frame = {
                    "observation.state": state,
                    "action": action,
                    # Side data, not glued to state yet (see transforms.py).
                    "observation.conditioning": np.asarray(f["conditioning"][t], dtype=np.float32),
                    "observation.oracle_state": np.asarray(f["oracle_state"][t], dtype=np.float32),
                    "observation.images.static_cam": _as_uint8_rgb(f["static_cam_rgb"][t]),
                    "observation.images.wrist_cam": _as_uint8_rgb(f["wrist_cam_rgb"][t]),
                }
                ds.add_frame(frame, task=task)
        ds.save_episode()
        with source_log.open("a") as sf:
            sf.write(
                json.dumps(
                    {
                        "episode_index": ep_i,
                        "h5": str(h5_path.resolve()),
                        "speed_m_s": float(meta.get("speed_m_s", -1)),
                        "category": meta.get("category"),
                    }
                )
                + "\n"
            )
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
    p.add_argument(
        "--clean-only",
        action="store_true",
        help="drop successes that retried, skipped the open stage, or dragged the object on release",
    )
    p.add_argument(
        "--skip-from-root",
        type=Path,
        default=None,
        help="skip the first N sorted clean episodes already written into this LeRobot root "
        "(N = that root's meta/info.json total_episodes). Write remaining to --root.",
    )
    p.add_argument(
        "--prefer-fast",
        action="store_true",
        help="after skip, convert highest speed_m_s first (keeps 20 cm/s if run dies early)",
    )
    args = p.parse_args()

    root = args.root or (Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
    pairs = list_success_episodes(
        args.input_dir, success_only=not args.include_failures, clean_only=args.clean_only
    )
    logger.info("found %d episodes under %s", len(pairs), args.input_dir)
    if args.skip_from_root is not None:
        n_done = episodes_done_in_root(args.skip_from_root)
        if n_done > len(pairs):
            raise RuntimeError(
                f"skip-from-root has {n_done} episodes but only {len(pairs)} convertible now"
            )
        pairs = pairs[n_done:]
        logger.info(
            "skip-from-root %s: skipped %d already converted, %d remaining",
            args.skip_from_root,
            n_done,
            len(pairs),
        )
    if args.prefer_fast:
        pairs = prefer_fast_pairs(pairs)
        if pairs:
            speeds = [float(json.loads(js.read_text()).get("speed_m_s", -1)) for _, js in pairs[:5]]
            logger.info("prefer-fast: first speeds %s ... last %s", speeds, float(json.loads(pairs[-1][1].read_text()).get("speed_m_s", -1)))
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
