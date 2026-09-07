"""Debug MP4: stitch camera views + state overlay."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Prefer third-person then wrist
_DEFAULT_CAM_KEYS = ("static_cam_rgb", "wrist_cam_rgb")


def _to_u8_rgb(frame: np.ndarray) -> np.ndarray:
    """Normalize one HxWxC frame to uint8 RGB."""
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=-1)
    elif frame.ndim == 3:
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        else:
            frame = frame[:, :, :3]
    else:
        raise ValueError(f"Unknown camera frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        if f.max() <= 1.0:
            f = f * 255.0
        frame = np.clip(f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _fmt_vec(v: np.ndarray, nd: int = 3) -> str:
    v = np.asarray(v).reshape(-1)
    return " ".join(f"{float(x):.3f}" for x in v[:nd])


def _state_lines(episode: dict[str, Any], t: int) -> list[str]:
    from sim.state_machine_logs import stage_label

    lines: list[str] = []
    if "stage" in episode:
        lines.append(f"Stage: {stage_label(int(episode['stage'][t]))}")
    if "holding" in episode:
        held = "holding" if float(episode["holding"][t]) > 0.5 else "not holding"
        lines.append(held)
    if "ee_pos" in episode:
        lines.append(f"Hand: {_fmt_vec(episode['ee_pos'][t])}")
    if "object_pos" in episode:
        lines.append(f"Object: {_fmt_vec(episode['object_pos'][t])}")
    if "object_vel" in episode:
        spd = 100.0 * float(np.linalg.norm(np.asarray(episode["object_vel"][t])[:3]))
        lines.append(f"Speed: {spd:.1f} cm/s")
    if "policy_replan" in episode and int(episode["policy_replan"][t]) == 1:
        lines.append("Policy replan")
    return lines


def _overlay_state(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw state text top-right."""
    if not lines:
        return frame
    import cv2

    out = np.ascontiguousarray(frame)
    margin, scale, thickness = 10, 0.5, 1
    color = (0, 255, 255)  # yellow (BGR) — reads better on light table views
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = margin
    img_w = out.shape[1]
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        x = max(margin, img_w - tw - margin)
        cv2.putText(out, line, (x, y + th), font, scale, color, thickness, cv2.LINE_AA)
        y += th + margin
    return out


def stitch_episode_frames(
    episode: dict[str, Any],
    cam_keys: tuple[str, ...] = _DEFAULT_CAM_KEYS,
    *,
    overlay: bool = True,
) -> list[np.ndarray]:
    """Build one stitched RGB frame per timestep from logged camera arrays."""
    cams: list[np.ndarray] = []
    for key in cam_keys:
        if key not in episode:
            continue
        arr = np.asarray(episode[key])
        if arr.ndim != 4 or arr.shape[0] == 0:
            continue
        cams.append(arr)

    if not cams:
        return []

    n = min(c.shape[0] for c in cams)
    frames: list[np.ndarray] = []
    for t in range(n):
        tiles = [_to_u8_rgb(c[t]) for c in cams]
        h = min(im.shape[0] for im in tiles)
        resized = []
        for im in tiles:
            if im.shape[0] != h:
                ys = (np.linspace(0, im.shape[0] - 1, h)).astype(np.int64)
                im = im[ys]
            resized.append(im)
        frame = np.concatenate(resized, axis=1)
        if overlay:
            frame = _overlay_state(frame, _state_lines(episode, t))
        frames.append(frame)
    return frames


def dump_video(frames: list[np.ndarray], output_path: Path | str, *, fps: int = 25) -> None:
    """Write stitched frames to libx264 MP4 (display-free)."""
    if not frames:
        return
    import imageio.v3

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.v3.imwrite(
        str(path),
        frames,
        fps=int(fps),
        codec="libx264",
        macro_block_size=1,
    )
    logger.info("Wrote video %s (%d frames)", path, len(frames))


def write_debug_mp4(
    episode: dict[str, Any],
    output_path: Path | str,
    *,
    fps: int = 25,
) -> bool:
    """Stitch static + wrist RGB (with state overlay) and write MP4."""
    frames = stitch_episode_frames(episode, overlay=True)
    if not frames:
        logger.warning("No camera frames to stitch for %s", output_path)
        return False
    dump_video(frames, output_path, fps=fps)
    return True
