"""Debug MP4: stitch camera views + state overlay.

Turns a logged episode into a video a human can watch (table cam + wrist cam,
with stage / speed text on top). Runs after an episode, only when we ask for video.
Must not: change the episode data or run inside the sim loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Prefer third-person then wrist
_DEFAULT_CAM_KEYS = ("static_cam_rgb", "wrist_cam_rgb")
_CFG_PATH = Path(__file__).resolve().parent / "phase0_cfg.yaml"
_YELLOW_BGR = (0, 255, 255)


def _to_u8_rgb(frame: np.ndarray) -> np.ndarray:
    """Turn any camera frame (gray, RGBA, float 0-1 or 0-255) into plain uint8 RGB."""
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
        # Floats in 0..1 need scaling up; floats already in 0..255 do not.
        if f.max() <= 1.0:
            f = f * 255.0
        frame = np.clip(f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _fmt_vec_mm(v: np.ndarray, nd: int = 3) -> str:
    """Format the first ``nd`` metres as whole millimetres for the overlay."""
    v = np.asarray(v).reshape(-1)
    return " ".join(f"{1000.0 * float(x):.0f}" for x in v[:nd])


def _mm(m: float) -> str:
    """One metre value as whole millimetres."""
    return f"{1000.0 * float(m):.0f}"


def gates_from_expert(expert: Any) -> dict[str, float]:
    """Copy the expert's one grasp gate for the video footer."""
    ok = np.asarray(expert.grasp_xyz_ok, dtype=np.float64).reshape(3)
    return {
        "grasp_dx": float(ok[0]),
        "grasp_dy": float(ok[1]),
        "grasp_dz": float(ok[2]),
        "close_gripper_s": float(expert.close_gripper_s),
        "close_depth_m": float(expert.close_depth_m),
        "dt": float(expert.dt),
    }


def _gates_from_cfg() -> dict[str, float]:
    """Yaml defaults when the caller did not pass live expert gates."""
    from interfaces.config import load_yaml

    cfg = load_yaml(_CFG_PATH)
    sm = cfg["state_machine_params"]
    scene = cfg.get("scene", {})
    physics = bool(scene.get("use_house", False)) and not bool(
        scene.get("use_weld_grasp", not bool(scene.get("use_house", False)))
    )
    close_s = float(sm["physics_close_s"] if physics else sm["close_gripper_s"])
    ok = sm["grasp_xyz_ok"]
    return {
        "grasp_dx": float(ok[0]),
        "grasp_dy": float(ok[1]),
        "grasp_dz": float(ok[2]),
        "close_gripper_s": close_s,
        "close_depth_m": float(sm.get("close_depth_m", 0.0)),
        "dt": float(cfg["sim"]["dt"]),
    }


def _resolve_gates(
    episode: dict[str, Any], gates: dict[str, float] | None
) -> dict[str, float]:
    """Prefer caller gates, then episode stash, then yaml defaults."""
    if gates:
        return {k: float(v) for k, v in gates.items()}
    stored = episode.get("overlay_gates")
    if isinstance(stored, dict) and stored:
        return {k: float(v) for k, v in stored.items()}
    return _gates_from_cfg()


def _state_lines(episode: dict[str, Any], t: int) -> list[str]:
    """Text lines for step ``t``. Skips any field the log does not have."""
    from sim.state_machine_logs import stage_label

    lines: list[str] = []
    if "stage" in episode:
        lines.append(f"Stage: {stage_label(int(episode['stage'][t]))}")
    if "holding" in episode:
        held = "holding" if float(episode["holding"][t]) > 0.5 else "not holding"
        lines.append(held)
    if "ee_pos" in episode:
        lines.append(f"Hand mm: {_fmt_vec_mm(episode['ee_pos'][t])}")
    if "object_pos" in episode:
        lines.append(f"Object mm: {_fmt_vec_mm(episode['object_pos'][t])}")
    if "object_vel" in episode:
        spd = 100.0 * float(np.linalg.norm(np.asarray(episode["object_vel"][t])[:3]))
        lines.append(f"Speed: {spd:.1f} cm/s")
    if "policy_replan" in episode and int(episode["policy_replan"][t]) == 1:
        lines.append("Policy replan")
    return lines


def _policy_input_lines(episode: dict[str, Any], t: int) -> list[str]:
    """What the policy was fed at step ``t`` (eval logs): text, 4 numbers, gripper."""
    lines = list(episode.get("overlay_info") or [])
    if "conditioning" in episode and len(episode["conditioning"]) > t:
        c = np.asarray(episode["conditioning"][t], dtype=np.float64)
        lines.append(
            f"Fed p-hat mm: {1000 * c[0]:.0f} {1000 * c[1]:.0f}   v-hat cm/s: {100 * c[2]:.1f} {100 * c[3]:.1f}"
        )
    if "gripper" in episode and len(episode["gripper"]) > t:
        cmd = "OPEN" if float(episode["gripper"][t]) > 0.5 else "CLOSE"
        gap = ""
        if "finger_gap" in episode and len(episode["finger_gap"]) > t:
            gap = f"   finger gap {1000 * float(episode['finger_gap'][t]):.0f} mm"
        lines.append(f"Gripper cmd: {cmd}{gap}")
    return lines


def _gate_lines(
    episode: dict[str, Any], t: int, gates: dict[str, float]
) -> list[str]:
    """Top-left gate report: live hand–object gap vs the one grasp_xyz_ok box."""
    if "ee_pos" not in episode or "object_pos" not in episode:
        return []

    ee = np.asarray(episode["ee_pos"][t], dtype=np.float64).reshape(-1)
    obj = np.asarray(episode["object_pos"][t], dtype=np.float64).reshape(-1)
    gap_dx = abs(float(ee[0] - obj[0]))
    gap_dy = abs(float(ee[1] - obj[1]))
    gap_z = abs(float(ee[2] - obj[2]))

    # Support old episode stashes that still use track_*/grasp_* keys.
    if "grasp_dx" in gates:
        gdx = float(gates["grasp_dx"])
        gdy = float(gates["grasp_dy"])
        gdz = float(gates["grasp_dz"])
    else:
        gdx = gdy = float(gates.get("grasp_xy_tol", gates.get("track_xy_tol", 0.01)))
        gdz = float(gates.get("grasp_z_tol", gates.get("approach_z_tol", 0.015)))

    return [
        f"gap dx/dy/z {_mm(gap_dx)}/{_mm(gap_dy)}/{_mm(gap_z)} mm   "
        f"gates mm: grasp {_mm(gdx)}/{_mm(gdy)}/{_mm(gdz)}",
    ]


def _overlay_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    anchor: str,
    color: tuple[int, int, int] = _YELLOW_BGR,
) -> np.ndarray:
    """Draw text on a dark box. ``anchor``: 'top' (top-right), 'top_left' or 'bottom_left'."""
    if not lines:
        return frame
    import cv2

    out = np.ascontiguousarray(frame)
    margin, pad, scale, thickness = 10, 4, 0.5, 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = out.shape[0], out.shape[1]
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    max_tw = max((tw for tw, _ in sizes), default=0)
    line_gap = margin
    block_h = sum(th for _, th in sizes) + line_gap * (len(lines) - 1) + 2 * pad

    if anchor in ("top_left", "bottom_left"):
        x0 = margin - pad
    else:
        x0 = max(0, img_w - max_tw - margin - pad)
    y0 = max(0, margin - pad) if anchor != "bottom_left" else max(0, img_h - block_h - margin)
    x1 = min(img_w, x0 + max_tw + 2 * pad)
    y1 = min(img_h, y0 + block_h)
    # Dark underlay so cyan text stays readable on bright house walls.
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    y = y0 + pad
    for line, (tw, th) in zip(lines, sizes):
        x = margin if anchor in ("top_left", "bottom_left") else max(margin, img_w - tw - margin)
        cv2.putText(out, line, (x, y + th), font, scale, color, thickness, cv2.LINE_AA)
        y += th + line_gap
    return out


def stitch_episode_frames(
    episode: dict[str, Any],
    cam_keys: tuple[str, ...] = _DEFAULT_CAM_KEYS,
    *,
    overlay: bool = True,
    gates: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Build one side-by-side RGB frame per timestep from logged camera arrays.

    Missing cameras are skipped. Returns [] if no camera has frames.
    """
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

    gate_vals = _resolve_gates(episode, gates) if overlay else {}

    # Cameras may log different lengths; use the shortest so tiles line up.
    n = min(c.shape[0] for c in cams)
    frames: list[np.ndarray] = []
    for t in range(n):
        tiles = [_to_u8_rgb(c[t]) for c in cams]
        # Side-by-side needs equal heights. Drop rows (no blur) to match the shortest.
        h = min(im.shape[0] for im in tiles)
        resized = []
        for im in tiles:
            if im.shape[0] != h:
                ys = (np.linspace(0, im.shape[0] - 1, h)).astype(np.int64)
                im = im[ys]
            resized.append(im)
        frame = np.concatenate(resized, axis=1)
        if overlay:
            frame = _overlay_text(frame, _state_lines(episode, t), anchor="top")
            frame = _overlay_text(
                frame, _gate_lines(episode, t, gate_vals), anchor="top_left", color=_YELLOW_BGR
            )
            frame = _overlay_text(frame, _policy_input_lines(episode, t), anchor="bottom_left")
        frames.append(frame)
    return frames


def dump_video(frames: list[np.ndarray], output_path: Path | str, *, fps: int = 25) -> None:
    """Write frames to an MP4 file. Needs no screen, so it works headless.

    Empty frame list = do nothing.
    """
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
    gates: dict[str, float] | None = None,
) -> bool:
    """One call: stitch table + wrist views with text overlay, save MP4.

    Returns False (and warns) if the log had no camera frames.
    Pass ``gates`` (or stash ``episode['overlay_gates']``) so the gate text
    matches the expert that ran; else yaml defaults are used.
    """
    frames = stitch_episode_frames(episode, overlay=True, gates=gates)
    if not frames:
        logger.warning("No camera frames to stitch for %s", output_path)
        return False
    dump_video(frames, output_path, fps=fps)
    return True
