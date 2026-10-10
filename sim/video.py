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
    """Copy the expert's descend / close gates for the video footer."""
    return {
        "track_xy_tol": float(expert.track_xy_tol),
        "approach_z_tol": float(expert.approach_z_tol),
        "track_hold_s": float(expert.track_hold_s),
        "grasp_xy_tol": float(expert.grasp_xy_tol),
        "grasp_z_tol": float(expert.grasp_z_tol),
        "grasp_tol_step_k": float(expert.grasp_tol_step_k),
        "close_gripper_s": float(expert.close_gripper_s),
        "hover_z": float(expert.hover_z),
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
    grasp_z = float(sm["physics_grasp_z_tol"] if physics else sm["grasp_z_tol"])
    close_s = float(sm["physics_close_s"] if physics else sm["close_gripper_s"])
    return {
        "track_xy_tol": float(sm["track_xy_tol"]),
        "approach_z_tol": float(sm["approach_z_tol"]),
        "track_hold_s": float(sm["track_hold_s"]),
        "grasp_xy_tol": float(sm["grasp_xy_tol"]),
        "grasp_z_tol": grasp_z,
        "grasp_tol_step_k": float(sm["grasp_tol_step_k"]),
        "close_gripper_s": close_s,
        "hover_z": float(sm["hover_z"]),
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


def _grasp_tol_xy_m(gates: dict[str, float], vel_xy_m_s: float) -> float:
    """Same floor as the expert: max(base, k * speed * dt)."""
    step = float(gates["grasp_tol_step_k"]) * float(vel_xy_m_s) * float(gates["dt"])
    return max(float(gates["grasp_xy_tol"]), step)


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


def _gate_lines(
    episode: dict[str, Any], t: int, gates: dict[str, float]
) -> list[str]:
    """Bottom footer: live hand–object gap vs the descend / close gates."""
    if "ee_pos" not in episode or "object_pos" not in episode:
        return []

    ee = np.asarray(episode["ee_pos"][t], dtype=np.float64).reshape(-1)
    obj = np.asarray(episode["object_pos"][t], dtype=np.float64).reshape(-1)
    gap_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
    gap_z = abs(float(ee[2] - obj[2]))

    vel_xy = 0.0
    if "object_vel" in episode:
        vel = np.asarray(episode["object_vel"][t], dtype=np.float64).reshape(-1)
        vel_xy = float(np.linalg.norm(vel[:2]))

    lock_xy = float(gates["track_xy_tol"])
    lock_z = float(gates["approach_z_tol"])
    close_xy = _grasp_tol_xy_m(gates, vel_xy)
    close_z = float(gates["grasp_z_tol"])
    hold_ms = 1000.0 * float(gates["track_hold_s"])
    dwell_ms = 1000.0 * float(gates["close_gripper_s"])
    hover_mm = _mm(gates["hover_z"])

    stage = int(episode["stage"][t]) if "stage" in episode else -1
    # Which gate is "active" for this stage (helps fine-tune that knob).
    # Intercept lock Z is vs the aim height (hover blend), not object top.
    if stage == 1:
        need = (
            f"need lock XY<{_mm(lock_xy)} + Z-to-aim<{_mm(lock_z)} "
            f"then hold {hold_ms:.0f} ms → descend"
        )
    elif stage == 2:
        need = f"need close XY<{_mm(close_xy)} Z<{_mm(close_z)} → grip"
    elif stage == 3:
        need = f"holding; close dwell {dwell_ms:.0f} ms → lift"
    else:
        need = (
            f"lock XY<{_mm(lock_xy)} Z-aim<{_mm(lock_z)} | "
            f"close XY<{_mm(close_xy)} Z<{_mm(close_z)}"
        )

    return [
        f"gap XY {_mm(gap_xy)} / Z {_mm(gap_z)} mm   {need}",
        (
            f"gates mm: lock {_mm(lock_xy)}/{_mm(lock_z)}  "
            f"close {_mm(close_xy)}/{_mm(close_z)}  "
            f"hold {hold_ms:.0f}ms  dwell {dwell_ms:.0f}ms  hover {hover_mm}"
        ),
    ]


def _overlay_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    anchor: str,
    color: tuple[int, int, int] = _YELLOW_BGR,
) -> np.ndarray:
    """Draw text top-right or bottom-left. ``anchor`` is 'top' or 'bottom'."""
    if not lines:
        return frame
    import cv2

    out = np.ascontiguousarray(frame)
    margin, scale, thickness = 10, 0.5, 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = out.shape[0], out.shape[1]
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_h = max((th for _, th in sizes), default=12) + margin

    if anchor == "bottom":
        y = img_h - margin - (len(lines) - 1) * line_h
        for line, (tw, th) in zip(lines, sizes):
            x = margin
            cv2.putText(out, line, (x, y + th), font, scale, color, thickness, cv2.LINE_AA)
            y += line_h
    else:
        y = margin
        for line, (tw, th) in zip(lines, sizes):
            x = max(margin, img_w - tw - margin)
            cv2.putText(out, line, (x, y + th), font, scale, color, thickness, cv2.LINE_AA)
            y += th + margin
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
                frame, _gate_lines(episode, t, gate_vals), anchor="bottom", color=_YELLOW_BGR
            )
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
    Pass ``gates`` (or stash ``episode['overlay_gates']``) so the footer
    matches the expert that ran; else yaml defaults are used.
    """
    frames = stitch_episode_frames(episode, overlay=True, gates=gates)
    if not frames:
        logger.warning("No camera frames to stitch for %s", output_path)
        return False
    dump_video(frames, output_path, fps=fps)
    return True
