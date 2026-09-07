"""Episode recording loop (scripted expert + sim-truth future numbers)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT, conditioning_vector
from policy.features import default_language
from sim.state_machine_logs import log_episode_begin
from sim.state_machine import STAGE_DONE, inference_window_steps

if TYPE_CHECKING:
    from sim.scene import Phase0Scene
    from sim.state_machine import PickPlaceStateMachine

logger = logging.getLogger(__name__)

# Per-frame series allowed in the JSON (no cameras / quats / full proprio).
JSON_TRACE_KEYS = (
    "timestamp",
    "stage",
    "holding",
    "object_pos",
    "object_vel",
    "ee_pos",
    "action_pos",
    "gripper",
    "conditioning",
    "policy_replan",
)
_JSON_NDIGITS = 4


def _json_leaf(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return round(v, _JSON_NDIGITS)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _json_array(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    if a.ndim == 0:
        return _json_leaf(a.item())
    return [_json_array(x) for x in a]


def episode_for_json(episode: dict[str, Any]) -> dict[str, Any]:
    """Lightweight metadata shared by collect and eval. No images."""
    arrays: dict[str, Any] = {}
    for k, v in episode.items():
        a = np.asarray(v)
        arrays[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}

    trace: dict[str, Any] = {}
    for k in JSON_TRACE_KEYS:
        if k not in episode:
            continue
        a = np.asarray(episode[k])
        if a.size == 0 or a.ndim >= 3:
            continue
        trace[k] = _json_array(a)

    vel = np.asarray(episode["object_vel"], dtype=np.float64) if "object_vel" in episode else None
    vel_xy_median = None
    if vel is not None and vel.size:
        spd = np.linalg.norm(vel.reshape(-1, vel.shape[-1])[:, :2], axis=1)
        moving = spd > 1e-4
        if np.any(moving):
            vel_xy_median = _json_leaf(float(np.median(spd[moving])))
        else:
            vel_xy_median = 0.0

    cond0 = None
    if "conditioning" in episode:
        c = np.asarray(episode["conditioning"])
        if c.ndim == 2 and c.shape[0] > 0:
            cond0 = _json_array(c[0])

    return {
        "conditioning_layout": list(CONDITIONING_LAYOUT),
        "conditioning_dim": int(CONDITIONING_DIM),
        "vel_xy_median": vel_xy_median,
        "cond0": cond0,
        "arrays": arrays,
        "trace": trace,
    }


def write_episode_h5(
    path: Any,
    episode: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the same episode arrays collect and eval log. Cond must be 4-D."""
    import h5py

    with h5py.File(path, "w") as fp:
        fp.attrs["conditioning_layout"] = ",".join(CONDITIONING_LAYOUT)
        fp.attrs["conditioning_dim"] = int(CONDITIONING_DIM)
        for k, v in episode.items():
            a = np.asarray(v)
            if a.size == 0:
                continue
            if k == "conditioning":
                if a.ndim != 2 or a.shape[1] != CONDITIONING_DIM:
                    raise ValueError(
                        f"conditioning must be (T, {CONDITIONING_DIM}), got {a.shape}"
                    )
            fp.create_dataset(k, data=a, compression="gzip")
        for k, v in (extra or {}).items():
            fp.create_dataset(k, data=np.asarray(v))


def run_episode(
    scene: Phase0Scene,
    expert: PickPlaceStateMachine,
    *,
    seed: int,
    speed: float,
    save_images: bool = False,
    delta: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one scripted episode. Returns (episode_arrays, meta).

    ``delta`` is the conditioning look-ahead for packing future XY — not the
    expert's aiming ``lookahead_s``. Default is ``conditioning_delta_s`` when
    present, else ``lookahead_s`` (getattr race-safe).
    """
    if delta is None:
        delta = float(getattr(expert, "conditioning_delta_s", expert.lookahead_s))
    meta = scene.reset_episode(seed=seed, speed=speed)
    meta["language"] = default_language(meta.get("category"), meta.get("container_category"))
    expert.reset()
    # Aim PLACE at container (or cfg fallback) for this episode.
    expert.set_box_poses(
        scene.get_carry_pose(),
        drop_pos=scene.get_drop_target(),
    )

    done_stage = int(getattr(expert, "stage_done", STAGE_DONE))
    n_steps = inference_window_steps(scene.cfg, scene.dt)
    state0 = scene.get_object_state()
    ee0, _ = scene.get_ee_pose()
    drop = scene.get_drop_target()
    log_episode_begin(
        logger,
        seed=seed,
        max_steps=n_steps,
        object_pos=state0.position,
        object_speed_m_s=float(np.linalg.norm(state0.velocity)),
        hand_pos=ee0,
        box_xy=scene.get_carry_pose()[:2],
        drop_z=float(drop[2]),
    )
    log: dict[str, list] = {
        "stage": [],
        "holding": [],
        "object_pos": [],
        "object_vel": [],
        "ee_pos": [],
        "ee_quat": [],
        "proprio": [],
        "action_pos": [],
        "action_quat": [],
        "gripper": [],
        "oracle_state": [],
        "conditioning": [],
        "timestamp": [],
        "container_pos": [],
    }
    if save_images:
        log["static_cam_rgb"] = []
        log["wrist_cam_rgb"] = []

    for t in range(n_steps):
        state = scene.get_object_state()
        ee_pos, ee_quat = scene.get_ee_pose()
        cmd = expert.step(state, ee_pos, ee_quat)

        scene.set_ee_target(cmd.position, cmd.quat_wxyz, cmd.gripper_open)
        scene.step(
            kinematic_object=cmd.kinematic_object,
            attach_object=cmd.attach_object,
            freeze_object=cmd.freeze_object,
        )

        # Log post-step state (world after action).
        state2 = scene.get_object_state()
        # Attached: zero vel for conditioning/logs (scene should already; belt+suspenders).
        if cmd.attach_object:
            state2.velocity = np.zeros(3, dtype=np.float64)
        ee_pos2, ee_quat2 = scene.get_ee_pose()
        cond = conditioning_vector(state2, delta=delta)

        log["stage"].append(cmd.stage)
        log["holding"].append(float(cmd.attach_object))
        log["object_pos"].append(state2.position.copy())
        log["object_vel"].append(state2.velocity.copy())
        log["ee_pos"].append(ee_pos2.copy())
        log["ee_quat"].append(ee_quat2.copy())
        log["proprio"].append(scene.get_proprio())
        log["action_pos"].append(cmd.position.copy())
        log["action_quat"].append(cmd.quat_wxyz.copy())
        log["gripper"].append(cmd.gripper_open)
        log["oracle_state"].append(state2.to_flat())
        log["conditioning"].append(cond)
        log["timestamp"].append(state2.timestamp)
        ctr = scene.get_container_pose()
        log["container_pos"].append(
            ctr[0].copy() if ctr is not None else np.full(3, np.nan)
        )

        if save_images:
            imgs = scene.get_images()
            if "static_cam" in imgs:
                log["static_cam_rgb"].append(imgs["static_cam"])
            if "wrist_cam" in imgs:
                log["wrist_cam_rgb"].append(imgs["wrist_cam"])

        # Early stop once retract-after-place completed (done stage = 9).
        if cmd.stage >= done_stage and t > 20:
            break

    episode = {k: np.asarray(v) for k, v in log.items() if len(v)}
    # Shape asserts at module boundary.
    t_len = episode["object_pos"].shape[0]
    assert episode["object_pos"].shape == (t_len, 3), episode["object_pos"].shape
    assert episode["conditioning"].shape == (t_len, CONDITIONING_DIM), episode[
        "conditioning"
    ].shape
    assert episode["oracle_state"].shape == (t_len, 14), episode["oracle_state"].shape
    meta["n_frames"] = int(t_len)
    meta["delta"] = float(delta)
    if hasattr(expert, "grasp_report"):
        meta.update(expert.grasp_report())
    return episode, meta
