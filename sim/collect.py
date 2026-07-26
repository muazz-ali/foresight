"""Episode collection loop for Phase 0 (oracle + conditioning logged)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from interfaces.state import conditioning_vector
from sim.expert import PickExpert
from sim.scene import Phase0Scene

logger = logging.getLogger(__name__)


def run_episode(
    scene: Phase0Scene,
    expert: PickExpert,
    *,
    seed: int,
    speed: float,
    save_images: bool = False,
    delta: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one scripted episode. Returns (episode_arrays, meta)."""
    meta = scene.reset_episode(seed=seed, speed=speed)
    expert.reset()

    n_steps = int(scene.cfg["expert"]["episode_steps"])
    log: dict[str, list] = {
        "sm_state": [],
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
        ee_pos2, ee_quat2 = scene.get_ee_pose()
        cond = conditioning_vector(state2, delta=delta)

        log["sm_state"].append(cmd.stage)
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

        if save_images:
            imgs = scene.get_images()
            if "static_cam" in imgs:
                log["static_cam_rgb"].append(imgs["static_cam"])
            if "wrist_cam" in imgs:
                log["wrist_cam_rgb"].append(imgs["wrist_cam"])

        # Early stop once place completed and a few settle frames.
        if cmd.stage >= 7 and t > 20:
            break

    episode = {k: np.asarray(v) for k, v in log.items() if len(v)}
    # Shape asserts at module boundary.
    t_len = episode["object_pos"].shape[0]
    assert episode["object_pos"].shape == (t_len, 3), episode["object_pos"].shape
    assert episode["conditioning"].shape == (t_len, 12), episode["conditioning"].shape
    assert episode["oracle_state"].shape == (t_len, 14), episode["oracle_state"].shape
    meta["n_frames"] = int(t_len)
    meta["delta"] = float(delta)
    return episode, meta
