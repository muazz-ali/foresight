"""Episode recording loop (scripted expert + sim-truth future numbers).

What it does: runs ONE episode — expert picks, scene steps, we log every frame —
and writes the result to JSON (light) and H5 (full arrays).
It runs from scripts/run_phase0.py (G0 score) and scripts/run_phase1_collect.py
(training demos). Policy eval uses the same writers so the files match.
It must NOT judge success or label failures (sim/success.py does that), and it
must NOT use DynamicVLA code.
"""

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
# Keeps the JSON small enough to skim; the H5 has everything.
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
    "truth_pack",
    "filter_pack",
    "filter_valid",
    "filter_redetect_counts",
    "policy_replan",
)
_JSON_NDIGITS = 4  # 0.1 mm precision is plenty for a skim file


def _json_leaf(x: Any) -> Any:
    """Turn one numpy number into plain JSON: round floats, NaN/inf → null."""
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
    """Turn a numpy array of any depth into nested lists of plain numbers."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return _json_leaf(a.item())
    return [_json_array(x) for x in a]


def episode_for_json(episode: dict[str, Any]) -> dict[str, Any]:
    """Human-readable episode summary for debugging. No images.

    Writes one row per frame (stage name, mm positions, gap, speed) instead of
    long parallel number columns. Full arrays stay in the ``.h5``.
    """
    from sim.state_machine_logs import stage_label

    arrays: dict[str, Any] = {}
    for k, v in episode.items():
        a = np.asarray(v)
        arrays[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}

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

    # Row-per-frame trace (easy to skim in an editor).
    n = 0
    for k in ("timestamp", "stage", "object_pos", "ee_pos"):
        if k in episode:
            n = max(n, int(np.asarray(episode[k]).shape[0]))
    frames: list[dict[str, Any]] = []
    highlights: list[dict[str, Any]] = []
    prev_stage = None
    for t in range(n):
        row: dict[str, Any] = {"i": t}
        if "timestamp" in episode:
            row["t_s"] = _json_leaf(float(np.asarray(episode["timestamp"])[t]))
        stage_i = None
        if "stage" in episode:
            stage_i = int(episode["stage"][t])
            row["stage"] = stage_label(stage_i)
        if "holding" in episode:
            row["holding"] = bool(float(episode["holding"][t]) > 0.5)
        if "ee_pos" in episode:
            ee = np.asarray(episode["ee_pos"][t], dtype=np.float64).reshape(-1)
            row["hand_mm"] = [int(round(1000.0 * float(x))) for x in ee[:3]]
        if "object_pos" in episode:
            obj = np.asarray(episode["object_pos"][t], dtype=np.float64).reshape(-1)
            row["obj_mm"] = [int(round(1000.0 * float(x))) for x in obj[:3]]
        if "ee_pos" in episode and "object_pos" in episode:
            ee = np.asarray(episode["ee_pos"][t], dtype=np.float64).reshape(-1)
            obj = np.asarray(episode["object_pos"][t], dtype=np.float64).reshape(-1)
            gap = obj[:3] - ee[:3]
            row["gap_mm"] = [int(round(1000.0 * float(x))) for x in gap]
        if "object_vel" in episode:
            v = np.asarray(episode["object_vel"][t], dtype=np.float64).reshape(-1)
            row["speed_cm_s"] = _json_leaf(100.0 * float(np.linalg.norm(v[:3])))
        if "gripper" in episode:
            row["grip_open"] = _json_leaf(float(episode["gripper"][t]))
        if "conditioning" in episode:
            c = np.asarray(episode["conditioning"][t], dtype=np.float64).reshape(-1)
            row["pack"] = _json_array(c[:CONDITIONING_DIM])
        frames.append(row)
        if stage_i is not None and stage_i != prev_stage:
            highlights.append(
                {
                    "i": t,
                    "t_s": row.get("t_s"),
                    "event": f"stage → {row.get('stage')}",
                }
            )
            prev_stage = stage_i
        if "object_pos" in episode and t > 0:
            z = float(np.asarray(episode["object_pos"][t])[2])
            z_prev = float(np.asarray(episode["object_pos"][t - 1])[2])
            if z_prev >= -0.01 and z < -0.02:
                highlights.append(
                    {"i": t, "t_s": row.get("t_s"), "event": "object dropped below table"}
                )

    return {
        "conditioning_layout": list(CONDITIONING_LAYOUT),
        "conditioning_dim": int(CONDITIONING_DIM),
        "vel_xy_median_m_s": vel_xy_median,
        "cond0": cond0,
        "arrays": arrays,
        "highlights": highlights,
        "frames": frames,
    }


def write_episode_h5(
    path: Any,
    episode: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the same episode arrays collect and eval log. Cond must be 4-D.

    One dataset per array, gzip-packed. ``extra`` adds small per-episode items
    (speed, seed, success, delta). Wrong-shaped policy numbers raise an error
    here, so bad data never lands on disk.
    """
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

    Each step: read true state → expert gives hand orders → scene moves → log
    the world AFTER the move. Stops at the "done" stage or the step cap.
    ``seed`` fixes the episode so it can be replayed. This function does not
    say if the run succeeded; the caller does that from the logged arrays.
    """
    if delta is None:
        delta = float(getattr(expert, "conditioning_delta_s", expert.lookahead_s))
    meta = scene.reset_episode(seed=seed, speed=speed)
    meta["language"] = default_language(meta.get("category"), meta.get("container_category"))
    expert.reset()
    # Tell the expert where this episode's box is, so carry / lower aim at it.
    # (Aim PLACE at container, or cfg fallback.)
    expert.set_box_poses(
        scene.get_carry_pose(),
        drop_pos=scene.get_drop_target(),
    )

    done_stage = int(getattr(expert, "stage_done", STAGE_DONE))
    # Hard cap on steps from the yaml, so a stuck run still ends.
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
    # One list per signal, one entry per step. Turned into arrays at the end.
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
        "wrist_depth_m": [],
    }
    if save_images:
        log["static_cam_rgb"] = []
        log["wrist_cam_rgb"] = []
        log["wrist_cam_depth"] = []

    # Phase 3 timing: last-cm reflex (opt-in). Default off until catch rate is
    # proven with it on; set state_machine_params.use_reflex: true to enable.
    reflex = None
    sm_cfg = scene.cfg.get("state_machine_params", {}) if isinstance(scene.cfg, dict) else {}
    want_reflex = bool(sm_cfg.get("use_reflex", False)) and bool(
        getattr(expert, "physics_grasp", False)
    )
    if want_reflex:
        from control import ReflexClose

        # close_time = how long the fingers take to reach the object (measured,
        # robot.gripper_close_time_s). NOT expert.close_gripper_s: that is a dwell.
        reflex = ReflexClose(
            close_time=float(scene.cfg["robot"]["gripper_close_time_s"]),
            radius=0.10,
            vmax=float(getattr(expert, "ee_speed_max", 0.5)),
        )
        reflex.reset()
    meta["reflex"] = bool(reflex is not None)

    # Main loop: expert looks → scene acts → we write down what happened.
    for t in range(n_steps):
        state = scene.get_object_state()
        ee_pos, ee_quat = scene.get_ee_pose()
        # Wrist depth (m) in front of the hand; NaN = no reading. Logged every step.
        depth_m = scene.get_wrist_depth_m() if "wrist_cam" in scene.cameras else float("nan")
        cmd = expert.step(state, ee_pos, ee_quat, wrist_depth_m=depth_m)
        if reflex is not None:
            from control import overlay_reflex

            cmd, _ = overlay_reflex(cmd, reflex, state, ee_pos)

        scene.set_ee_target(cmd.position, cmd.quat_wxyz, cmd.gripper_open)
        scene.step(
            kinematic_object=cmd.kinematic_object,
            attach_object=cmd.attach_object,
            freeze_object=cmd.freeze_object,
        )

        # Log post-step state (world after action).
        state2 = scene.get_object_state()
        # Weld only: a welded object has no table speed (scene already zeroes it; belt+suspenders).
        # Physics grasp: keep the real velocity, so a slipping object shows up in the logs.
        if cmd.attach_object and scene.weld_grasp:
            state2.velocity = np.zeros(3, dtype=np.float64)
        ee_pos2, ee_quat2 = scene.get_ee_pose()
        # The 4 numbers the policy will see: future XY spot + XY speed.
        # Built from sim truth here; same helper as at eval, so train = test.
        cond = conditioning_vector(state2, delta=delta)

        log["stage"].append(cmd.stage)
        log["holding"].append(float(cmd.attach_object))  # "grasp commanded" (stages 3-6), not contact
        log["wrist_depth_m"].append(depth_m)
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
        # NaN (not 0) when there is no box, so "no box" cannot pass for a real spot.
        log["container_pos"].append(
            ctr[0].copy() if ctr is not None else np.full(3, np.nan)
        )

        if save_images:
            imgs = scene.get_images()
            if "static_cam" in imgs:
                log["static_cam_rgb"].append(imgs["static_cam"])
            if "wrist_cam" in imgs:
                log["wrist_cam_rgb"].append(imgs["wrist_cam"])
                log["wrist_cam_depth"].append(scene.get_wrist_depth().astype(np.float16))

        # Early stop once retract-after-place completed (done stage = 9).
        # t > 20: never stop in the first few steps, even if stage reads "done".
        if cmd.stage >= done_stage and t > 20:
            break
        # Miss: object left the table — do not keep chasing under the floor.
        if scene.off_table:
            logger.info("  stop: object off the table")
            break
        # Safety: hand left the table, or coast path went through the box / base.
        abort = scene.check_safety_abort(attached=bool(cmd.attach_object))
        if abort:
            logger.info("  stop: safety abort (%s)", abort)
            break

    episode = {k: np.asarray(v) for k, v in log.items() if len(v)}
    # Shape asserts at module boundary. 14 = flat ObjectState:
    # pos 3 + vel 3 + covariance diagonal 6 + timestamp 1 + valid 1.
    t_len = episode["object_pos"].shape[0]
    assert episode["object_pos"].shape == (t_len, 3), episode["object_pos"].shape
    assert episode["conditioning"].shape == (t_len, CONDITIONING_DIM), episode[
        "conditioning"
    ].shape
    assert episode["oracle_state"].shape == (t_len, 14), episode["oracle_state"].shape
    meta["n_frames"] = int(t_len)
    meta["off_table"] = bool(scene.off_table)  # object left the table footprint (a real miss)
    meta["abort_reason"] = getattr(scene, "abort_reason", None)
    meta["delta"] = float(delta)
    if hasattr(expert, "grasp_report"):
        meta.update(expert.grasp_report())
    return episode, meta
