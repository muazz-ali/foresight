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
    meta["language"] = default_language(
        meta.get("category"),
        meta.get("container_category"),
        seed=int(seed),
    )
    expert.reset()
    # Real mesh half-height + table Z → close near object centre (not in the air).
    hh = meta.get("object_half_height")
    if hh is not None:
        expert.set_object_half_height(
            float(hh),
            z_table=float(getattr(scene, "_table_z", meta.get("table_z", 0.0))),
        )
    # Tell the expert where this episode's box is, so carry / lower aim at it.
    # (Aim PLACE at container, or cfg fallback.)
    expert.set_box_poses(
        scene.get_carry_pose(),
        drop_pos=scene.get_drop_target(),
    )
    # Clip intercept aims inside the table, inset by the ee-off-table margin,
    # and outside the robot-base keepout so DiffIK cannot overshoot into abort.
    # Escape/catch box stays the full table (keepout on −X only) so a far-rim
    # spawn is still chaseable — aim inset alone made lemon seed-50194 go home
    # on frame 1 ("left aim box" at x just past inset xmax).
    if hasattr(expert, "set_aim_xy_bounds"):
        table_box = getattr(scene, "_table_box", None)
        if table_box is not None:
            lo = np.asarray(table_box[0], dtype=np.float64).copy()
            hi = np.asarray(table_box[1], dtype=np.float64).copy()
            margin = float(getattr(scene, "_ee_table_margin_m", 0.02))
            keepout = float(getattr(scene, "_robot_base_keepout_m", 0.12))
            esc_lo = lo.copy()
            esc_hi = hi.copy()
            esc_lo[0] = max(float(esc_lo[0]), keepout)
            aim_lo = lo + margin
            aim_hi = hi - margin
            aim_lo[0] = max(float(aim_lo[0]), keepout)
            expert.set_aim_xy_bounds(
                [[float(aim_lo[0]), float(aim_hi[0])], [float(aim_lo[1]), float(aim_hi[1])]]
            )
            if hasattr(expert, "set_escape_xy_bounds"):
                expert.set_escape_xy_bounds(
                    [
                        [float(esc_lo[0]), float(esc_hi[0])],
                        [float(esc_lo[1]), float(esc_hi[1])],
                    ]
                )
        else:
            expert.set_aim_xy_bounds(None)
            if hasattr(expert, "set_escape_xy_bounds"):
                expert.set_escape_xy_bounds(None)

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
        # Step 1 claim timing (filled only when claim_timing.enabled).
        "capture_t0": [],
        "applied_delta_s": [],
        "reported_delta_s": [],
    }
    if save_images:
        log["static_cam_rgb"] = []
        log["wrist_cam_rgb"] = []
        log["wrist_cam_depth"] = []
        log["static_cam_depth"] = []
        log["wrist_cam_z"] = []

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

    # Step 1: delay injection. When enabled, log t0 / applied / reported Δ and
    # (if enforce_apply_at) hold motions until apply_at = t0 + applied_Δ.
    delay = None
    ct = scene.cfg.get("claim_timing", {}) if isinstance(scene.cfg, dict) else {}
    ct = ct if isinstance(ct, dict) else {}
    if bool(ct.get("enabled", False)):
        from control import DelayInjection

        delay = DelayInjection.from_cfg(scene.cfg, seed=int(seed))
    enforce_apply = bool(delay is not None) and bool(ct.get("enforce_apply_at", True))
    use_noisy = bool(ct.get("use_noisy_gt", False))
    close_guard = bool(ct.get("close_speed_guard", False))
    meta["claim_timing"] = bool(delay is not None)
    meta["enforce_apply_at"] = bool(enforce_apply)
    meta["use_noisy_gt"] = bool(use_noisy)
    meta["close_speed_guard"] = bool(close_guard)

    noisy_filt = None
    if use_noisy:
        from perception.noisy_gt import WindowedCVFilter

        noisy_filt = WindowedCVFilter(dt=float(scene.dt))

    from sim.state_machine import STAGE_CLOSE

    # Pending expert cmds waiting for apply_at; held = last released cmd.
    pending: list[tuple[float, Any, Any]] = []
    held_cmd = None
    prev_ee = None

    # Main loop: expert looks → (optional delay) → scene acts → log.
    for t in range(n_steps):
        state = scene.get_object_state()
        ee_pos, ee_quat = scene.get_ee_pose()
        depth_m = scene.get_wrist_depth_m() if "wrist_cam" in scene.cameras else float("nan")
        depth_img = scene.get_wrist_depth() if "wrist_cam" in scene.cameras else None
        static_depth = scene.get_static_depth() if "static_cam" in scene.cameras else None
        static_z = scene.static_cam_z() if static_depth is not None else None
        wrist_pose = (
            scene.get_wrist_cam_pose() if "wrist_cam" in scene.cameras else None
        )
        wrist_cam_z = float(wrist_pose[0][2]) if wrist_pose is not None else None
        planned = expert.step(
            state,
            ee_pos,
            ee_quat,
            wrist_depth_m=depth_m,
            wrist_depth_img=depth_img,
            static_depth_img=static_depth,
            static_cam_z=static_z,
            wrist_cam_z=wrist_cam_z,
        )
        if reflex is not None:
            from control import overlay_reflex

            planned, _ = overlay_reflex(planned, reflex, state, ee_pos)

        # Optional close-speed: if hand and object slide apart, ride object XY.
        if close_guard and int(planned.stage) == int(STAGE_CLOSE):
            from control import close_speed_check

            dt = float(scene.dt)
            if prev_ee is None:
                v_hand = np.asarray(state.velocity, dtype=np.float64)
            else:
                v_hand = (np.asarray(ee_pos, dtype=np.float64) - prev_ee) / max(dt, 1e-6)
            chk = close_speed_check(v_hand, state.velocity)
            if not chk.ok:
                pos = np.asarray(planned.position, dtype=np.float64).copy()
                pos[0:2] = np.asarray(state.position, dtype=np.float64)[0:2]
                planned.position = pos

        # Obs stamp for this plan: last camera capture (or current sim time).
        t0 = float(
            getattr(scene, "last_capture_t0", getattr(scene, "_sim_time", state.timestamp))
        )
        pack_delta = float(delta)
        applied_s = float("nan")
        reported_s = float("nan")
        sample = None
        if delay is not None:
            sample = delay.sample(t0, mode="train")
            pack_delta = float(sample.reported_delta_s)
            applied_s = float(sample.applied_delta_s)
            reported_s = float(sample.reported_delta_s)
            if enforce_apply:
                pending.append((float(sample.apply_at), planned, sample))
            else:
                held_cmd = planned
        else:
            held_cmd = planned

        now = float(getattr(scene, "_sim_time", t0))
        if enforce_apply and pending:
            still: list[tuple[float, Any, Any]] = []
            for apply_at, cmd_i, samp_i in pending:
                if now + 1e-9 >= float(apply_at):
                    held_cmd = cmd_i
                else:
                    still.append((apply_at, cmd_i, samp_i))
            pending = still
        if held_cmd is None:
            held_cmd = planned

        cmd = held_cmd
        scene.set_ee_target(cmd.position, cmd.quat_wxyz, cmd.gripper_open)
        scene.step(
            kinematic_object=cmd.kinematic_object,
            attach_object=cmd.attach_object,
            freeze_object=cmd.freeze_object,
        )
        prev_ee = np.asarray(ee_pos, dtype=np.float64).copy()

        # Log post-step state (world after action).
        state2 = scene.get_object_state()
        if cmd.attach_object and scene.weld_grasp:
            state2.velocity = np.zeros(3, dtype=np.float64)
        ee_pos2, ee_quat2 = scene.get_ee_pose()
        t0_log = float(getattr(scene, "last_capture_t0", state2.timestamp))

        pack_state = state2
        if noisy_filt is not None:
            from perception.noisy_gt import add_gt_noise, measure_noise_std

            rng = np.random.default_rng(int(seed) + t)
            std = measure_noise_std(ee_pos2, state2.position)
            noisy = add_gt_noise(state2, rng, pos_std=std)
            filt_st = noisy_filt.step(
                float(state2.timestamp),
                noisy.position[:2],
                plane_z=float(state2.position[2]),
            )
            if filt_st is not None:
                pack_state = filt_st

        # Pack height: depth latch (p_hat_z) when enabled; else expert aim z_grasp.
        p_hat_z = getattr(expert, "p_hat_z", None)
        if p_hat_z is None:
            p_hat_z = getattr(expert, "z_grasp", None)
        cond = conditioning_vector(pack_state, delta=pack_delta, p_hat_z=p_hat_z)

        log["stage"].append(cmd.stage)
        log["holding"].append(float(cmd.attach_object))
        log["wrist_depth_m"].append(depth_m)
        log["object_pos"].append(state2.position.copy())
        log["object_vel"].append(state2.velocity.copy())
        log["ee_pos"].append(ee_pos2.copy())
        log["ee_quat"].append(ee_quat2.copy())
        log["proprio"].append(scene.get_proprio())
        # Log the *planned* action (what the expert wanted), not the delayed hold.
        log["action_pos"].append(planned.position.copy())
        log["action_quat"].append(planned.quat_wxyz.copy())
        log["gripper"].append(planned.gripper_open)
        log["oracle_state"].append(state2.to_flat())
        log["conditioning"].append(cond)
        log["timestamp"].append(state2.timestamp)
        log["capture_t0"].append(t0_log)
        log["applied_delta_s"].append(applied_s)
        log["reported_delta_s"].append(reported_s)
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
            wd = scene.get_wrist_depth()
            if wd is not None and "wrist_cam_depth" in log:
                log["wrist_cam_depth"].append(wd.astype(np.float16))
            sd = scene.get_static_depth()
            if sd is not None and "static_cam_depth" in log:
                log["static_cam_depth"].append(sd.astype(np.float16))
            if "wrist_cam_z" in log:
                log["wrist_cam_z"].append(
                    float(wrist_cam_z) if wrist_cam_z is not None else float("nan")
                )

        if planned.stage >= done_stage and t > 20:
            break
        if scene.off_table:
            logger.info("  stop: object off the table")
            break
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
