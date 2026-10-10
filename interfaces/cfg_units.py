"""Human yaml units → SI for the sim.

Yaml knobs use millimetres (mm), milliseconds (ms), and centimetres/second (cm/s).
Every loader calls ``normalize_cfg_units`` so the rest of the code still sees
metres and seconds (Isaac / expert units).

Legacy metre / second / m/s keys in the yaml are rejected — rename them.
"""

from __future__ import annotations

from typing import Any

_MM = 1e-3  # mm → m
_MS = 1e-3  # ms → s
_CM = 1e-2  # cm/s → m/s


def _scale(val: Any, factor: float) -> Any:
    """Scale a number or a nested list of numbers."""
    if isinstance(val, (list, tuple)):
        return [_scale(v, factor) for v in val]
    return float(val) * factor


def _take(section: dict, human: str, si: str, factor: float) -> None:
    """Move ``human`` → ``si`` with a unit scale. Reject leftover SI-only keys."""
    if human in section:
        section[si] = _scale(section.pop(human), factor)
        return
    if si in section:
        raise ValueError(
            f"legacy key {si!r} is not supported; write {human!r} instead "
            f"(mm / ms / cm/s in the yaml)"
        )


# Old yaml knob → new yaml knob (human-facing names only).
_SM_RENAMES: dict[str, str] = {
    "conditioning_delta_ms": "policy_look_ahead_ms",
    "settle_ms": "wait_at_home_ms",
    "lookahead_ms": "max_aim_ahead_ms",
    "robot_approach_pad_ms": "tip_lag_pad_ms",
    "servo_lag_ms": "tip_lag_pad_ms",
    "servo_lag_s": "tip_lag_pad_ms",
    "ee_speed_max_cm_s": "hand_speed_max_cm_s",
    "intercept_iters": "aim_refine_iters",
    "hover_z_mm": "hover_above_object_mm",
    "z_release_radius_mm": "start_lowering_xy_mm",
    "track_xy_tol_mm": "riding_xy_ok_mm",
    "approach_z_tol_mm": "riding_z_ok_mm",
    "track_hold_ms": "riding_hold_ms",
    "bounce_guard_ms": "min_wait_after_bounce_ms",
    "wait_after_bounce_ms": "min_wait_after_bounce_ms",
    "min_clear_before_wall_ms": "min_wait_after_bounce_ms",
    "ee_settle_tol_cm_s": "object_still_cm_s",
    "grasp_xy_tol_mm": "grasp_xy_ok_mm",
    "grasp_z_tol_mm": "grasp_z_ok_mm",
    "grasp_tol_step_k": "grasp_tol_grows_k",
    "physics_grasp_z_tol_mm": "physics_grasp_z_ok_mm",
    "close_gripper_ms": "hold_closed_weld_ms",
    "physics_close_ms": "hold_closed_physics_ms",
    "close_depth_mm": "wrist_close_depth_mm",
    "slip_dist_mm": "slip_if_farther_mm",
    "servo_xy_tol_mm": "near_enough_xy_mm",
    "drop_z_gap_mm": "drop_above_rest_mm",
    "drop_rim_clear_mm": "drop_below_rim_mm",
    "place_xy_tol_mm": "in_box_xy_ok_mm",
    "home_z_tol_mm": "home_z_ok_mm",
    "inference_window_ms": "episode_max_ms",
    "stage_timeout_ms": "stall_timeout_ms",
    "grasp_stage_timeout_ms": "chase_stall_timeout_ms",
}


def _reject_renamed_sm_keys(sm: dict) -> None:
    """Fail fast if yaml still uses old knob names."""
    for old, new in _SM_RENAMES.items():
        if old in sm:
            raise ValueError(
                f"legacy key {old!r} renamed to {new!r} "
                f"(see state_machine_params in sim/phase0_cfg.yaml)"
            )


def _take_pose_mm(section: dict, human: str, si: str) -> None:
    """``[x_mm, y_mm, z_mm, qw, qx, qy, qz]`` → metres + unchanged quat."""
    if human in section:
        raw = list(section.pop(human))
        if len(raw) < 3:
            raise ValueError(f"{human} needs at least xyz; got {raw!r}")
        pose = [_scale(raw[0], _MM), _scale(raw[1], _MM), _scale(raw[2], _MM)]
        pose.extend(float(x) for x in raw[3:])
        section[si] = pose
        return
    if si in section:
        raise ValueError(
            f"legacy key {si!r} is not supported; write {human!r} "
            f"(xyz in mm, then quat)"
        )


def normalize_cfg_units(cfg: dict[str, Any]) -> dict[str, Any]:
    """Rewrite human-unit keys in-place to the SI keys the sim reads."""
    robot = cfg.get("robot")
    if isinstance(robot, dict):
        _take(robot, "tcp_offset_mm", "tcp_offset", _MM)
        _take(robot, "gripper_open_mm", "gripper_open", _MM)
        _take(robot, "gripper_close_mm", "gripper_close", _MM)
        _take_pose_mm(robot, "init_pose_mm", "init_pose")
        _take(robot, "max_reach_dist_mm", "max_reach_dist", _MM)
        _take(robot, "gripper_close_time_ms", "gripper_close_time_s", _MS)

    scene = cfg.get("scene")
    if isinstance(scene, dict):
        _take(scene, "table_height_mm", "table_height", _MM)
        _take(scene, "table_size_mm", "table_size", _MM)
        _take(scene, "table_pos_mm", "table_pos", _MM)
        _take(scene, "object_radius_mm", "object_radius", _MM)
        _take(scene, "place_hover_above_container_mm", "place_hover_above_container", _MM)
        _take(scene, "spawn_xy_pad_mm", "spawn_xy_pad", _MM)
        _take(scene, "container_jitter_xy_mm", "container_jitter_xy", _MM)
        _take(scene, "container_xy_clip_mm", "container_xy_clip", _MM)
        _take(scene, "attach_z_offset_max_mm", "attach_z_offset_max", _MM)
        _take(scene, "object_xy_bounds_mm", "object_xy_bounds", _MM)
        _take(scene, "max_object_width_mm", "max_object_width_m", _MM)
        _take(scene, "min_container_clearance_mm", "min_container_clearance_m", _MM)
        _take(scene, "min_container_xy_margin_mm", "min_container_xy_margin_m", _MM)
        _take(scene, "robot_base_keepout_mm", "robot_base_keepout_m", _MM)
        _take(scene, "ee_table_margin_mm", "ee_table_margin_m", _MM)
        _take(scene, "container_xy_mm", "container_xy", _MM)
        if "cameras" in scene:
            raise ValueError(
                "legacy key scene.cameras removed; put pose + lens under "
                "top-level 'static_camera' (see sim/phase0_cfg.yaml)"
            )

    if "camera" in cfg:
        raise ValueError(
            "legacy key 'camera' split into top-level 'static_camera' and "
            "'wrist_camera' (see sim/phase0_cfg.yaml)"
        )

    static = cfg.get("static_camera")
    if isinstance(static, dict):
        _take(static, "position_mm", "position", _MM)
        clip = static.get("clip")
        if isinstance(clip, dict):
            _take(clip, "near_mm", "near", _MM)
            _take(clip, "far_mm", "far", _MM)

    wrist = cfg.get("wrist_camera")
    if isinstance(wrist, dict):
        _take(wrist, "pos_mm", "pos", _MM)
        _take(wrist, "clipping_range_mm", "clipping_range", _MM)

    lighting = cfg.get("lighting")
    if isinstance(lighting, dict):
        _take(lighting, "behind_mm", "behind_m", _MM)
        _take(lighting, "above_mm", "above_m", _MM)
        _take(lighting, "side_mm", "side_m", _MM)

    sm = cfg.get("state_machine_params")
    if isinstance(sm, dict):
        # Fail first if yaml still uses old human names.
        _reject_renamed_sm_keys(sm)
        # times (clear yaml name → SI key the expert still reads)
        for human, si in (
            ("policy_look_ahead_ms", "conditioning_delta_s"),
            ("wait_at_home_ms", "settle_s"),
            ("max_aim_ahead_ms", "lookahead_s"),
            ("tip_lag_pad_ms", "robot_approach_pad_s"),
            ("riding_hold_ms", "track_hold_s"),
            ("min_wait_after_bounce_ms", "bounce_guard_s"),
            ("hold_closed_weld_ms", "close_gripper_s"),
            ("hold_closed_physics_ms", "physics_close_s"),
            ("episode_max_ms", "inference_window_s"),
            ("stall_timeout_ms", "stage_timeout_s"),
            ("chase_stall_timeout_ms", "grasp_stage_timeout_s"),
        ):
            _take(sm, human, si, _MS)
        # lengths
        for human, si in (
            ("hover_above_object_mm", "hover_z"),
            ("start_lowering_xy_mm", "z_release_radius"),
            ("riding_xy_ok_mm", "track_xy_tol"),
            ("riding_z_ok_mm", "approach_z_tol"),
            ("grasp_xy_ok_mm", "grasp_xy_tol"),
            ("grasp_z_ok_mm", "grasp_z_tol"),
            ("physics_grasp_z_ok_mm", "physics_grasp_z_tol"),
            ("wrist_close_depth_mm", "close_depth_m"),
            ("lift_height_mm", "lift_height"),
            ("slip_if_farther_mm", "slip_dist_m"),
            ("near_enough_xy_mm", "servo_xy_tol"),
            ("drop_above_rest_mm", "drop_z_gap"),
            ("drop_below_rim_mm", "drop_rim_clear_m"),
            ("in_box_xy_ok_mm", "place_xy_tol"),
            ("home_z_ok_mm", "home_z_tol"),
        ):
            _take(sm, human, si, _MM)
        # speeds (cm/s → m/s)
        _take(sm, "object_still_cm_s", "ee_settle_tol", _CM)
        _take(sm, "hand_speed_max_cm_s", "ee_speed_max", _CM)
        # unitless / ints (rename in place; SI name kept for the expert)
        if "aim_refine_iters" in sm:
            sm["intercept_iters"] = int(sm.pop("aim_refine_iters"))
        elif "intercept_iters" not in sm:
            pass  # optional; expert defaults to 3
        if "grasp_tol_grows_k" in sm:
            sm["grasp_tol_step_k"] = float(sm.pop("grasp_tol_grows_k"))
        elif "grasp_tol_step_k" in sm:
            raise ValueError(
                "legacy key 'grasp_tol_step_k' renamed to 'grasp_tol_grows_k' "
                "(see state_machine_params in sim/phase0_cfg.yaml)"
            )

    sim = cfg.get("sim")
    if isinstance(sim, dict):
        _take(sim, "dt_ms", "dt", _MS)

    return cfg
