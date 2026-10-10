"""Human mm/ms/cm/s yaml → SI metres/seconds for the sim."""

from __future__ import annotations

import copy

import pytest

from interfaces.cfg_units import normalize_cfg_units
from interfaces.config import load_yaml, DEFAULT_SIM_CFG


def test_normalize_converts_mm_ms_cm_s():
    raw = {
        "robot": {
            "gripper_open_mm": 40.0,
            "gripper_close_time_ms": 80,
            "init_pose_mm": [450.0, 0.0, 400.0, 0.0, 1.0, 0.0, 0.0],
            "tcp_offset_mm": [0.0, 0.0, 107.0],
        },
        "scene": {
            "max_object_width_mm": 75.0,
            "min_container_clearance_mm": 10.0,
            "container_xy_mm": [500.0, 300.0],
            "object_xy_bounds_mm": [[480.0, 720.0], [-200.0, 100.0]],
        },
        "static_camera": {
            "position_mm": [1000.0, 0.0, 600.0],
            "clip": {"near_mm": 10.0, "far_mm": 100000.0},
        },
        "wrist_camera": {
            "pos_mm": [65.0, 0.0, 0.0],
            "clipping_range_mm": [10.0, 2000.0],
        },
        "state_machine_params": {
            "hover_above_object_mm": 80.0,
            "riding_hold_ms": 80,
            "hand_speed_max_cm_s": 90.0,
            "object_still_cm_s": 5.0,
            "physics_grasp_z_ok_mm": 8.0,
            "hold_closed_physics_ms": 300,
            "aim_refine_iters": 3,
            "grasp_tol_grows_k": 0.6,
        },
        "sim": {"dt_ms": 40},
    }
    cfg = normalize_cfg_units(copy.deepcopy(raw))
    assert cfg["robot"]["gripper_open"] == 0.04
    assert cfg["robot"]["gripper_close_time_s"] == 0.08
    assert cfg["robot"]["init_pose"][:3] == [0.45, 0.0, 0.4]
    assert cfg["robot"]["init_pose"][3:] == [0.0, 1.0, 0.0, 0.0]
    assert cfg["robot"]["tcp_offset"] == [0.0, 0.0, 0.107]
    assert cfg["scene"]["max_object_width_m"] == 0.075
    assert cfg["scene"]["min_container_clearance_m"] == 0.01
    assert cfg["scene"]["container_xy"] == [0.5, 0.3]
    assert cfg["scene"]["object_xy_bounds"] == [[0.48, 0.72], [-0.2, 0.1]]
    assert cfg["static_camera"]["position"] == [1.0, 0.0, 0.6]
    assert cfg["static_camera"]["clip"]["near"] == 0.01
    assert cfg["static_camera"]["clip"]["far"] == 100.0
    assert cfg["wrist_camera"]["pos"] == [0.065, 0.0, 0.0]
    assert cfg["wrist_camera"]["clipping_range"] == [0.01, 2.0]
    assert cfg["state_machine_params"]["hover_z"] == 0.08
    assert cfg["state_machine_params"]["track_hold_s"] == 0.08
    assert cfg["state_machine_params"]["ee_speed_max"] == 0.9
    assert cfg["state_machine_params"]["ee_settle_tol"] == 0.05
    assert cfg["state_machine_params"]["physics_grasp_z_tol"] == 0.008
    assert cfg["state_machine_params"]["physics_close_s"] == 0.3
    assert cfg["state_machine_params"]["intercept_iters"] == 3
    assert cfg["state_machine_params"]["grasp_tol_step_k"] == 0.6
    assert cfg["sim"]["dt"] == 0.04
    assert "gripper_open_mm" not in cfg["robot"]
    assert "hand_speed_max_cm_s" not in cfg["state_machine_params"]
    assert "dt_ms" not in cfg["sim"]


def test_legacy_si_keys_are_rejected():
    with pytest.raises(ValueError, match="legacy key"):
        normalize_cfg_units({"robot": {"gripper_open": 0.04}})
    with pytest.raises(ValueError, match="legacy key"):
        normalize_cfg_units({"sim": {"dt": 0.04}})
    with pytest.raises(ValueError, match="legacy key"):
        normalize_cfg_units({"state_machine_params": {"ee_speed_max": 0.9}})


def test_legacy_human_knob_names_are_rejected():
    with pytest.raises(ValueError, match="min_wait_after_bounce_ms"):
        normalize_cfg_units({"state_machine_params": {"bounce_guard_ms": 100}})
    with pytest.raises(ValueError, match="min_wait_after_bounce_ms"):
        normalize_cfg_units({"state_machine_params": {"wait_after_bounce_ms": 100}})
    with pytest.raises(ValueError, match="min_wait_after_bounce_ms"):
        normalize_cfg_units({"state_machine_params": {"min_clear_before_wall_ms": 100}})
    with pytest.raises(ValueError, match="tip_lag_pad_ms"):
        normalize_cfg_units({"state_machine_params": {"robot_approach_pad_ms": 160}})


def test_phase0_yaml_loads_to_expected_si():
    cfg = load_yaml(DEFAULT_SIM_CFG)
    assert cfg["robot"]["init_pose"][:3] == [0.45, 0.0, 0.4]
    assert abs(cfg["robot"]["gripper_close_time_s"] - 0.08) < 1e-12
    assert abs(cfg["scene"]["max_object_width_m"] - 0.075) < 1e-12
    assert abs(cfg["state_machine_params"]["hover_z"] - 0.08) < 1e-12
    assert abs(cfg["state_machine_params"]["ee_speed_max"] - 0.9) < 1e-12
    assert abs(cfg["state_machine_params"]["ee_settle_tol"] - 0.05) < 1e-12
    assert abs(cfg["state_machine_params"]["bounce_guard_s"] - 0.10) < 1e-12
    assert abs(cfg["state_machine_params"]["conditioning_delta_s"] - 0.25) < 1e-12
    assert cfg["state_machine_params"]["intercept_iters"] == 3
    assert abs(cfg["sim"]["dt"] - 0.04) < 1e-12
    assert "camera" not in cfg
    assert "cameras" not in cfg["scene"]
    assert cfg["static_camera"]["position"] == [1.0, 0.0, 0.6]
    assert cfg["static_camera"]["clip"]["near"] == 0.01
    assert cfg["wrist_camera"]["pos"] == [0.065, 0.0, 0.0]
    assert cfg["wrist_camera"]["clipping_range"] == [0.01, 2.0]


def test_legacy_camera_sections_are_rejected():
    with pytest.raises(ValueError, match="static_camera"):
        normalize_cfg_units({"camera": {"fps": 25}})
    with pytest.raises(ValueError, match="static_camera"):
        normalize_cfg_units({"scene": {"cameras": [{"name": "static_cam"}]}})
