"""SI yaml loads as-is; train/eval delay folds into claim_timing."""

from __future__ import annotations

import pytest

from interfaces.config import load_yaml, DEFAULT_SIM_CFG, FORESIGHT_ROOT


def test_phase0_yaml_loads_si_keys():
    cfg = load_yaml(DEFAULT_SIM_CFG)
    assert cfg["robot"]["init_pose"][:3] == pytest.approx([0.45, 0.0, 0.4])
    assert cfg["robot"]["gripper_open"] == pytest.approx(0.04)
    assert cfg["robot"]["gripper_close_time_s"] == pytest.approx(0.180)
    assert cfg["robot"]["tcp_offset"] == pytest.approx([0.0, 0.0, 0.1034])
    ga = cfg["robot"]["gripper_actuator"]
    assert ga["v_finger_m_s"] == pytest.approx(0.05)
    assert ga["t_cmd_s"] == pytest.approx(0.030)
    assert ga["t_force_s"] == pytest.approx(0.050)
    assert cfg["scene"]["max_object_width_m"] == pytest.approx(0.070)
    assert cfg["scene"]["min_aim_z_above_table_m"] == pytest.approx(0.015)
    assert cfg["state_machine_params"]["grasp_xyz_ok_m"] == pytest.approx(
        [0.01, 0.01, 0.012]
    )
    assert cfg["state_machine_params"]["close_xy_ok_m"] == pytest.approx(0.002)
    assert cfg["state_machine_params"]["robot_speed_max_m_s"] == pytest.approx(2.5)
    assert cfg["state_machine_params"]["max_aim_ahead_s"] == pytest.approx(0.40)
    assert "policy_look_ahead_s" not in cfg["state_machine_params"]
    assert cfg["state_machine_params"]["home_radius_m"] == pytest.approx(0.10)
    assert cfg["state_machine_params"]["move_up_above_container_m"] == pytest.approx(0.05)
    assert cfg["state_machine_params"]["release_below_rim_m"] == pytest.approx(0.01)
    assert cfg["state_machine_params"]["episode_max_s"] == pytest.approx(8.0)
    assert cfg["state_machine_params"]["aim_refine_iters"] == 3
    assert cfg["state_machine_params"]["robot_base_safe_zone_m"] == pytest.approx(0.10)
    assert cfg["sim"]["dt"] == pytest.approx(1.0 / 30.0)
    assert cfg["claim_timing"]["delay_enabled"] is True
    assert cfg["claim_timing"]["min_delta_delay_s"] == pytest.approx(0.0)
    assert cfg["claim_timing"]["max_delta_delay_s"] == pytest.approx(0.200)
    assert cfg["claim_timing"]["apply_delta_delay_to_robot"] is False
    assert cfg["claim_timing"]["enabled_noise_on_replan"] is False
    assert cfg["claim_timing"]["replan_meas_noise_s"] == pytest.approx(0.330)
    assert "policy_training_delay" not in cfg
    assert "evaluation" not in cfg
    ch = cfg["claim_height"]
    assert ch.get("use_object_height_from_depth", True) is True
    assert ch["table_clear_m"] == pytest.approx(0.005)
    assert ch["z_slack_m"] == pytest.approx(0.005)
    assert ch["stop_trusting_depth_below_m"] == pytest.approx(0.07)
    assert ch["use_depth_below_m"] == pytest.approx(0.20)
    assert ch["depth_rmse_max_m"] == pytest.approx(0.01)
    assert ch["depth_z_bias_m"] == pytest.approx(0.0)
    assert cfg["static_camera"]["position"] == pytest.approx([1.0, 0.0, 0.7])
    assert cfg["wrist_camera"]["pos"] == pytest.approx([0.065, 0.0, 0.0])
    assert cfg["wrist_camera"]["clipping_range"] == pytest.approx([0.01, 2.0])
    assert "dt_ms" not in cfg["sim"]
    assert "gripper_open_mm" not in cfg["robot"]
    assert "max_aim_ahead_ms" not in cfg["state_machine_params"]
    assert "robot_speed_max_cm_s" not in cfg["state_machine_params"]


def test_mode2_yaml_overrides_shared_defaults():
    cfg = load_yaml(FORESIGHT_ROOT / "sim" / "mode2_cfg.yaml")
    assert cfg["scene"]["mode"] == 2
    assert cfg["scene"]["object_categories"] == [
        "cube_blue",
        "cube_red",
        "cube_green",
        "cube_yellow",
    ]
    assert cfg["scene"]["mode2_min_separation_m"] == pytest.approx(0.12)
    assert cfg["scene"]["mode2_container_clearance_m"] == pytest.approx(0.15)
    assert cfg["state_machine_params"]["lift_height_m"] == pytest.approx(0.05)
    assert cfg["state_machine_params"]["robot_distance_to_object_m"] == pytest.approx(0.05)
    assert cfg["claim_timing"]["delay_enabled"] is True
    assert cfg["claim_height"]["use_object_height_from_depth"] is True


def test_timing_fold_conflict_raises():
    from interfaces.config import _fold_timing_sections

    with pytest.raises(ValueError, match="conflicts"):
        _fold_timing_sections(
            {
                "claim_timing": {"delay_enabled": True},
                "policy_training_delay": {"delay_enabled": False},
            }
        )
