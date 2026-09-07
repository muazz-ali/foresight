#!/usr/bin/env python3
"""Unit tests for frozen ObjectState (no Isaac required)."""

from __future__ import annotations

import numpy as np

from interfaces.state import conditioning_vector, oracle_from_gt, pack_conditioning
from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT
from sim.motion import kinematic_forecast


def test_oracle_predict_and_condition():
    st = oracle_from_gt([0.4, 0.0, 0.05], [0.2, 0.0, 0.0], timestamp=1.0)
    assert st.valid
    fut = st.predict(0.25)
    np.testing.assert_allclose(fut.position, [0.45, 0.0, 0.05], atol=1e-9)
    vec = conditioning_vector(st, delta=0.25)
    assert vec.shape == (CONDITIONING_DIM,)
    assert CONDITIONING_LAYOUT == ("p_hat_x", "p_hat_y", "v_hat_x", "v_hat_y")
    np.testing.assert_allclose(vec[:2], fut.position[:2])
    np.testing.assert_allclose(vec[2:4], fut.velocity[:2])


def test_flat_roundtrip_len():
    st = oracle_from_gt([0, 0, 0], [0, 0, 0], 0.0)
    assert st.to_flat().shape == (14,)


def test_kinematic_forecast_matches_cv_away_from_wall():
    pos = np.array([0.5, 0.0, 0.05])
    vel = np.array([0.2, 0.0, 0.0])
    bounds = ((0.30, 0.70), (-0.30, 0.30))
    fp, fv = kinematic_forecast(pos, vel, 0.25, 0.04, xy_bounds=bounds)
    np.testing.assert_allclose(fp, pos + vel * 0.25, atol=1e-9)
    np.testing.assert_allclose(fv, vel, atol=1e-9)


def test_episode_for_json_is_light():
    from sim.collect import JSON_TRACE_KEYS, episode_for_json

    T = 5
    episode = {
        "stage": np.arange(T, dtype=np.int64),
        "holding": np.zeros(T),
        "object_pos": np.zeros((T, 3)),
        "object_vel": np.tile([0.2, 0.0, 0.0], (T, 1)),
        "ee_pos": np.zeros((T, 3)),
        "gripper": np.ones(T),
        "conditioning": np.tile([0.45, 0.0, 0.2, 0.0], (T, 1)),
        "static_cam_rgb": np.zeros((T, 8, 8, 3), dtype=np.uint8),
    }
    d = episode_for_json(episode)
    assert d["conditioning_dim"] == CONDITIONING_DIM
    assert d["conditioning_layout"] == list(CONDITIONING_LAYOUT)
    assert abs(d["vel_xy_median"] - 0.2) < 1e-6
    assert d["cond0"] == [0.45, 0.0, 0.2, 0.0]
    assert "static_cam_rgb" not in d["trace"]
    assert d["arrays"]["static_cam_rgb"]["shape"] == [T, 8, 8, 3]
    for k in ("stage", "holding", "object_vel", "ee_pos", "gripper"):
        assert k in JSON_TRACE_KEYS
        assert k in d["trace"]
        assert len(d["trace"][k]) == T


def test_write_episode_h5_is_4d_not_12():
    import tempfile
    from pathlib import Path

    import h5py

    from sim.collect import write_episode_h5

    T = 3
    ep = {
        "object_pos": np.zeros((T, 3)),
        "object_vel": np.zeros((T, 3)),
        "conditioning": np.zeros((T, CONDITIONING_DIM)),
        "static_cam_rgb": np.zeros((T, 4, 4, 3), dtype=np.uint8),
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ep.h5"
        write_episode_h5(p, ep, extra={"speed_m_s": np.array([0.2], dtype=np.float32)})
        with h5py.File(p, "r") as f:
            assert f.attrs["conditioning_dim"] == CONDITIONING_DIM
            assert f["conditioning"].shape == (T, CONDITIONING_DIM)
            assert "static_cam_rgb" in f
            assert abs(float(f["speed_m_s"][0]) - 0.2) < 1e-6
        bad = dict(ep)
        bad["conditioning"] = np.zeros((T, 12))
        try:
            write_episode_h5(p, bad)
        except ValueError as ex:
            assert "4" in str(ex)
        else:
            raise AssertionError("expected 12-D conditioning to fail")


def test_kinematic_forecast_bounces_unlike_cv():
    # Sim motion bounces; policy pack must stay CV (can leave the table in numbers).
    pos = np.array([0.68, 0.0, 0.05])
    vel = np.array([0.4, 0.0, 0.0])
    bounds = ((0.30, 0.70), (-0.30, 0.30))
    fp, fv = kinematic_forecast(pos, vel, 0.25, 0.04, xy_bounds=bounds)
    cv = pos + vel * 0.25
    assert fp[0] <= 0.70 + 1e-9
    assert abs(fp[0] - cv[0]) > 1e-3
    assert fv[0] < 0.0  # bounced in the world model
    st = oracle_from_gt(pos, vel, timestamp=0.0)
    vec = conditioning_vector(st, delta=0.25)
    np.testing.assert_allclose(vec[:2], cv[:2])
    np.testing.assert_allclose(vec[2:4], vel[:2])
    bounce_pack = pack_conditioning(oracle_from_gt(fp, fv, timestamp=0.25), 0.25)
    assert abs(float(bounce_pack[0] - vec[0])) > 1e-3


def test_sm_stage_order_and_success_latch():
    from sim.state_machine import (
        STAGE_APPROACH,
        STAGE_CARRY,
        STAGE_CLOSE,
        STAGE_DONE,
        STAGE_LIFT,
        STAGE_OPEN,
        STAGE_SCHEMA,
        PickPlaceStateMachine,
    )
    from sim.success import episode_success

    assert STAGE_APPROACH < STAGE_CLOSE < STAGE_LIFT < STAGE_CARRY < STAGE_OPEN < STAGE_DONE
    assert STAGE_SCHEMA.done == STAGE_DONE == 9
    T = 8
    ep = {
        "stage": np.array([0, 1, 2, 4, 5, 6, 7, 9], dtype=np.int64),
        "object_pos": np.zeros((T, 3)),
        "container_pos": np.tile([0.50, 0.30, 0.05], (T, 1)),
    }
    ep["object_pos"][:, 2] = [0.02, 0.02, 0.02, 0.18, 0.18, 0.12, 0.06, 0.06]
    ep["object_pos"][-1, :2] = [0.50, 0.30]
    assert episode_success(ep) is True
    ep_miss = dict(ep)
    ep_miss["stage"] = np.array([0, 1, 2, 4, 4, 4, 4, 4], dtype=np.int64)
    assert episode_success(ep_miss) is False



if __name__ == "__main__":
    test_oracle_predict_and_condition()
    test_flat_roundtrip_len()
    test_kinematic_forecast_matches_cv_away_from_wall()
    test_kinematic_forecast_bounces_unlike_cv()
    test_episode_for_json_is_light()
    test_write_episode_h5_is_4d_not_12()
    test_sm_stage_order_and_success_latch()
    print("interfaces/state + motion forecast OK")
