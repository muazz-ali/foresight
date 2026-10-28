"""Isaac-free checks for claim Steps 2–5 (noise filter, forecast, relabel, close)."""

from __future__ import annotations

import numpy as np
import pytest

from control import ActionBuffer, close_speed_check, max_speed_diff_m_s
from control.exec_chunk import push_forecast_chunk
from interfaces.forecast import conditioning_at_measured_delta, forecast_horizon
from interfaces.state import ObjectState, conditioning_vector, oracle_from_gt
from perception.noisy_gt import (
    STATIC_POS_STD_M,
    WRIST_POS_STD_M,
    WindowedCVFilter,
    add_gt_noise,
    filter_noisy_trajectory,
    measure_noise_std,
)
from policy.relabel import (
    actions_from_forecast_rel,
    actions_to_forecast_rel,
    relabel_frame,
)


DT = 1.0 / 30.0


def _moving_state(v=0.20, t=0.0) -> ObjectState:
    return oracle_from_gt(
        np.array([0.50, 0.0, 0.03]),
        np.array([float(v), 0.0, 0.0]),
        float(t),
    )


# ----- Step 2 -----


def test_noise_std_switches_at_10cm():
    obj = np.array([0.5, 0.0, 0.03])
    assert measure_noise_std(obj + [0.05, 0, 0], obj) == WRIST_POS_STD_M
    assert measure_noise_std(obj + [0.20, 0, 0], obj) == STATIC_POS_STD_M


def test_add_gt_noise_only_shakes_xy():
    rng = np.random.default_rng(0)
    s = _moving_state()
    n = add_gt_noise(s, rng, pos_std=0.004)
    assert n.position[2] == pytest.approx(s.position[2])
    np.testing.assert_allclose(n.velocity, s.velocity)
    assert not np.allclose(n.position[:2], s.position[:2])


def test_step2_conveyor_vel_error_and_nis():
    """At 20 cm/s, filter |v̂ − v| ≲ 6.5 mm/s; residual within ~3σ after warm-up."""
    rng = np.random.default_rng(1)
    v = np.array([0.20, 0.0, 0.0])
    T = 60  # 2 s at 30 Hz
    pos0 = np.array([0.40, 0.0, 0.03])
    positions = np.stack([pos0 + v * (i * DT) for i in range(T)])
    velocities = np.tile(v, (T, 1))
    # Far hand → static 4 mm noise.
    ee = positions + np.array([0.30, 0.0, 0.10])
    noisy, filtered = filter_noisy_trajectory(
        positions, velocities, dt=DT, rng=rng, ee_positions=ee
    )
    # After 0.5 s warm-up (16 frames), check velocity error.
    warm = WindowedCVFilter(dt=DT).window_frames
    vel_err = [
        float(np.linalg.norm(filtered[i].velocity[:2] - v[:2]))
        for i in range(warm, T)
    ]
    med_vel_err = float(np.median(vel_err))
    assert med_vel_err <= 0.0065 + 1e-9, f"vel err {med_vel_err*1000:.2f} mm/s"

    # Prediction residual vs next noisy measurement, in units of meas σ.
    # Use filter predict to next step vs noisy XY; |y|/σ ≲ 3 most of the time.
    sig = STATIC_POS_STD_M
    nis_ok = 0
    n_check = 0
    for i in range(warm, T - 1):
        pred = filtered[i].position[:2] + filtered[i].velocity[:2] * DT
        y = noisy[i + 1].position[:2] - pred
        # Per-axis |y|/σ; both axes within 3σ counts as ok for this check.
        if np.all(np.abs(y) <= 3.0 * sig):
            nis_ok += 1
        n_check += 1
    frac = nis_ok / max(n_check, 1)
    assert frac >= 0.90, f"only {frac:.0%} residuals within 3σ"


# ----- Step 3 -----


def test_step3_forecast_leads_33mm_at_165ms():
    """v=20 cm/s, Δ=165 ms → p̂0 leads current p by 33 mm."""
    s = _moving_state(0.20)
    delta = 0.165
    phat0 = forecast_horizon(s, delta, dt=DT, n=1)[0]
    lead = phat0[0] - s.position[0]
    assert lead == pytest.approx(0.20 * 0.165, abs=1e-9)
    # conditioning pack uses the same lead on XY; p̂z = state z (no latch here).
    cond = conditioning_at_measured_delta(s, delta)
    assert cond.shape == (5,)
    assert cond[0] == pytest.approx(s.position[0] + 0.20 * 0.165)
    assert cond[2] == pytest.approx(s.position[2])
    assert cond[3] == pytest.approx(0.20)
    # Matches conditioning_vector with the same Δ.
    np.testing.assert_allclose(cond, conditioning_vector(s, delta))


def test_forecast_horizon_length_and_spacing():
    s = _moving_state(0.10)
    f = forecast_horizon(s, 0.1, dt=DT, n=25)
    assert f.shape == (25, 3)
    # Spacing between consecutive p̂ ≈ v · dt
    gaps = np.diff(f[:, 0])
    np.testing.assert_allclose(gaps, 0.10 * DT, atol=1e-12)


# ----- Step 4 -----


def test_step4_relabel_roundtrip_exact():
    """Convert abs→rel→abs must reproduce the expert motion exactly."""
    T = 80
    v = 0.15
    ep = {
        "timestamp": np.arange(T) * DT,
        "object_pos": np.stack([[0.4 + v * i * DT, 0.0, 0.03] for i in range(T)]),
        "object_vel": np.tile([v, 0.0, 0.0], (T, 1)),
        # Expert hand rides 2 cm above object XY, same speed.
        "action_pos": np.stack([[0.4 + v * i * DT, 0.0, 0.12] for i in range(T)]),
        "action_quat": np.tile([0.0, 1.0, 0.0, 0.0], (T, 1)),
        "gripper": np.ones(T),
    }
    out = relabel_frame(ep, frame_i=10, delta_s=0.100, dt=DT, chunk=25)
    back = actions_from_forecast_rel(out["action_rel"], out["forecasts"])
    np.testing.assert_allclose(back, out["action_abs"], atol=1e-12)
    # Sanity: relative XYZ z component ≈ hand_z − obj_z (forecast z = obj z).
    assert out["action_rel"][0, 2] == pytest.approx(0.09, abs=1e-9)


def test_actions_to_forecast_rel_shapes():
    targets = np.zeros((5, 8))
    targets[:, 0:3] = 1.0
    forecasts = np.zeros((5, 3))
    forecasts[:, 0] = 0.25
    rel = actions_to_forecast_rel(targets, forecasts)
    np.testing.assert_allclose(rel[:, 0], 0.75)
    np.testing.assert_allclose(rel[:, 3:], targets[:, 3:])


# ----- Step 5 -----


def test_step5_close_speed_limit_is_3p1_cm_s():
    lim = max_speed_diff_m_s(slack_m=0.0025, close_time_s=0.08)
    assert lim == pytest.approx(0.03125)
    ok = close_speed_check([0.20, 0.0, 0.0], [0.20, 0.0, 0.0])
    assert ok.ok
    bad = close_speed_check([0.20, 0.0, 0.0], [0.25, 0.0, 0.0])
    assert not bad.ok
    assert bad.speed_diff_m_s == pytest.approx(0.05)


def test_step5_scripted_close_at_30cms_stays_under_limit():
    """Hand XY matched to object at 30 cm/s → under 3.1 cm/s bar during close."""
    v_obj = np.array([0.30, 0.0, 0.0])
    v_hand = v_obj.copy()  # matched
    check = close_speed_check(v_hand, v_obj, close_time_s=0.08, slack_m=0.0025)
    assert check.ok
    assert check.speed_diff_m_s < 0.03125


def test_step5_buffer_drops_past_and_rebuilds_abs():
    buf = ActionBuffer(max_age=0.5)
    s = _moving_state(0.20, t=1.0)
    # Relative: stay 5 cm above object forecast.
    rel = np.zeros((10, 8))
    rel[:, 2] = 0.05
    rel[:, 4] = 1.0  # quat y
    apply_at = 1.10  # t0 + Δ
    # Drop slots 0..2 (push now midway to slot 3); execute at slot 3.
    push_now = apply_at + 2.5 * DT
    exec_now = apply_at + 3 * DT
    abs_chunk = push_forecast_chunk(
        buf,
        rel_actions=rel,
        state=s,
        delta_s=0.10,
        apply_at=apply_at,
        now=push_now,
        dt=DT,
    )
    assert abs_chunk.shape == (10, 8)
    assert len(buf) < 10
    # Slots older than one step before push are gone; the slot due at push time is kept.
    assert buf.t[0] > push_now - DT
    act = buf.pop_due(exec_now)
    assert act is not None
    phat = forecast_horizon(s, 0.10, dt=DT, n=10)[3]
    np.testing.assert_allclose(act[0:3], phat + np.array([0.0, 0.0, 0.05]), atol=1e-9)
