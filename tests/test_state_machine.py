"""Isaac-free tests for the intercept-servo expert.

Run from the repo root:  pytest tests/test_state_machine.py -v

Single-step and unit checks only. Closed-loop behavior (arm lag, reach,
contact) is checked in Isaac (C30 / B1 grid), not with a surrogate arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interfaces.state import oracle_from_gt
from sim.state_machine import (
    AIM_BRANCHES,
    STAGE_APPROACH,
    STAGE_CARRY,
    STAGE_CLOSE,
    STAGE_DESCEND,
    STAGE_DONE,
    STAGE_HOME,
    STAGE_LIFT,
    STAGE_LOWER,
    STAGE_OPEN,
    STAGE_RESET,
    PickPlaceStateMachine,
)

_REPO = Path(__file__).resolve().parents[1]
_YAML = _REPO / "sim" / "phase0_cfg.yaml"

CARRY = np.array([0.50, 0.30, 0.12], dtype=np.float64)
DROP = np.array([0.50, 0.30, 0.06], dtype=np.float64)


def load_cfg() -> dict:
    from interfaces.config import load_yaml

    return load_yaml(_YAML)


@pytest.fixture
def cfg() -> dict:
    """Shipped yaml (house throw + GT weld grasp)."""
    return load_cfg()


@pytest.fixture
def sm(cfg) -> PickPlaceStateMachine:
    m = PickPlaceStateMachine(cfg, dt=float(cfg["sim"]["dt"]))
    m.set_box_poses(CARRY, drop_pos=DROP)
    return m


def _state(pos, vel, t=0.0):
    return oracle_from_gt(np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64), t)


def _riding(sm, ee, vel):
    """Hand was one tick behind on the same track → |v_hand − v_obj| = 0 (B3 gate)."""
    sm._prev_ee = np.asarray(ee, dtype=np.float64) - np.asarray(vel, dtype=np.float64) * sm.dt
    return ee


# --------------------------------------------------------------- interception


def test_lead_collapses_to_pad_when_on_top(sm):
    """Riding the object → horizon is pure lag compensation, not 0.25 s ahead.

    This is the v1 bug: a fixed 0.25 s lead parks the hand ``0.25 * v`` in front
    of the object (20 cm at 80 cm/s) and it never closes.
    """
    obj = np.array([0.55, 0.0, 0.03])
    for speed in (0.0, 0.20, 0.40, 0.80):
        vel = np.array([speed, 0.0, 0.0])
        ee_on_top = obj.copy()
        lead = sm._intercept_lead_s(ee_on_top, obj, vel)
        assert lead == pytest.approx(sm.robot_approach_pad_s, abs=1e-6), f"speed {speed}"


def test_delayed_ride_aims_at_p_now_not_base_tau(sm):
    """Under Δ: ride aim = p_now = p_delayed + v·Δ; no leftover +dt lead.

    Why: base_tau = Δ+tip_lag+dt left a v·dt ≈ 10 mm park line outside the
    10 mm ride box at 30 cm/s. DiffIK FF carries v̂_obj while aim sits on now.
    """
    sm.stage = STAGE_APPROACH
    sm.z_grasp = 0.036
    sm.set_aim_xy_bounds([[0.20, 0.90], [-0.40, 0.40]])
    delta = 0.30
    vel = np.array([0.30, 0.0, 0.0])
    # Delayed obs: object was here Δ ago; true now is + v·Δ.
    p_delayed = np.array([0.50, 0.0, 0.036])
    p_now = p_delayed + vel * delta
    ee = p_now.copy()  # already riding on now
    ee[2] = sm.z_grasp
    sm._prev_ee = ee - vel * sm.dt  # matched speed for gate
    cmd = sm.step(
        _state(p_delayed, vel),
        ee,
        sm._down_quat,
        delta_meas_s=delta,
        ee_vel=vel,
    )
    assert cmd.stage in (STAGE_APPROACH, STAGE_CLOSE)
    # Inside the 2 mm box aim = p_now (= ee here); DiffIK FF carries v̂_obj.
    assert cmd.position[0] == pytest.approx(float(p_now[0]), abs=1e-6)
    assert cmd.velocity is not None
    assert cmd.velocity[0] == pytest.approx(0.30, abs=1e-9)  # DiffIK FF = v̂_obj


def test_inside_close_box_aims_at_object_not_hand(sm):
    """No hold rule: 1.5 mm inside the close box, aim = object_now, not the hand."""
    sm.stage = STAGE_APPROACH
    sm.z_grasp = 0.036
    sm.set_aim_xy_bounds([[0.20, 0.90], [-0.40, 0.40]])
    vel = np.array([0.30, 0.0, 0.0])
    obj = np.array([0.55, 0.0, 0.036])
    ee = obj + np.array([0.0015, -0.0015, 0.0])
    sm._prev_ee = ee.copy()  # hand still: |Δv| = 30 cm/s, so the close gate stays shut
    cmd = sm.step(
        _state(obj, vel), ee, sm._down_quat,
        delta_meas_s=0.0, ee_vel=np.zeros(3),
    )
    assert cmd.stage == STAGE_APPROACH
    assert sm.aim_branch == "settle"
    assert cmd.position[0] == pytest.approx(float(obj[0]), abs=1e-6)
    assert cmd.position[1] == pytest.approx(float(obj[1]), abs=1e-6)


def test_delayed_ride_settle_is_p_now(sm):
    """Delayed ride aim = p_now: no forecast(delayed, Δ+dt), no hand-written +dt."""
    sm.stage = STAGE_APPROACH
    sm.z_grasp = 0.036
    sm.ride_k = 0.0  # isolate the settle lead
    sm.set_aim_xy_bounds([[0.20, 0.90], [-0.40, 0.40]])
    delta = 0.30
    vel = np.array([0.30, 0.0, 0.0])
    p_delayed = np.array([0.50, 0.0, 0.036])
    p_now = p_delayed + vel * delta
    # Outside 2 mm hold, inside chase box.
    ee = p_now + np.array([-0.005, 0.0, 0.0])
    ee[2] = sm.z_grasp
    sm._prev_ee = ee - vel * sm.dt
    cmd = sm.step(
        _state(p_delayed, vel),
        ee,
        sm._down_quat,
        delta_meas_s=delta,
        ee_vel=vel,
    )
    assert cmd.position[0] == pytest.approx(float(p_now[0]), abs=1e-6)


def test_zero_delay_ride_settle_is_p_now(sm):
    """Δ≈0 near_ok settle matches delayed: aim = p_now, no +dt.

    Why: with DiffIK FF a +v·dt settle parks the hand ahead of the 2 mm box.
    """
    sm.stage = STAGE_APPROACH
    sm.z_grasp = 0.036
    sm.ride_k = 0.0  # isolate the settle lead
    sm.set_aim_xy_bounds([[0.20, 0.90], [-0.40, 0.40]])
    vel = np.array([0.30, 0.0, 0.0])
    obj = np.array([0.55, 0.0, 0.036])
    # Sit in the chase box, outside the 2 mm close box (ride pull off here).
    ee = obj + np.array([-0.005, 0.0, 0.0])
    ee[2] = sm.z_grasp
    sm._prev_ee = ee - vel * sm.dt
    cmd = sm.step(
        _state(obj, vel),
        ee,
        sm._down_quat,
        delta_meas_s=0.0,
        ee_vel=vel,
    )
    assert cmd.position[0] == pytest.approx(float(obj[0]), abs=1e-6)


def test_delayed_chase_travel_from_p_now_no_double_delta(sm):
    """Far chase under Δ: travel solved from p_now; aim ≠ p + v·(Δ+dt) leftover."""
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.20, 1.20], [-0.40, 0.40]])
    delta = 0.30
    vel = np.array([0.30, 0.0, 0.0])
    p_delayed = np.array([0.70, 0.0, 0.03])
    p_now = p_delayed + vel * delta
    ee = np.array([0.40, 0.0, 0.40])  # far behind
    cmd = sm.step(
        _state(p_delayed, vel),
        ee,
        sm._down_quat,
        delta_meas_s=delta,
    )
    # Aim should be ahead of p_now by travel·v, not stuck at v·dt past p_now.
    # Full double-count lead would put aim near p_delayed + v*(Δ + travel + tau).
    assert cmd.position[0] > float(p_now[0]) - 1e-6
    leftover_dt = float(p_now[0] + vel[0] * sm.dt)
    # Must not be the old park line (p_now + v·dt) when still far — travel binds.
    assert abs(float(cmd.position[0]) - leftover_dt) > 0.02


def _far_chase_cmd(sm, delta):
    """One far-chase step with the object's true *now* fixed at x = 0.79."""
    sm.reset()
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.20, 1.20], [-0.40, 0.40]])
    vel = np.array([0.30, 0.0, 0.0])
    p_now = np.array([0.79, 0.0, 0.03])
    p_obs = p_now - vel * delta
    ee = np.array([0.40, 0.0, 0.40])  # far behind
    cmd = sm.step(
        _state(p_obs, vel),
        ee,
        sm._down_quat,
        delta_meas_s=delta,
    )
    return cmd, sm.aim_branch


def test_far_chase_same_formula_for_zero_and_delayed(sm):
    """Δ=0 and Δ>0 far chase both aim at p_now + v·(travel + pad).

    Why: a Δ=0-only +dt lead made Δ=0 cells run different code from Δ>0.
    """
    cmd0, br0 = _far_chase_cmd(sm, 0.0)
    cmd3, br3 = _far_chase_cmd(sm, 0.30)
    assert br0 == br3 == "far_chase"
    assert cmd0.position[:2] == pytest.approx(cmd3.position[:2], abs=1e-9)


def test_aim_branch_flag_per_tick(sm):
    """Expert reports which aim branch ran; 'none' when no chase aim is built."""
    sm.stage = STAGE_RESET
    sm.step(_state([0.55, 0.0, 0.03], [0.0, 0.0, 0.0]), np.array([0.4, 0.0, 0.4]), sm._down_quat)
    assert sm.aim_branch == "none"
    assert sm.aim_branch in AIM_BRANCHES
    _, br = _far_chase_cmd(sm, 0.10)
    assert br == "far_chase"


def test_lead_is_capped_at_lookahead_when_far(sm):
    """Far away → the yaml look-ahead cap binds (plus approach pad, no +dt).

    Hand at 2.5 m/s needs >1.0 m of travel before the 400 ms cap binds.
    """
    obj = np.array([1.20, 0.50, 0.03])
    ee = np.array([0.20, 0.0, 0.55])
    assert sm.lookahead_s == pytest.approx(0.40, abs=1e-9)
    for speed in (0.20, 0.40, 0.80):
        lead = sm._intercept_lead_s(ee, obj, np.array([0.0, speed, 0.0]))
        assert lead == pytest.approx(sm.lookahead_s + sm.robot_approach_pad_s, abs=1e-6)


def test_approach_aims_ahead_of_a_moving_object(sm):
    """APPROACH command leads the object; Z aims at z_grasp (p̂z), not centre."""
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.08, 0.03])
    vel = np.array([0.0, -0.40, 0.0])
    cmd = sm.step(_state(obj, vel), np.array([0.45, 0.0, 0.40]), sm._down_quat)
    assert cmd.position[1] < obj[1] - 0.05, "command must lead a 40 cm/s object"
    assert sm.z_grasp is not None
    assert cmd.position[2] == pytest.approx(sm.z_grasp)


def test_z_grasp_is_object_centre(sm):
    """Expert close height is object centre (height/2), never a depth latch."""
    sm.set_object_half_height(0.036, z_table=0.0)
    obj = np.array([0.55, 0.0, 0.036])
    # Even with a high hand, p̂z must stay on the fruit centre (smoke bug was ~18 cm).
    zg = sm._update_z_grasp(np.array([0.55, 0.0, 0.40]), obj)
    assert zg == pytest.approx(0.036, abs=1e-6)
    assert sm.close_at_object_center is True


def test_close_gate_requires_hand_at_object_centre(sm):
    """Fingers only shut when EE Z is near object centre — not 14 cm above."""
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.0, 0.036])
    vel = np.array([0.0, 0.0, 0.0])
    # Mid-air (old bug height): must NOT enter CLOSE.
    high = np.array([obj[0], obj[1], 0.177])
    cmd_hi = sm.step(_state(obj, vel), high, sm._down_quat)
    assert cmd_hi.stage == STAGE_APPROACH
    # At centre: CLOSE.
    at = np.array([obj[0], obj[1], obj[2]])
    cmd = sm.step(_state(obj, vel), at, sm._down_quat)
    assert cmd.stage == STAGE_CLOSE


def test_z_gate_honors_yaml_not_z_slack_half(sm):
    """Close Z tol is grasp_xyz_ok[2] (12 mm), not claim z_slack/2 (2.5 mm).

    Why: a 30 cm/s smoke near-hit (8.8 mm above centre) never closed when the
    gate was clamped to 2.5 mm, then the object hit the pedestal.
    """
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.0, 0.036])
    vel = np.zeros(3)
    zg = sm._update_z_grasp(obj.copy(), obj)
    assert sm.grasp_xyz_ok[2] == pytest.approx(0.012, abs=1e-9)
    # 8 mm above centre: inside yaml box + XY settled → CLOSE.
    near = _riding(sm, np.array([obj[0], obj[1], zg + 0.008]), vel)
    assert sm.step(_state(obj, vel), near, sm._down_quat).stage == STAGE_CLOSE
    # 13 mm above: outside yaml Z box → stay intercept.
    sm.stage = STAGE_APPROACH
    sm.timer_s = 0.0
    far = _riding(sm, np.array([obj[0], obj[1], zg + 0.013]), vel)
    assert sm.step(_state(obj, vel), far, sm._down_quat).stage == STAGE_APPROACH


def test_close_requires_xy_settle_2mm(sm):
    """B5: near the chase box is not enough — |dx|,|dy| must be ≤ close_xy_ok."""
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.0, 0.036])
    vel = np.zeros(3)
    zg = sm._update_z_grasp(obj.copy(), obj)
    # 5 mm off in X: inside grasp_xyz_ok (10 mm) but outside close_xy_ok (2 mm).
    almost = _riding(sm, np.array([obj[0] + 0.005, obj[1], zg]), vel)
    assert sm.step(_state(obj, vel), almost, sm._down_quat).stage == STAGE_APPROACH
    # 1 mm off: settled → CLOSE.
    sm.stage = STAGE_APPROACH
    sm.timer_s = 0.0
    tight = _riding(sm, np.array([obj[0] + 0.001, obj[1], zg]), vel)
    assert sm.step(_state(obj, vel), tight, sm._down_quat).stage == STAGE_CLOSE


def test_predictive_close_waits_when_approach_would_overshoot(sm):
    """|e|≤2 mm is not enough if e+Δv·t_contact still lands outside the box.

    Hand 1 mm short but still charging at +5 cm/s → predicted gap at
    t_contact≈130 ms is past the object → stay in intercept.
    """
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.0, 0.036])
    vel = np.zeros(3)
    zg = sm._update_z_grasp(obj.copy(), obj)
    # Hand behind object, approaching fast (not speed-matched).
    ee = np.array([obj[0] - 0.001, obj[1], zg])
    sm._prev_ee = ee - np.array([0.05, 0.0, 0.0]) * sm.dt  # v_hand ≈ +5 cm/s
    assert sm.step(_state(obj, vel), ee, sm._down_quat).stage == STAGE_APPROACH
    # Same pose, matched speed → e_pred ≈ e → CLOSE.
    sm.stage = STAGE_APPROACH
    sm.timer_s = 0.0
    ee = _riding(sm, ee, vel)
    assert sm.step(_state(obj, vel), ee, sm._down_quat).stage == STAGE_CLOSE


def test_close_rides_object_xy(sm):
    """During CLOSE hold hand XY; DiffIK FF coasts with v̂_obj (no τ snap)."""
    sm.stage = STAGE_CLOSE
    sm.z_grasp = 0.036
    sm.timer_s = 0.0
    sm.set_aim_xy_bounds([[0.40, 0.90], [-0.20, 0.20]])
    obj = np.array([0.55, 0.0, 0.036])
    vel = np.array([0.50, 0.0, 0.0])
    ee = obj.copy()
    cmd = sm.step(_state(obj, vel), ee, sm._down_quat)
    assert cmd.stage == STAGE_CLOSE
    assert cmd.gripper_open == 0.0
    # Physics: coast until fingers shut; free only after finger_close_s.
    assert sm.physics_grasp
    assert not cmd.attach_object and cmd.kinematic_object
    # Hold ee XY — forecast(tau) snap jerked s1003 off-center.
    assert cmd.position[0] == pytest.approx(float(ee[0]), abs=1e-6)
    assert cmd.position[2] == pytest.approx(0.036, abs=1e-6)
    # After finger travel, free to PhysX (attach).
    sm.stage = STAGE_CLOSE
    sm.timer_s = sm.finger_close_s + 1e-6
    cmd2 = sm.step(_state(obj, vel), ee, sm._down_quat)
    assert cmd2.attach_object and not cmd2.kinematic_object


def test_close_ff_is_xy_only(sm):
    """CLOSE feeds v̂_obj forward in XY only; a popping object's v_z is dropped.

    Why: the z FF fed the squeeze pop back into the hand, so close/lift labels
    depended on Δ (pop 13.5 mm at Δ0 vs 2.4 mm at Δ300).
    """
    sm.stage = STAGE_CLOSE
    sm.z_grasp = 0.036
    sm.timer_s = 0.0
    sm.set_aim_xy_bounds([[0.40, 0.90], [-0.20, 0.20]])
    obj = np.array([0.55, 0.0, 0.036])
    vel = np.array([0.30, -0.10, 0.25])  # squeeze pop: object moving up
    cmd = sm.step(_state(obj, vel), obj.copy(), sm._down_quat)
    assert cmd.stage == STAGE_CLOSE and cmd.kinematic_object
    np.testing.assert_allclose(cmd.velocity, [0.30, -0.10, 0.0], atol=1e-12)
    # Chase keeps the full v̂_obj.
    sm.reset()
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.20, 1.20], [-0.40, 0.40]])
    cmd = sm.step(_state(obj, vel), np.array([0.40, 0.0, 0.40]), sm._down_quat)
    np.testing.assert_allclose(cmd.velocity, vel, atol=1e-12)


def test_p_now_uses_delta_meas(sm):
    """F2: p_now = p_delayed + v·Δ_meas, and the expert logs it."""
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.20, 1.20], [-0.40, 0.40]])
    vel = np.array([0.30, 0.10, 0.0])
    p = np.array([0.60, 0.0, 0.03])
    sm.step(_state(p, vel), np.array([0.40, 0.0, 0.40]), sm._down_quat, delta_meas_s=0.137)
    np.testing.assert_allclose(sm.p_now[:2], p[:2] + vel[:2] * 0.137, atol=1e-12)


def test_intercept_aim_clips_to_table(sm):
    """Long lead must not command the hand past the table (ee-off-table)."""
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.20, 0.80], [-0.25, 0.25]])
    obj = np.array([0.55, 0.0, 0.03])
    vel = np.array([0.0, 0.80, 0.0])  # 80 cm/s toward +Y
    ee = np.array([0.45, 0.0, 0.40])
    cmd = sm.step(_state(obj, vel), ee, sm._down_quat)
    assert cmd.position[1] <= 0.25 + 1e-9
    assert cmd.position[0] >= 0.20 - 1e-9
    assert cmd.position[0] <= 0.80 + 1e-9


def test_far_rim_spawn_still_chased(sm):
    """Object just past aim inset but inside escape box → keep chasing.

    Smoke lemon seed-50194 sat on the far rim past the 2 cm aim inset and the
    old escape check sent the expert home on frame 1.
    """
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.12, 0.76], [-0.23, 0.23]])  # inset
    sm.set_escape_xy_bounds([[0.12, 0.80], [-0.25, 0.25]])  # full table
    obj = np.array([0.782, 0.072, 0.034])  # past aim xmax, inside escape
    vel = np.array([-0.15, -0.05, 0.0])
    ee = np.array([0.45, 0.0, 0.40])
    cmd = sm.step(_state(obj, vel), ee, sm._down_quat)
    assert cmd.stage == STAGE_APPROACH
    assert not sm._object_left_catch_box(obj)


def test_past_keepout_stops_chase(sm):
    """Object past the near-base escape edge → go home (no ee-off-table chase)."""
    sm.stage = STAGE_APPROACH
    sm.set_aim_xy_bounds([[0.12, 0.76], [-0.23, 0.23]])
    sm.set_escape_xy_bounds([[0.12, 0.80], [-0.25, 0.25]])
    obj = np.array([0.10, -0.14, 0.03])  # x < escape xmin keepout
    vel = np.array([-0.20, 0.0, 0.0])
    ee = np.array([0.45, 0.0, 0.40])
    cmd = sm.step(_state(obj, vel), ee, sm._down_quat)
    assert cmd.stage == STAGE_HOME


def test_forecast_coasts_straight(sm):
    """Forecast is constant-velocity coast — can leave the play band."""
    obj = np.array([0.65, 0.0, 0.03])
    vel = np.array([0.80, 0.0, 0.0])
    pred = sm._forecast(obj, vel, 0.25)
    np.testing.assert_allclose(pred, obj + vel * 0.25, atol=1e-9)


# ------------------------------------------------------------------- gating


def test_chase_gate_is_measured_against_the_object_not_the_aim(sm):
    """Hand on object XY at z_grasp closes, even though the aim leads."""
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.0, 0.03])
    vel = np.array([0.80, 0.0, 0.0])
    zg = sm._update_z_grasp(obj.copy(), obj)
    ee = _riding(sm, np.array([obj[0], obj[1], zg]), vel)
    cmd = sm.step(_state(obj, vel), ee, sm._down_quat)
    assert cmd.stage == STAGE_CLOSE
    assert sm.grasp_err_xy is not None and sm.grasp_err_xy < 1e-6


def test_close_waits_for_speed_match(sm):
    """B3: on the object but slower than it → keep riding; matched → CLOSE."""
    assert sm.close_speed_gate  # claim_timing.close_speed_guard: true
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.0, 0.03])
    vel = np.array([0.30, 0.0, 0.0])
    zg = sm._update_z_grasp(obj.copy(), obj)
    ee = np.array([obj[0], obj[1], zg])
    # First tick: no hand velocity yet → no close.
    sm._prev_ee = None
    assert sm.step(_state(obj, vel), ee, sm._down_quat).stage == STAGE_APPROACH
    # Hand at 25 cm/s under a 30 cm/s object: Δv·80 ms = 4 mm > 2.5 mm → wait.
    sm._prev_ee = ee - np.array([0.25, 0.0, 0.0]) * sm.dt
    assert sm.step(_state(obj, vel), ee, sm._down_quat).stage == STAGE_APPROACH
    # Hand at 29 cm/s: 0.8 mm → close.
    sm._prev_ee = ee - np.array([0.29, 0.0, 0.0]) * sm.dt
    assert sm.step(_state(obj, vel), ee, sm._down_quat).stage == STAGE_CLOSE


def test_grasp_xyz_ok_is_a_box(sm):
    """Gate uses |dx|, |dy|, |dz| — not a single XY norm alone."""
    obj = np.array([0.55, 0.0, 0.03])
    ok, _, _ = sm._grasp_gate_ok(obj.copy(), obj)
    assert ok
    just_out_x = obj + np.array([sm.grasp_xyz_ok[0] + 1e-4, 0.0, 0.0])
    assert not sm._grasp_gate_ok(just_out_x, obj)[0]
    just_out_z = obj + np.array([0.0, 0.0, sm.grasp_xyz_ok[2] + 1e-4])
    assert not sm._grasp_gate_ok(just_out_z, obj)[0]


def test_place_tol_is_not_the_servo_tol(sm):
    assert sm.place_xy_tol > sm.servo_xy_tol
    assert sm._in_container(np.array([0.50 + sm.place_xy_tol - 0.005, 0.30, 0.05]))
    assert not sm._in_container(np.array([0.50 + sm.place_xy_tol + 0.005, 0.30, 0.05]))


def test_arrived_z_tol_is_looser_than_grasp_z(sm):
    """Carry / lower arrival (~30 mm) must not reuse the grasp close Z (~20 mm)."""
    assert sm.place_z_tol == pytest.approx(0.03, abs=1e-9)
    assert sm.place_z_tol > sm.grasp_z_tol


def test_go_home_retreats_straight_up_before_crossing(sm):
    """Physics grasp does not pin the placed object: leave vertically first.

    A big Z jump made DiffIK swing ~5 cm in XY and the open fingers swept the
    fruit out of the box (34% of g0_object_smoke successes).
    """
    sm.stage = STAGE_HOME
    ee = np.array([0.50, 0.30, 0.06])
    obj = np.array([0.50, 0.30, 0.04])
    start = ee.copy()
    for i in range(60):
        cmd = sm.step(_state(obj, np.zeros(3), i * sm.dt), ee, sm._down_quat)
        if ee[2] < start[2] + sm.release_retreat_m - sm.release_step_m:
            # Retreat leg: XY pinned to where home began, Z leads by one small step.
            assert np.allclose(cmd.position[:2], start[:2])
            assert cmd.position[2] - ee[2] <= sm.release_step_m + 1e-9
        ee = ee + 0.5 * (cmd.position - ee)  # crude lagging arm
    assert ee[2] > start[2] + sm.release_retreat_m - sm.release_step_m
    assert np.linalg.norm(ee[:2] - sm.init_pose[:2]) < np.linalg.norm(start[:2] - sm.init_pose[:2])


def test_blocked_lower_over_box_still_opens(sm):
    """Lower that cannot reach the drop height but sits over the box releases there."""
    sm.lift_z = 0.20
    sm.stage = STAGE_LOWER
    ee = np.array([0.50, 0.30, 0.15])  # stuck 9 cm above the drop point, on the box XY
    obj = ee - np.array([0.0, 0.0, 0.01])
    stages = []
    for i in range(int((sm.stage_timeout_s + 0.5) / sm.dt)):
        stages.append(sm.step(_state(obj, np.zeros(3), i * sm.dt), ee, sm._down_quat).stage)
    assert STAGE_OPEN in stages
    assert STAGE_HOME in stages and stages.index(STAGE_OPEN) < stages.index(STAGE_HOME)


def test_open_holds_the_hand_still(sm):
    sm.stage = STAGE_OPEN
    ee = np.array([0.50, 0.30, 0.09])
    obj = np.array([0.50, 0.30, 0.05])
    first = sm.step(_state(obj, np.zeros(3), 0.0), ee, sm._down_quat).position
    second = sm.step(_state(obj, np.zeros(3), sm.dt), ee + 0.003, sm._down_quat).position
    assert np.allclose(first, ee) and np.allclose(second, ee)


def test_stage_timeout_retries_then_gives_up(sm):
    """Intercept stalls re-intercept up to ``max_retries``, then go home."""
    obj = np.array([1.40, 0.0, 0.03])  # unreachable
    ee = np.array([0.45, 0.0, 0.40])
    stages = []
    for i in range(int(30.0 / sm.dt)):
        cmd = sm.step(_state(obj, np.zeros(3), i * sm.dt), ee, sm._down_quat)
        stages.append(cmd.stage)
    assert sm.retries > 0, "never retried"
    assert sm.retries <= sm.max_retries + 1
    assert max(stages) >= STAGE_APPROACH


def test_timeouts_are_per_stage_not_the_episode_window(sm):
    assert sm.stage_timeout_s < sm.inference_window_s
    assert sm.grasp_stage_timeout_s < sm.inference_window_s


def test_stage_schema_matches_the_constants():
    from sim.success import classify_failure
    from sim.state_machine import STAGE_SCHEMA

    assert STAGE_SCHEMA.done == STAGE_DONE
    assert STAGE_SCHEMA.lift == STAGE_LIFT
    assert sorted(STAGE_SCHEMA.names) == list(range(STAGE_RESET, STAGE_DONE + 1))
    # v1's numbering would mislabel a finished v2 episode as a failure.
    assert (
        classify_failure(
            success=True,
            max_stage=STAGE_DONE,
            object_z_min=0.03,
            object_z_max=0.20,
            object_z_end=0.06,
            schema=STAGE_SCHEMA,
        )
        == "success"
    )
    assert (
        classify_failure(
            success=False,
            max_stage=STAGE_DESCEND,
            object_z_min=0.03,
            object_z_max=0.03,
            object_z_end=0.03,
            schema=STAGE_SCHEMA,
        )
        == "too-late"
    )


def test_motion_coasts_straight():
    from sim.motion import kinematic_forecast

    pos, vel = np.array([0.69, 0.0, 0.03]), np.array([0.4, 0.0, 0.0])
    p, v = kinematic_forecast(pos, vel, 0.5, 0.04)
    np.testing.assert_allclose(p, pos + vel * 0.5, atol=1e-9)
    np.testing.assert_allclose(v, vel)


def test_slip_goes_back_to_intercept(sm):
    """Physics grasp: object far from the hand during lift → re-intercept, not carry on."""
    sm.stage = STAGE_LIFT
    sm.lift_z = 0.2
    ee = np.array([0.55, 0.0, 0.15])
    obj = np.array([0.55, 0.0, 0.03])  # 12 cm below the hand
    cmd = sm.step(_state(obj, np.zeros(3)), ee, sm._down_quat)
    assert cmd.stage == STAGE_APPROACH and sm.retries == 1
    # Held (object at the hand) keeps lifting.
    sm.stage = STAGE_LIFT
    cmd = sm.step(_state(ee.copy(), np.zeros(3)), ee, sm._down_quat)
    assert cmd.stage == STAGE_LIFT


def test_wrist_depth_cannot_override_gt_z(cfg):
    """Wrist depth must not replace Z gate (seed-260 stuck-arm / early-close bug)."""
    cfg["state_machine_params"]["close_depth_m"] = 0.08
    sm = PickPlaceStateMachine(cfg, dt=float(cfg["sim"]["dt"]))
    sm.set_box_poses(CARRY, drop_pos=DROP)
    obj = np.array([0.55, 0.0, 0.03])
    zg = sm._update_z_grasp(obj.copy(), obj)
    high = np.array([0.55, 0.0, zg + 0.06])  # too high vs z_grasp
    on_grasp = np.array([0.55, 0.0, zg])
    sm.stage = STAGE_APPROACH
    # Shallow wrist used to force a close here — that is no longer allowed.
    assert sm.step(_state(obj, np.zeros(3)), high, sm._down_quat, wrist_depth_m=0.05).stage == STAGE_APPROACH
    sm.stage = STAGE_APPROACH
    assert sm.step(_state(obj, np.zeros(3)), on_grasp, sm._down_quat, wrist_depth_m=0.30).stage == STAGE_CLOSE
    sm.stage = STAGE_APPROACH
    assert sm.step(_state(obj, np.zeros(3)), on_grasp, sm._down_quat, wrist_depth_m=float("nan")).stage == STAGE_CLOSE


def test_hover_over_bowl_is_not_success():
    """Stalled lower that freezes the object mid-air over the bowl must fail."""
    from sim.success import episode_success

    n = 20
    stages = np.full(n, 9, dtype=np.int64)
    # Lifted, finished stages, XY over bowl — but hovering at carry height.
    pos = np.tile([0.50, 0.30, 0.15], (n, 1)).astype(np.float64)
    pos[:5, 2] = 0.20  # peak lift
    cpos = np.tile([0.50, 0.30, 0.06], (n, 1)).astype(np.float64)
    ep = {"stage": stages, "object_pos": pos, "container_pos": cpos}
    assert episode_success(ep, container_xy=np.array([0.50, 0.30])) is False
    # Same run but object resting in/on the bowl.
    pos[:, 2] = 0.08
    pos[:5, 2] = 0.20
    assert episode_success(ep, container_xy=np.array([0.50, 0.30])) is True


def test_physics_grasp_helpers():
    from sim.physics_grasp import gripper_depth_m, xy_outside_box

    d = np.full((40, 60), np.inf, dtype=np.float32)
    assert np.isnan(gripper_depth_m(d)) and np.isnan(gripper_depth_m(None))
    d[35, 30] = 0.07  # bottom centre, between the fingertips
    d[20, 30] = 0.01  # image centre (table) is outside the window
    d[39, 2] = 0.01  # a fingertip in the bottom corner is outside too
    assert gripper_depth_m(d[..., None]) == pytest.approx(0.07, abs=1e-6)
    lo, hi = np.array([0.0, -0.5]), np.array([1.2, 0.5])
    assert not xy_outside_box([0.6, 0.0], lo, hi)
    assert xy_outside_box([1.3, 0.0], lo, hi) and xy_outside_box([0.6, -0.6], lo, hi)


def test_off_table_flag_labels_a_lost_object():
    from sim.success import classify_failure

    kw = dict(success=False, max_stage=8, object_z_min=0.03, object_z_max=0.2,
              object_z_end=0.03, schema=None)
    assert classify_failure(**kw) == "timeout"
    assert classify_failure(**kw, off_table=True) == "off-table"


def test_off_table_during_chase_not_too_late():
    """Fall mid-intercept used to be labelled too-late; name the fall."""
    from sim.success import classify_failure

    assert (
        classify_failure(
            success=False,
            max_stage=2,
            object_z_min=-0.07,
            object_z_max=0.04,
            object_z_end=-0.07,
            off_table=True,
        )
        == "off-table"
    )


def test_abort_reason_is_safety_abort():
    from sim.success import classify_failure

    assert (
        classify_failure(
            success=False,
            max_stage=2,
            object_z_min=0.04,
            object_z_max=0.04,
            object_z_end=0.04,
            abort_reason="object-through-container",
        )
        == "safety-abort"
    )


def test_release_drag_rule_needs_the_hand_to_move():
    """Fruit rolling in the box under a still hand is not a drag; a sweep is."""
    from sim.success import release_disturbed

    stage = np.array([7, 8, 8, 8, 8, 8, 8])
    still_hand = np.tile([0.49, -0.207, 0.09], (7, 1)) + np.linspace(0, 0.001, 7)[:, None]
    rolling = np.array([[0.490 - 0.004 * i, -0.206 + 0.004 * i, 0.035] for i in range(7)])
    assert not release_disturbed(stage, rolling, still_hand)
    sweeping_hand = np.array([[0.49 - 0.015 * i, -0.207, 0.09 + 0.005 * i] for i in range(7)])
    dragged = np.array([[0.49 - 0.015 * i, -0.207, 0.035] for i in range(7)])
    assert release_disturbed(stage, dragged, sweeping_hand)


def test_slip_check_uses_hand_at_observation_time(sm):
    """B1: a held object seen Δ late trails the live hand; that is not a slip."""
    sm.stage = STAGE_CARRY
    sm.lift_z = 0.15
    vel = np.array([0.0, 0.40, 0.0])  # carrying at 40 cm/s
    delta = 0.30
    k = int(round(delta / sm.dt))
    ees = [np.array([0.5, 0.0 + 0.40 * sm.dt * i, 0.15]) for i in range(k + 5)]
    cmd = None
    for i, ee in enumerate(ees):
        # Delayed frame: object sat in the hand k ticks ago (12 cm behind now).
        obj = ees[max(i - k, 0)]
        cmd = sm.step(_state(obj, vel), ee, sm._down_quat, delta_meas_s=delta)
    assert cmd.stage == STAGE_CARRY
    # A real slip (object left behind 15 cm from where the hand was) still triggers.
    far = ees[-1 - k] + np.array([0.0, -0.15, -0.10])
    cmd = sm.step(_state(far, np.zeros(3)), ees[-1] + [0, 0.40 * sm.dt, 0], sm._down_quat, delta_meas_s=delta)
    assert cmd.stage != STAGE_CARRY
