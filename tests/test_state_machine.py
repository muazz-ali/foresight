"""Isaac-free tests for the intercept-servo expert.

Run from the repo root:  pytest tests/test_state_machine.py -v

The speed tests use ``FakeArm``, a first-order surrogate for the IK + joint-PD
chain (``ee += (cmd - ee) * dt / tau`` with a TCP speed cap) plus the scene's
own ``kinematic_advance`` for the object. It is not the real sim, but it has
tracking lag and wall bounces — so a change that cannot pass here will not
pass on GPU either. Treat it as the cheap pre-filter, never as gate evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from interfaces.state import oracle_from_gt
from sim.motion import kinematic_advance
from sim.state_machine import (
    STAGE_APPROACH,
    STAGE_CLOSE,
    STAGE_DESCEND,
    STAGE_DONE,
    STAGE_LIFT,
    STAGE_RESET,
    PickPlaceStateMachine,
)

_REPO = Path(__file__).resolve().parents[1]
_YAML = _REPO / "sim" / "phase0_cfg.yaml"

CARRY = np.array([0.50, 0.30, 0.12], dtype=np.float64)
DROP = np.array([0.50, 0.30, 0.06], dtype=np.float64)


def load_cfg() -> dict:
    with _YAML.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def cfg() -> dict:
    return load_cfg()


@pytest.fixture
def sm(cfg) -> PickPlaceStateMachine:
    m = PickPlaceStateMachine(cfg, dt=float(cfg["sim"]["dt"]))
    m.set_box_poses(CARRY, drop_pos=DROP)
    return m


def _state(pos, vel, t=0.0):
    return oracle_from_gt(np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64), t)


# --------------------------------------------------------------- interception


def test_lead_collapses_to_tau_when_on_top(sm):
    """Riding the object → horizon is pure lag compensation, not 0.25 s ahead.

    This is the v1 bug: a fixed 0.25 s lead parks the hand ``0.25 * v`` in front
    of the object (20 cm at 80 cm/s) and it never closes.
    """
    obj = np.array([0.55, 0.0, 0.03])
    for speed in (0.0, 0.20, 0.40, 0.80):
        vel = np.array([speed, 0.0, 0.0])
        ee_on_top = np.array([obj[0], obj[1], obj[2] + sm.hover_z])
        lead = sm._intercept_lead_s(ee_on_top, obj, vel)
        assert lead == pytest.approx(sm.tau, abs=1e-6), f"speed {speed}"


def test_lead_is_capped_at_lookahead_when_far(sm):
    """Far away → the plan's 0.25 s look-ahead cap binds (plus tau)."""
    obj = np.array([0.70, 0.08, 0.03])
    ee = np.array([0.45, 0.0, 0.40])
    for speed in (0.20, 0.40, 0.80):
        lead = sm._intercept_lead_s(ee, obj, np.array([0.0, speed, 0.0]))
        assert lead == pytest.approx(sm.lookahead_s + sm.tau, abs=1e-6)


def test_approach_aims_ahead_of_a_moving_object(sm):
    """APPROACH command leads the object; the gate still reads the object now."""
    sm.stage = STAGE_APPROACH
    obj = np.array([0.55, 0.08, 0.03])
    vel = np.array([0.0, -0.40, 0.0])  # away from the near wall, so no bounce
    cmd = sm.step(_state(obj, vel), np.array([0.45, 0.0, 0.40]), sm._down_quat)
    assert cmd.position[1] < obj[1] - 0.05, "command must lead a 40 cm/s object"
    assert cmd.position[2] == pytest.approx(obj[2] + sm.hover_z)


def test_forecast_bounces_off_the_play_band(sm):
    """0.25 s at 80 cm/s is 20 cm — longer than the band, so it must bounce.

    A straight-line predict aims outside the table and the hand chases a point
    the object can never reach.
    """
    (xmin, xmax), _ = sm.xy_bounds
    obj = np.array([xmax - 0.05, 0.0, 0.03])
    vel = np.array([0.80, 0.0, 0.0])
    pred = sm._forecast(obj, vel, 0.25)
    assert pred[0] <= xmax + 1e-9, "prediction left the play band"
    assert pred[0] < (obj + vel * 0.25)[0], "prediction did not bounce"


# ------------------------------------------------------------------- gating


def test_descend_gate_is_measured_against_the_object_not_the_aim(sm):
    """Hand sitting exactly on a fast object passes, even though the aim leads."""
    sm.stage = STAGE_DESCEND
    obj = np.array([0.55, 0.0, 0.03])
    vel = np.array([0.80, 0.0, 0.0])
    cmd = sm.step(_state(obj, vel), obj.copy(), sm._down_quat)
    assert cmd.stage == STAGE_CLOSE
    assert sm.grasp_err_xy is not None and sm.grasp_err_xy < 1e-6


def test_grasp_tolerance_floors_with_speed(sm):
    """A gate tighter than one control step of object travel is unmeasurable."""
    slow = sm._grasp_tol_xy(np.zeros(3))
    fast = sm._grasp_tol_xy(np.array([2.0, 0.0, 0.0]))
    assert slow == pytest.approx(sm.grasp_xy_tol)
    assert fast > slow
    assert fast == pytest.approx(sm.grasp_tol_step_k * 2.0 * sm.dt)


def test_grasp_window_is_only_the_residual_descend(sm):
    """After attach the object is welded to the hand, so walls stop mattering.

    Only the leftover Z has to fit in a straight run — not the whole descend
    plus close, which is what makes 80 cm/s look impossible on paper.
    """
    obj = np.array([0.55, 0.0, 0.03])
    on_top = np.array([0.55, 0.0, 0.03 + sm.grasp_z_tol])
    high = np.array([0.55, 0.0, 0.03 + sm.hover_z])
    assert sm._grasp_window_s(on_top, obj) == pytest.approx(sm.bounce_guard_s)
    assert sm._grasp_window_s(high, obj) > sm._grasp_window_s(on_top, obj)
    assert sm._grasp_window_s() == pytest.approx(sm.bounce_guard_s)


def test_bounce_guard_blocks_a_descend_into_a_wall(sm):
    """Do not commit when the object reverses inside the grasp window."""
    (_, _), (ymin, ymax) = sm.xy_bounds
    obj = np.array([0.55, ymax - 0.01, 0.03])
    on_top = np.array([obj[0], obj[1], obj[2] + sm.grasp_z_tol])
    high = np.array([obj[0], obj[1], obj[2] + sm.hover_z])
    toward = np.array([0.0, 0.80, 0.0])  # reverses in 12 ms
    away = np.array([0.0, -0.80, 0.0])  # whole band ahead of it

    assert not sm._bounce_clear(obj, toward, on_top)
    assert sm._bounce_clear(obj, away, on_top)
    # A hand still at hover height needs more straight run than one already down,
    # which is exactly why APPROACH walks Z with the XY lock.
    assert sm._time_to_wall_s(obj, away) > sm._grasp_window_s(on_top, obj)
    assert sm._grasp_window_s(high, obj) > sm._grasp_window_s(on_top, obj)
    assert sm._bounce_clear(obj, np.zeros(3), high), "a still object is always clear"


def test_place_tol_is_not_the_servo_tol(sm):
    assert sm.place_xy_tol > sm.servo_xy_tol
    assert sm._in_container(np.array([0.50 + sm.place_xy_tol - 0.005, 0.30, 0.05]))
    assert not sm._in_container(np.array([0.50 + sm.place_xy_tol + 0.005, 0.30, 0.05]))


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


# --------------------------------------------------------- closed-loop check


class FakeArm:
    """First-order TCP tracker with a speed cap. Surrogate for IK + joint PD.

    Commands take effect one step late, matching the real loop: ``collect.py``
    reads the pose, calls the expert, writes the target, *then* steps physics.
    That is why the expert's ``tau`` is ``servo_lag_s + dt`` and not just the
    servo lag — drop the delay here and the surrogate flatters the expert by
    exactly ``v * dt`` (1.6 cm at 40 cm/s, more than ``track_xy_tol``).
    """

    def __init__(self, pos, *, tau: float, speed_max: float, dt: float):
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.tau = float(tau)
        self.speed_max = float(speed_max)
        self.dt = float(dt)
        self._pending = self.pos.copy()

    def follow(self, cmd: np.ndarray) -> np.ndarray:
        step = (self._pending - self.pos) * (self.dt / self.tau)
        n = float(np.linalg.norm(step))
        cap = self.speed_max * self.dt
        if n > cap:
            step *= cap / n
        self.pos = self.pos + step
        self._pending = np.asarray(cmd, dtype=np.float64).copy()
        return self.pos.copy()


def run_fake_episode(
    cfg: dict,
    speed: float,
    *,
    seed: int = 0,
    tau_true: float | None = None,
    speed_max_true: float | None = None,
    max_s: float = 10.0,
) -> dict:
    """Drive the expert against FakeArm + the scene's own object integrator."""
    dt = float(cfg["sim"]["dt"])
    sm = PickPlaceStateMachine(cfg, dt=dt)
    sm.set_box_poses(CARRY, drop_pos=DROP)

    rng = np.random.default_rng(seed)
    (xmin, xmax), (ymin, ymax) = sm.xy_bounds
    obj = np.array(
        [rng.uniform(xmin + 0.03, xmax - 0.03), rng.uniform(ymin + 0.03, ymax - 0.03), 0.03]
    )
    th = rng.uniform(0.0, 2.0 * np.pi)
    vel = np.array([speed * np.cos(th), speed * np.sin(th), 0.0])

    arm = FakeArm(
        cfg["robot"]["init_pose"][:3],
        tau=float(tau_true if tau_true is not None else sm.servo_lag_s),
        speed_max=float(speed_max_true if speed_max_true is not None else sm.ee_speed_max),
        dt=dt,
    )
    ee = arm.pos.copy()
    max_stage, z_max, attached = 0, float(obj[2]), False
    for i in range(int(max_s / dt)):
        cmd = sm.step(_state(obj, np.zeros(3) if attached else vel, i * dt), ee, sm._down_quat)
        ee = arm.follow(cmd.position)
        attached = bool(cmd.attach_object)
        if attached:
            obj = ee.copy()  # scene welds the object to the TCP
        elif cmd.freeze_object:
            pass
        elif cmd.kinematic_object:
            obj, vel = kinematic_advance(obj, vel, dt, xy_bounds=sm.xy_bounds)
            obj[2] = 0.03
        max_stage = max(max_stage, cmd.stage)
        z_max = max(z_max, float(obj[2]))
        if cmd.stage >= STAGE_DONE:
            break
    return {
        "max_stage": max_stage,
        "z_max": z_max,
        "done": max_stage >= STAGE_DONE,
        "grasp_err_xy": sm.grasp_err_xy,
        "retries": sm.retries,
        "grasp_window_s": sm._grasp_window_s(),
    }


@pytest.mark.parametrize("speed", [0.0, 0.20, 0.40])
def test_closed_loop_completes_at_speed(cfg, speed):
    """0 / 20 / 40 cm/s must finish from every seed with the shipped knobs.

    80 cm/s is covered by ``test_80cms_is_bounded_by_the_arm_not_the_logic``:
    it needs a faster arm, not different expert logic.
    """
    runs = [run_fake_episode(cfg, speed, seed=s) for s in range(10)]
    done = sum(r["done"] for r in runs)
    assert done == len(runs), [
        {k: r[k] for k in ("max_stage", "retries", "grasp_err_xy")} for r in runs
    ]
    for r in runs:
        assert r["z_max"] >= 0.12, "object never lifted"


def test_grasp_error_stays_honest_at_speed(cfg):
    """The scene welds object XY to the hand, so the gate is the only guard."""
    for speed in (0.20, 0.40):
        for seed in range(5):
            r = run_fake_episode(cfg, speed, seed=seed)
            if r["grasp_err_xy"] is None:
                continue
            assert r["grasp_err_xy"] <= 0.03, (speed, seed, r)


def test_80cms_is_bounded_by_the_arm_not_the_logic(cfg):
    """At 80 cm/s the binding constraint is arm bandwidth, not the state machine.

    The object reverses off a wall roughly every 0.34 s in the stock 24 x 30 cm
    band. An arm with a 0.16 s tracking lag needs ~3 tau (0.5 s) to re-converge
    after each reversal, so it never holds a lock long enough to commit — and no
    tolerance, look-ahead or gate change fixes that. Cutting the lag does.

    Surrogate sweep (done/10 at 80 cm/s):

        lag / TCP cap   stock band   wide band
        0.16 / 0.9        0/10         0/10
        0.12 / 1.3        6/10        10/10
        0.08 / 0.9        9/10        10/10

    So: get measured lag <= 0.08 s, or <= 0.12 s with a >= 1.3 m/s TCP cap and
    the wider play band. This test pins both ends so a future change cannot
    quietly claim 80 cm/s without moving one of those numbers.
    """
    stock = [run_fake_episode(cfg, 0.80, seed=s) for s in range(10)]
    assert sum(r["done"] for r in stock) == 0, "stock arm should not reach 80 cm/s"
    assert all(r["max_stage"] <= STAGE_DESCEND for r in stock), "expected an intercept stall"

    fast = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
    fast["state_machine_params"]["servo_lag_s"] = 0.08
    tuned = [
        run_fake_episode(fast, 0.80, seed=s, tau_true=0.08, speed_max_true=0.90)
        for s in range(10)
    ]
    assert sum(r["done"] for r in tuned) >= 8, [r["max_stage"] for r in tuned]
    for r in tuned:
        if r["grasp_err_xy"] is not None:
            assert r["grasp_err_xy"] <= 0.03, r


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
