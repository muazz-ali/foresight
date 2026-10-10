"""Isaac-free tests for control.ReflexClose."""

import numpy as np
import pytest

from control import ReflexClose
from interfaces.state import ObjectState


def obj(p, v=(0, 0, 0), t=0.0, valid=True):
    return ObjectState(p, v, np.full(6, 1e-4), t, valid)


def test_far_or_invalid_is_inactive():
    r = ReflexClose()
    assert not r.step(obj([1, 0, 0]), [0, 0, 0], 0.0).active
    out = r.step(obj([0.05, 0, 0], valid=False), [0, 0, 0], 0.0)
    assert not out.active and not out.close


def test_steers_at_meeting_point_ahead_of_moving_object():
    r = ReflexClose()
    out = r.step(obj([0.08, 0, 0], v=(0, 0.1, 0)), [0, 0, 0], 0.0)
    assert out.active and out.target[1] > 0  # aimed ahead of the object
    assert out.vel_cmd[1] > 0 and out.vel_cmd[0] > 0
    assert np.linalg.norm(out.vel_cmd) <= r.vmax + 1e-9


def test_extrapolates_stale_reading():
    r = ReflexClose()
    out = r.step(obj([0.0, 0.0, 0.0], v=(0.1, 0, 0), t=0.0), [0.05, 0, 0], 0.5)
    # reading is old: object really at x=0.05 now, on top of the hand -> close
    assert out.close


def test_closes_only_when_tta_below_close_time():
    r = ReflexClose(close_time=0.15)
    # object rushing at hand at 0.5 m/s from 0.09 m: tta 0.18 s -> not yet
    o = r.step(obj([0.09, 0, 0], v=(-0.5, 0, 0)), [0, 0, 0], 0.0)
    assert o.tta == pytest.approx(0.18) and not o.close
    # 0.02 s later: object at 0.08 -> tta 0.16... keep going to 0.07 -> 0.14 s
    o = r.step(obj([0.09, 0, 0], v=(-0.5, 0, 0)), [0, 0, 0], 0.04)
    assert o.tta == pytest.approx(0.14) and o.close


def test_close_is_latched_until_reset():
    r = ReflexClose()
    r.step(obj([0.01, 0, 0]), [0, 0, 0], 0.0)  # touching
    assert r.closed
    out = r.step(obj([0.09, 0, 0]), [0, 0, 0], 0.0)
    assert out.close
    r.reset()
    assert not r.step(obj([0.09, 0, 0]), [0, 0, 0], 0.0).close


def test_moving_away_never_closes():
    r = ReflexClose()
    out = r.step(obj([0.08, 0, 0], v=(0.3, 0, 0)), [0, 0, 0], 0.0)
    assert out.tta == np.inf and not out.close


def test_future_timestamp_raises():
    with pytest.raises(ValueError):
        ReflexClose().step(obj([0.05, 0, 0], t=1.0), [0, 0, 0], 0.0)


def test_closed_loop_catches_moving_object():
    """Hand follows vel_cmd; object slides past. Gripper must close with hand near it."""
    r, dt = ReflexClose(), 0.002
    ee, p, v = np.array([0.0, -0.08, 0.0]), np.array([0.06, -0.08, 0.0]), np.array([-0.15, 0.1, 0.0])
    for k in range(1000):
        now = k * dt
        out = r.step(obj(p, v, now), ee, now)
        if out.close:
            break
        ee = ee + out.vel_cmd * dt
        p = p + v * dt
    assert out.close and np.linalg.norm(p - ee) < 0.03
