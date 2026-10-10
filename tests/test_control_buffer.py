"""Isaac-free tests for control.ActionBuffer."""

import numpy as np
import pytest

from control import ActionBuffer


def chunk(v, n=4, d=2):
    return np.full((n, d), float(v))


def test_pop_due_order_and_none():
    b = ActionBuffer()
    b.push_chunk(np.arange(8.0).reshape(4, 2), t0=1.0, dt=0.1)
    assert b.pop_due(0.9) is None
    assert b.pop_due(1.0).tolist() == [0.0, 1.0]
    assert b.pop_due(1.0) is None  # already used
    assert b.pop_due(1.12).tolist() == [2.0, 3.0]


def test_pop_skips_old_slots_and_returns_newest_due():
    b = ActionBuffer(max_age=1.0)
    b.push_chunk(np.arange(8.0).reshape(4, 2), t0=0.0, dt=0.1)
    assert b.pop_due(0.25).tolist() == [4.0, 5.0]  # slot at 0.2
    assert len(b) == 1  # only slot 0.3 left


def test_stale_action_not_used():
    b = ActionBuffer(max_age=0.05)
    b.push_chunk(chunk(1), t0=0.0, dt=0.1)
    assert b.pop_due(0.5) is None  # slot 0.3 is 0.2 s old
    assert len(b) == 0


def test_new_chunk_overwrites_future():
    b = ActionBuffer()
    b.push_chunk(chunk(1, n=10), t0=0.0, dt=0.1)  # slots 0.0..0.9
    b.push_chunk(chunk(2, n=3), t0=0.35, dt=0.1)  # slots 0.35, 0.45, 0.55
    assert len(b) == 4 + 3  # old 0.0..0.3 kept, old 0.4+ gone
    assert b.pop_due(0.6)[0] == 2.0
    assert len(b) == 0


def test_now_drops_past_slots():
    b = ActionBuffer()
    b.push_chunk(chunk(1, n=10), t0=0.0, dt=0.1, now=0.5)
    assert len(b) == 5
    assert b.t[0] == pytest.approx(0.5)


def test_blend_ramps_from_old_to_new():
    b = ActionBuffer()
    b.push_chunk(chunk(0, n=10), t0=0.0, dt=0.1)
    b.push_chunk(chunk(1, n=5), t0=0.2, dt=0.1, blend=3)
    vals = [b.a[np.isclose(b.t, t)][0, 0] for t in (0.2, 0.3, 0.4, 0.5)]
    assert vals == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_bad_inputs():
    b = ActionBuffer()
    with pytest.raises(ValueError):
        b.push_chunk(np.zeros(3), 0.0, 0.1)
    with pytest.raises(ValueError):
        b.push_chunk(chunk(1), 0.0, 0.0)
    b.push_chunk(chunk(1, d=2), 0.0, 0.1)
    with pytest.raises(ValueError):
        b.push_chunk(chunk(1, d=3), 1.0, 0.1)
