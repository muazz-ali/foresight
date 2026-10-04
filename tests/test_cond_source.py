"""Oracle vs filter pack choice. No Isaac."""

from __future__ import annotations

import numpy as np

from eval.cond_source import FILTER_WARMUP_BEFORE, choose_pack
from interfaces.state import oracle_from_gt


def _state(xy, vxy, *, valid=True):
    return oracle_from_gt(
        [xy[0], xy[1], 0.03],
        [vxy[0], vxy[1], 0.0],
        timestamp=0.0,
        valid=valid,
    )


def test_warmup_is_at_least_two_frames():
    assert FILTER_WARMUP_BEFORE >= 2


def test_oracle_pack_matches_truth():
    obj = _state((0.5, 0.1), (0.2, 0.0))
    out = choose_pack(source="oracle", oracle_state=obj, filter_state=None, delta=0.25, zero=False)
    np.testing.assert_allclose(out["cond"], out["truth"])
    assert out["filter"] is None
    # p̂ = p + v·Δ
    np.testing.assert_allclose(out["cond"][:2], [0.5 + 0.05, 0.1])


def test_filter_pack_uses_filter_state():
    oracle = _state((0.50, 0.10), (0.20, 0.0))
    filt = _state((0.51, 0.11), (0.18, 0.01))
    out = choose_pack(source="filter", oracle_state=oracle, filter_state=filt, delta=0.25, zero=False)
    np.testing.assert_allclose(out["cond"], out["filter"])
    assert not np.allclose(out["cond"], out["truth"])
    assert out["valid"] == 1.0


def test_missing_filter_is_zeros_and_invalid():
    oracle = _state((0.5, 0.1), (0.2, 0.0))
    out = choose_pack(source="filter", oracle_state=oracle, filter_state=None, delta=0.25, zero=False)
    assert np.all(out["cond"] == 0.0)
    assert out["valid"] == 0.0
    assert not np.allclose(out["truth"], 0.0)


def test_invalid_filter_still_packs_its_numbers():
    oracle = _state((0.5, 0.1), (0.2, 0.0))
    filt = _state((0.4, 0.0), (0.0, 0.0), valid=False)
    out = choose_pack(source="filter", oracle_state=oracle, filter_state=filt, delta=0.25, zero=False)
    assert out["valid"] == 0.0
    np.testing.assert_allclose(out["cond"][:2], [0.4, 0.0])


def test_zero_does_not_mix_with_filter():
    oracle = _state((0.5, 0.1), (0.2, 0.0))
    try:
        choose_pack(source="filter", oracle_state=oracle, filter_state=None, delta=0.25, zero=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
