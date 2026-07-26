#!/usr/bin/env python3
"""Unit tests for frozen ObjectState (no Isaac required)."""

from __future__ import annotations

import numpy as np

from interfaces.state import conditioning_vector, oracle_from_gt


def test_oracle_predict_and_condition():
    st = oracle_from_gt([0.4, 0.0, 0.05], [0.2, 0.0, 0.0], timestamp=1.0)
    assert st.valid
    fut = st.predict(0.25)
    np.testing.assert_allclose(fut.position, [0.45, 0.0, 0.05], atol=1e-9)
    vec = conditioning_vector(st, delta=0.25)
    assert vec.shape == (12,)
    np.testing.assert_allclose(vec[:3], fut.position)
    assert vec[6] == 0.25
    assert vec[10] == 1.0  # valid


def test_flat_roundtrip_len():
    st = oracle_from_gt([0, 0, 0], [0, 0, 0], 0.0)
    assert st.to_flat().shape == (14,)


if __name__ == "__main__":
    test_oracle_predict_and_condition()
    test_flat_roundtrip_len()
    print("interfaces/state OK")
