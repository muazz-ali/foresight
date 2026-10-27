"""Pinch point = pad midpoint from Franka URDF geometry."""

from __future__ import annotations

import numpy as np
import pytest

from sim.pinch_point import (
    FINGER_JOINT_ORIGIN_Z_M,
    FINGERTIP_OFFSET_Z_M,
    PINCH_Z_IN_HAND_M,
    pinch_from_finger_bodies,
    pinch_in_hand_m,
    pinch_world_from_hand,
    tcp_minus_pinch_in_hand_m,
)


def test_pinch_in_hand_is_pad_midpoint_from_urdf():
    """Hand-frame pinch Z = finger origin + fingertip offset (0.1034 m)."""
    assert FINGER_JOINT_ORIGIN_Z_M == pytest.approx(0.0584)
    assert FINGERTIP_OFFSET_Z_M == pytest.approx(0.045)
    assert PINCH_Z_IN_HAND_M == pytest.approx(0.1034)
    p = pinch_in_hand_m()
    np.testing.assert_allclose(p, [0.0, 0.0, 0.1034], atol=1e-12)


def test_tcp_107mm_minus_pinch_is_plus_3p6mm_along_hand_z():
    """Legacy DiffIK TCP 107 mm sat 3.6 mm past the pad midpoint along hand Z."""
    d = tcp_minus_pinch_in_hand_m([0.0, 0.0, 0.107])
    np.testing.assert_allclose(d, [0.0, 0.0, 0.0036], atol=1e-12)


def test_tcp_103p4mm_matches_pinch():
    """Production tcp_offset = pad midpoint → tcp − pinch = 0."""
    d = tcp_minus_pinch_in_hand_m([0.0, 0.0, 0.1034])
    np.testing.assert_allclose(d, [0.0, 0.0, 0.0], atol=1e-12)


def test_pinch_from_finger_bodies_is_midpoint():
    left = np.array([1.0, 0.04, 0.2])
    right = np.array([1.0, -0.04, 0.2])
    np.testing.assert_allclose(
        pinch_from_finger_bodies(left, right), [1.0, 0.0, 0.2], atol=1e-12
    )


def test_pinch_world_from_identity_hand():
    """Identity hand pose → world pinch equals hand-frame pinch."""
    h = pinch_world_from_hand(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(h, pinch_in_hand_m(), atol=1e-12)
