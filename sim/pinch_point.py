"""Geometric gripper pinch point (pad midpoint) vs configured TCP.

Plain words: DiffIK aims at ``tcp_offset`` from ``panda_hand``. 
The real grasp point is actually between the finger pads. Franka URDF (IsaacLab lula_franka_gen):

  finger joint origin on hand:  z = 0.0584 m
  fingertip fixed on finger:    z = 0.045 m
  → pinch mid-depth in hand:    z = 0.1034 m  (XY = 0 when fingers symmetric)

Configured TCP is often ``[0, 0, 0.107]``. 
Callers use this module to measure the offset between the configured TCP and the real grasp point in the hand frame,


Usage::

    from sim.pinch_point import pinch_in_hand_m, pinch_world_from_hand

    h_hand = pinch_in_hand_m()                    # (3,) in panda_hand
    h_w = pinch_world_from_hand(hand_pos, hand_quat_wxyz)
"""

from __future__ import annotations

import numpy as np

# URDF constants (metres). Do not change without updating the unit test.
FINGER_JOINT_ORIGIN_Z_M = 0.0584 # the z-coordinate of the finger joint origin on the hand frame (58.4 mm)
FINGERTIP_OFFSET_Z_M = 0.045 # the z-coordinate of the fingertip fixed on the finger (45.0 mm)
# Midpoint between fingertip frames when fingers are symmetric (pad mid-depth).
PINCH_Z_IN_HAND_M = FINGER_JOINT_ORIGIN_Z_M + FINGERTIP_OFFSET_Z_M  # 0.1034


def pinch_in_hand_m() -> np.ndarray:
    """Pinch point in ``panda_hand`` frame (metres). Always ``[0, 0, 0.1034]``."""
    return np.array([0.0, 0.0, float(PINCH_Z_IN_HAND_M)], dtype=np.float64)


def tcp_minus_pinch_in_hand_m(tcp_offset_m: np.ndarray | list[float]) -> np.ndarray:
    """Configured TCP minus geometric pinch, in the hand frame (metres)."""
    tcp = np.asarray(tcp_offset_m, dtype=np.float64).reshape(3)
    return tcp - pinch_in_hand_m()


def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation matrix (hand → world)."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pinch_world_from_hand(
    hand_pos_w: np.ndarray,
    hand_quat_wxyz: np.ndarray,
    *,
    pinch_hand_m: np.ndarray | None = None,
) -> np.ndarray:
    """World-frame pinch = hand pose ⊕ pinch vector in the hand frame."""
    p = pinch_in_hand_m() if pinch_hand_m is None else np.asarray(pinch_hand_m, np.float64)
    R = _quat_wxyz_to_rot(hand_quat_wxyz)
    return np.asarray(hand_pos_w, dtype=np.float64).reshape(3) + R @ p.reshape(3)


def pinch_from_finger_bodies(
    left_pos_w: np.ndarray,
    right_pos_w: np.ndarray,
) -> np.ndarray:
    """Pinch = midpoint of left/right finger (or fingertip) body positions."""
    a = np.asarray(left_pos_w, dtype=np.float64).reshape(3)
    b = np.asarray(right_pos_w, dtype=np.float64).reshape(3)
    return 0.5 * (a + b)
