"""Step 5: hand–object speed match while the gripper closes.

Plain words: fingers take ~80 ms to shut. In that time the object must not
slide more than ~2.5 mm across the pads, so the hand speed and object speed
must stay close:

  |v_hand − v_obj|  <  2.5 mm / 0.08 s  ≈  3.1 cm/s

Use this as a check (scripted or policy) during the close window. It does not
move the robot by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Plan numbers: 2.5 mm slack over 80 ms close → 3.125 cm/s.
DEFAULT_SLACK_M = 0.0025
DEFAULT_CLOSE_TIME_S = 0.080


def max_speed_diff_m_s(
    *,
    slack_m: float = DEFAULT_SLACK_M,
    close_time_s: float = DEFAULT_CLOSE_TIME_S,
) -> float:
    """Largest allowed |v_hand − v_obj| (m/s) during close."""
    ct = float(close_time_s)
    if ct <= 0.0:
        raise ValueError(f"close_time_s must be > 0, got {ct}")
    return float(slack_m) / ct


@dataclass(frozen=True)
class CloseSpeedCheck:
    """One sample of the close-speed rule."""

    speed_diff_m_s: float
    limit_m_s: float
    ok: bool


def close_speed_check(
    v_hand: np.ndarray,
    v_obj: np.ndarray,
    *,
    slack_m: float = DEFAULT_SLACK_M,
    close_time_s: float = DEFAULT_CLOSE_TIME_S,
) -> CloseSpeedCheck:
    """Compare hand vs object velocity (usually XY; Z included in the norm)."""
    vh = np.asarray(v_hand, dtype=np.float64).reshape(-1)
    vo = np.asarray(v_obj, dtype=np.float64).reshape(-1)
    n = min(vh.size, vo.size)
    diff = float(np.linalg.norm(vh[:n] - vo[:n]))
    limit = max_speed_diff_m_s(slack_m=slack_m, close_time_s=close_time_s)
    return CloseSpeedCheck(speed_diff_m_s=diff, limit_m_s=limit, ok=diff <= limit)


def match_hand_xy_to_object(
    v_obj: np.ndarray,
    v_hand: np.ndarray,
) -> np.ndarray:
    """Copy object XY speed onto the hand command; keep hand Z speed.

    Helper for a scripted close that must stay under the 3.1 cm/s bar.
    """
    vo = np.asarray(v_obj, dtype=np.float64).reshape(3)
    vh = np.asarray(v_hand, dtype=np.float64).reshape(3).copy()
    vh[0:2] = vo[0:2]
    return vh
