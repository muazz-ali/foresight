"""Step 5: predictive close gate + hand–object speed match.

Plain words: fingers take ``t_contact`` ≈ 130 ms to reach the pads (cmd +
travel; force ramp is extra). Close when the hand–object gap *at contact*
is still inside 2 mm, and speeds already match:

  |e + Δv_rel · t_contact| ≤ 2 mm   (per XY axis)
  |v_hand − v_obj| ≤ 2.5 mm / t_close ≈ 1.39 cm/s   (t_close = 180 ms)

``e = hand_xy − object_xy`` (+ = hand ahead on that axis).
``Δv_rel = v_hand − v_obj``. Coast both for ``t_contact`` → predicted gap.

Always pass ``close_time_s`` / ``t_contact_s`` from the actuator model
(``sim.gripper_actuator``), not hard-coded magic numbers at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Plan numbers: 2.5 mm slack over SPEC-DERIVED t_close (180 ms → 1.39 cm/s).
DEFAULT_SLACK_M = 0.0025
DEFAULT_CLOSE_TIME_S = 0.180
# Pad contact (no force ramp): 30 ms cmd + 5 mm / 50 mm/s = 130 ms.
DEFAULT_T_CONTACT_S = 0.130
DEFAULT_XY_OK_M = 0.002


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


@dataclass(frozen=True)
class PredictiveCloseCheck:
    """Predictive XY gap at pad contact + speed match."""

    e_now_m: np.ndarray
    e_pred_m: np.ndarray
    speed_diff_m_s: float
    speed_limit_m_s: float
    xy_ok_m: float
    t_contact_s: float
    ok_xy: bool
    ok_speed: bool
    ok: bool


def predictive_close_check(
    hand_xy: np.ndarray,
    obj_xy: np.ndarray,
    v_hand: np.ndarray,
    v_obj: np.ndarray,
    *,
    t_contact_s: float = DEFAULT_T_CONTACT_S,
    xy_ok_m: float = DEFAULT_XY_OK_M,
    slack_m: float = DEFAULT_SLACK_M,
    close_time_s: float = DEFAULT_CLOSE_TIME_S,
) -> PredictiveCloseCheck:
    """Close when predicted gap at contact is inside ``xy_ok`` and speeds match.

    ``e = hand − object`` (metres). Predicted: ``e_pred = e + (v_hand − v_obj)·t``.
    Per-axis ``|e_pred| ≤ xy_ok`` (same shape as the old 2 mm settle box).
    """
    h = np.asarray(hand_xy, dtype=np.float64).reshape(-1)[:2]
    o = np.asarray(obj_xy, dtype=np.float64).reshape(-1)[:2]
    vh = np.asarray(v_hand, dtype=np.float64).reshape(-1)[:2]
    vo = np.asarray(v_obj, dtype=np.float64).reshape(-1)[:2]
    tc = float(t_contact_s)
    if tc < 0.0:
        raise ValueError(f"t_contact_s must be ≥ 0, got {tc}")
    e = h - o
    e_pred = e + (vh - vo) * tc
    spd = close_speed_check(vh, vo, slack_m=slack_m, close_time_s=close_time_s)
    # Both now and at contact must sit in the 2 mm box. Predicting alone would
    # fire while still 2.5 mm short; requiring |e_now| too waits until the hand
    # is actually inside before the pads start moving.
    ok_xy = bool(
        abs(float(e[0])) <= float(xy_ok_m) + 1e-12
        and abs(float(e[1])) <= float(xy_ok_m) + 1e-12
        and abs(float(e_pred[0])) <= float(xy_ok_m) + 1e-12
        and abs(float(e_pred[1])) <= float(xy_ok_m) + 1e-12
    )
    return PredictiveCloseCheck(
        e_now_m=e.copy(),
        e_pred_m=e_pred.copy(),
        speed_diff_m_s=float(spd.speed_diff_m_s),
        speed_limit_m_s=float(spd.limit_m_s),
        xy_ok_m=float(xy_ok_m),
        t_contact_s=tc,
        ok_xy=ok_xy,
        ok_speed=bool(spd.ok),
        ok=bool(ok_xy and spd.ok),
    )


def match_hand_xy_to_object(
    v_obj: np.ndarray,
    v_hand: np.ndarray,
) -> np.ndarray:
    """Copy object XY speed onto the hand command; keep hand Z speed.

    Helper for a scripted close that must stay under the t_close speed bar.
    """
    vo = np.asarray(v_obj, dtype=np.float64).reshape(3)
    vh = np.asarray(v_hand, dtype=np.float64).reshape(3).copy()
    vh[0:2] = vo[0:2]
    return vh
