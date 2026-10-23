"""Grasp-capture probability P_s and the close-timing rule (Stage 1).

P_s = chance the object sits inside the two-finger capture region when the close is
sent. The region comes from geometry (foresight_docs/RESEARCH.md §3.2, revised by
Stage 0):

  closing axis  |e_n - c_n| <= open_half - R - m_n    (fits between the fingers)
  pad axis      |e_t - c_t| <= a_t                    (inside the pad)
  height        h <= h_cyl + R * sin(atan(mu))        (contact inside the friction cone)
                h >= 0.005 - z_centre                 (fingertips above the table)

e is object minus TCP in the gripper frame (x = pad axis, y = closing axis,
z = down, so e_z = TCP height above the object centre). Each axis is a Gaussian
interval with std sigma. Stage 0 showed look-ahead hurts, so the offset is the one
at the moment of the close.

Rule: close at the first step with P_s >= p_min and P_s one step ahead <= P_s now
(the peak has come). The one-step-ahead offset uses the object and hand velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import ndtr

# Fit on 1,074 tangerine06 closes, horizon 0, 5-fold CV AUC 0.971, ECE 0.02.
# Source: data/eval/stage0/horizon0/stage0.json (calibrated_params_all).
PARAMS_STAGE0 = {"c_t": 0.0008, "c_n": 0.0012, "m_n": -0.0059, "a_t": 0.0195,
                 "mu": 1.105, "sigma": 0.0036, "p_max": 1.0}


@lru_cache(maxsize=64)
def collider_dims(usd_path: str) -> tuple[float, float]:
    """(radius, half-length of the straight part) of the object's USD capsule, metres."""
    from pxr import Usd

    st = Usd.Stage.Open(str(usd_path))
    scale = float(st.GetPrimAtPath("/Object").GetAttribute("xformOp:scale").Get()[2])
    col = st.GetPrimAtPath("/Object/collider")
    if not col or col.GetTypeName() != "Capsule":
        raise RuntimeError(f"{usd_path}: expected a Capsule collider at /Object/collider")
    r = float(col.GetAttribute("radius").Get()) * scale
    h = float(col.GetAttribute("height").Get()) * scale
    return r, h / 2.0


def gripper_offset(tcp: np.ndarray, quat_wxyz: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Object minus TCP in the gripper frame: (pad axis, closing axis, TCP height above object)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    rot = Rotation.from_quat(q[[1, 2, 3, 0]]).inv()
    return rot.apply(np.asarray(obj, dtype=np.float64) - np.asarray(tcp, dtype=np.float64))


def _interval(mu, lo, hi, s):
    return np.clip(ndtr((hi - mu) / s) - ndtr((lo - mu) / s), 0.0, 1.0)


def p_success(e: np.ndarray, radius: float, half_straight: float, z_centre: float,
              params: dict, open_half: float = 0.040) -> np.ndarray:
    """P_s for offsets e (..., 3)."""
    e = np.atleast_2d(np.asarray(e, dtype=np.float64))
    a_n = max(open_half - radius - params["m_n"], 1e-4)
    h_hi = half_straight + radius * np.sin(np.arctan(params["mu"]))
    h_lo = 0.005 - z_centre
    s = params["sigma"]
    pt = _interval(e[:, 0] - params["c_t"], -params["a_t"], params["a_t"], s)
    pn = _interval(e[:, 1] - params["c_n"], -a_n, a_n, s)
    ph = _interval(e[:, 2], h_lo, h_hi, s)
    return params["p_max"] * pt * pn * ph


@dataclass
class CaptureRule:
    """Stateless P_s evaluation plus the peak test for one object."""

    radius: float
    half_straight: float
    p_min: float = 0.5
    open_half: float = 0.040
    params: dict = field(default_factory=lambda: dict(PARAMS_STAGE0))

    def evaluate(self, tcp, quat_wxyz, obj_pos, obj_vel, hand_vel, dt: float) -> tuple[float, float]:
        """(P_s now, P_s one step ahead at current velocities)."""
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        tcp = np.asarray(tcp, dtype=np.float64)
        v_obj = np.asarray(obj_vel, dtype=np.float64).copy()
        v_obj[2] = 0.0  # object rides on the table until the close
        e_now = gripper_offset(tcp, quat_wxyz, obj_pos)
        e_next = gripper_offset(tcp + np.asarray(hand_vel, dtype=np.float64) * dt, quat_wxyz,
                                obj_pos + v_obj * dt)
        z_c = float(obj_pos[2])
        both = p_success(np.stack([e_now, e_next]), self.radius, self.half_straight, z_c,
                         self.params, self.open_half)
        return float(both[0]), float(both[1])

    def fire(self, ps_now: float, ps_next: float) -> bool:
        return ps_now >= self.p_min and ps_next <= ps_now + 1e-9
