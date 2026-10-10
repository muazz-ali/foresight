"""Isaac-free: reflex overlay rebuilds an ExpertCommand-like dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control import ReflexClose, overlay_reflex
from interfaces.state import ObjectState


@dataclass
class _Cmd:
    position: np.ndarray
    quat_wxyz: np.ndarray
    gripper_open: float
    stage: int
    kinematic_object: bool
    attach_object: bool
    freeze_object: bool = False


def test_overlay_inactive_far_away():
    reflex = ReflexClose(radius=0.10)
    obj = ObjectState(
        position=np.array([1.0, 0.0, 0.05]),
        velocity=np.zeros(3),
        covariance=np.eye(6) * 1e-4,
        timestamp=0.0,
        valid=True,
    )
    cmd = _Cmd(
        position=np.array([0.0, 0.0, 0.4]),
        quat_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
        gripper_open=1.0,
        stage=2,
        kinematic_object=True,
        attach_object=False,
    )
    out_cmd, rout = overlay_reflex(cmd, reflex, obj, cmd.position)
    assert rout is not None and not rout.active
    assert out_cmd.gripper_open == 1.0
    assert not out_cmd.attach_object


def test_overlay_closes_when_near():
    reflex = ReflexClose(close_time=0.2, radius=0.10, touch_dist=0.02, vmax=0.5)
    ee = np.array([0.50, 0.0, 0.05])
    obj = ObjectState(
        position=np.array([0.505, 0.0, 0.05]),
        velocity=np.zeros(3),
        covariance=np.eye(6) * 1e-4,
        timestamp=0.0,
        valid=True,
    )
    # Stage 2 (descend): may steer, but must not force attach yet.
    cmd = _Cmd(
        position=ee.copy(),
        quat_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
        gripper_open=1.0,
        stage=2,
        kinematic_object=True,
        attach_object=False,
    )
    out_cmd, rout = overlay_reflex(cmd, reflex, obj, ee)
    assert rout is not None and rout.active
    assert not out_cmd.attach_object
    # Stage 3 (close): may force the grasp.
    cmd.stage = 3
    out_cmd, rout = overlay_reflex(cmd, reflex, obj, ee)
    assert rout is not None and rout.close
    assert out_cmd.gripper_open == 0.0
    assert out_cmd.attach_object
    assert not out_cmd.kinematic_object
