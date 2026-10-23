"""P_s capture model + close-timing rule (control/capture.py)."""

import numpy as np

from control.capture import PARAMS_STAGE0, CaptureRule, gripper_offset, p_success

DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, hand pointing down (init pose)
R, HC = 0.035, 0.00175  # tangerine06 capsule
ZC = 0.0255


def _ps(dx=0.0, dy=0.0, dz=0.015):
    obj = np.array([0.5, 0.1, ZC])
    tcp = obj + np.array([dx, dy, dz])
    return float(p_success(gripper_offset(tcp, DOWN, obj), R, HC, ZC, PARAMS_STAGE0)[0])


def test_offset_axes_and_height_sign():
    obj = np.array([0.5, 0.1, ZC])
    e = gripper_offset(obj + np.array([0.0, 0.0, 0.02]), DOWN, obj)
    assert abs(e[2] - 0.02) < 1e-9  # TCP 2 cm above centre -> +2 cm
    e = gripper_offset(obj + np.array([0.01, 0.0, 0.0]), DOWN, obj)
    assert abs(abs(e[0]) - 0.01) < 1e-9 and abs(e[1]) < 1e-9  # world x = pad axis


def test_capture_edges_match_stage0():
    assert _ps() > 0.9
    assert _ps(dx=0.03) < 0.05  # past the 2 cm pad edge
    assert _ps(dy=0.02) < 0.05  # past the fitted ~1.1 cm closing-axis slack
    assert _ps(dz=0.035) < 0.05  # above the friction-cone height
    assert _ps(dx=0.01) > 0.8


def test_rule_fires_at_peak_not_before():
    rule = CaptureRule(R, HC, p_min=0.5)
    obj, v_obj = np.array([0.5, 0.1, ZC]), np.zeros(3)
    v_hand = np.array([-0.10, 0.0, 0.0])  # hand sweeps along the pad axis toward the object
    fired_at = None
    for k in range(20):
        tcp = obj + np.array([0.05 + v_hand[0] * 0.04 * k, 0.0, 0.015])
        now, nxt = rule.evaluate(tcp, DOWN, obj, v_obj, v_hand, 0.04)
        if rule.fire(now, nxt):
            fired_at = tcp[0] - obj[0]
            break
    assert fired_at is not None and abs(fired_at) <= 0.004  # closes at the centre, not at the edge
