"""B2: DESCEND / intercept command Z tracks z_grasp (Isaac-free).

Run: pytest tests/test_b2_descend_z_grasp.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b2_descend_z_grasp import (  # noqa: E402
    _MIN_SPAN_M,
    _PROBE_HEIGHTS_M,
    _s_z_half_m,
    probe_sm,
    summarize_demos,
)
from interfaces.config import load_yaml  # noqa: E402
from interfaces.state import oracle_from_gt  # noqa: E402
from sim.state_machine import STAGE_DESCEND, PickPlaceStateMachine  # noqa: E402

_YAML = ROOT / "sim" / "phase0_cfg.yaml"
_CARRY = np.array([0.50, 0.30, 0.12], dtype=np.float64)
_DROP = np.array([0.50, 0.30, 0.06], dtype=np.float64)


@pytest.fixture
def cfg() -> dict:
    return load_yaml(_YAML)


def test_s_z_half_is_half_of_yaml_slack(cfg):
    """Budget for d_z spread is s_z/2 (5 mm slack → 2.5 mm)."""
    assert _s_z_half_m(cfg) == pytest.approx(0.0025, abs=1e-9)


def test_descend_cmd_z_equals_z_grasp_for_tall_and_short(cfg):
    """STAGE_DESCEND aim Z follows object height — never one desk height."""
    dt = float(cfg["sim"]["dt"])
    cmd_zs = []
    for half in (0.02, 0.08):
        sm = PickPlaceStateMachine(cfg, dt=dt)
        sm.set_box_poses(_CARRY, drop_pos=_DROP)
        sm.set_object_half_height(half, z_table=0.0)
        sm.stage = STAGE_DESCEND
        obj = np.array([0.55, 0.0, half])
        ee = np.array([0.45, 0.0, 0.40])
        cmd = sm.step(
            oracle_from_gt(obj, np.array([0.0, -0.2, 0.0]), 0.0),
            ee,
            sm._down_quat,
        )
        assert sm.z_grasp == pytest.approx(half)
        assert cmd.position[2] == pytest.approx(sm.z_grasp)
        cmd_zs.append(float(cmd.position[2]))
    assert cmd_zs[1] - cmd_zs[0] == pytest.approx(0.06, abs=1e-6)


def test_probe_sm_passes_on_shipped_expert(cfg):
    """Full B2 SM probe (APPROACH + DESCEND height sweep) must pass."""
    rep = probe_sm(cfg)
    assert rep["pass"]
    assert len(_PROBE_HEIGHTS_M) >= 2
    for s in rep["summaries"]:
        assert s["pass_tracks_z_grasp"]
        assert s["pass_not_fixed"]
        assert s["cmd_z_span_m"] >= _MIN_SPAN_M


def test_summarize_demos_fails_fixed_hand_z():
    """Hand Z stuck while object height moves → fail (fixed-height fingerprint)."""
    rows = [
        {
            "closed": True,
            "d_z_m": 0.05 - 0.02,
            "ee_z_m": 0.05,
            "z_grasp_m": 0.02,
        },
        {
            "closed": True,
            "d_z_m": 0.05 - 0.08,
            "ee_z_m": 0.05,
            "z_grasp_m": 0.08,
        },
    ]
    rep = summarize_demos(rows, s_z_half=0.0025, label="fake")
    assert rep["fixed_height_like"] is True
    assert rep["pass"] is False


def test_summarize_demos_passes_tight_d_z():
    """Hand tracks centre within s_z/2 → pass."""
    rows = [
        {
            "closed": True,
            "d_z_m": 0.001,
            "ee_z_m": 0.021,
            "z_grasp_m": 0.020,
        },
        {
            "closed": True,
            "d_z_m": -0.001,
            "ee_z_m": 0.079,
            "z_grasp_m": 0.080,
        },
    ]
    rep = summarize_demos(rows, s_z_half=0.0025, label="fake")
    assert rep["fixed_height_like"] is False
    assert rep["pass"] is True
