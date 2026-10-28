"""B4: d_k invariance across speeds / delays / heights (Isaac-free).

Run: pytest tests/test_b4_dk_invariance.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b4_dk_invariance import (  # noqa: E402
    _ABS_SPAN_MIN_M,
    _D_TRUE,
    _budgets_m,
    expert_close_tau_s,
    summarize_demos,
    synthetic_grid,
)
from interfaces.config import load_yaml  # noqa: E402
from interfaces.forecast import forecast_horizon  # noqa: E402
from interfaces.state import oracle_from_gt  # noqa: E402

_YAML = ROOT / "sim" / "phase0_cfg.yaml"


@pytest.fixture
def cfg() -> dict:
    return load_yaml(_YAML)


def test_budgets_match_yaml_half_slack(cfg):
    """XY = grasp_xyz_ok/2 (5 mm); Z = z_slack/2 (2.5 mm)."""
    s_xy, s_z = _budgets_m(cfg)
    assert s_xy == pytest.approx(0.005, abs=1e-9)
    assert s_z == pytest.approx(0.0025, abs=1e-9)


def test_expert_close_tau_is_pad_only(cfg):
    """CLOSE lead = approach pad; no hand-written control tick."""
    pad = float(cfg["state_machine_params"]["robot_approach_pad_s"])
    assert expert_close_tau_s(cfg) == pytest.approx(pad, abs=1e-12)


def test_synthetic_grid_recovers_fixed_offset(cfg):
    """Tracking expert with fixed d_true → relabel d flat across the grid."""
    s_xy, s_z = _budgets_m(cfg)
    rep = synthetic_grid(s_xy=s_xy, s_z=s_z, d_true=_D_TRUE)
    assert rep["pass_match"]
    assert rep["pass_sigma"]
    assert rep["pass_abs_moves"]
    assert rep["pass_naive_leaks"]
    assert rep["pass"]
    assert rep["max_abs_err_mm"] == pytest.approx(0.0, abs=1e-9)
    assert rep["abs_span_mm"][0] >= 1000.0 * _ABS_SPAN_MIN_M
    assert rep["abs_span_mm"][2] >= 1000.0 * _ABS_SPAN_MIN_M


def test_naive_no_delta_absorbs_speed_times_delay():
    """Without baking Δ into p̂, d_x grows as v·Δ (the bug B4 catches)."""
    v, delta, z = 0.30, 0.300, 0.05
    d_true = np.array([0.0, 0.0, 0.03])
    p0 = np.array([0.50, 0.0, z])
    vel = np.array([v, 0.0, 0.0])
    state = oracle_from_gt(p0, vel, 0.0)
    phat = forecast_horizon(state, delta, n=1, p_hat_z=z)[0]
    target = phat + d_true
    naive = target - p0
    assert (target - phat)[0] == pytest.approx(0.0, abs=1e-12)
    assert naive[0] == pytest.approx(v * delta + d_true[0], abs=1e-12)
    assert abs(naive[0]) > 0.05


def test_summarize_demos_fails_fixed_height_fingerprint():
    """Hand Z stuck while object height moves → fail (same idea as B2)."""
    rows = [
        {
            "closed": True,
            "d_m": [0.0, 0.0, 0.05 - 0.02],
            "action_m": [0.5, 0.0, 0.05],
            "z_grasp_m": 0.02,
            "speed_m_s": 0.2,
        },
        {
            "closed": True,
            "d_m": [0.0, 0.0, 0.05 - 0.08],
            "action_m": [0.5, 0.0, 0.05],
            "z_grasp_m": 0.08,
            "speed_m_s": 0.2,
        },
    ]
    rep = summarize_demos(rows, s_xy=0.005, s_z=0.0025, label="fake", min_n=2)
    assert rep["fixed_height_like"] is True
    assert rep["pass_pooled"] is False
    assert rep["pass"] is False


def test_summarize_demos_passes_tight_d_across_speeds():
    """Tight close d at two speeds → pooled + per-speed pass."""
    rows = []
    for spd, zg in ((0.0, 0.02), (0.0, 0.08), (0.2, 0.03), (0.2, 0.07)):
        for _ in range(3):
            rows.append(
                {
                    "closed": True,
                    "d_m": [0.001, -0.001, 0.0005],
                    "action_m": [0.50, 0.0, zg + 0.0005],
                    "z_grasp_m": zg,
                    "speed_m_s": spd,
                }
            )
    rep = summarize_demos(rows, s_xy=0.005, s_z=0.0025, label="fake", min_n=3)
    assert rep["fixed_height_like"] is False
    assert rep["pass_pooled"]
    assert rep["pass_speeds"]
    assert rep["pass"]
    assert len(rep["speed_bins"]) == 2
