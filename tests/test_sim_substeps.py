"""Physics-substep math (no Isaac): rate choice, substep clock, scripted motion per substep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.motion import kinematic_advance, physics_substeps, substep_times

CFG = Path(__file__).resolve().parents[1] / "sim" / "phase0_cfg.yaml"


def _cfg(use_house: bool, n=None) -> dict:
    sim = {"dt": 0.04}
    if n is not None:
        sim["physics_substeps"] = n
    return {"scene": {"use_house": use_house}, "sim": sim}


def test_floor_mode_is_always_one_substep():
    # Old G1 / G2b numbers were measured at 25 Hz physics: floor must not change.
    assert physics_substeps(_cfg(False, 4)) == 1
    assert physics_substeps(_cfg(False)) == 1


def test_house_mode_reads_knob_and_defaults_to_one():
    assert physics_substeps(_cfg(True, 4)) == 4
    assert physics_substeps(_cfg(True)) == 1
    with pytest.raises(ValueError):
        physics_substeps(_cfg(True, 0))


def test_shipped_cfg_house_physics_is_100_to_200_hz():
    from interfaces.config import load_yaml

    cfg = load_yaml(CFG)
    dt = float(cfg["sim"]["dt"])
    assert dt == pytest.approx(0.04)  # control rate stays 25 Hz
    n = physics_substeps(cfg)
    assert 100.0 <= 1.0 / (dt / n) <= 200.0


def test_shipped_cfg_close_dwell_covers_measured_close():
    from interfaces.config import load_yaml

    cfg = load_yaml(CFG)
    close_s = float(cfg["robot"]["gripper_close_time_s"])
    assert 0.0 < close_s < float(cfg["state_machine_params"]["physics_close_s"])


def test_substep_times_tile_one_control_step():
    ts = substep_times(1.20, 0.04, 4)
    assert ts == pytest.approx([1.20, 1.21, 1.22, 1.23])
    assert ts[-1] + 0.04 / 4 == pytest.approx(1.24)
    assert substep_times(0.0, 0.04, 1) == [0.0]


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_scripted_coast_moves_exactly_speed_times_dt(n):
    # Throw mode (no walls): n substeps of dt/n == one step of dt == v * dt.
    dt, v = 0.04, np.array([0.20, -0.05, 0.0])
    pos = np.array([0.5, 0.0, 0.03])
    p, vel = pos.copy(), v.copy()
    for _ in range(n):
        p, vel = kinematic_advance(p, vel, dt / n, xy_bounds=None)
    np.testing.assert_allclose(p[:2], pos[:2] + v[:2] * dt, atol=1e-12)
    np.testing.assert_allclose(vel, v)
    # A whole second at the control rate: drift-free.
    p = pos.copy()
    for _ in range(25 * n):
        p, _ = kinematic_advance(p, v, dt / n, xy_bounds=None)
    np.testing.assert_allclose(p[:2], pos[:2] + v[:2] * 1.0, atol=1e-9)
