"""Step 1 check: delay injection logs t0 / reported Δ / applied Δ; jitter ~10 ms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from control.delay_injection import DelayInjection
from interfaces.config import load_yaml

CFG = Path(__file__).resolve().parents[1] / "sim" / "phase0_cfg.yaml"
DT_30HZ = 1.0 / 30.0


def test_from_cfg_reads_claim_timing():
    cfg = load_yaml(CFG)
    d = DelayInjection.from_cfg(cfg, seed=0)
    # Training yaml omits jitter → default 0.
    assert d.jitter_std_s == pytest.approx(0.0)
    assert d.train_delta_min_s == pytest.approx(0.0)
    assert d.train_delta_max_s == pytest.approx(0.200)
    assert d.eval_delta_s == pytest.approx(0.0)


def test_shipped_cfg_is_30hz_with_120hz_house_physics():
    cfg = load_yaml(CFG)
    dt = float(cfg["sim"]["dt"])
    assert dt == pytest.approx(DT_30HZ, rel=1e-6)
    from sim.motion import physics_substeps

    n = physics_substeps(cfg)
    phys_hz = 1.0 / (dt / n)
    assert phys_hz == pytest.approx(120.0, rel=1e-5)
    assert int(cfg["static_camera"]["fps"]) == 30
    assert int(cfg["wrist_camera"]["fps"]) == 30


def test_apply_at_is_t0_plus_applied():
    d = DelayInjection(jitter_std_s=0.0, seed=1)
    d.set_eval_delta(0.165)
    s = d.sample(1.0, mode="eval")
    assert s.applied_delta_s == pytest.approx(0.165)
    assert s.reported_delta_s == pytest.approx(0.165)
    assert s.apply_at == pytest.approx(1.165)


def test_step1_jitter_spread_about_10ms_over_1000_cycles():
    """Plan check: log 1000 cycles; jitter std ≈ 10 ms."""
    d = DelayInjection(
        jitter_std_s=0.010,
        train_delta_min_s=0.0,
        train_delta_max_s=0.200,
        seed=42,
    )
    t0 = 0.0
    for _ in range(1000):
        d.sample(t0, mode="train")
        t0 += DT_30HZ
    arrays = d.log_arrays()
    assert len(arrays["t0"]) == 1000
    assert arrays["t0"][0] == pytest.approx(0.0)
    assert arrays["t0"][-1] == pytest.approx(999 * DT_30HZ, rel=1e-9)
    # Reported − applied must be the logged jitter.
    np.testing.assert_allclose(
        arrays["reported_delta_s"] - arrays["applied_delta_s"],
        arrays["jitter_s"],
        atol=1e-12,
    )
    # Spread ≈ 10 ms (allow sampling noise on N=1000; ~8–12 ms).
    std = d.jitter_std()
    assert 0.008 <= std <= 0.012, f"jitter std {std * 1000:.2f} ms not ~10 ms"


def test_eval_mode_uses_fixed_applied_with_jitter_on_reported():
    d = DelayInjection(jitter_std_s=0.010, eval_delta_s=0.100, seed=7)
    samples = [d.sample(0.5, mode="eval") for _ in range(200)]
    applied = np.asarray([s.applied_delta_s for s in samples])
    assert np.all(applied == pytest.approx(0.100))
    # Reported still jitters around the fixed applied.
    assert d.jitter_std() == pytest.approx(0.010, abs=0.003)
