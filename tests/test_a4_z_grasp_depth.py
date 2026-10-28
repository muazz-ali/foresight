"""A4 helpers: σ_z gate math (Isaac-free)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.a4_z_grasp_depth import SIGMA_MAX_MM, _stats_mm  # noqa: E402


def test_a4_sigma_gate_passes_tight_noise():
    rng = np.random.default_rng(0)
    err = rng.normal(0.0, 0.001, size=500)  # 1 mm σ
    s = _stats_mm(err)
    assert s["sigma_mm"] <= SIGMA_MAX_MM
    assert s["pass_sigma"]


def test_a4_sigma_gate_fails_wide_noise():
    rng = np.random.default_rng(1)
    err = rng.normal(0.0, 0.010, size=500)  # 10 mm σ
    s = _stats_mm(err)
    assert s["sigma_mm"] > SIGMA_MAX_MM
    assert not s["pass_sigma"]
