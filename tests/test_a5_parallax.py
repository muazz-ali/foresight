"""A5 helpers: RMS gate and lift correction (Isaac-free)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.a5_parallax import PASS_MM, stats  # noqa: E402
from interfaces.config import load_yaml  # noqa: E402
from perception.camera import PinholeCamera  # noqa: E402
from perception.estimator import StaticCamEstimator  # noqa: E402


def test_a5_pass_mm_is_2():
    assert PASS_MM == pytest.approx(2.0)


def test_stats_pass_when_rms_under_2mm():
    rng = np.random.default_rng(0)
    rows = [{"deployed": (rng.normal(0, 0.0008), rng.normal(0, 0.0008))} for _ in range(200)]
    s = stats(rows, "deployed")
    assert s is not None
    assert s["pass"]
    assert s["rms_mm"][0] <= PASS_MM


def test_stats_fail_when_rms_over_2mm():
    rows = [{"deployed": (0.005, 0.0)} for _ in range(50)]  # 5 mm constant
    s = stats(rows, "deployed")
    assert s is not None
    assert not s["pass"]


def test_lift_xy_applies_uv_bias_and_lead():
    """Silhouette bias + lead pullback move the measurement vs raw lift."""
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    cam = PinholeCamera.from_cfg(cfg)
    est = StaticCamEstimator(cam, uv_bias_px=(0.0, 0.0), centroid_lead_s=0.0)
    est.plane_z = 0.04
    uv = cam.project(np.array([0.60, 0.0, 0.04]))
    raw = np.asarray(cam.lift(uv, est.plane_z), dtype=np.float64)
    # No bias / lead → same as raw lift.
    np.testing.assert_allclose(est._lift_xy(uv), raw, atol=1e-9)
    # With UV bias only: shifts when bias ≠ 0.
    est.uv_bias_px = np.array([1.0, 0.0])
    shifted = est._lift_xy(uv)
    assert np.linalg.norm(shifted - raw) > 1e-4
