"""B5: mean d at close within ±2 mm (Isaac-free).

Run: pytest tests/test_b5_close_d_bias.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b5_close_d_bias import BIAS_MAX_MM, _bias_pass, summarize  # noqa: E402


def test_bias_pass_at_limit():
    assert _bias_pass([2.0, -2.0, 0.0], lim_mm=BIAS_MAX_MM)
    assert not _bias_pass([2.01, 0.0, 0.0], lim_mm=BIAS_MAX_MM)


def test_summarize_fails_trailing_bias():
    """Systematic +5 mm on x (hand trails / τ too long) → fail."""
    rows = [
        {
            "closed": True,
            "d_m": [0.005, 0.0, 0.0],
            "speed_m_s": 0.2,
        }
        for _ in range(4)
    ]
    rep = summarize(rows, label="fake", lim_mm=2.0, min_n=3)
    assert rep["pass_pooled"] is False
    assert abs(rep["pooled"]["bias_mm"][0] - 5.0) < 1e-6
    assert rep["pass"] is False


def test_summarize_passes_near_zero():
    rows = []
    for spd in (0.0, 0.2):
        for dx in (0.001, -0.001, 0.0005):
            rows.append(
                {
                    "closed": True,
                    "d_m": [dx, -dx, 0.0],
                    "speed_m_s": spd,
                }
            )
    rep = summarize(rows, label="fake", lim_mm=2.0, min_n=3)
    assert rep["pass_pooled"]
    assert rep["pass_speeds"]
    assert rep["pass"]
