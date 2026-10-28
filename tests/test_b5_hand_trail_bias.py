"""B5: mean d at close within ±2 mm (Isaac-free).

Run: pytest tests/test_b5_hand_trail_bias.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b5_hand_trail_bias import (  # noqa: E402
    BIAS_MAX_M,
    along_track_m,
    summarize,
)


def test_along_track_hand_ahead_is_positive():
    """Object moving +x; hand further along +x → positive d (lead, not trail)."""
    d = along_track_m([0.12, 0.0], [0.10, 0.0], [0.20, 0.0])
    assert d == pytest.approx(0.02)


def test_along_track_hand_trails_is_negative():
    """Hand behind the object along velocity → negative d (the old ~2 cm bug)."""
    d = along_track_m([0.08, 0.0], [0.10, 0.0], [0.20, 0.0])
    assert d == pytest.approx(-0.02)


def test_along_track_nan_when_static():
    assert np.isnan(along_track_m([0.1, 0.0], [0.1, 0.0], [0.0, 0.0]))


def test_summarize_passes_near_zero_bias():
    rows = []
    for spd, cat in ((0.1, "lemon"), (0.2, "onion")):
        for _ in range(4):
            rows.append(
                {
                    "closed": True,
                    "moving": True,
                    "speed_m_s": spd,
                    "category": cat,
                    "d_along_m": 0.001,
                    "d_x_m": 0.001,
                    "d_y_m": 0.0,
                    "d_cmd_along_m": 0.03,
                    "pass": True,
                }
            )
    rep = summarize(rows, bias_max_m=BIAS_MAX_M, label="fake", min_n=3)
    assert rep["pass"]
    assert abs(rep["pooled_d_along_mean_mm"]) <= 1000.0 * BIAS_MAX_M + 1e-6


def test_summarize_fails_trailing_bias():
    """~2 cm trail (old demos) → fail."""
    rows = []
    for _ in range(4):
        rows.append(
            {
                "closed": True,
                "moving": True,
                "speed_m_s": 0.2,
                "category": "lemon",
                "d_along_m": -0.021,
                "d_x_m": -0.021,
                "d_y_m": 0.0,
                "d_cmd_along_m": 0.03,
                "pass": False,
            }
        )
    rep = summarize(rows, bias_max_m=BIAS_MAX_M, label="fake", min_n=3)
    assert not rep["pass"]
    assert rep["bins"][0]["d_mean_mm"] == pytest.approx(-21.0)


def test_summarize_static_uses_xy_means():
    rows = [
        {
            "closed": True,
            "moving": False,
            "speed_m_s": 0.0,
            "category": "egg",
            "d_along_m": float("nan"),
            "d_x_m": 0.001,
            "d_y_m": -0.0005,
            "d_cmd_along_m": float("nan"),
            "pass": True,
        }
        for _ in range(4)
    ]
    rep = summarize(rows, bias_max_m=BIAS_MAX_M, label="fake", min_n=3)
    assert rep["pass"]
    assert rep["bins"][0]["kind"] == "static_xy"
