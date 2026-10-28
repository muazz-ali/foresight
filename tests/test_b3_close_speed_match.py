"""B3: |Δv| × t_close ≤ 2.5 mm helpers (Isaac-free).

Run: pytest tests/test_b3_close_speed_match.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.close_guard import max_speed_diff_m_s  # noqa: E402
from eval.b3_close_speed_match import SLACK_M, slip_m, summarize  # noqa: E402
from interfaces.config import load_yaml  # noqa: E402


def test_slip_equals_delta_v_times_t_close():
    """2.5 mm slack over 80 ms → 3.125 cm/s; matched speeds → 0 slip."""
    t = 0.08
    assert slip_m([0.20, 0.0], [0.20, 0.0], t_close=t) == pytest.approx(0.0)
    # 5 cm/s mismatch → 4 mm slide
    assert slip_m([0.25, 0.0], [0.20, 0.0], t_close=t) == pytest.approx(0.004)
    lim = max_speed_diff_m_s(slack_m=SLACK_M, close_time_s=t)
    assert lim == pytest.approx(0.03125)
    assert slip_m([lim, 0.0], [0.0, 0.0], t_close=t) == pytest.approx(SLACK_M)


def test_slip_ignores_z():
    """Pad slide is XY; a pure Z speed gap does not count."""
    assert slip_m([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], t_close=0.08) == pytest.approx(0.0)


def test_summarize_passes_when_all_bins_under_slack():
    rows = []
    for spd, cat in ((0.0, "lemon"), (0.2, "onion"), (0.3, "egg")):
        for _ in range(4):
            rows.append(
                {
                    "closed": True,
                    "speed_m_s": spd,
                    "category": cat,
                    "slip_ee_mm": 1.5,
                    "slip_cmd_mm": 0.2,
                    "pass": True,
                }
            )
    rep = summarize(rows, slack_m=0.0025, t_close=0.08, label="fake", min_n=3)
    assert rep["pass"]
    assert all(b["pass"] for b in rep["bins"])


def test_summarize_fails_fast_bin():
    """One speed×object bin over 2.5 mm median → overall fail."""
    rows = []
    for _ in range(4):
        rows.append(
            {
                "closed": True,
                "speed_m_s": 0.0,
                "category": "lemon",
                "slip_ee_mm": 1.0,
                "slip_cmd_mm": 0.1,
                "pass": True,
            }
        )
        rows.append(
            {
                "closed": True,
                "speed_m_s": 0.3,
                "category": "onion",
                "slip_ee_mm": 5.0,
                "slip_cmd_mm": 0.4,
                "pass": False,
            }
        )
    rep = summarize(rows, slack_m=0.0025, t_close=0.08, label="fake", min_n=3)
    assert not rep["pass"]
    by = {(b["speed_m_s"], b["category"]): b for b in rep["bins"]}
    assert by[(0.0, "lemon")]["pass"]
    assert not by[(0.3, "onion")]["pass"]


def test_t_close_from_yaml():
    from eval.b3_close_speed_match import _t_close_s

    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    assert _t_close_s(cfg) == pytest.approx(0.180)
