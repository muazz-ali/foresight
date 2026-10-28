"""B1: oracle expert ≥ 95% per grid cell (Isaac-free).

Run: pytest tests/test_b1_oracle_grid.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b1_oracle_grid import (  # noqa: E402
    required_cells,
    snap_delay_s,
    summarize,
    wilson,
)


def test_wilson_bounds_sensible():
    lo, hi = wilson(19, 20)
    assert 0.0 <= lo <= 0.95 <= hi <= 1.0


def test_snap_delay_to_grid():
    assert snap_delay_s(0.097, (0.0, 0.100, 0.200)) == pytest.approx(0.100)
    assert snap_delay_s(0.0, (0.0, 0.100, 0.200)) == pytest.approx(0.0)
    assert snap_delay_s(0.350, (0.0, 0.100, 0.200)) is None  # far from grid


def test_required_cells_speed_and_plan():
    assert len(required_cells("speed")) == 4
    assert all(d is None for _, d in required_cells("speed"))
    assert len(required_cells("plan")) == 6
    assert len(required_cells("smoke")) == 12


def test_summarize_passes_perfect_speed_grid():
    rows = []
    for spd in (0.0, 0.10, 0.20, 0.30):
        for _ in range(20):
            rows.append({"speed_m_s": spd, "delay_s": None, "success": True})
    rep = summarize(rows, grid="speed", label="fake", min_n=20)
    assert rep["pass"]
    assert all(c["pass"] for c in rep["required"])


def test_summarize_fails_low_rate_cell():
    rows = []
    for spd in (0.0, 0.10, 0.20, 0.30):
        for i in range(20):
            # 0.30 cell: only 18/20 = 90% → fail
            ok = True if spd < 0.30 else (i < 18)
            rows.append({"speed_m_s": spd, "delay_s": None, "success": ok})
    rep = summarize(rows, grid="speed", label="fake", min_n=20)
    assert rep["pass"] is False
    bad = [c for c in rep["required"] if c["speed_m_s"] == 0.30][0]
    assert bad["rate"] == pytest.approx(0.9)
    assert bad["pass"] is False


def test_summarize_fails_missing_plan_cell():
    rows = [
        {"speed_m_s": 0.10, "delay_s": 0.100, "success": True}
        for _ in range(20)
    ]
    rep = summarize(rows, grid="plan", label="fake", min_n=20)
    assert rep["pass"] is False
    assert len(rep["missing"]) == 5
