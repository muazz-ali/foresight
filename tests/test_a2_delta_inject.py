"""A2 gate: inject 100/200/300 ms → timestamp-measured Δ bias≈0, σ≤5 ms."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.a2_delta_inject import BIAS_MAX_MS, SIGMA_MAX_MS, run_cell  # noqa: E402


@pytest.mark.parametrize("inject_ms", [100, 200, 300])
def test_a2_measured_matches_injected(inject_ms: int):
    """apply_at−t0 and enforce-release−t0 recover the inject; pack uses reported Δ."""
    cell = run_cell(
        inject_ms * 1e-3,
        n=500,
        seed=inject_ms,
        jitter_std_s=0.010,
    )
    book = cell["bookkeeping_apply_at_minus_t0"]
    rel = cell["enforce_release_minus_t0"]
    assert abs(book["bias_ms"]) <= BIAS_MAX_MS, book
    assert book["sigma_ms"] <= SIGMA_MAX_MS, book
    assert abs(rel["bias_ms"]) <= BIAS_MAX_MS, rel
    assert rel["sigma_ms"] <= SIGMA_MAX_MS, rel
    assert cell["pack_lead_ok"], cell["pack_lead_err_max_mm"]
    assert cell["pass"]


def test_a2_reported_jitter_is_about_10ms_not_gated():
    """Reported−inject spreads with yaml jitter (~10 ms); A2 does not gate that."""
    cell = run_cell(0.200, n=1000, seed=7, jitter_std_s=0.010)
    info = cell["reported_minus_inject"]
    assert abs(info["bias_ms"]) < 2.0  # mean of N(0,10) noise
    assert 7.0 <= info["sigma_ms"] <= 13.0
