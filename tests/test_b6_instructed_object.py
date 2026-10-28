"""B6: expert closes on the instructed object (Isaac-free).

Run: pytest tests/test_b6_instructed_object.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.b6_instructed_object import (  # noqa: E402
    classify_close,
    language_names_target,
    probe_sm,
    summarize,
)
from interfaces.config import load_yaml  # noqa: E402

_YAML = ROOT / "sim" / "phase0_cfg.yaml"


@pytest.fixture
def cfg() -> dict:
    return load_yaml(_YAML)


def test_language_names_target_strips_digits():
    assert language_names_target("Pick the lemon and place it in the bowl.", "lemon")
    assert language_names_target("Grab the apple and set it in the box.", "apple12")
    assert not language_names_target("Pick the lemon and place it in the bowl.", "avocado")


def test_classify_close_prefers_nearer_object():
    ee = [0.50, 0.01]
    assert classify_close(ee, [0.50, 0.0], [0.50, 0.12], radius_m=0.06) == "target"
    assert classify_close(ee, [0.50, 0.12], [0.50, 0.0], radius_m=0.06) == "distractor"
    assert classify_close([0.80, 0.80], [0.50, 0.0], [0.50, 0.12], radius_m=0.06) == "neither"
    assert classify_close(ee, [0.50, 0.0], None, radius_m=0.06) == "target"


def test_probe_sm_aims_at_target_not_distractor(cfg):
    rep = probe_sm(cfg)
    assert rep["pass"]
    assert rep["dist_to_forecast_mm"] < rep["dist_to_distractor_mm"]


def test_summarize_requires_100_percent():
    rows = [
        {"closed": True, "which": "target", "lang_ok": True, "pass": True, "has_distractor": False},
        {"closed": True, "which": "target", "lang_ok": True, "pass": True, "has_distractor": False},
    ]
    ok = summarize(rows, label="fake", radius_m=0.06)
    assert ok["pass"] and ok["pass_rate"] == pytest.approx(1.0)

    bad = [
        {"closed": True, "which": "target", "lang_ok": True, "pass": True, "has_distractor": True},
        {
            "closed": True,
            "which": "distractor",
            "lang_ok": True,
            "pass": False,
            "has_distractor": True,
        },
    ]
    rep = summarize(bad, label="fake", radius_m=0.06)
    assert rep["pass"] is False
    assert rep["n_distractor"] == 1
    assert rep["pass_rate"] == pytest.approx(0.5)
