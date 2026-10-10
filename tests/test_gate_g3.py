"""Isaac-free: Gate G3 scorer on fake eval run folders."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.gate_g3 import load_run, score, wilson

ROOT = Path(__file__).resolve().parents[1]


def write_run(folder: Path, spec: dict, jerk: float) -> Path:
    """spec: speed -> list of timing labels (one per seed). success = label 'success'."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for speed, labels in spec.items():
        for i, lab in enumerate(labels):
            rows.append({
                "speed_m_s": speed, "seed": 5000 + int(speed * 1000) + i,
                "success": lab == "success",
                "timing": {"timing_label": lab,
                           "pre_grasp": {"path_len_m": 0.5, "jerk_rms": jerk}},
            })
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def labels(n_ok, n_late=0, n_early=0, n_short=0, n_aim=0, n=20):
    out = ["success"] * n_ok + ["late"] * n_late + ["early"] * n_early + ["short-close"] * n_short
    out += ["aim"] * n_aim
    return out + ["never-closed"] * (n - len(out))


def test_wilson_known_values():
    lo, hi = wilson(84, 100)
    assert lo == pytest.approx(0.755, abs=0.002) and hi == pytest.approx(0.899, abs=0.002)
    assert wilson(0, 0) == (0.0, 1.0)


def test_pass_when_timing_halves_smoother_faster_and_zero_collapses(tmp_path):
    base = write_run(tmp_path / "off", {0.15: labels(10, 4, 4), 0.20: labels(8, 6, 4)}, jerk=30.0)
    on = write_run(tmp_path / "on", {0.15: labels(14, 1, 1), 0.20: labels(13, 2, 1)}, jerk=20.0)
    zero = write_run(tmp_path / "zero", {0.15: labels(1), 0.20: labels(0)}, jerk=25.0)
    rep = score(load_run(base), load_run(on), load_run(zero))
    assert rep["checks"] == {"timing_halved": True, "smoother": True,
                             "success_20_up": True, "zero_out_collapse": True}
    assert rep["gate_g3_pass"] and not rep["claim_ready"]  # n=20 is a scout
    assert rep["timing_fail_pooled"]["baseline"]["k"] == 18
    assert rep["timing_fail_pooled"]["treatment"]["k"] == 5
    assert all(r["seeds_match"] for r in rep["per_speed"])


def test_fail_when_timing_only_drops_a_little(tmp_path):
    base = write_run(tmp_path / "off", {0.20: labels(8, 6, 4)}, jerk=30.0)
    on = write_run(tmp_path / "on", {0.20: labels(10, 5, 3)}, jerk=20.0)
    rep = score(load_run(base), load_run(on))
    assert rep["checks"]["timing_halved"] is False
    assert rep["checks"]["zero_out_collapse"] is None  # no zero run given
    assert not rep["gate_g3_pass"]


def test_short_close_counts_aim_does_not(tmp_path):
    base = write_run(tmp_path / "off", {0.15: labels(5, n_short=4, n_aim=6)}, jerk=1.0)
    rep = score(load_run(base), load_run(base))
    assert rep["timing_fail_pooled"]["baseline"]["k"] == 4


def test_cli_writes_json_and_markdown(tmp_path):
    base = write_run(tmp_path / "off", {0.20: labels(8, 6, 4)}, jerk=30.0)
    on = write_run(tmp_path / "on", {0.20: labels(13, 2, 1)}, jerk=20.0)
    out = tmp_path / "rep"
    subprocess.run([sys.executable, str(ROOT / "eval" / "gate_g3.py"), "--baseline", str(base),
                    "--treatment", str(on), "-o", str(out)], check=True, capture_output=True)
    rep = json.loads((out / "gate_g3_report.json").read_text())
    assert rep["gate"] == "G3" and rep["per_speed"][0]["speed_m_s"] == 0.2
    md = (out / "gate_g3_report.md").read_text()
    assert "20 cm/s" in md and "13/20" in md
