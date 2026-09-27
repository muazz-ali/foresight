"""Thin YAML helpers — no Isaac imports (safe for train + sim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
DEFAULT_POLICY_B_CFG = FORESIGHT_ROOT / "policy" / "configs" / "smolvla_b.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def conditioning_delta_s(
    cfg: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> float:
    """Policy look-ahead Δ (s) from ``state_machine_params.conditioning_delta_s``."""
    if cfg is None:
        cfg = load_yaml(path or DEFAULT_SIM_CFG)
    sm = cfg["state_machine_params"]
    if "conditioning_delta_s" in sm:
        return float(sm["conditioning_delta_s"])
    return float(sm["lookahead_s"])


def gripper_limits(
    cfg: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> tuple[float, float]:
    """Return (gripper_open, gripper_close) meters from ``robot`` yaml."""
    if cfg is None:
        cfg = load_yaml(path or DEFAULT_SIM_CFG)
    robot = cfg["robot"]
    return float(robot["gripper_open"]), float(robot["gripper_close"])


def policy_aug_defaults(
    policy_path: str | Path | None = None,
    *,
    sim_path: str | Path | None = None,
) -> dict[str, float]:
    """Train aug: Δ from phase0 yaml; noise from ``smolvla_b.yaml``."""
    pol = load_yaml(policy_path or DEFAULT_POLICY_B_CFG)
    delta = conditioning_delta_s(path=sim_path or DEFAULT_SIM_CFG)
    return {
        "delta_min": delta,
        "delta_max": delta,
        "pos_noise_std": float(pol.get("pos_noise_std", 0.015)),
        "vel_noise_std": float(pol.get("vel_noise_std", 0.03)),
    }
