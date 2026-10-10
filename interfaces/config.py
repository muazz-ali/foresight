"""Thin YAML helpers — no Isaac imports (safe for train + sim).

Why this file exists:
  Sim, training, and eval all need the same few settings (look-ahead Δ, gripper
  limits, noise sizes). They read them from yaml through these helpers, so every
  stage uses the same number. Only needs ``yaml`` — works in both conda envs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = one folder above ``interfaces/``. Lets us find yaml files from anywhere.
FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
# Sim settings: robot, gripper, look-ahead Δ.
DEFAULT_SIM_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"
# Policy training settings: noise sizes for the 4 numbers.
DEFAULT_POLICY_CFG = FORESIGHT_ROOT / "policy" / "configs" / "smolvla.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    # Read one yaml file into a dict, then turn mm/ms knobs into metres/seconds.
    from interfaces.cfg_units import normalize_cfg_units

    with open(path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    return normalize_cfg_units(cfg)


def conditioning_delta_s(
    cfg: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> float:
    """Policy look-ahead Δ (s) from ``state_machine_params.conditioning_delta_s``.

    Δ = how many seconds ahead we point the policy. Collect, train, and eval
    must all read it from here, or the numbers will not match.
    """
    # Pass an already-loaded cfg to skip re-reading the file.
    if cfg is None:
        cfg = load_yaml(path or DEFAULT_SIM_CFG)
    sm = cfg["state_machine_params"]
    if "conditioning_delta_s" in sm:
        return float(sm["conditioning_delta_s"])
    # Old yaml files only had ``lookahead_s``; use it so they still work.
    return float(sm["lookahead_s"])


def gripper_limits(
    cfg: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> tuple[float, float]:
    """Return (gripper_open, gripper_close) meters from ``robot`` yaml."""
    # Used to turn the policy's gripper output into "open" or "closed".
    if cfg is None:
        cfg = load_yaml(path or DEFAULT_SIM_CFG)
    robot = cfg["robot"]
    return float(robot["gripper_open"]), float(robot["gripper_close"])


def policy_aug_defaults(
    policy_path: str | Path | None = None,
    *,
    sim_path: str | Path | None = None,
) -> dict[str, float]:
    """Train aug: Δ from phase0 yaml; noise from ``smolvla.yaml``.

    "Aug" = we add small random errors to the 4 numbers while training, so the
    model still works when the camera filter is a bit off later.
    delta_min == delta_max because Δ is fixed (not random) for now.
    pos_noise_std: random error on position (meters). vel_noise_std: on speed (m/s).
    """
    pol = load_yaml(policy_path or DEFAULT_POLICY_CFG)
    delta = conditioning_delta_s(path=sim_path or DEFAULT_SIM_CFG)
    return {
        "delta_min": delta,
        "delta_max": delta,
        "pos_noise_std": float(pol.get("pos_noise_std", 0.015)),
        "vel_noise_std": float(pol.get("vel_noise_std", 0.03)),
    }
