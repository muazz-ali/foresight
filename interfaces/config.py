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
# Policy training settings: steps / batch / lr / val + noise sizes.
DEFAULT_POLICY_CFG = FORESIGHT_ROOT / "policy" / "configs" / "smolvla.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` into ``base`` in place. Nested dicts merge; else replace."""
    for key, value in overlay.items():
        if key == "include":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_raw(path: Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read yaml + resolve ``include:`` list. Values stay as written (SI)."""
    path = path.resolve()
    if path in _stack:
        chain = " → ".join(str(p) for p in _stack + (path,))
        raise ValueError(f"yaml include cycle: {chain}")
    with open(path) as fp:
        raw = yaml.load(fp, Loader=yaml.FullLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root must be a mapping, got {type(raw).__name__}")
    includes = raw.get("include") or []
    if not isinstance(includes, list):
        raise ValueError(f"{path}: include must be a list of relative paths")
    merged: dict[str, Any] = {}
    for rel in includes:
        inc = (path.parent / str(rel)).resolve()
        if not inc.is_file():
            raise FileNotFoundError(f"{path}: include not found: {rel} → {inc}")
        _deep_merge(merged, _load_yaml_raw(inc, _stack=_stack + (path,)))
    _deep_merge(merged, raw)
    return merged


def _fold_timing_sections(cfg: dict[str, Any]) -> None:
    """Merge ``policy_training_delay`` + ``evaluation`` into flat ``claim_timing``.

    Split yaml files keep train vs eval readable. Runtime still reads one dict.
    """
    ct = cfg.get("claim_timing")
    if ct is None:
        ct = {}
        cfg["claim_timing"] = ct
    if not isinstance(ct, dict):
        raise ValueError("claim_timing must be a mapping")

    for section_name in ("policy_training_delay", "evaluation"):
        block = cfg.pop(section_name, None)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"{section_name} must be a mapping")
        for key, value in block.items():
            if key in ct:
                raise ValueError(
                    f"{section_name}.{key} conflicts with claim_timing.{key}"
                )
            ct[key] = value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one yaml (with optional includes). Knobs are already SI."""
    cfg = _load_yaml_raw(Path(path))
    _fold_timing_sections(cfg)
    return cfg


def policy_look_ahead_s(
    cfg: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> float:
    """Policy look-ahead Δ (s) from ``state_machine_params.policy_look_ahead_s``.

    Δ = how many seconds ahead we point the policy. Collect, train, and eval
    must all read it from here, or the numbers will not match.
    """
    # Pass an already-loaded cfg to skip re-reading the file.
    if cfg is None:
        cfg = load_yaml(path or DEFAULT_SIM_CFG)
    sm = cfg["state_machine_params"]
    if "policy_look_ahead_s" in sm:
        return float(sm["policy_look_ahead_s"])
    return float(sm["max_aim_ahead_s"])


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
    delta = policy_look_ahead_s(path=sim_path or DEFAULT_SIM_CFG)
    return {
        "delta_min": delta,
        "delta_max": delta,
        "pos_noise_std": float(pol.get("pos_noise_std", 0.015)),
        "vel_noise_std": float(pol.get("vel_noise_std", 0.03)),
        "ee_noise_std": float(pol.get("ee_noise_std", 0.0)),
    }


def policy_train_defaults(policy_path: str | Path | None = None) -> dict[str, Any]:
    """Run length / batch / lr / log+val knobs from ``smolvla.yaml``.

    CLI flags override these when set. Missing keys fall back to the same
    numbers the trainer used to hard-code.
    """
    pol = load_yaml(policy_path or DEFAULT_POLICY_CFG)
    return {
        "pretrained": str(pol.get("pretrained", "lerobot/smolvla_base")),
        "steps": int(pol.get("steps", 20000)),
        "batch_size": int(pol.get("batch_size", 8)),
        "lr": float(pol.get("lr", 1.0e-4)),
        "log_every": int(pol.get("log_every", 50)),
        "eval_every": int(pol.get("eval_every", 100)),
        "val_fraction": float(pol.get("val_fraction", 0.05)),
        "chunk_size": int(pol.get("chunk_size", 25)),
        "n_action_steps": int(pol.get("n_action_steps", 10)),
    }
