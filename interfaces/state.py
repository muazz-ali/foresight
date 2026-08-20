"""Frozen state message shared by simulator truth (Phase 0- 1) and camera track (Phase 2+).

Swap simulator truth → camera track later with a config flag; do not change this schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

STATE_DIM = 3
COVARIANCE_DIM = 6  # [x,y,z,vx,vy,vz] diagonal-friendly layout


@dataclass
class ObjectState:
    """Object place + speed + how sure we are (same layout for sim truth and camera track)."""

    position: np.ndarray  # (3,) meters, world/table frame
    velocity: np.ndarray  # (3,) m/s
    covariance: np.ndarray  # (6, 6) or (6,) diag; uncertainty on [p, v]
    timestamp: float  # seconds (sim time or wall clock)
    valid: bool


    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        cov = np.asarray(self.covariance, dtype=np.float64)
        if cov.ndim == 1:
            if cov.shape != (COVARIANCE_DIM,):
                raise ValueError(f"diag covariance must be ({COVARIANCE_DIM},), got {cov.shape}")
            self.covariance = np.diag(cov)
        elif cov.shape != (COVARIANCE_DIM, COVARIANCE_DIM):
            raise ValueError(f"covariance must be (6,6), got {cov.shape}")
        else:
            self.covariance = cov
        self.timestamp = float(self.timestamp)
        self.valid = bool(self.valid)

    def predict(self, dt: float) -> ObjectState:
        """Constant-velocity predict (oracle and filter share this kinematics)."""
        dt = float(dt)
        pos = self.position + self.velocity * dt
        # Grow uncertainty while coasting (simple process noise).
        q = 1e-4 * max(dt, 0.0)
        cov = self.covariance.copy()
        cov[0, 0] += q
        cov[1, 1] += q
        cov[2, 2] += q
        return ObjectState(pos, self.velocity.copy(), cov, self.timestamp + dt, self.valid)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["position"] = self.position.tolist()
        d["velocity"] = self.velocity.tolist()
        d["covariance"] = self.covariance.tolist()
        return d

    def to_flat(self) -> np.ndarray:
        """Pack for logging: pos(3)+vel(3)+diag_cov(6)+t(1)+valid(1) = 14."""
        return np.concatenate(
            [
                self.position,
                self.velocity,
                np.diag(self.covariance),
                [self.timestamp, 1.0 if self.valid else 0.0],
            ]
        )


def oracle_from_gt(
    position: np.ndarray,
    velocity: np.ndarray,
    timestamp: float,
    *,
    valid: bool = True,
    position_std: float = 1e-4,
    velocity_std: float = 1e-4,
) -> ObjectState:
    """Build ObjectState from **simulator truth** (perfect pose/speed).

    Plan nickname: “oracle”. Tiny covariance marks “we are sure.”
    """
    diag = np.array(
        [position_std] * 3 + [velocity_std] * 3,
        dtype=np.float64,
    )
    return ObjectState(position, velocity, diag, timestamp, valid)


# Phase-1 live layout only. Dropped as CONSTANT / dead in oracle demos:
# p_hat_z, v_hat_z, delta (rebuilt at train via random Δ), sigma_*, valid, coast_time.
CONDITIONING_DIM = 4
CONDITIONING_LAYOUT = ("p_hat_x", "p_hat_y", "v_hat_x", "v_hat_y")


def conditioning_vector(
    state: ObjectState,
    delta: float,
    *,
    workspace_scale: float = 1.0,
) -> np.ndarray:
    """4 live numbers for the policy: future XY place + XY speed.

    Layout: [p_hat_x, p_hat_y, v_hat_x, v_hat_y] (workspace-scaled).
    ``delta`` still drives the CV predict; it is not packed (train samples Δ).
    """
    future = state.predict(delta)
    return pack_conditioning(future, delta, valid=state.valid, workspace_scale=workspace_scale)


def pack_conditioning(
    future: ObjectState,
    delta: float,
    *,
    valid: bool | None = None,
    workspace_scale: float = 1.0,
) -> np.ndarray:
    """Pack future ObjectState into the 4-D live conditioning layout.

    ``delta`` / ``valid`` kept for call-site compatibility; unused in the pack.
    """
    del delta, valid  # call-site compat; not in live layout
    scale = float(workspace_scale) if workspace_scale else 1.0
    return np.concatenate(
        [
            future.position[:2] / scale,
            future.velocity[:2] / scale,
        ]
    ).astype(np.float64)


FLAT_STATE_KEYS = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "cov_px",
    "cov_py",
    "cov_pz",
    "cov_vx",
    "cov_vy",
    "cov_vz",
    "timestamp",
    "valid",
)
