"""Frozen state message shared by simulator truth (Phase 0- 1) and camera track (Phase 2+).

Why this file exists:
  The policy needs to know "where is the object, and how fast is it moving?"
  In Phase 0-1 the simulator tells us the exact answer (the "oracle").
  In Phase 2+ a camera + Kalman filter guesses it.
  Both sides fill the SAME ObjectState, so the policy cannot tell them apart.

Swap simulator truth → camera track later with a config flag; do not change this schema.
(Why a flag and not new code: if the swap needs no code change, then any score drop
after the swap is caused by the filter being less accurate — not by a bug in glue.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Position and velocity each have 3 parts (x, y, z).
STATE_DIM = 3
# Covariance covers position (3) + velocity (3) = 6 numbers.
COVARIANCE_DIM = 6  # [x,y,z,vx,vy,vz] diagonal-friendly layout


@dataclass
class ObjectState:
    """Object place + speed + how sure we are (same layout for sim truth and camera track).

    These five fields are FROZEN. Add new fields only with a handoff note.
    """

    # Where the object center is. Meters. Table/world frame.
    position: np.ndarray  # (3,) meters, world/table frame
    # How fast it moves right now. Meters per second.
    velocity: np.ndarray  # (3,) m/s
    # How unsure we are. Small = very sure (oracle). Big = guessing (filter after a gap).
    # You may pass just the 6 diagonal numbers; we turn them into a 6x6 grid for you.
    covariance: np.ndarray  # (6, 6) or (6,) diag; uncertainty on [p, v]
    # When this reading was true, in seconds. Needed so we can move it forward/back in time.
    timestamp: float  # seconds (sim time or wall clock)
    # False means "do not trust this" (e.g. camera lost the object).
    valid: bool

    def __post_init__(self) -> None:
        # Force fixed shapes and float64 so a bad shape fails here, not deep inside the policy.
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        cov = np.asarray(self.covariance, dtype=np.float64)
        if cov.ndim == 1:
            # Short form: 6 numbers -> put them on the diagonal of a 6x6 grid.
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
        """Constant-velocity predict (oracle and filter share this kinematics).

        Plain words: "if it keeps going at this speed, where is it dt seconds later?"
        No bounce, no friction. Oracle and filter both use this, so they agree.
        """
        dt = float(dt)
        pos = self.position + self.velocity * dt
        # Grow uncertainty while coasting (simple process noise).
        # Further into the future = less sure. Only position terms grow here.
        q = 1e-4 * max(dt, 0.0)
        cov = self.covariance.copy()
        cov[0, 0] += q
        cov[1, 1] += q
        cov[2, 2] += q
        return ObjectState(pos, self.velocity.copy(), cov, self.timestamp + dt, self.valid)

    def as_dict(self) -> dict[str, Any]:
        # Plain lists instead of numpy arrays, so it can be dumped to JSON / logs.
        d = asdict(self)
        d["position"] = self.position.tolist()
        d["velocity"] = self.velocity.tolist()
        d["covariance"] = self.covariance.tolist()
        return d

    def to_flat(self) -> np.ndarray:
        """Pack for logging: pos(3)+vel(3)+diag_cov(6)+t(1)+valid(1) = 14.

        Order matches ``FLAT_STATE_KEYS`` at the bottom of this file.
        Only the diagonal of covariance is kept (off-diagonal is dropped to keep logs small).
        """
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
    The Kalman filter (Phase 2) returns the same ObjectState with a bigger covariance.
    """
    # Same tiny "sureness" for all 3 position axes, and for all 3 velocity axes.
    diag = np.array(
        [position_std] * 3 + [velocity_std] * 3,
        dtype=np.float64,
    )
    return ObjectState(position, velocity, diag, timestamp, valid)


# Live pack = 5 numbers. Δ drives XY predict (p̂ = p + v·Δ) and is NOT packed.
# p̂z is grasp height (latched; vz = 0) — height changes p̂, never the relative d_z.
# Claim path: measured/reported Δ via DelayInjection. Legacy: fixed yaml Δ.
# Still dropped: v_hat_z, sigma_*, valid, coast_time.
#
# Plain words: the policy gets:
#   p_hat_x, p_hat_y = where the object WILL be (XY) in Δ seconds
#   p_hat_z           = grasp height z_grasp (constant while sliding)
#   v_hat_x, v_hat_y = how fast it moves now (XY)
CONDITIONING_DIM = 5
CONDITIONING_LAYOUT = ("p_hat_x", "p_hat_y", "p_hat_z", "v_hat_x", "v_hat_y")


def conditioning_vector(
    state: ObjectState,
    delta: float,
    *,
    workspace_scale: float = 1.0,
    p_hat_z: float | None = None,
) -> np.ndarray:
    """5 live numbers: future XY + grasp Z + XY speed (workspace-scaled).

    ``delta`` drives the CV XY predict; it is not in the vector.
    ``p_hat_z`` overrides future Z when set (latched z_grasp). Else uses
    ``state.position[2]`` after predict (vz usually 0).
    """
    future = state.predict(delta)
    if p_hat_z is not None and np.isfinite(p_hat_z):
        future.position = future.position.copy()
        future.position[2] = float(p_hat_z)
    return pack_conditioning(future, delta, valid=state.valid, workspace_scale=workspace_scale)


def pack_conditioning(
    future: ObjectState,
    delta: float,
    *,
    valid: bool | None = None,
    workspace_scale: float = 1.0,
) -> np.ndarray:
    """Pack future ObjectState into the 5-D live conditioning layout.

    Layout: [p̂x, p̂y, p̂z, vx, vy]. ``delta`` / ``valid`` unused in the pack.
    """
    del delta, valid  # call-site compat; not in live layout
    scale = float(workspace_scale) if workspace_scale else 1.0
    return np.concatenate(
        [
            future.position[:3] / scale,
            future.velocity[:2] / scale,
        ]
    ).astype(np.float64)


# Column names for ``ObjectState.to_flat()``. Same order. Use these for log headers.
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
