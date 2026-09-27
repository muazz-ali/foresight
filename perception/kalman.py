"""Constant-velocity Kalman filter on table XY → ``ObjectState`` (plan Job 1).

State [x, y, vx, vy]. Predict is the same straight-line physics the oracle pack
uses; update takes one table point per camera frame. Two plain additions:

  bounce — a point far outside the predicted spread (a wall hit flips the
           velocity) re-runs that predict step with a large acceleration
           allowance, so the same update also re-learns the speed.
  coast  — no point this frame: predict only. Past ``max_coast_s`` blind,
           ``valid`` goes False and the filter holds the last place (speed 0)
           until the tracker re-detects (plan §6 coast rule).
"""

from __future__ import annotations

import numpy as np

from interfaces.state import ObjectState

_H = np.eye(2, 4)
_CHI2_2DOF_999 = 13.8  # 99.9% gate for a 2-D residual


def _transition(dt: float, accel_std: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    # White-noise acceleration, per axis: [[dt⁴/4, dt³/2], [dt³/2, dt²]] · σ_a².
    q = accel_std**2
    Q = np.zeros((4, 4))
    for i in (0, 1):
        Q[i, i] = q * dt**4 / 4
        Q[i, i + 2] = Q[i + 2, i] = q * dt**3 / 2
        Q[i + 2, i + 2] = q * dt**2
    return F, Q


class CVKalman:
    # Defaults from `scripts/g2_perception_report.py --tune` on demo speeds 12/14/16/18 cm/s
    # (held out of the report). accel_std 2–4 is a flat optimum; 2 keeps v̂ calmer.
    def __init__(
        self,
        *,
        meas_std: float = 0.002,
        accel_std: float = 2.0,
        bounce_accel_std: float = 10.0,
        init_vel_std: float = 0.3,
        bounce_nis: float = _CHI2_2DOF_999,
        max_coast_s: float = 0.5,
    ) -> None:
        self.R = np.eye(2) * meas_std**2
        self.meas_std = float(meas_std)
        self.accel_std = float(accel_std)
        self.bounce_accel_std = float(bounce_accel_std)
        self.init_vel_std = float(init_vel_std)
        self.bounce_nis = float(bounce_nis)
        self.max_coast_s = float(max_coast_s)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t = 0.0
        self.last_meas_t = 0.0
        self.n_bounces = 0

    @property
    def started(self) -> bool:
        return self.x is not None

    def predict_xy(self, t: float) -> np.ndarray:
        """Position guess at time ``t`` (no state change) — used to gate the tracker."""
        assert self.x is not None
        return self.x[:2] + self.x[2:] * max(float(t) - self.t, 0.0)

    def step(self, t: float, z_xy: np.ndarray | None) -> None:
        """Advance to time ``t``; fold in the table point ``z_xy`` if the tracker saw one."""
        t = float(t)
        if self.x is None:
            if z_xy is not None:
                self.x = np.r_[np.asarray(z_xy, dtype=np.float64), 0.0, 0.0]
                self.P = np.diag([self.meas_std**2] * 2 + [self.init_vel_std**2] * 2)
                self.t = self.last_meas_t = t
            return
        dt = max(t - self.t, 0.0)
        F, Q = _transition(dt, self.accel_std)
        x_prev, P_prev = self.x, self.P
        self.x, self.P, self.t = F @ x_prev, F @ P_prev @ F.T + Q, t
        if z_xy is None:
            if self.coast_s() > self.max_coast_s:
                # Blind too long: a straight line through unseen wall hits only gets
                # worse. Hold the last place (plan §6: hold, re-detect).
                self.x[2:] = 0.0
            return
        z = np.asarray(z_xy, dtype=np.float64)
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + self.R
        if float(y @ np.linalg.solve(S, y)) > self.bounce_nis:
            # Wall hit: redo this step allowing a sharp speed change.
            _, Qb = _transition(dt, self.bounce_accel_std)
            self.P = F @ P_prev @ F.T + Qb
            S = _H @ self.P @ _H.T + self.R
            self.n_bounces += 1
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        IKH = np.eye(4) - K @ _H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T  # Joseph form: stays symmetric
        self.last_meas_t = t

    def coast_s(self) -> float:
        return self.t - self.last_meas_t

    def state(self, plane_z: float) -> ObjectState:
        """Shared message: position / velocity / 6×6 covariance / timestamp / valid."""
        assert self.x is not None and self.P is not None
        cov = np.zeros((6, 6))
        idx = [0, 1, 3, 4]  # x, y, vx, vy inside [x, y, z, vx, vy, vz]
        cov[np.ix_(idx, idx)] = self.P
        cov[2, 2] = cov[5, 5] = 1e-8  # on-table: z and vz are known
        return ObjectState(
            position=np.array([self.x[0], self.x[1], plane_z]),
            velocity=np.array([self.x[2], self.x[3], 0.0]),
            covariance=cov,
            timestamp=self.t,
            valid=self.coast_s() <= self.max_coast_s,
        )
