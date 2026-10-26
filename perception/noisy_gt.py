"""Step 2: noisy ground-truth object state + short constant-velocity filter.

Take the simulator's true place/speed, add a little camera-like
noise, then run the same CV Kalman we use on real pixels for the last 0.5 s
(~16 frames at 30 Hz). Output is still the frozen ``ObjectState`` message.

Noise (plan):
  * static camera: 4 mm std on XY
  * wrist camera within 10 cm of the object: 2 mm std on XY
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.state import ObjectState, oracle_from_gt
from perception.kalman import CVKalman

STATIC_POS_STD_M = 0.004  # 4 mm
WRIST_POS_STD_M = 0.002  # 2 mm
WRIST_NEAR_M = 0.10  # use wrist noise when hand is this close
WINDOW_S = 0.5  # filter memory the plan asks for
DEFAULT_DT_S = 1.0 / 30.0 # 30 Hz control step size (i control step = 33.3 ms is ~16 frames at 30 Hz)


def measure_noise_std(
    ee_pos: np.ndarray,
    obj_pos: np.ndarray,
    *,
    wrist_near_m: float = WRIST_NEAR_M,
) -> float:
    """Pick static vs wrist noise from hand–object distance (metres)."""
    ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    obj = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(ee - obj)) <= float(wrist_near_m):
        return WRIST_POS_STD_M
    return STATIC_POS_STD_M


def add_gt_noise(
    state: ObjectState,
    rng: np.random.Generator,
    *,
    ee_pos: np.ndarray | None = None,
    pos_std: float | None = None,
    wrist_near_m: float = WRIST_NEAR_M,
) -> ObjectState:
    """Copy ``state`` with Gaussian noise on XY position (Z / velocity clean).

    Pass ``ee_pos`` to auto-pick 4 mm vs 2 mm, or set ``pos_std`` yourself.
    """
    if pos_std is None:
        if ee_pos is None:
            pos_std = STATIC_POS_STD_M
        else:
            pos_std = measure_noise_std(ee_pos, state.position, wrist_near_m=wrist_near_m)
    pos = np.asarray(state.position, dtype=np.float64).copy()
    pos[:2] = pos[:2] + rng.normal(0.0, float(pos_std), size=2)
    # Tiny cov so callers see "noisy measurement", not oracle 1e-4.
    return oracle_from_gt(
        pos,
        state.velocity.copy(),
        state.timestamp,
        valid=state.valid,
        position_std=float(pos_std),
        velocity_std=0.05,
    )


@dataclass
class WindowedCVFilter:
    """Feed one noisy XY per control step; fit constant velocity on the last 0.5 s.

    keep the last ~16 points, draw a straight-line fit for place and
    speed. 
    That matches the claim Step-2 check (conveyor velocity error ≲ 6.5 mm/s).
    A high-process-noise recursive Kalman is kept available via ``use_kalman``.
    """

    dt: float = DEFAULT_DT_S
    window_s: float = WINDOW_S
    meas_std: float = STATIC_POS_STD_M
    accel_std: float = 0.5  # low: conveyor is nearly constant velocity
    use_kalman: bool = False

    def __post_init__(self) -> None:
        self.kf = CVKalman(meas_std=float(self.meas_std), accel_std=float(self.accel_std))
        self.plane_z = 0.0
        self.n_updates = 0 # number of updates to the filter, used to check if the filter has started 
        self._t: list[float] = [] # list of timestamps of the updates
        self._xy: list[np.ndarray] = [] # list of XY positions of the updates

    @property
    def window_frames(self) -> int:
        """How many control steps fit in ``window_s`` (plan: ~16 at 30 Hz)."""
        return max(1, int(round(float(self.window_s) / float(self.dt))))

    def reset(self, *, plane_z: float = 0.0) -> None:
        """Clear filter state for a new episode."""
        self.kf = CVKalman(meas_std=float(self.meas_std), accel_std=float(self.accel_std))
        self.plane_z = float(plane_z)
        self.n_updates = 0
        self._t.clear()
        self._xy.clear()

    def _fit_cv(self, t_now: float) -> ObjectState:
        """Least-squares constant-velocity fit on the stored window → ObjectState."""
        t = np.asarray(self._t, dtype=np.float64)
        xy = np.asarray(self._xy, dtype=np.float64)  # (N, 2)
        # Fit each axis: x(t) = x0 + vx * (t - t_ref), t_ref = latest time.
        t_ref = float(t_now)
        tau = t - t_ref
        # Design matrix [1, tau]
        A = np.column_stack([np.ones_like(tau), tau])
        # Solve A @ [p, v]^T = xy for both axes (lstsq).
        beta, _, _, _ = np.linalg.lstsq(A, xy, rcond=None)  # (2, 2): rows = [p, v], cols = x/y
        p_xy = beta[0]
        v_xy = beta[1]
        # Covariance from residual + design (simple diagonal).
        resid = xy - A @ beta
        n = max(len(t) - 2, 1)
        var = float(np.mean(resid**2)) + float(self.meas_std) ** 2
        # Position var ~ var; velocity var ~ var / mean(tau^2) scale.
        tau2 = float(np.mean(tau**2)) + 1e-12
        v_var = var / tau2
        diag = np.array(
            [var, var, 1e-8, v_var, v_var, 1e-8],
            dtype=np.float64,
        )
        return ObjectState(
            position=np.array([p_xy[0], p_xy[1], self.plane_z], dtype=np.float64),
            velocity=np.array([v_xy[0], v_xy[1], 0.0], dtype=np.float64),
            covariance=diag,
            timestamp=t_ref,
            valid=True,
        )

    def step(self, t: float, z_xy: np.ndarray | None, *, plane_z: float | None = None) -> ObjectState | None:
        """One measurement at sim time ``t``. Returns None until the first hit."""
        if plane_z is not None:
            self.plane_z = float(plane_z)
        t = float(t)
        if z_xy is not None:
            z = np.asarray(z_xy, dtype=np.float64).reshape(2)
            self._t.append(t)
            self._xy.append(z.copy())
            self.n_updates += 1
            # Drop points older than the window.
            t_cut = t - float(self.window_s) - 1e-12
            while self._t and self._t[0] < t_cut:
                self._t.pop(0)
                self._xy.pop(0)
        if self.use_kalman:
            self.kf.step(t, None if z_xy is None else np.asarray(z_xy, dtype=np.float64))
            if not self.kf.started:
                return None
            return self.kf.state(self.plane_z)
        if not self._xy:
            return None
        return self._fit_cv(t)


def filter_noisy_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    dt: float = DEFAULT_DT_S,
    rng: np.random.Generator,
    ee_positions: np.ndarray | None = None,
    window_s: float = WINDOW_S,
) -> tuple[list[ObjectState], list[ObjectState]]:
    """Run noise + windowed filter over a conveyor trajectory (Isaac-free check).

    ``positions`` / ``velocities`` shape (T, 3). Returns (noisy_states, filtered).
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (T, 3), got {pos.shape}")
    T = pos.shape[0]
    filt = WindowedCVFilter(dt=dt, window_s=window_s)
    filt.reset(plane_z=float(pos[0, 2]))
    noisy: list[ObjectState] = []
    filtered: list[ObjectState] = []
    for i in range(T):
        t = i * float(dt)
        gt = oracle_from_gt(pos[i], vel[i], t, valid=True)
        ee = None if ee_positions is None else ee_positions[i]
        n = add_gt_noise(gt, rng, ee_pos=ee)
        noisy.append(n)
        out = filt.step(t, n.position[:2], plane_z=float(n.position[2]))
        assert out is not None
        filtered.append(out)
    return noisy, filtered
