"""Frame-delay buffer for B1 oracle expert.

Plain words: the expert only sees the world as it was Δ seconds ago.
Live GT is pushed every tick; the expert reads a delayed pack. Motions
run on the current tick (Δ is NOT also an enforce_apply_at hold).

Leak audit: any live-GT read outside the success judge must increment
``live_gt_leaks``. B1 asserts that counter stays 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from interfaces.state import ObjectState


@dataclass
class DelayedFrame:
    """One captured frame the expert is allowed to see."""

    t_capture: float
    state: ObjectState
    wrist_depth_m: float
    wrist_depth_img: np.ndarray | None
    static_depth_img: np.ndarray | None
    static_cam_z: float | None
    wrist_cam_z: float | None


@dataclass
class GtFallbackCounters:
    """Counts every place that can slip back to live/sim GT."""

    z_top: int = 0
    object_xy: int = 0
    velocity: int = 0
    delta: int = 0

    def total(self) -> int:
        return int(self.z_top + self.object_xy + self.velocity + self.delta)

    def as_dict(self) -> dict[str, int]:
        return {
            "z_top": int(self.z_top),
            "object_xy": int(self.object_xy),
            "velocity": int(self.velocity),
            "delta": int(self.delta),
            "total": int(self.total()),
        }


@dataclass
class ObsDelayBuffer:
    """Ring of captured frames. Expert reads ``get(now - Δ_true)``.

    Call ``push`` once per control step with the live capture, then
    ``get_delayed`` for the expert. Rounding of Δ to the nearest sim step
    is logged on ``last_round_dt_s``.
    """

    dt: float
    frames: list[DelayedFrame] = field(default_factory=list)
    last_round_dt_s: float = 0.0
    last_capture_t: float = float("nan")
    live_gt_leaks: int = 0
    fallbacks: GtFallbackCounters = field(default_factory=GtFallbackCounters)
    _allow_live: bool = False  # True only inside success judge

    def reset(self) -> None:
        self.frames.clear()
        self.last_round_dt_s = 0.0
        self.last_capture_t = float("nan")
        self.live_gt_leaks = 0
        self.fallbacks = GtFallbackCounters()

    def allow_live_gt(self, on: bool) -> None:
        """Success judge may flip this on while scoring; expert must leave it off."""
        self._allow_live = bool(on)

    def note_live_gt_read(self, where: str) -> None:
        """Call when code touches live object GT. Expert path must never call this."""
        if self._allow_live:
            return
        self.live_gt_leaks += 1

    def push(self, frame: DelayedFrame) -> None:
        """Store one capture. Keeps ~2 s of history (enough for Δ ≤ 300 ms)."""
        self.frames.append(frame)
        self.last_capture_t = float(frame.t_capture)
        max_keep = max(8, int(round(2.0 / max(self.dt, 1e-6))) + 2)
        if len(self.frames) > max_keep:
            self.frames = self.frames[-max_keep:]

    def get_delayed(self, now: float, delta_true_s: float) -> DelayedFrame:
        """Return the frame nearest to ``now − Δ``, and log the time round error."""
        if not self.frames:
            raise RuntimeError("ObsDelayBuffer.get_delayed called before any push")
        target = float(now) - float(delta_true_s)
        # Round target to nearest capture time (captures sit on sim steps).
        best = min(self.frames, key=lambda f: abs(float(f.t_capture) - target))
        self.last_round_dt_s = float(best.t_capture) - target
        return best


@dataclass
class DeltaMeasSampler:
    """Draw η ~ N(0, σ) once per replan cycle; hold Δ_meas until the next.

    Replan period defaults to 0.33 s (≈ 10 ticks at 30 Hz). Δ_meas is clamped ≥ 0.
    """

    delta_true_s: float
    noise_std_s: float = 0.005
    replan_period_s: float = 0.33
    seed: int = 0
    _rng: Any = field(init=False, repr=False)
    _last_replan_t: float = field(default=-1e9, init=False)
    _delta_meas_s: float = field(default=0.0, init=False)
    _eta_s: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._delta_meas_s = max(0.0, float(self.delta_true_s))
        self._eta_s = 0.0

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._last_replan_t = -1e9
        self._delta_meas_s = max(0.0, float(self.delta_true_s))
        self._eta_s = 0.0

    def update(self, now: float) -> float:
        """Maybe redraw η; return current Δ_meas (≥ 0)."""
        now = float(now)
        if now - self._last_replan_t >= float(self.replan_period_s) - 1e-12:
            self._eta_s = float(self._rng.normal(0.0, float(self.noise_std_s)))
            self._delta_meas_s = max(0.0, float(self.delta_true_s) + self._eta_s)
            self._last_replan_t = now
        return float(self._delta_meas_s)

    @property
    def delta_meas_s(self) -> float:
        return float(self._delta_meas_s)

    @property
    def eta_s(self) -> float:
        return float(self._eta_s)


def finger_axis_width_m(
    object_size_xyz: np.ndarray,
    *,
    grasp_yaw_rad: float = 0.0,
) -> float:
    """Object width along the finger-open axis at grasp yaw (metres).

    Fingers open along world Y when yaw=0. Rotating the hand by yaw swaps the
    projected XY extents. Use this for w_pre, not raw bbox Y.
    """
    s = np.asarray(object_size_xyz, dtype=np.float64).reshape(3)
    # Half-extents in object frame; for upright fruit, XY are the open axes.
    hx, hy = 0.5 * float(s[0]), 0.5 * float(s[1])
    c, sn = float(np.cos(grasp_yaw_rad)), float(np.sin(grasp_yaw_rad))
    # World unit along finger-open (hand Y after yaw about Z).
    axis = np.array([-sn, c], dtype=np.float64)
    # Width of an ellipse (round fruit seen from above) along that axis. Same as
    # the bbox at 0° / 90°; the bbox read a 4×4.2 cm egg as 5.8 cm at 45° and
    # freed it ~130 ms before the pads touched (yawed grasp).
    return float(2.0 * np.hypot(axis[0] * hx, axis[1] * hy))


def w_pre_m(
    object_size_xyz: np.ndarray,
    *,
    grasp_yaw_rad: float = 0.0,
    slack_m: float = 0.005,
    max_open_m: float = 0.080,
) -> tuple[float, bool, float]:
    """Pre-shape opening: w_obj + 2·s, clamped at Franka open width.

    Returns (w_pre, clamped, raw_before_clamp).
    """
    w_obj = finger_axis_width_m(object_size_xyz, grasp_yaw_rad=grasp_yaw_rad)
    raw = float(w_obj) + 2.0 * float(slack_m)
    clamped = raw > float(max_open_m) + 1e-12
    return float(min(raw, float(max_open_m))), bool(clamped), float(raw)
