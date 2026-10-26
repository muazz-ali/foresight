"""Grasp height from object top (claim Z update).

Plain words:
  z_grasp = max(z_top − g, z_table + c)

  * z_top — object top height (m). From wrist depth (high percentile of object
    pixels) or from sim GT (centre + half-height).
  * g — how far the fingers push down past the top (insertion).
  * c — keep fingertips off the table.
  * Collect: while hand is ≥ 20 cm above the top, keep reading z_grasp every
    step and store them (wrist depth is usually good this far).
  * Lock: when hand drops below 10 cm, freeze to the median of those readings
    (D405 gets bad in the last ~7 cm — stop trusting new depth).

p̂z in the policy pack is this z_grasp (vz = 0), so approach offsets d_z stay
the same for short and tall objects.

Dual-cam fuse (optional): wrist RGB-D + static depth → one z_top per step
via ``fuse_z_top`` (average when both good; prefer wrist when close).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Defaults (metres). Override from yaml via from_cfg.
DEFAULT_INSERT_M = 0.010  # g: 10 mm into the object from the top
DEFAULT_CLEAR_M = 0.005  # c: 5 mm above table
DEFAULT_Z_SLACK_M = 0.005  # s_z: 5 mm vertical slack
DEFAULT_LATCH_LO_M = 0.10  # lock when hand–top gap drops below this
DEFAULT_LATCH_HI_M = 0.20  # collect while gap is at or above this
DEFAULT_DEPTH_PERCENTILE = 90.0  # high percentile, not max (outliers)
# Cap how many approach samples we keep (~2 s at 30 Hz) so old heights fade.
DEFAULT_MAX_SAMPLES = 60


def z_grasp_m(
    z_top: float,
    *,
    z_table: float = 0.0,
    insert_m: float = DEFAULT_INSERT_M,
    clear_m: float = DEFAULT_CLEAR_M,
) -> float:
    """Closing height: into the object from the top, but never through the table."""
    return float(max(float(z_top) - float(insert_m), float(z_table) + float(clear_m)))


def z_top_from_gt_center(
    z_center: float,
    *,
    half_height_m: float,
) -> float:
    """Sim GT: object top ≈ centre Z + half-height (sphere/fruit approx)."""
    return float(z_center) + float(half_height_m)


def z_top_from_wrist_depth(
    depth: np.ndarray | None,
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
    *,
    cam_z: float | None = None,
    cam_above_tcp_m: float = 0.107,
    percentile: float = DEFAULT_DEPTH_PERCENTILE,
    rows: tuple[float, float] = (0.55, 0.95),
    cols: tuple[float, float] = (0.30, 0.70),
) -> float:
    """Object top Z from wrist depth image (metres, world frame).

    Takes a high percentile of valid depths in the finger window (not the single
    minimum/maximum — those are noisy). Depth is distance along the camera look.

    Prefer real ``cam_z`` from ``scene.get_wrist_cam_pose()``. Fallback:
    ``ee_z + cam_above_tcp_m`` (Franka hand cam is above the TCP). Using plain
    ``ee_z − depth`` was ~12 cm too low.

    ``ee_quat_wxyz`` reserved for a full cam-ray lift later.
    Returns NaN if there is no usable depth.
    """
    del ee_quat_wxyz  # full optical-axis lift later; Z-down approx is enough now
    if depth is None:
        return float("nan")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {d.shape}")
    h, w = d.shape
    crop = d[
        int(h * rows[0]) : int(h * rows[1]),
        int(w * cols[0]) : int(w * cols[1]),
    ]
    ok = crop[np.isfinite(crop) & (crop > 0.05) & (crop < 2.0)]  # D405 useful band
    if ok.size < 8:
        return float("nan")
    # Nearer = smaller depth. High percentile of *near* side ≈ top surface toward cam.
    # Use low percentile of depth (closer points) as the top facing the wrist.
    depth_top = float(np.percentile(ok, 100.0 - float(percentile)))
    ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    if cam_z is not None and np.isfinite(float(cam_z)):
        cz = float(cam_z)
    else:
        cz = float(ee[2]) + float(cam_above_tcp_m)
    return float(cz - depth_top)


def z_top_from_static_depth(
    depth: np.ndarray | None,
    *,
    cam_z: float,
    look_down_cos: float = 0.5,
    percentile: float = DEFAULT_DEPTH_PERCENTILE,
    rows: tuple[float, float] = (0.35, 0.75),
    cols: tuple[float, float] = (0.30, 0.70),
) -> float:
    """Rough object-top Z from a fixed camera depth map (metres).

    Uses a centre crop percentile, then ``cam_z − depth · cos`` (camera looking
    partly down). Good enough to fuse with wrist; not a full ray cast.
    Returns NaN if depth is missing or empty.
    """
    if depth is None:
        return float("nan")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {d.shape}")
    h, w = d.shape
    crop = d[
        int(h * rows[0]) : int(h * rows[1]),
        int(w * cols[0]) : int(w * cols[1]),
    ]
    ok = crop[np.isfinite(crop) & (crop > 0.05) & (crop < 5.0)]
    if ok.size < 8:
        return float("nan")
    depth_top = float(np.percentile(ok, 100.0 - float(percentile)))
    c = float(np.clip(look_down_cos, 0.1, 1.0))
    return float(cam_z) - depth_top * c


def fuse_z_top(
    wrist_z: float,
    static_z: float,
    *,
    gap_m: float | None = None,
    latch_hi_m: float = DEFAULT_LATCH_HI_M,
) -> float:
    """Combine wrist + static top heights into one z_top.

    * Both bad → NaN
    * One good → that one
    * Both good, hand still high (gap ≥ latch_hi): mean
    * Both good, closer: prefer wrist (D405 more local)
    """
    w_ok = bool(np.isfinite(wrist_z))
    s_ok = bool(np.isfinite(static_z))
    if not w_ok and not s_ok:
        return float("nan")
    if w_ok and not s_ok:
        return float(wrist_z)
    if s_ok and not w_ok:
        return float(static_z)
    gap = float(gap_m) if gap_m is not None and np.isfinite(gap_m) else float(latch_hi_m)
    if gap >= float(latch_hi_m):
        return 0.5 * (float(wrist_z) + float(static_z))
    return float(wrist_z)


@dataclass
class ZLatch:
    """Collect z_grasp while high (≥20 cm); lock median when close (<10 cm).

    Call ``update`` every control step during approach. Returns the live median
    while collecting, then the frozen value after lock.
    """

    insert_m: float = DEFAULT_INSERT_M
    clear_m: float = DEFAULT_CLEAR_M
    z_table: float = 0.0
    latch_lo_m: float = DEFAULT_LATCH_LO_M
    latch_hi_m: float = DEFAULT_LATCH_HI_M
    z_slack_m: float = DEFAULT_Z_SLACK_M
    max_samples: int = DEFAULT_MAX_SAMPLES
    latched_z_grasp: float | None = None
    latched_z_top: float | None = None
    samples_z_grasp: list[float] = field(default_factory=list)
    samples_z_top: list[float] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, cfg: dict) -> ZLatch:
        """Read claim_height / state_machine_params knobs (SI after normalize)."""
        ch = cfg.get("claim_height") if isinstance(cfg, dict) else None
        ch = ch if isinstance(ch, dict) else {}
        sm = cfg.get("state_machine_params") if isinstance(cfg, dict) else None
        sm = sm if isinstance(sm, dict) else {}
        scene = cfg.get("scene") if isinstance(cfg, dict) else None
        scene = scene if isinstance(scene, dict) else {}
        z_table = float(scene.get("table_height", 0.0) or 0.0)
        return cls(
            insert_m=float(ch.get("finger_insert_m", sm.get("finger_insert_m", DEFAULT_INSERT_M))),
            clear_m=float(ch.get("table_clear_m", sm.get("table_clear_m", DEFAULT_CLEAR_M))),
            z_table=z_table,
            latch_lo_m=float(ch.get("latch_lo_m", DEFAULT_LATCH_LO_M)),
            latch_hi_m=float(ch.get("latch_hi_m", DEFAULT_LATCH_HI_M)),
            z_slack_m=float(ch.get("z_slack_m", sm.get("z_slack_m", DEFAULT_Z_SLACK_M))),
            max_samples=int(ch.get("max_samples", DEFAULT_MAX_SAMPLES)),
        )

    def reset(self) -> None:
        """Clear samples and lock for a new episode."""
        self.latched_z_grasp = None
        self.latched_z_top = None
        self.samples_z_grasp.clear()
        self.samples_z_top.clear()

    def _append_sample(self, zg: float, z_top: float) -> None:
        """Store one good reading; drop the oldest if over the cap."""
        self.samples_z_grasp.append(float(zg))
        self.samples_z_top.append(float(z_top))
        n = int(self.max_samples)
        if n > 0 and len(self.samples_z_grasp) > n:
            overflow = len(self.samples_z_grasp) - n
            del self.samples_z_grasp[:overflow]
            del self.samples_z_top[:overflow]

    def _median_pair(self) -> tuple[float, float] | None:
        """Median of stored (z_grasp, z_top), or None if empty."""
        if not self.samples_z_grasp:
            return None
        return (
            float(np.median(np.asarray(self.samples_z_grasp, dtype=np.float64))),
            float(np.median(np.asarray(self.samples_z_top, dtype=np.float64))),
        )

    def _lock(self, zg: float, z_top: float) -> float:
        """Freeze height: prefer sample median, else this frame's value."""
        med = self._median_pair()
        if med is not None:
            self.latched_z_grasp, self.latched_z_top = med
        else:
            self.latched_z_grasp = float(zg)
            self.latched_z_top = float(z_top)
        return float(self.latched_z_grasp)

    def update(
        self,
        *,
        z_top: float,
        ee_z: float,
    ) -> float:
        """Update collect/lock and return the height to use this step.

        * gap ≥ latch_hi (20 cm): append sample; return running median.
        * latch_lo ≤ gap < latch_hi: do not add (depth getting riskier);
          return median so far, or live z_grasp if no samples yet.
        * gap < latch_lo (10 cm): lock once to median (or live if empty).
        """
        if self.latched_z_grasp is not None:
            return float(self.latched_z_grasp)

        if not np.isfinite(z_top):
            med = self._median_pair()
            return float(med[0]) if med is not None else float("nan")

        zg = z_grasp_m(
            z_top,
            z_table=self.z_table,
            insert_m=self.insert_m,
            clear_m=self.clear_m,
        )
        gap = float(ee_z) - float(z_top)

        if gap >= self.latch_hi_m:
            self._append_sample(zg, z_top)
            med = self._median_pair()
            return float(med[0]) if med is not None else zg

        if gap < self.latch_lo_m:
            return self._lock(zg, z_top)

        # Mid band (10–20 cm): hold the collect median; no new samples.
        med = self._median_pair()
        return float(med[0]) if med is not None else zg

    def close_z_ok(self, ee_z: float, p_hat_z: float) -> bool:
        """Guard: |z_hand − p̂z| ≤ s_z / 2 (not part of the learned policy)."""
        if not np.isfinite(ee_z) or not np.isfinite(p_hat_z):
            return False
        return abs(float(ee_z) - float(p_hat_z)) <= 0.5 * float(self.z_slack_m)


def hit_prob_z(s_z: float, sigma_z: float) -> float:
    """P_z(hit) = erf(s_z / (σ_z √2)). Isaac-free check helper."""
    from math import erf, sqrt

    if sigma_z <= 0.0:
        return 1.0
    return float(erf(float(s_z) / (float(sigma_z) * sqrt(2.0))))
