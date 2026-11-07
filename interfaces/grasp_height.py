"""Grasp height from object top (claim Z update).

Plain words:
  Expert aims object centre. From a measured top:
  z_grasp = max(z_top − half_height, z_table + c)

  * z_top — object top (wrist depth or GT centre + half-height).
  * half_height — from the mesh each episode (not a yaml insert knob).
  * c — keep fingertips off the table (table_clear).
  * z_slack — only close-band: |hand Z − target| ≤ z_slack/2.
  * Collect while hand is high; lock when close (D405 bad in the last ~7 cm).

p̂z in the policy pack is this z_grasp (vz = 0).

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
DEFAULT_DEPTH_PERCENTILE = 90.0  # high → near-side depth via 100−p (outliers)
# Cap how many approach samples we keep (~2 s at 30 Hz) so old heights fade.
DEFAULT_MAX_SAMPLES = 60
# Franka defaults: TCP from panda_hand; wrist cam on the hand (phase0_cfg.yaml).
# Pad midpoint (pinch) in panda_hand — matches robot.tcp_offset_mm after Fix1.
DEFAULT_TCP_OFFSET_M = np.array([0.0, 0.0, 0.1034], dtype=np.float64)
DEFAULT_WRIST_POS_M = np.array([0.065, 0.0, 0.0], dtype=np.float64)
DEFAULT_WRIST_QUAT_WXYZ = np.array([0.0, 0.7071068, 0.7071068, 0.0], dtype=np.float64)
# Skip depths closer than this (fingers). Reject the frame if the near cluster
# is too mixed (table+object) — A4: latch σ ≤ 2 mm needs this gate.
DEFAULT_FINGER_MAX_M = 0.05
DEFAULT_MAX_SPREAD_M = 0.010


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


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def wrist_cam_pose_from_ee(
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
    *,
    tcp_offset_m: np.ndarray | None = None,
    wrist_pos_m: np.ndarray | None = None,
    wrist_quat_wxyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """World (cam_pos, look_unit) of the wrist cam from TCP pose + known offsets.

    TCP = panda_hand ⊕ tcp_offset. Cam = panda_hand ⊕ wrist_pos (OpenGL −Z look
    after wrist_quat). Use this when ``Camera.data.pos_w`` is stale (common).
    """
    tcp = np.asarray(
        DEFAULT_TCP_OFFSET_M if tcp_offset_m is None else tcp_offset_m, dtype=np.float64
    ).reshape(3)
    wpos = np.asarray(
        DEFAULT_WRIST_POS_M if wrist_pos_m is None else wrist_pos_m, dtype=np.float64
    ).reshape(3)
    wq = np.asarray(
        DEFAULT_WRIST_QUAT_WXYZ if wrist_quat_wxyz is None else wrist_quat_wxyz,
        dtype=np.float64,
    ).reshape(4)
    ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    R = quat_wxyz_to_rotmat(ee_quat_wxyz)
    # ee_quat ≈ hand quat when tcp quat is identity (our DiffIK default).
    hand_pos = ee - R @ tcp
    cam_pos = hand_pos + R @ wpos
    look = -(R @ quat_wxyz_to_rotmat(wq))[:, 2]
    n = float(np.linalg.norm(look))
    if n > 1e-9:
        look = look / n
    return cam_pos, look


def z_top_from_wrist_depth(
    depth: np.ndarray | None,
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
    *,
    cam_z: float | None = None,
    cam_above_tcp_m: float = 0.1034,
    percentile: float = DEFAULT_DEPTH_PERCENTILE,
    rows: tuple[float, float] = (0.55, 0.95),
    cols: tuple[float, float] = (0.30, 0.70),
    finger_max_m: float = DEFAULT_FINGER_MAX_M,
    max_spread_m: float = DEFAULT_MAX_SPREAD_M,
    tcp_offset_m: np.ndarray | None = None,
    wrist_pos_m: np.ndarray | None = None,
    z_bias_m: float = 0.0,
) -> float:
    """Object top Z from wrist depth image (metres, world frame).

    Near-side percentile in the finger/object window. Prefer live ``cam_z``;
    else lift from TCP⊕hand⊕wrist offsets (not bare ``ee_z`` — that was ~12 cm
    low). Reject the frame when the near-depth cluster is mixed (table+object):
    ``std(depths ≤ depth_top+2 cm) > max_spread_m`` → NaN so the latch skips it.
    ``z_bias_m`` (claim_height.depth_z_bias) is added last — A4 fruit latch was
    ~6.7 mm low; set the yaml knob from that report, do not invent per-frame.

    Returns NaN if there is no usable depth.
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
    # D405 band; drop finger-close hits so the near percentile is the object top.
    ok = crop[
        np.isfinite(crop)
        & (crop > float(finger_max_m))
        & (crop < 2.0)
    ]
    if ok.size < 8:
        return float("nan")
    # Nearer = smaller depth. yaml percentile 90 → use 10th percentile of depth.
    depth_top = float(np.percentile(ok, 100.0 - float(percentile)))
    near = ok[ok <= depth_top + 0.02]
    if (
        float(max_spread_m) > 0.0
        and near.size >= 8
        and float(np.std(near)) > float(max_spread_m)
    ):
        return float("nan")

    if cam_z is not None and np.isfinite(float(cam_z)):
        cz = float(cam_z)
        look_z = -1.0
    else:
        # Geometric cam from EE (fixes stale Camera.data.pos_w in logs).
        tcp = (
            np.asarray(tcp_offset_m, dtype=np.float64).reshape(3)
            if tcp_offset_m is not None
            else DEFAULT_TCP_OFFSET_M.copy()
        )
        # Legacy kw: only Z of tcp when caller still passes cam_above_tcp_m alone.
        if tcp_offset_m is None and abs(float(cam_above_tcp_m) - 0.107) > 1e-9:
            tcp = np.array([0.0, 0.0, float(cam_above_tcp_m)], dtype=np.float64)
        cam_pos, look = wrist_cam_pose_from_ee(
            ee_pos,
            ee_quat_wxyz,
            tcp_offset_m=tcp,
            wrist_pos_m=wrist_pos_m,
        )
        cz = float(cam_pos[2])
        look_z = float(look[2])
    # OpenGL look ≈ −Z when hanging down: z_top = cam_z − depth.
    if look_z < -0.5:
        z_top = float(cz - depth_top)
    else:
        z_top = float(cz + depth_top * look_z)
    return z_top + float(z_bias_m)


def wrist_intrinsics(cfg: dict) -> tuple[float, float, float]:
    """(f_px, cx, cy) of the wrist camera from ``wrist_camera`` (square pixels)."""
    wc = cfg.get("wrist_camera") or {}
    w, h = int(wc.get("width", 480)), int(wc.get("height", 360))
    f_px = float(wc.get("focal_length", 2.3)) / float(wc.get("horizontal_aperture", 4.6)) * w
    return f_px, w / 2.0, h / 2.0


def z_top_from_wrist_cloud(
    depth: np.ndarray | None,
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
    obj_xy: np.ndarray,
    *,
    f_px: float,
    cx: float,
    cy: float,
    z_table: float = 0.0,
    tcp_offset_m: np.ndarray | None = None,
    wrist_pos_m: np.ndarray | None = None,
    wrist_quat_wxyz: np.ndarray | None = None,
    radius_m: float = 0.04,
    percentile: float = 99.0,
    min_points: int = 20,
    stride: int = 2,
    z_bias_m: float = 0.0,
    top_frac: float = 0.05,
    use_top_median: bool = True,
) -> float:
    """Object top Z (world, m) from the whole wrist depth image. NaN if not seen.

    Every depth pixel is lifted to a world point with the full camera pose
    (hand pose ⊕ wrist mount, OpenGL axes; Isaac depth = distance to the image
    plane). Points within ``radius_m`` of the object XY (from the static camera
    / filter), above the table and below the fingertips are the object; its top
    is the **median of the top 5% of Z** (robust to one-pixel spikes). Pass
    ``use_top_median=False`` to fall back to a raw high percentile.
    """
    if depth is None:
        return float("nan")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    d = d[::stride, ::stride]
    vv, uu = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
    uu, vv = uu * stride, vv * stride
    tcp = DEFAULT_TCP_OFFSET_M if tcp_offset_m is None else np.asarray(tcp_offset_m, np.float64)
    wpos = DEFAULT_WRIST_POS_M if wrist_pos_m is None else np.asarray(wrist_pos_m, np.float64)
    wq = DEFAULT_WRIST_QUAT_WXYZ if wrist_quat_wxyz is None else np.asarray(wrist_quat_wxyz, np.float64)
    ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    R = quat_wxyz_to_rotmat(ee_quat_wxyz)
    cam = ee - R @ tcp.reshape(3) + R @ wpos.reshape(3)
    R_wc = R @ quat_wxyz_to_rotmat(wq)
    ok = np.isfinite(d) & (d > DEFAULT_FINGER_MAX_M) & (d < 2.0)
    d, uu, vv = d[ok], uu[ok], vv[ok]
    pc = np.stack([(uu - cx) / f_px * d, -(vv - cy) / f_px * d, -d], axis=-1)
    pw = pc @ R_wc.T + cam
    xy = np.asarray(obj_xy, dtype=np.float64).reshape(-1)[:2]
    sel = (
        (np.linalg.norm(pw[:, :2] - xy, axis=1) < float(radius_m))
        & (pw[:, 2] > float(z_table) + 0.005)  # not the table
        & (pw[:, 2] < float(ee[2]) - 0.01)  # not the fingers / hand
    )
    if int(sel.sum()) < int(min_points):
        return float("nan")
    zs = pw[sel, 2]
    if bool(use_top_median):
        # Median of the tallest top_frac points — less spike-prone than p99.
        k = max(1, int(np.ceil(float(top_frac) * float(zs.size))))
        top = np.partition(zs, -k)[-k:]
        z_top = float(np.median(top))
    else:
        z_top = float(np.percentile(zs, float(percentile)))
    return z_top + float(z_bias_m)


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
        """Read claim_height knobs (SI after normalize).

        ``insert_m`` starts as a placeholder; the expert overwrites it each
        episode with mesh half-height so top − insert ≈ object centre.
        """
        ch = cfg.get("claim_height") if isinstance(cfg, dict) else None
        ch = ch if isinstance(ch, dict) else {}
        sm = cfg.get("state_machine_params") if isinstance(cfg, dict) else None
        sm = sm if isinstance(sm, dict) else {}
        scene = cfg.get("scene") if isinstance(cfg, dict) else None
        scene = scene if isinstance(scene, dict) else {}
        z_table = float(scene.get("table_height", 0.0) or 0.0)
        # Prefer scene object_radius as the pre-episode half-height guess.
        half_guess = float(scene.get("object_radius") or DEFAULT_INSERT_M)
        return cls(
            insert_m=half_guess,
            clear_m=float(ch.get("table_clear_m", sm.get("table_clear_m", DEFAULT_CLEAR_M))),
            z_table=z_table,
            latch_lo_m=float(
                ch.get(
                    "stop_trusting_depth_below_m",
                    ch.get("latch_lo_m", DEFAULT_LATCH_LO_M),
                )
            ),
            latch_hi_m=float(
                ch.get(
                    "use_depth_below_m",
                    ch.get("latch_hi_m", DEFAULT_LATCH_HI_M),
                )
            ),
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
