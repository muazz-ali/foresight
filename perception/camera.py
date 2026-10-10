"""PIPELINE STAGE: camera (geometry only, no images, no filtering).

Static-camera pinhole model: world point → pixel, pixel → point on a table plane.

Why it exists: the tracker works in pixels, the filter and policy work in metres.
This file is the bridge between the two. Its table points feed the Kalman filter,
which makes the frozen ``position, velocity, covariance, timestamp, valid`` message.

Built from ``sim/phase0_cfg.yaml`` → ``static_camera`` — the same numbers
``sim/scene.py`` spawns the table cam with (XYZ Euler degrees, OpenGL frame:
looks down −Z, +Y up, +X right).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PinholeCamera:
    """One fixed camera: lens numbers (pixels) plus where it sits and points (metres)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    position: np.ndarray  # (3,) camera centre, world frame (m)
    R_wc: np.ndarray  # (3, 3) camera axes in world coordinates (OpenGL)

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], name: str = "static_cam") -> PinholeCamera:
        """Read ``static_camera`` (lens + pose) from the sim yaml.

        ``name`` is kept for call-site compatibility; only ``static_cam`` is supported
        (wrist geometry is not a fixed world pinhole).
        """
        if name != "static_cam":
            raise ValueError(
                f"PinholeCamera.from_cfg only builds static_cam; got {name!r}"
            )
        cam = cfg["static_camera"]
        w, h = int(cam["width"]), int(cam["height"])
        # Square pixels: Isaac derives the vertical aperture from the aspect ratio.
        f_px = float(cam["focal_length"]) / float(cam["horizontal_aperture"]) * w
        R_wc = Rotation.from_euler("XYZ", cam["rotation_deg"], degrees=True).as_matrix()
        pos = np.asarray(cam["position"], dtype=np.float64)
        return cls(f_px, f_px, w / 2.0, h / 2.0, w, h, pos, R_wc)

    def project(self, p_world: np.ndarray) -> np.ndarray:
        """World point (3,) → pixel (u, v)."""
        # Step 1: move the point into the camera's own axes (shift to camera, then rotate).
        x, y, z = self.R_wc.T @ (np.asarray(p_world, dtype=np.float64).reshape(3) - self.position)
        # Step 2: divide by depth (-z, since the camera looks down -Z) to get the
        # pixel. Pixel v grows downward, but camera y grows upward, so v uses a minus.
        return np.array([self.cx + self.fx * x / -z, self.cy - self.fy * y / -z])

    def lift(self, uv: np.ndarray, plane_z: float) -> np.ndarray:
        """Pixel → world XY where its ray meets the plane ``z = plane_z``.

        Use the object's centre height, not the table top: the blob centre is the
        object's middle, and at this camera tilt a 3 cm height error is a ~2–3 cm
        XY error.
        """
        u, v = float(uv[0]), float(uv[1])
        # Ray direction through this pixel: camera axes first, then rotated to world.
        # (This undoes ``project``; depth is unknown, so we only get a direction.)
        d = self.R_wc @ np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        # The ray must point down (world z shrinking) or it never reaches the table.
        if d[2] >= -1e-9:
            raise ValueError(f"pixel ({u:.1f}, {v:.1f}) looks at or above the horizon")
        # Walk along the ray ``s`` steps until its height equals ``plane_z``.
        s = (float(plane_z) - self.position[2]) / d[2]
        # Keep only X, Y (metres). Z is already ``plane_z`` by construction.
        return (self.position + s * d)[:2]
