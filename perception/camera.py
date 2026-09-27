"""Static-camera pinhole model: world point → pixel, pixel → point on a table plane.

Built from ``sim/phase0_cfg.yaml`` — the same numbers ``sim/scene.py`` spawns the
camera with (XYZ Euler degrees, OpenGL frame: looks down −Z, +Y up, +X right).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PinholeCamera:
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
        """Read ``camera.*`` (lens) and ``scene.cameras[name]`` (pose) from the sim yaml."""
        cam = cfg["camera"]
        w, h = int(cam["width"]), int(cam["height"])
        # Square pixels: Isaac derives the vertical aperture from the aspect ratio.
        f_px = float(cam["focal_length"]) / float(cam["horizontal_aperture"]) * w
        entry = next(c for c in cfg["scene"]["cameras"] if c["name"] == name)
        R_wc = Rotation.from_euler("XYZ", entry["rotation_deg"], degrees=True).as_matrix()
        pos = np.asarray(entry["position"], dtype=np.float64)
        return cls(f_px, f_px, w / 2.0, h / 2.0, w, h, pos, R_wc)

    def project(self, p_world: np.ndarray) -> np.ndarray:
        """World point (3,) → pixel (u, v)."""
        x, y, z = self.R_wc.T @ (np.asarray(p_world, dtype=np.float64).reshape(3) - self.position)
        return np.array([self.cx + self.fx * x / -z, self.cy - self.fy * y / -z])

    def lift(self, uv: np.ndarray, plane_z: float) -> np.ndarray:
        """Pixel → world XY where its ray meets the plane ``z = plane_z``.

        Use the object's centre height, not the table top: the blob centre is the
        object's middle, and at this camera tilt a 3 cm height error is a ~2–3 cm
        XY error.
        """
        u, v = float(uv[0]), float(uv[1])
        d = self.R_wc @ np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        if d[2] >= -1e-9:
            raise ValueError(f"pixel ({u:.1f}, {v:.1f}) looks at or above the horizon")
        s = (float(plane_z) - self.position[2]) / d[2]
        return (self.position + s * d)[:2]
