"""Blob tracker for the static camera (plan §3 tracker ladder, rung 1).

Detector-once hands over a start pixel (in sim: the projected spawn pose).
After that the tracker uses pixels only: each frame it takes the blob nearest
the filter's guess, inside a gate. The look is picked once, on the first frame:

  colour — saturated pixels. Every fruit except egg (table, robot, grid and
           bowl are grey / white).
  bright — white pixels, then an opening that erases thin grid lines (~5 px)
           but keeps an egg (~16 px wide).

``search`` is the re-detect for the plan §6 coast rule: after a long blind spell
(e.g. the hand parked between camera and object) look for the same look and size
anywhere in the play area, not just near the stale guess.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

SAT_MIN, VAL_MIN = 90, 50  # colour look
BRIGHT_VAL_MIN, BRIGHT_SAT_MAX = 150, 40  # bright look
OPEN_PX = 9  # wider than a grid line, narrower than an egg
MIN_AREA_PX = 15
MAX_AREA_GROWTH = 3.0  # × first-frame area


class BlobTracker:
    def __init__(self, gate_px: float = 25.0) -> None:
        self.gate_px = float(gate_px)
        self.mode: str | None = None
        self.ref_area: float | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))

    def _mask(self, rgb: np.ndarray, mode: str) -> np.ndarray:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        if mode == "colour":
            return ((s > SAT_MIN) & (v > VAL_MIN)).astype(np.uint8)
        m = ((v > BRIGHT_VAL_MIN) & (s < BRIGHT_SAT_MAX)).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)

    def _nearest(
        self,
        rgb: np.ndarray,
        mode: str,
        uv_guess: np.ndarray,
        *,
        gate_px: float | None = None,
        area_range: tuple[float, float] | None = None,
        inside: Callable[[np.ndarray], bool] | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """(centre, area) of the blob nearest ``uv_guess`` that passes the checks, else None."""
        n, _, stats, cents = cv2.connectedComponentsWithStats(self._mask(rgb, mode))
        if n <= 1:
            return None
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        dist = np.linalg.norm(cents[1:] - np.asarray(uv_guess, dtype=np.float64), axis=1)
        ok = area >= MIN_AREA_PX
        if gate_px is not None:
            ok &= dist < gate_px
        if area_range is not None:
            ok &= (area >= area_range[0]) & (area <= area_range[1])
        if inside is not None:
            ok &= np.array([inside(c) for c in cents[1:]], dtype=bool)
        if not ok.any():
            return None
        j = int(np.argmin(np.where(ok, dist, np.inf)))
        return cents[1 + j], float(area[j])

    def start(self, rgb: np.ndarray, uv0: np.ndarray) -> bool:
        """Detector-once: keep the first look that finds a blob at ``uv0``."""
        self.mode = self.ref_area = None
        for mode in ("colour", "bright"):
            hit = self._nearest(rgb, mode, uv0, gate_px=self.gate_px)
            if hit is not None:
                self.mode, self.ref_area = mode, hit[1]
                return True
        return False

    def step(self, rgb: np.ndarray, uv_guess: np.ndarray) -> np.ndarray | None:
        """Object centre pixel this frame, or None (lost / hidden)."""
        if self.mode is None:
            return None
        grown = (0.0, MAX_AREA_GROWTH * self.ref_area)  # bigger = merged with the gripper
        hit = self._nearest(rgb, self.mode, uv_guess, gate_px=self.gate_px, area_range=grown)
        return None if hit is None else hit[0]

    def search(
        self, rgb: np.ndarray, uv_near: np.ndarray, inside: Callable[[np.ndarray], bool]
    ) -> np.ndarray | None:
        """Re-detect after a long blind spell: same look, similar size, anywhere ``inside``."""
        if self.mode is None:
            return None
        similar = (0.4 * self.ref_area, 2.5 * self.ref_area)
        hit = self._nearest(rgb, self.mode, uv_near, area_range=similar, inside=inside)
        return None if hit is None else hit[0]
