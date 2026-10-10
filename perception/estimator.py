"""PIPELINE STAGE: glue (camera picture in, ``ObjectState`` out).

Camera frames → ``ObjectState``: the drop-in for ``scene.get_object_state()``.
Output is the frozen message: ``position, velocity, covariance, timestamp, valid``.

Per frame: guess where the object is → tracker finds the blob → lift pixel to the
table → Kalman filter smooths it. No new maths lives here, only the wiring.

Wires detector-once + ``BlobTracker`` + ``PinholeCamera.lift`` + ``CVKalman``.
The policy pack is then built by the same ``interfaces.state.conditioning_vector``
the oracle uses, so the swap changes the source of the numbers and nothing else.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interfaces.state import ObjectState
from perception.camera import PinholeCamera
from perception.kalman import CVKalman
from perception.track import BlobTracker

SEARCH_MARGIN_M = 0.03  # re-detect looks this far outside the object play area
MAX_HALF_WIDTH_M = 0.035  # cans / bottles / cups stand tall but are at most ~7 cm wide


class StaticCamEstimator:
    """Runs track → lift → filter once per camera frame. Call ``start`` once, then ``step``."""

    def __init__(
        self,
        camera: PinholeCamera,
        *,
        search_xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        tracker: BlobTracker | None = None,
        kalman_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.camera = camera
        self.search_xy_bounds = search_xy_bounds
        self.tracker = tracker or BlobTracker()
        self._kalman_kwargs = dict(kalman_kwargs or {})
        self.kf = CVKalman(**self._kalman_kwargs)
        self.plane_z = 0.0  # height (m) of the object centre; set in ``start``
        self._uv_last: np.ndarray | None = None  # last pixel we trusted
        self.last_uv: np.ndarray | None = None  # tracker output this frame (None = missed)
        self.n_redetects = 0

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], **kwargs: Any) -> StaticCamEstimator:
        """Static camera + object play area (``scene.object_xy_bounds``) from the sim yaml."""
        bounds = cfg["scene"].get("object_xy_bounds")
        return cls(
            PinholeCamera.from_cfg(cfg),
            search_xy_bounds=None if bounds is None else tuple(map(tuple, bounds)),
            **kwargs,
        )

    def start(self, rgb: np.ndarray, spawn_xyz: np.ndarray) -> bool:
        """Detector-once at the spawn pose; ``plane_z`` = object centre height.

        In sim the spawn pose stands in for an open-vocabulary detector (plan §3).
        It is used once; every later frame is pixels only.
        """
        self.kf = CVKalman(**self._kalman_kwargs)  # fresh filter for a fresh episode
        self.n_redetects = 0
        p = np.asarray(spawn_xyz, dtype=np.float64).reshape(3)
        self.plane_z = float(p[2])
        self._uv_last = self.camera.project(p)
        # Rough object size in pixels. Resting on the table, the centre height is
        # the half-height; a fruit is as wide as it is tall, a bottle is not.
        r_m = min(self.plane_z, MAX_HALF_WIDTH_M)
        r_px = self.camera.fx * r_m / float(np.linalg.norm(p - self.camera.position))
        # Tall objects: the upright axis, minus a half-width at each end, in pixels.
        up = np.array([0.0, 0.0, self.plane_z - r_m])
        seg = np.array([self.camera.project(p + up), self.camera.project(p - up)])
        return self.tracker.start(rgb, self._uv_last, r_px=r_px, seg=seg)

    def _in_play_area(self, uv: np.ndarray) -> bool:
        """True if this pixel, lifted to the table, lands in the play area (+ small margin)."""
        if self.search_xy_bounds is None:
            return True
        (x0, x1), (y0, y1) = self.search_xy_bounds
        try:
            x, y = self.camera.lift(uv, self.plane_z)
        except ValueError:  # a wall or ceiling pixel (house scenes): never on the table
            return False
        m = SEARCH_MARGIN_M
        return x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m

    def step(self, rgb: np.ndarray, t: float) -> ObjectState | None:
        """One camera frame at time ``t``. None until the object has been seen once."""
        # "Blind" = filter exists but ``valid`` is False (unseen too long).
        blind = self.kf.started and not self.kf.state(self.plane_z).valid
        if blind:
            # Coast rule: past max_coast_s the old guess is stale — search the play area.
            uv = self.tracker.search(rgb, self._uv_last, self._in_play_area)
            if uv is not None:
                self.kf = CVKalman(**self._kalman_kwargs)  # fresh track: speed unknown
                self.n_redetects += 1
        else:
            # Normal case: filter predicts where the object is now (metres); project
            # that to a pixel so the tracker looks in the right spot.
            if self.kf.started:
                guess = self.kf.predict_xy(t)
                uv_guess = self.camera.project(np.r_[guess, self.plane_z])
            else:
                uv_guess = self._uv_last
            uv = self.tracker.step(rgb, uv_guess)
        self.last_uv = uv
        # Pixel → table point (metres). None means "not seen": the filter will coast.
        z = None if uv is None else self.camera.lift(uv, self.plane_z)
        if uv is not None:
            self._uv_last = uv
        self.kf.step(t, z)
        return self.kf.state(self.plane_z) if self.kf.started else None
