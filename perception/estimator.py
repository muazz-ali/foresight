"""Camera frames → ``ObjectState``: the drop-in for ``scene.get_object_state()``.

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


class StaticCamEstimator:
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
        self.plane_z = 0.0
        self._uv_last: np.ndarray | None = None
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
        self.kf = CVKalman(**self._kalman_kwargs)
        self.n_redetects = 0
        self.plane_z = float(np.asarray(spawn_xyz, dtype=np.float64)[2])
        self._uv_last = self.camera.project(spawn_xyz)
        return self.tracker.start(rgb, self._uv_last)

    def _in_play_area(self, uv: np.ndarray) -> bool:
        if self.search_xy_bounds is None:
            return True
        (x0, x1), (y0, y1) = self.search_xy_bounds
        x, y = self.camera.lift(uv, self.plane_z)
        m = SEARCH_MARGIN_M
        return x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m

    def step(self, rgb: np.ndarray, t: float) -> ObjectState | None:
        """One camera frame at time ``t``. None until the object has been seen once."""
        blind = self.kf.started and not self.kf.state(self.plane_z).valid
        if blind:
            # Coast rule: past max_coast_s the old guess is stale — search the play area.
            uv = self.tracker.search(rgb, self._uv_last, self._in_play_area)
            if uv is not None:
                self.kf = CVKalman(**self._kalman_kwargs)  # fresh track: speed unknown
                self.n_redetects += 1
        else:
            if self.kf.started:
                guess = self.kf.predict_xy(t)
                uv_guess = self.camera.project(np.r_[guess, self.plane_z])
            else:
                uv_guess = self._uv_last
            uv = self.tracker.step(rgb, uv_guess)
        self.last_uv = uv
        z = None if uv is None else self.camera.lift(uv, self.plane_z)
        if uv is not None:
            self._uv_last = uv
        self.kf.step(t, z)
        return self.kf.state(self.plane_z) if self.kf.started else None
