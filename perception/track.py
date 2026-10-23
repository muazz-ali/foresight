"""PIPELINE STAGE: track (pixels in, one pixel out; no metres, no velocity).

Blob tracker for the static camera (plan §3 tracker ladder, rung 2).
Why it exists: the filter needs a measurement each frame. This file turns a
colour picture into "the object is at pixel (u, v)", or "I can't see it".
It does not fill the frozen message (``position, velocity, covariance,
timestamp, valid``); that happens in kalman.py after the pixel is lifted to the table.

Detector-once hands over a start pixel (in sim: the projected spawn pose) and
the object's rough size in pixels. On that first frame we learn what the object
looks like compared with the rest of the picture:

  hist   — count HSV colours inside a small disk on the object, and in the rest
           of the frame (table, robot, bowl, walls). A colour is "object" when
           the object owns most of its pixels. Each later frame keeps the pixels
           whose colour scores high (histogram back-projection). Works on wood
           tables, where "any saturated pixel" covers the whole table.
  colour — saturated pixels (old rung 1). Fallback only.
  bright — white pixels minus thin grid lines (old rung 1, the egg). Fallback only.

A look is kept only if its blob at the start pixel is about the object's size
(not a speck, not the whole table). After that: nearest blob to the filter's
guess, inside a gate, not much bigger than on frame 0 (else merged with the hand).

``search`` is the re-detect for the plan §6 coast rule: after a long blind spell
look for the same look and size anywhere in the play area.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

# HSV (hue / saturation / value) cut-offs for the fallback looks.
SAT_MIN, VAL_MIN = 90, 50  # colour look: colourful and not dark
BRIGHT_VAL_MIN, BRIGHT_SAT_MAX = 150, 40  # bright look: very bright and not colourful
OPEN_PX = 9  # wider than a grid line, narrower than an egg
# Histogram look: bins per channel (hue, saturation, value). Coarse = forgiving of light.
BINS = np.array([16, 8, 6])
OBJ_SHARE = 0.5  # keep a pixel when the object owns at least this share of its colour
OBJ_DISK = 1.0  # learn the object's colours inside this × its half-width
REST_WEIGHT = 10.0  # a colour is "object" only if this many times commoner on it than elsewhere
MAX_AREA_GROWTH = 3.0  # × first-frame area
SIZE_OK = (0.3, 6.0)  # start blob area must be this many × the expected disk area


class BlobTracker:
    """Finds the one object blob each frame. Remembers its look and size from frame 0."""

    def __init__(self, gate_px: float = 25.0) -> None:
        # gate_px: only accept a blob this close (pixels) to where we expect the object.
        self.gate_px = float(gate_px)
        self.mode: str | None = None  # "hist", "colour" or "bright"; set by ``start``
        self.ref_area: float | None = None  # blob size on frame 0 (pixels²)
        self.min_area = 4.0  # ignore specks smaller than this (pixels)
        self._lut: np.ndarray | None = None  # object share per colour bin ("hist" look)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))

    @staticmethod
    def _bins(hsv: np.ndarray) -> np.ndarray:
        """Colour-bin number of every pixel (one integer per pixel)."""
        h = hsv[..., 0].astype(np.int32) * BINS[0] // 180  # OpenCV hue runs 0..179
        s = hsv[..., 1].astype(np.int32) * BINS[1] // 256
        v = hsv[..., 2].astype(np.int32) * BINS[2] // 256
        return (h * BINS[1] + s) * BINS[2] + v

    def _learn(self, hsv: np.ndarray, seg: np.ndarray, r_px: float) -> None:
        """Object share per colour: pixels on the object vs everything far from it.

        "On the object" = within ``r_px`` of the segment ``seg`` (2 end pixels): a
        disk for a fruit, a tall capsule for a bottle (so its neck counts too).
        """
        yy, xx = np.mgrid[: hsv.shape[0], : hsv.shape[1]]
        a, ab = seg[0], seg[1] - seg[0]
        # Distance from every pixel to the nearest point of the segment.
        k = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d2 = (xx - a[0] - k * ab[0]) ** 2 + (yy - a[1] - k * ab[1]) ** 2
        b = self._bins(hsv)
        n = int(np.prod(BINS))
        obj = np.bincount(b[d2 <= max(OBJ_DISK * r_px, 2.0) ** 2], minlength=n).astype(np.float64)
        # Leave a border (soft edges, shadow) out of "rest".
        rest = np.bincount(b[d2 > (2.5 * r_px) ** 2], minlength=n).astype(np.float64)
        # Scale the rest to the object's pixel count: small object, big table.
        rest *= obj.sum() / max(rest.sum(), 1.0) * REST_WEIGHT
        self._lut = obj / np.maximum(obj + rest, 1e-9)

    def _mask(self, rgb: np.ndarray, mode: str) -> np.ndarray:
        """Black/white picture: 1 where a pixel looks like the object, 0 elsewhere."""
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        if mode == "hist":
            m = (self._lut[self._bins(hsv)] >= OBJ_SHARE).astype(np.uint8)
            k3 = np.ones((3, 3), np.uint8)
            # "Open" (shrink, grow) erases thin lines such as soft shadow edges that
            # cling to the blob; "close" (grow, shrink) then fills small holes.
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
            return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)
        if mode == "colour":
            return ((s > SAT_MIN) & (v > VAL_MIN)).astype(np.uint8)
        m = ((v > BRIGHT_VAL_MIN) & (s < BRIGHT_SAT_MAX)).astype(np.uint8)
        # "Open" = shrink then grow. Thin white grid lines vanish; the fat egg stays.
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
        # Split the mask into separate blobs. Label 0 is background, so skip it ([1:]).
        n, _, stats, cents = cv2.connectedComponentsWithStats(self._mask(rgb, mode))
        if n <= 1:
            return None
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        # Association: distance (pixels) from each blob to where we expect the object.
        dist = np.linalg.norm(cents[1:] - np.asarray(uv_guess, dtype=np.float64), axis=1)
        # Throw out blobs that fail any check; keep the rest as candidates.
        ok = area >= self.min_area
        if gate_px is not None:  # too far from the guess
            ok &= dist < gate_px
        if area_range is not None:  # wrong size
            ok &= (area >= area_range[0]) & (area <= area_range[1])
        if inside is not None:  # outside the allowed region
            ok &= np.array([inside(c) for c in cents[1:]], dtype=bool)
        if not ok.any():
            return None
        # Winner = the closest candidate to the guess.
        j = int(np.argmin(np.where(ok, dist, np.inf)))
        return cents[1 + j], float(area[j])

    def start(
        self, rgb: np.ndarray, uv0: np.ndarray, r_px: float = 9.0, seg: np.ndarray | None = None
    ) -> bool:
        """Detector-once: keep the first look whose blob at ``uv0`` is object-sized.

        ``r_px`` = object half-width in pixels. ``seg`` = 2 pixels, the object's
        upright axis shortened by ``r_px`` at each end (tall objects; default: none).
        """
        self.mode = self.ref_area = None
        uv0 = np.asarray(uv0, dtype=np.float64)
        seg = np.array([uv0, uv0] if seg is None else seg, dtype=np.float64)
        length = float(np.linalg.norm(seg[1] - seg[0]))
        disk = np.pi * r_px**2 + 2 * r_px * length  # expected blob area (pixels²)
        self.min_area = max(4.0, 0.1 * disk)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
        self._learn(hsv, seg, r_px)
        size_ok = (SIZE_OK[0] * disk, SIZE_OK[1] * disk)
        gate = max(self.gate_px, length / 2 + r_px)
        for mode in ("hist", "colour", "bright"):
            hit = self._nearest(rgb, mode, uv0, gate_px=gate, area_range=size_ok)
            if hit is not None:
                self.mode, self.ref_area = mode, hit[1]
                return True
        return False

    def step(self, rgb: np.ndarray, uv_guess: np.ndarray) -> np.ndarray | None:
        """Object centre pixel this frame, or None (lost / hidden).

        ``uv_guess`` comes from the Kalman prediction, so the gate follows the object.
        """
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
        # No distance gate here (the old guess is stale), so size and region do the filtering.
        similar = (0.4 * self.ref_area, 2.5 * self.ref_area)
        hit = self._nearest(rgb, self.mode, uv_near, area_range=similar, inside=inside)
        return None if hit is None else hit[0]


if __name__ == "__main__":
    """Standalone smoke: one demo episode → overlays under perception/track_debug/.

    Runs the real glue (StaticCamEstimator = BlobTracker → lift → Kalman), same
    as eval ``--cond-source filter``. No Isaac.

      python perception/track.py
      python perception/track.py path/to/episode.h5
    """
    import json
    import sys
    from pathlib import Path

    import h5py

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from interfaces.config import load_yaml
    from perception.estimator import StaticCamEstimator

    default = Path("/home/gpuadmin/Desktop/foresight/data/eval/p3_throw_n100/eval_model_v0.20_s9293_cond.h5")
    if not default and len(sys.argv) < 2:
        raise SystemExit("No demo .h5 under data/p3_throw; pass a path with static_cam_rgb")
    #/home/gpuadmin/Desktop/foresight/data/eval/p3_throw_n100/eval_model_v0.20_s9293_cond.h5
    h5_path = Path(sys.argv[1] if len(sys.argv) > 1 else default)
    out_dir = Path(__file__).resolve().parent / "track_debug" / h5_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        rgb = np.asarray(f["static_cam_rgb"])  # (T,H,W,3) RGB uint8
        oracle = np.asarray(f["oracle_state"], dtype=np.float64)
        ts = np.asarray(f["timestamp"], dtype=np.float64).reshape(-1)
        holding = (
            np.asarray(f["holding"]).reshape(-1).astype(bool) if "holding" in f else None
        )

    import imageio.v3

    cfg = load_yaml(root / "sim" / "phase0_cfg.yaml")
    est = StaticCamEstimator.from_cfg(cfg)
    start_ok = est.start(rgb[0], oracle[0, :3])
    fps = int(cfg.get("static_camera", {}).get("fps", 25))

    hits, rows, frames = 0, [], []
    for t in range(len(rgb)):
        if holding is not None and bool(holding[t]):
            break  # free-object frames only (object not in the hand)
        est.step(rgb[t], float(ts[t]))
        uv = est.last_uv
        uv_true = est.camera.project(oracle[t, :3])
        frame = np.ascontiguousarray(rgb[t][..., :3]).copy()
        # Green = sim truth pixel. Red ring = tracker blob (missing = lost).
        cv2.circle(
            frame,
            (int(round(uv_true[0])), int(round(uv_true[1]))),
            6,
            (40, 220, 40),
            2,
        )
        hit = uv is not None
        hits += int(hit)
        if hit:
            cv2.circle(
                frame,
                (int(round(uv[0])), int(round(uv[1]))),
                10,
                (220, 40, 40),
                2,
            )
            err_px = float(np.linalg.norm(uv - uv_true))
        else:
            err_px = float("nan")
        label = f"t={t} mode={est.tracker.mode} hit={hit} err_px={err_px:.1f}"
        cv2.putText(
            frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        frames.append(frame)  # RGB — imageio writes H.264 (IDE-playable)
        rows.append(
            {
                "t": t,
                "hit": hit,
                "err_px": err_px,
                "uv": None if uv is None else [float(uv[0]), float(uv[1])],
            }
        )
        # Keep a few stills: start, mid-ish, and the first miss (if any).
        if t in (0, 10, 20) or (not hit and not any(not r["hit"] for r in rows[:-1])):
            cv2.imwrite(
                str(out_dir / f"frame_{t:04d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
    # libx264 (not OpenCV mp4v) so Cursor / browsers can play it.
    imageio.v3.imwrite(
        str(out_dir / "overlay.mp4"),
        frames,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )

    err = np.asarray([r["err_px"] for r in rows], dtype=np.float64)
    summary = {
        "episode": str(h5_path),
        "out_dir": str(out_dir),
        "start_ok": bool(start_ok),
        "mode": est.tracker.mode,
        "ref_area": None if est.tracker.ref_area is None else float(est.tracker.ref_area),
        "n_free_frames": len(rows),
        "track_rate": hits / max(len(rows), 1),
        "median_err_px": float(np.nanmedian(err)) if len(rows) else None,
        "n_redetects": int(est.n_redetects),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


