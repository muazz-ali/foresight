"""Delay injection.

Plain words: the camera sees the world at time t0. 
The robot should move as if
that picture is Δ seconds old. We store two Δ values:

  applied  — the real wait before motions from that frame start (t0 + applied)
  reported — what the rest of the stack is told (applied + ~10 ms noise)

Train draws applied each cycle; eval uses one fixed applied per test cell.
Isaac-free: pure math + numpy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

Mode = Literal["train", "eval"]


@dataclass(frozen=True)
class DelaySample:
    """One cycle's delay bookkeeping."""

    t0: float  # capture / obs sim time (s)
    applied_delta_s: float  # real wait before motions from this frame run
    reported_delta_s: float  # what forecast / policy are told
    jitter_s: float  # reported - applied (≈ N(0, jitter_std))

    @property
    def apply_at(self) -> float:
        """Sim time when the first motion from this frame may run."""
        return float(self.t0) + float(self.applied_delta_s)

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["apply_at"] = self.apply_at
        return {k: float(v) for k, v in d.items()}


class DelayInjection:
    """Sample applied/reported Δ and keep a cycle log for the Step-1 check."""

    def __init__(
        self,
        *,
        jitter_std_s: float = 0.010,
        train_delta_min_s: float = 0.0,
        train_delta_max_s: float = 0.200,
        eval_delta_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        if jitter_std_s < 0.0:
            raise ValueError(f"jitter_std_s must be >= 0, got {jitter_std_s}")
        if train_delta_max_s < train_delta_min_s:
            raise ValueError("train_delta_max_s must be >= train_delta_min_s")
        if eval_delta_s < 0.0:
            raise ValueError(f"eval_delta_s must be >= 0, got {eval_delta_s}")
        self.jitter_std_s = float(jitter_std_s)
        self.train_delta_min_s = float(train_delta_min_s)
        self.train_delta_max_s = float(train_delta_max_s)
        self.eval_delta_s = float(eval_delta_s)
        self._rng = np.random.default_rng(int(seed))
        self.log: list[DelaySample] = []

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], *, seed: int = 0) -> DelayInjection:
        """Build from ``claim_timing`` (preferred) or ``sim`` fallbacks."""
        ct = cfg.get("claim_timing") if isinstance(cfg, dict) else None
        ct = ct if isinstance(ct, dict) else {}
        sim = cfg.get("sim") if isinstance(cfg, dict) else None
        sim = sim if isinstance(sim, dict) else {}
        # Prefer SI keys after normalize_cfg_units; accept ms aliases if raw.
        jitter = ct.get("delta_jitter_s", None)
        if jitter is None and "delta_jitter_ms" in ct:
            jitter = float(ct["delta_jitter_ms"]) * 1e-3
        if jitter is None:
            jitter = 0.010
        tmin = ct.get("train_delta_min_s", None)
        if tmin is None and "train_delta_min_ms" in ct:
            tmin = float(ct["train_delta_min_ms"]) * 1e-3
        if tmin is None:
            tmin = 0.0
        tmax = ct.get("train_delta_max_s", None)
        if tmax is None and "train_delta_max_ms" in ct:
            tmax = float(ct["train_delta_max_ms"]) * 1e-3
        if tmax is None:
            tmax = 0.200
        ed = ct.get("eval_delta_s", None)
        if ed is None and "eval_delta_ms" in ct:
            ed = float(ct["eval_delta_ms"]) * 1e-3
        if ed is None:
            ed = float(sim.get("inject_latency_s", 0.0) or 0.0)
        return cls(
            jitter_std_s=float(jitter),
            train_delta_min_s=float(tmin),
            train_delta_max_s=float(tmax),
            eval_delta_s=float(ed),
            seed=seed,
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.log.clear()

    def set_eval_delta(self, delta_s: float) -> None:
        """Fix applied Δ for one eval cell (0 / 100 / 200 / 300 ms)."""
        if delta_s < 0.0:
            raise ValueError(f"eval delta must be >= 0, got {delta_s}")
        self.eval_delta_s = float(delta_s)

    def sample(self, t0: float, *, mode: Mode = "train") -> DelaySample:
        """Draw one cycle. Appends to ``log``."""
        t0 = float(t0)
        if mode == "eval":
            applied = float(self.eval_delta_s)
        elif mode == "train":
            applied = float(
                self._rng.uniform(self.train_delta_min_s, self.train_delta_max_s)
            )
        else:
            raise ValueError(f"mode must be train|eval, got {mode!r}")
        jitter = float(self._rng.normal(0.0, self.jitter_std_s))
        reported = applied + jitter
        # Reported delay used by forecast must stay non-negative.
        if reported < 0.0:
            reported = 0.0
            jitter = reported - applied
        sample = DelaySample(
            t0=t0,
            applied_delta_s=applied,
            reported_delta_s=reported,
            jitter_s=jitter,
        )
        self.log.append(sample)
        return sample

    def jitter_std(self) -> float:
        """Sample std of (reported − applied). Empty log → nan."""
        if not self.log:
            return float("nan")
        j = np.asarray([s.jitter_s for s in self.log], dtype=np.float64)
        return float(np.std(j, ddof=1)) if len(j) > 1 else 0.0

    def log_arrays(self) -> dict[str, np.ndarray]:
        """Columnar log for the Step-1 check (t0, reported, applied, jitter)."""
        if not self.log:
            empty = np.zeros(0, dtype=np.float64)
            return {
                "t0": empty,
                "reported_delta_s": empty,
                "applied_delta_s": empty,
                "jitter_s": empty,
                "apply_at": empty,
            }
        return {
            "t0": np.asarray([s.t0 for s in self.log], dtype=np.float64),
            "reported_delta_s": np.asarray(
                [s.reported_delta_s for s in self.log], dtype=np.float64
            ),
            "applied_delta_s": np.asarray(
                [s.applied_delta_s for s in self.log], dtype=np.float64
            ),
            "jitter_s": np.asarray([s.jitter_s for s in self.log], dtype=np.float64),
            "apply_at": np.asarray([s.apply_at for s in self.log], dtype=np.float64),
        }
