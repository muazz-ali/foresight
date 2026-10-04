"""Pick the 4 numbers Model B sees: simulator truth, or the camera filter.

Both packs are built by ``conditioning_vector``. The only change is which
object message goes in.
"""

from __future__ import annotations

import numpy as np

from interfaces.state import CONDITIONING_DIM, ObjectState, conditioning_vector

# Frames the filter sees before the frame Model B plans from.
# The filter starts at speed 0, so the first plan would miss by speed × Δ.
FILTER_WARMUP_BEFORE = 2


def live_pack(state: ObjectState, delta: float) -> np.ndarray:
    """4 numbers, same helper the oracle and the filter share."""
    pack = conditioning_vector(state, delta=float(delta)).astype(np.float32)
    if pack.shape != (CONDITIONING_DIM,):
        raise ValueError(f"live pack must be ({CONDITIONING_DIM},), got {pack.shape}")
    return pack


def choose_pack(
    *,
    source: str,
    oracle_state: ObjectState,
    filter_state: ObjectState | None,
    delta: float,
    zero: bool,
) -> dict[str, np.ndarray | float | None]:
    """Pack B will see, plus the sim-truth pack beside it.

    ``source`` is ``oracle`` or ``filter``. ``zero`` is the sanity check that
    turns the pack off. It does not mix with ``filter``.
    """
    if source not in ("oracle", "filter"):
        raise ValueError(f"cond source must be oracle or filter, got {source!r}")
    if zero and source == "filter":
        raise ValueError("zero conditioning and filter source cannot mix")

    truth = live_pack(oracle_state, delta)
    filt: np.ndarray | None = None
    valid: float | None = None
    if zero:
        cond = np.zeros(CONDITIONING_DIM, dtype=np.float32)
    elif source == "oracle":
        cond = truth.copy()
    else:
        if filter_state is None:
            filt = np.zeros(CONDITIONING_DIM, dtype=np.float32)
            valid = 0.0
        else:
            filt = live_pack(filter_state, delta)
            valid = 1.0 if filter_state.valid else 0.0
        cond = filt.copy()
    return {"cond": cond, "truth": truth, "filter": filt, "valid": valid}
