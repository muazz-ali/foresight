"""Pick the 4 numbers the policy sees: simulator truth, or the camera filter.

Both packs are built by ``conditioning_vector``. The only change is which
object message goes in.

Why a flag: the policy must work with camera numbers, not only sim truth.
  - ``oracle``: feed the policy sim truth. Clean test of "does the policy use
    the numbers?" with no camera error mixed in.
  - ``filter``: feed it the tracker+Kalman output from the simulated camera.
    This tests the input path intended for later real use.
Same helper and shape keep the policy input layout unchanged.
We also return the truth pack every time, so callers can log filter error.
"""

from __future__ import annotations

import numpy as np

from interfaces.state import CONDITIONING_DIM, ObjectState, conditioning_vector

# Frames the filter sees before the frame the policy plans from.
# The filter starts at speed 0, so the first plan would miss by speed × Δ.
FILTER_WARMUP_BEFORE = 2


def live_pack(state: ObjectState, delta: float) -> np.ndarray:
    """4 numbers, same helper the oracle and the filter share.

    Oracle and filter use this same helper, which reduces layout drift.
    The shape check catches a wrong-size pack early, before it hits the policy.
    """
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
    """Pack the policy will see, plus the sim-truth pack beside it.

    ``source`` is ``oracle`` or ``filter``. ``zero`` is the sanity check that
    turns the pack off. It does not mix with ``filter``.
    """
    if source not in ("oracle", "filter"):
        raise ValueError(f"cond source must be oracle or filter, got {source!r}")
    if zero and source == "filter":
        raise ValueError("zero conditioning and filter source cannot mix")

    # Truth pack is always built, even when the policy does not see it.
    truth = live_pack(oracle_state, delta)
    filt: np.ndarray | None = None
    valid: float | None = None
    if zero:
        # Zero-out check: hide the numbers (G1 needs success to drop).
        cond = np.zeros(CONDITIONING_DIM, dtype=np.float32)
    elif source == "oracle":
        cond = truth.copy()
    else:
        # Filter has no estimate yet: send zeros and mark valid=0.
        # We log this instead of quietly copying the truth.
        if filter_state is None:
            filt = np.zeros(CONDITIONING_DIM, dtype=np.float32)
            valid = 0.0
        else:
            filt = live_pack(filter_state, delta)
            valid = 1.0 if filter_state.valid else 0.0
        cond = filt.copy()
    # cond = what the policy sees. truth/filter/valid = for logging only.
    return {"cond": cond, "truth": truth, "filter": filt, "valid": valid}
