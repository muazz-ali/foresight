"""Foresight public interfaces.

This folder is the "shared plug" between sim, perception, policy, and control.
Everyone talks to each other through these few names, so nobody has to know
how the other side works inside.

  ObjectState         — the one message: where the object is, how fast, how sure
  oracle_from_gt      — perfect ObjectState from the simulator (the "oracle")
  conditioning_vector — 4 live numbers for the policy (XY p̂ + XY v̂)
"""

from interfaces.state import (
    CONDITIONING_DIM,
    CONDITIONING_LAYOUT,
    FLAT_STATE_KEYS,
    ObjectState,
    conditioning_vector,
    oracle_from_gt,
    pack_conditioning,
)

# Keep this list in sync with the import above. It is what `from interfaces import *` exposes.
__all__ = [
    "CONDITIONING_DIM",
    "CONDITIONING_LAYOUT",
    "FLAT_STATE_KEYS",
    "ObjectState",
    "conditioning_vector",
    "oracle_from_gt",
    "pack_conditioning",
]
