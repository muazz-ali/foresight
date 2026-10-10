"""Phase 1 policy helpers: convert demos, train SmolVLA.

Pipeline: H5 demos -> convert_h5.py -> LeRobot dataset -> train_smolvla.py -> checkpoint.
All three steps share ONE number layout (see features.py). If you change the
layout in one place, change it everywhere and retrain.
"""

# Re-export the sizes so other code can write `from policy import STATE_DIM`.
from policy.features import ACTION_DIM, CONDITIONING_DIM, PROPRIO_DIM, STATE_DIM

__all__ = [
    "ACTION_DIM",
    "CONDITIONING_DIM",
    "PROPRIO_DIM",
    "STATE_DIM",
]
