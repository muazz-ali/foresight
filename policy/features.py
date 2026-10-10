"""Frozen LeRobot feature schema for Foresight Phase 1.

This file is the one place that says what the numbers mean.
Convert, train, and infer must all read the same layout from here.
If they disagree, the model reads the wrong slots and fails with no error.

What the policy sees as "state" (12 numbers):
  * 8 robot numbers: hand XYZ (3), hand rotation (4), gripper (1).
  * 4 "future object" numbers: where the object will be (XY position),
    and how fast it moves (XY velocity).
The 4 numbers come from `interfaces.state` (CONDITIONING_LAYOUT).

Delta (the look-ahead time) is NOT in the pack while it is a fixed value.
Only when the real delay can change, add it as a 5th number. Then train
and eval must both pack it.
"""

from __future__ import annotations

from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT

# Sizes. Everything else in policy/ asserts against these.
PROPRIO_DIM = 8  # ee_pos(3) + ee_quat_wxyz(4) + gripper(1)
ORACLE_DIM = 14  # perfect sim object state saved per frame (see ORACLE_NAMES)
ACTION_DIM = 8   # same layout as PROPRIO: target hand pose + gripper
STATE_DIM = PROPRIO_DIM + CONDITIONING_DIM  # 12 (8 proprio + 4 live cond)

# Names are only labels for humans and LeRobot meta. Order must match the data.
PROPRIO_NAMES = [
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
    "ee_rot_qw",
    "ee_rot_qx",
    "ee_rot_qy",
    "ee_rot_qz",
    "gripper",
]
ACTION_NAMES = list(PROPRIO_NAMES)  # action = the pose we want next, same slots
CONDITIONING_NAMES = list(CONDITIONING_LAYOUT)  # the 4 future-object numbers
# Oracle = perfect object info from the sim. Saved so training can rebuild the
# 4 numbers with any look-ahead and fresh noise. The policy never sees it directly.
ORACLE_NAMES = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "cov_px",
    "cov_py",
    "cov_pz",
    "cov_vx",
    "cov_vy",
    "cov_vz",
    "timestamp",
    "valid",
]

# Two cameras: a fixed front one and one on the wrist. Same size as collected.
IMG_H = 360
IMG_W = 480
DEFAULT_FPS = 25


def make_features(*, include_aux: bool = True) -> dict:
    """LeRobot feature dict. Aux = conditioning + oracle_state (for train-time aug).

    Plain words: state here is only the 8 robot numbers. The 4 object numbers
    are stored on the side ("aux") and glued on at train time by
    policy/transforms.py, so we can add noise to them.
    """
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (PROPRIO_DIM,),
            "names": PROPRIO_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ACTION_NAMES,
        },
        "observation.images.static_cam": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_cam": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
    }
    if include_aux:
        # Clean 4 numbers as logged by the sim (what the policy would see with no noise).
        features["observation.conditioning"] = {
            "dtype": "float32",
            "shape": (CONDITIONING_DIM,),
            "names": CONDITIONING_NAMES,
        }
        features["observation.oracle_state"] = {
            "dtype": "float32",
            "shape": (ORACLE_DIM,),
            "names": ORACLE_NAMES,
        }
    return features


def default_language(object_category: str | None, container_category: str | None) -> str:
    # Fallback task text when the episode JSON has no "language" field.
    obj = object_category or "object"
    ctr = container_category or "container"
    return f"Pick the {obj} and place it in the {ctr}."
