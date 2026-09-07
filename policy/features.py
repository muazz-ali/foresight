"""Frozen LeRobot feature schema for Foresight Phase 1."""

from __future__ import annotations

from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT

PROPRIO_DIM = 8  # ee_pos(3) + ee_quat_wxyz(4) + gripper(1)
ORACLE_DIM = 14
ACTION_DIM = 8
STATE_DIM_B = PROPRIO_DIM + CONDITIONING_DIM  # 12 (8 proprio + 4 live cond)

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
ACTION_NAMES = list(PROPRIO_NAMES)
CONDITIONING_NAMES = list(CONDITIONING_LAYOUT)
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

IMG_H = 360
IMG_W = 480
DEFAULT_FPS = 25


def make_features(*, include_aux: bool = True) -> dict:
    """LeRobot feature dict. Aux = conditioning + oracle_state (for train-time aug)."""
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
    obj = object_category or "object"
    ctr = container_category or "container"
    return f"Pick the {obj} and place it in the {ctr}."
