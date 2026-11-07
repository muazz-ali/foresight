"""Frozen LeRobot feature schema for Foresight 

This file is the one place that says what the numbers mean.
Convert, train, and infer must all read the same layout from here.
If they disagree, the model reads the wrong slots and fails with no error.

What the policy sees as "state" (13 numbers today):
  * 8 robot numbers: hand XYZ (3), hand rotation (4), gripper (1).
  * 5 "future object" numbers: p̂x, p̂y, p̂z, vx, vy
    (p̂z = grasp height; height changes p̂, not relative d_z).
Numbers come from `interfaces.state` (CONDITIONING_LAYOUT).

Delta (look-ahead) is baked into p̂ XY and is NOT packed separately.
"""

from __future__ import annotations

from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT

# Sizes. Everything else in policy/ asserts against these.
PROPRIO_DIM = 8  # ee_pos(3) + ee_quat_wxyz(4) + gripper(1)
ORACLE_DIM = 14  # perfect sim object state saved per frame (see ORACLE_NAMES)
ACTION_DIM = 8   # same layout as PROPRIO: target hand pose + gripper
STATE_DIM = PROPRIO_DIM + CONDITIONING_DIM  # 13 (8 proprio + 5 live cond)

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


_INSTR_CACHE: dict | None = None


def _load_instructions() -> dict:
    """Load ``policy/configs/instructions.yaml`` once (templates per fruit)."""
    global _INSTR_CACHE
    if _INSTR_CACHE is not None:
        return _INSTR_CACHE
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parent / "configs" / "instructions.yaml"
    if not path.is_file():
        _INSTR_CACHE = {}
        return _INSTR_CACHE
    with path.open() as f:
        _INSTR_CACHE = yaml.safe_load(f) or {}
    return _INSTR_CACHE


# Longer names first so "tangerine" is not read as "orange".
# Mode-2 cubes first so "red cube" / ``cube_red`` map before fruit stems.
_OBJECT_KEYS = (
    "cube_blue",
    "cube_red",
    "cube_green",
    "cube_yellow",
    "tangerine",
    "avocado",
    "orange",
    "apple",
    "peach",
    "lemon",
    "onion",
    "kiwi",
    "lime",
    "egg",
    "can",
)
_CONTAINER_KEYS = ("basket", "bowl", "box")
_CUBE_COLORS = ("blue", "red", "green", "yellow")


def _stem(name: str | None, fallback: str) -> str:
    """Letters only, so ``apple12`` becomes ``apple``."""
    raw = (name or fallback).strip().lower()
    return "".join(ch for ch in raw if ch.isalpha()) or fallback


def object_key(name: str | None) -> str:
    """Map a category or a full sentence onto a template key."""
    low = (name or "").strip().lower()
    # Mode 2: ``cube_red`` folder or "Grasp the red cube …"
    for color in _CUBE_COLORS:
        token = f"cube_{color}"
        if token in low.replace(" ", "_") or f"{color} cube" in low:
            return token
        if _stem(low, "") == f"cube{color}":
            return token
    key = _stem(name, "object")
    for name_i in _OBJECT_KEYS:
        if name_i.startswith("cube_"):
            continue
        if key == name_i or key.startswith(name_i) or name_i in low:
            return name_i
    return key


def container_word(name: str | None) -> str:
    """Bowl / box / basket word used inside a task sentence."""
    data = _load_instructions()
    containers = (data.get("containers") or {}) if isinstance(data, dict) else {}
    key = _stem(name, "bowl")
    low = (name or "").lower()
    for name_i in _CONTAINER_KEYS:
        if key == name_i or name_i in low:
            key = name_i
            break
    return str(containers.get(key, containers.get("default", "bowl")))


def _format_task(obj_key: str, ctr_word: str, *, seed: int | None) -> str:
    """One random wording for this object + container. Noun stays in the sentence."""
    data = _load_instructions()
    templates = (data.get("templates") or {}) if isinstance(data, dict) else {}
    lines = templates.get(obj_key) or templates.get("default") or []
    if not lines:
        return f"Pick the {obj_key} and place it in the {ctr_word}."
    rng = __import__("random").Random(
        None if seed is None else int(seed)
    )
    line = str(rng.choice(list(lines)))
    return line.format(object=obj_key, container=ctr_word)


def default_language(
    object_category: str | None,
    container_category: str | None,
    *,
    seed: int | None = None,
) -> str:
    """Pick a task string that names the object (apple ≠ tangerine) and container.

    Templates live in ``policy/configs/instructions.yaml``. Same object gets
    many wordings so the model cannot memorize one sentence. No seed → a
    stable pick (eval). Train redraws wording via ``paraphrase_task``.
    """
    obj = object_key(object_category)
    if seed is None:
        seed = hash(obj) & 0xFFFF
    return _format_task(obj, container_word(container_category), seed=seed)


def paraphrase_task(task: str, *, seed: int | None = None) -> str:
    """Replace a saved task sentence with another wording of the same ask.

    Train calls this each batch. The object noun and the container stay, so
    the model has to read the words and match them to the picture — not one
    frozen phrase from the demo file.
    """
    text = str(task).strip()
    if not text:
        return text
    return _format_task(object_key(text), container_word(text), seed=seed)
