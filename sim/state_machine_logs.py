"""Plain Phase 0 logs for people. JSON / H5 keep the old field names."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Same integers as sim.state_machine. One spoken name each (no aliases).
STAGE_PLAIN = {
    0: "start",
    1: "intercept",
    2: "go down",
    3: "close fingers",
    4: "lift",
    5: "carry to box",
    6: "lower into box",
    7: "open fingers",
    8: "go home",
    9: "done",
}

STAGE_WHY = {
    0: "Arm at rest. Object still on the table.",
    1: "Hand flies to where the object will be when the hand gets there (0.25 s cap).",
    2: "Hand rides on the object and comes down onto it.",
    3: "Fingers shut. Object is held (snapped to the hand).",
    4: "Raise the object.",
    5: "Move high, over the box.",
    6: "Come down inside the box.",
    7: "Let go. Object should stay in the box.",
    8: "Arm returns to rest. Object stays in the box.",
    9: "Finished.",
}

# JSON still stores the short code. This is only for the human file.
WHY_FAILED = {
    "success": "success",
    "never-engaged": "never reached the object",
    "too-late": "reached the object but never lifted",
    "early-close": "closed too soon (barely left the table)",
    "grasp-slip": "lifted, then lost the object",
    "off-table": "object fell off the table",
    "timeout": "ran out of time",
    "safety-abort": "run crashed",
    "wrong-object": "grabbed the wrong object",
}


class HumanFormatter(logging.Formatter):
    """Skip the prefix when extra={'bare': True} so banners stay clean."""

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "bare", False):
            return record.getMessage()
        return super().format(record)


def _bare(log: logging.Logger, message: str) -> None:
    log.info(message, extra={"bare": True})


def log_blank(log: logging.Logger) -> None:
    _bare(log, "")


def log_section(log: logging.Logger, title: str) -> None:
    log_blank(log)
    log_blank(log)
    _bare(log, f"========  {title}  ========")
    log_blank(log)


def log_event(log: logging.Logger, message: str) -> None:
    log.info(message)


def log_note(log: logging.Logger, message: str) -> None:
    log.info("    %s", message)


def fmt_xyz(p: Any) -> str:
    v = np.asarray(p, dtype=float).reshape(-1)
    return f"({float(v[0]):.3f}, {float(v[1]):.3f}, {float(v[2]):.3f}) m"


def fmt_size_cm(size_m: Any) -> str:
    s = np.asarray(size_m, dtype=float).reshape(-1)
    return " x ".join(f"{100.0 * float(x):.1f}" for x in s[:3]) + " cm"


def stage_label(stage: int, names: dict | None = None) -> str:
    """``name (id)``. Pass ``expert.stage_schema.names`` when overriding the table."""
    n = int(stage)
    table = STAGE_PLAIN if names is None else names
    return f"{table.get(n, n)} ({n})"


def why_failed(code: str) -> str:
    return WHY_FAILED.get(str(code), str(code))


def speed_title(speed_m_s: float) -> str:
    cms = float(speed_m_s) * 100.0
    if cms < 0.5:
        return "SPEED 0 cm/s  (object sits still)"
    if abs(cms - round(cms)) < 0.05:
        return f"SPEED {int(round(cms))} cm/s"
    return f"SPEED {cms:.1f} cm/s"


def log_stage(
    log: logging.Logger,
    stage: int,
    *,
    hand: Any,
    obj: Any,
    speed_m_s: float,
    holding: bool,
) -> None:
    hand_v = np.asarray(hand, dtype=float).reshape(3)
    obj_v = np.asarray(obj, dtype=float).reshape(3)
    gap_cm = 100.0 * float(np.linalg.norm(hand_v - obj_v))
    held = "holding" if holding else "not holding"
    log.info(
        "  %-20s  hand %s   object %s   gap %.1f cm   speed %.1f cm/s   %s",
        stage_label(stage),
        fmt_xyz(hand_v),
        fmt_xyz(obj_v),
        gap_cm,
        100.0 * float(speed_m_s),
        held,
    )


def log_how_to_read(
    log: logging.Logger,
    *,
    lookahead_s: float,
    names: dict | None = None,
    why: dict | None = None,
) -> None:
    name_table = STAGE_PLAIN if names is None else names
    why_table = STAGE_WHY if why is None else why
    log_blank(log)
    _bare(log, "How to read stages (in order):")
    for n in sorted(name_table):
        _bare(log, f"  {stage_label(n, name_table):<20}  {why_table.get(n, '')}")
    log_blank(log)
    ms = 1000.0 * float(lookahead_s)
    _bare(
        log,
        f"Intercept: the hand aims where the object will be when the hand arrives (cap {ms:.0f} ms), with lag compensation.",
    )
    _bare(log, "Holding: we snap the object to the hand (not a real physics grab).")
    _bare(
        log,
        "No wait-still: the hand rides the moving object; we only freeze after place/open/home.",
    )
    log_blank(log)


def log_episode_begin(
    log: logging.Logger,
    *,
    seed: int,
    max_steps: int,
    object_pos: Any,
    object_speed_m_s: float,
    hand_pos: Any,
    box_xy: Any,
    drop_z: float,
) -> None:
    box = np.asarray(box_xy, dtype=float).reshape(-1)
    cms = 100.0 * float(object_speed_m_s)
    log.info(
        "Object at %s, speed %.1f cm/s.",
        fmt_xyz(object_pos),
        cms,
    )
    log.info(
        "Hand at %s. Box at (%.3f, %.3f) m. Drop height %.1f cm.",
        fmt_xyz(hand_pos),
        float(box[0]),
        float(box[1]),
        100.0 * float(drop_z),
    )
    log.info("Up to %d steps (seed %s).", int(max_steps), seed)
    log_blank(log)


def log_result(
    log: logging.Logger,
    *,
    failure: str,
    max_stage: int,
    z_max: float,
    z_end: float,
    n_frames: int,
    video: str | Path | None = None,
    names: dict | None = None,
) -> None:
    log_blank(log)
    log.info("Result: %s", why_failed(failure))
    log.info("  Farthest stage: %s", stage_label(max_stage, names))
    log.info(
        "  Object height: peak %.1f cm, ended %.1f cm",
        100.0 * float(z_max),
        100.0 * float(z_end),
    )
    log.info("  Frames: %d", int(n_frames))
    if video is not None:
        log.info("  Video: %s", video)
    log_blank(log)
