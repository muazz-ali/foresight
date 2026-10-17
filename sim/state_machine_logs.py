"""Plain Phase 0 logs for people. JSON / H5 keep the old field names.

What it does: turns stage ids and failure codes into short, readable text, and
prints the banners and per-episode lines in the run log.
It runs whenever the expert or the run scripts write a log line.
It must NOT change any saved data (JSON / H5 / metrics keep the short codes) or
decide success. It only words things.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Same integers as sim.state_machine. One spoken name each (no aliases).
# Change the ids there and here together, or logs will name the wrong stage.
STAGE_PLAIN = {
    0: "start",
    1: "intercept",
    2: "go down",  # unused happy-path id; intercept closes directly
    3: "close fingers",
    4: "lift",
    5: "carry to box",
    6: "lower into box",
    7: "open fingers",
    8: "go home",
    9: "done",
}

# One line per stage: what the hand is trying to do. Printed in the "how to read" banner.
STAGE_WHY = {
    0: "Arm at rest. Object still on the table.",
    1: "Chase the object (0.25 s ahead). Close when inside grasp_xyz_ok.",
    2: "Unused. Old go-down stage; chase now closes on the same gate.",
    3: "Fingers shut. Object is held (snapped to the hand).",
    4: "Raise the object.",
    5: "Move high, over the box.",
    6: "Come down inside the box.",
    7: "Let go. Object should stay in the box.",
    8: "Arm rises straight off the object, then returns to rest.",
    9: "Finished.",
}

# JSON still stores the short code. This is only for the human file.
# One entry per expert label in sim/success.py FAILURE_LABELS — keep them in sync.
# Policy runs use their own labels and never reach this map.
WHY_FAILED = {
    "success": "success",
    "never-engaged": "never reached the object",
    "too-late": "reached the object but never lifted",
    "early-close": "closed too soon (barely left the table)",
    "grasp-slip": "lifted, then lost the object",
    "off-table": "object fell off the table",
    "timeout": "ran out of time",
    "safety-abort": "stopped early (hand left table, or object went through box/base)",
    "wrong-object": "grabbed the wrong object",
}


class HumanFormatter(logging.Formatter):
    """Skip the prefix when extra={'bare': True} so banners stay clean."""

    def format(self, record: logging.LogRecord) -> str:
        """Plain message for banner lines; normal timestamped line for the rest."""
        if getattr(record, "bare", False):
            return record.getMessage()
        return super().format(record)


def _bare(log: logging.Logger, message: str) -> None:
    """Log a line with no timestamp prefix (see HumanFormatter)."""
    log.info(message, extra={"bare": True})


def log_blank(log: logging.Logger) -> None:
    """Empty line, to space out the log."""
    _bare(log, "")


def log_section(log: logging.Logger, title: str) -> None:
    """Big ``====  title  ====`` banner, e.g. one per speed bin."""
    log_blank(log)
    log_blank(log)
    _bare(log, f"========  {title}  ========")
    log_blank(log)


def log_event(log: logging.Logger, message: str) -> None:
    """Normal log line (with timestamp)."""
    log.info(message)


def log_note(log: logging.Logger, message: str) -> None:
    """Indented side note under an event line."""
    log.info("    %s", message)


def fmt_xyz(p: Any) -> str:
    """Point as ``(x, y, z) m`` with 3 decimals."""
    v = np.asarray(p, dtype=float).reshape(-1)
    return f"({float(v[0]):.3f}, {float(v[1]):.3f}, {float(v[2]):.3f}) m"


def fmt_size_cm(size_m: Any) -> str:
    """Size in metres → ``a x b x c cm`` (first 3 values)."""
    s = np.asarray(size_m, dtype=float).reshape(-1)
    return " x ".join(f"{100.0 * float(x):.1f}" for x in s[:3]) + " cm"


def stage_label(stage: int, names: dict | None = None) -> str:
    """``name (id)``. Pass ``expert.stage_schema.names`` when overriding the table."""
    n = int(stage)
    table = STAGE_PLAIN if names is None else names
    return f"{table.get(n, n)} ({n})"


def why_failed(code: str) -> str:
    """Failure code → plain sentence. Unknown codes pass through unchanged."""
    return WHY_FAILED.get(str(code), str(code))


def speed_title(speed_m_s: float) -> str:
    """Banner title for a speed bin, in cm/s."""
    cms = float(speed_m_s) * 100.0
    # Under half a cm/s counts as "not moving".
    if cms < 0.5:
        return "SPEED 0 cm/s  (object sits still)"
    # Whole numbers print without decimals (15, not 15.0).
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
    """One line when a stage starts: hand spot, object spot, gap, speed, holding or not."""
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
    """Print the stage list once at the start, so a reader can follow the log."""
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
    """Start-of-episode summary: where things are, and the step cap. Seed is logged for replay."""
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
    """End-of-episode summary. Shows the farthest stage and object height (peak / end).

    Why those two: they let a reader tell WHY a run failed (never reached it,
    never lifted, lost it) instead of trusting the label alone.
    """
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
