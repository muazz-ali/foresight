"""Why a physics-grasp (throw) episode failed, from the logged tracks only.

No Isaac. Inputs are the per-frame arrays eval already saves:
``ee_pos`` (T,3), ``object_pos`` (T,3), ``gripper`` (T,) commanded open 0/1,
``holding`` (T,) 0/1, and the step time ``dt``.

Labels (checked in this order, first match wins):
  success      the episode passed the success rule.
  never-closed object stayed on the table, the hand never closed and never
               got near it.
  off-table    object left the table before the hand ever closed or got near.
  aim          hand closed, but was never within NEAR_M (3-D) of the object.
  early        hand got near only while already closed (shut too soon and
               did not re-open in time).
  late         hand was near while open, then closed only after the closest
               approach had passed (or never closed again).
  short-close  hand closed near the object at about the right time, but the
               episode still failed (slipped, dropped, never lifted).
Only frames before the object left the table count: what happens after a
fall cannot be a timing cause. ``off_table`` is also logged on its own.
"""

from __future__ import annotations

import numpy as np

# "Near" = hand point (TCP) within this many metres (3-D) of the object centre.
# Same 5 cm as the eval attach latch.
NEAR_M = 0.05
# A close counts as "late" only if the hand was already this much farther away
# than at its closest point (one 25 Hz frame of jitter must not flip the label).
LATE_SLACK_M = 0.01
# Object this far below the table top = fell off (same margin as sim.success).
OFF_TABLE_DZ = 0.04

TIMING_FAILURES = ("late", "early", "short-close")


def _time_of(i: int | None, dt: float) -> float | None:
    return None if i is None else float(i) * float(dt)


def smoothness(ee_pos, dt: float, end: int | None = None) -> dict:
    """Path length (m) and RMS jerk (m/s^3) of the hand path in frames [0, end)."""
    p = np.asarray(ee_pos, dtype=np.float64).reshape(-1, 3)
    if end is not None:
        p = p[: max(int(end), 0)]
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
    jerk = None
    if len(p) > 3:
        j = np.diff(p, n=3, axis=0) / float(dt) ** 3
        jerk = float(np.sqrt(np.mean(np.sum(j * j, axis=1))))
    return {"path_len_m": path, "jerk_rms": jerk, "n_frames": int(len(p))}


def timing_label(
    *,
    ee_pos,
    object_pos,
    gripper,
    holding=None,
    success: bool,
    dt: float,
    table_z: float = 0.0,
    off_table_flag: bool = False,
) -> dict:
    """Return {"timing_label", distances, close vs closest-approach times, ...}."""
    ee = np.asarray(ee_pos, dtype=np.float64).reshape(-1, 3)
    ob = np.asarray(object_pos, dtype=np.float64).reshape(-1, 3)
    closed = np.asarray(gripper, dtype=np.float64).reshape(-1) < 0.5
    n = min(len(ee), len(ob), len(closed))
    ee, ob, closed = ee[:n], ob[:n], closed[:n]
    held = np.zeros(n, dtype=bool) if holding is None else np.asarray(holding).reshape(-1)[:n] > 0.5

    # Cut at the first frame the object is below the table: later frames are a fall.
    below = ob[:, 2] < float(table_z) - OFF_TABLE_DZ
    i_off = int(np.argmax(below)) if below.any() else None
    off_table = bool(off_table_flag or i_off is not None)
    m = n if i_off is None else i_off

    d3 = np.linalg.norm(ee[:m] - ob[:m], axis=1)
    dxy = np.linalg.norm(ee[:m, :2] - ob[:m, :2], axis=1)
    c = closed[:m]
    near = d3 < NEAR_M
    near_open = near & ~c

    i_first_close = int(np.argmax(c)) if c.any() else None
    # Closest approach while the hand was still open (the chance it had).
    i_closest = None
    if (~c).any():
        open_idx = np.flatnonzero(~c)
        i_closest = int(open_idx[np.argmin(d3[open_idx])])
    # The close that matters: first close after the hand first got near while open.
    i_close = i_first_close
    if near_open.any():
        i_near = int(np.argmax(near_open))
        after = np.flatnonzero(c[i_near:])
        i_close = (i_near + int(after[0])) if after.size else None
        # Closest point in the near-open window, up to and including the close.
        stop = m if i_close is None else i_close + 1
        i_closest = i_near + int(np.argmin(d3[i_near:stop]))

    if success:
        label = "success"
    elif not c.any() and not near_open.any():
        label = "off-table" if off_table else "never-closed"
    elif not near_open.any() and not near.any():
        label = "aim"
    elif not near_open.any():
        label = "early"
    elif i_close is None:
        label = "late"  # was near and open, never closed in time
    elif d3[i_close] >= NEAR_M or d3[i_close] - d3[i_closest] > LATE_SLACK_M:
        label = "late"
    else:
        label = "short-close"

    t_close = _time_of(i_close, dt)
    t_closest = _time_of(i_closest, dt)
    z = ob[:, 2] if n else np.zeros(1)
    return {
        "timing_label": label,
        "z_max": float(z.max()),
        "off_table": off_table,
        "off_table_t": _time_of(i_off, dt),
        "min_dist_3d_m": float(d3.min()) if m else None,
        "min_dist_xy_m": float(dxy.min()) if m else None,
        "close_t": t_close,
        "closest_t": t_closest,
        "close_minus_closest_s": (
            None if t_close is None or t_closest is None else t_close - t_closest
        ),
        "dist_at_close_m": None if i_close is None else float(d3[i_close]),
        "ever_held": bool(held.any()),
        "near_m": NEAR_M,
        # Hand path before the grasp (up to the close that mattered).
        "pre_grasp": smoothness(ee, dt, end=(i_close if i_close is not None else m)),
    }
