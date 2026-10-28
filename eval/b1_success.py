"""B1 success + failure labels (expert only).

Success = lifted + held + placed (Oct 29):
  - force-place off
  - lifted: object ≥ 5 cm above the table after LIFT starts
  - held: from the first LIFT frame, 1.0 s with hand-frame slip ≤ 10 mm
  - placed: object ends on/in the container (sim/success.py rule: XY within
    12 cm of the container, resting between table + 2 cm and container + 5 cm)

Close-time checks (closed_above, closed_empty_xy, knocked) are diagnostics:
they name the cause of a failure, they never fail a success.
``first_attempt`` = success with exactly one CLOSE entry (no retry).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.state_machine import STAGE_CLOSE, STAGE_LIFT

# B1 latch numbers (do not loosen without a new gate version).
LIFT_MIN_M = 0.05
HOLD_MIN_S = 1.0
SLIP_MAX_M = 0.010
CLOSE_XY_EMPTY_M = 0.005
KNOCK_M = 0.010
CLOSED_ABOVE_SZ_M = 0.005  # s_z default; caller may override
# Placed (same numbers as sim.success.episode_success).
PLACE_XY_TOL_M = 0.12
PLACE_Z_MIN_ABOVE_TABLE_M = 0.02
PLACE_Z_MAX_ABOVE_CONTAINER_M = 0.05


FAILURE_MODES = (
    "unreachable",
    "never_closed",
    "closed_above",
    "closed_empty_xy",
    "knocked",
    "slipped",
    "not_placed",
    "other",
)


def _arr(ep: dict, key: str) -> np.ndarray:
    return np.asarray(ep.get(key, []), dtype=np.float64)


def close_diagnostics(
    episode: dict[str, Any], *, dt: float = 1.0 / 30.0, s_z_m: float = CLOSED_ABOVE_SZ_M
) -> dict[str, Any]:
    """Hand vs object at the first CLOSE frame + knock check (diagnostic only)."""
    stages = np.asarray(episode.get("stage", []), dtype=np.int64)
    obj = _arr(episode, "object_pos")
    ee = _arr(episode, "ee_pos")
    n = min(len(stages), len(obj), len(ee))
    closed = np.where(stages[:n] == STAGE_CLOSE)[0]
    entries = int(np.sum((stages[1:n] == STAGE_CLOSE) & (stages[: n - 1] != STAGE_CLOSE)))
    entries += int(n > 0 and stages[0] == STAGE_CLOSE)
    out: dict[str, Any] = {"n_close_entries": entries, "closed_above": False,
                           "closed_empty_xy": False, "knocked": False}
    if closed.size == 0:
        return out
    c0 = int(closed[0])
    if c0 > 2 and "object_vel" in episode:
        # Object left its coast path by > 10 mm while the hand was near
        # (vs the constant-velocity prediction, not absolute travel).
        vel = _arr(episode, "object_vel")
        for i in range(1, min(c0, len(vel))):
            if float(np.linalg.norm(ee[i, :2] - obj[i, :2])) >= 0.03:
                continue
            pred = obj[i - 1, :2] + vel[i - 1, :2] * float(dt)
            if float(np.linalg.norm(obj[i, :2] - pred)) > float(KNOCK_M) + 1e-9:
                out["knocked"] = True
                break
    # ee_pos is the pinch frame; z_err > 0 = pinch above the object centre.
    z_err = float(ee[c0, 2] - obj[c0, 2])
    out["close_z_err_m"] = z_err
    out["close_xy_err_m"] = float(np.linalg.norm(ee[c0, :2] - obj[c0, :2]))
    out["closed_above"] = bool(z_err > float(s_z_m) + 1e-9)
    out["closed_empty_xy"] = bool(out["close_xy_err_m"] > float(CLOSE_XY_EMPTY_M) + 1e-9)
    return out


def b1_success(
    episode: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
    *,
    z_table: float = 0.0,
    force_place: bool | None = None,
    dt: float = 1.0 / 30.0,
    s_z_m: float = CLOSED_ABOVE_SZ_M,
) -> tuple[bool, str | None]:
    """Return (ok, failure_mode). failure_mode is None on success.

    May read live logged GT (object/ee) — this is the success judge, so leak
    audit allows it.
    """
    meta = meta or {}
    if force_place is None:
        force_place = bool(meta.get("force_place", False))
    if force_place or episode is None:
        return False, "other"

    stages = np.asarray(episode.get("stage", []), dtype=np.int64)
    obj = _arr(episode, "object_pos")
    ee = _arr(episode, "ee_pos")
    if stages.size == 0 or obj.size == 0 or ee.size == 0:
        return False, "other"
    n = min(len(stages), len(obj), len(ee))
    stages, obj, ee = stages[:n], obj[:n], ee[:n]

    if not np.any(stages == STAGE_CLOSE):
        return False, ("unreachable" if bool(meta.get("unreachable", False)) else "never_closed")
    diag = close_diagnostics(episode, dt=dt, s_z_m=s_z_m)

    def _close_cause() -> str:
        # Name why the grasp did not hold: close-time diagnostics first.
        for k in ("closed_above", "closed_empty_xy", "knocked"):
            if diag.get(k):
                return k
        return "slipped"

    lift = np.where(stages == STAGE_LIFT)[0]
    if lift.size == 0:
        return False, _close_cause()
    i0 = int(lift[0])
    z = obj[:, 2]
    # Lifted: ≥ 5 cm above the table at some point after LIFT starts.
    if float(np.max(z[i0:])) < float(z_table) + float(LIFT_MIN_M) - 1e-9:
        return False, _close_cause()
    # Held: 1 s from the first LIFT frame, hand-frame slip ≤ 10 mm.
    n_hold = max(1, int(round(float(HOLD_MIN_S) / max(float(dt), 1e-6))))
    if i0 + n_hold > n:
        return False, _close_cause()
    rel0 = obj[i0] - ee[i0]
    for i in range(i0, i0 + n_hold):
        if float(np.linalg.norm((obj[i] - ee[i]) - rel0)) > float(SLIP_MAX_M) + 1e-9:
            return False, _close_cause()

    # Placed: ends on/in the container.
    cont = _arr(episode, "container_pos")
    if cont.size == 0 or not np.all(np.isfinite(cont[-1])):
        return False, "not_placed"
    c = cont[min(len(cont), n) - 1]
    near = float(np.linalg.norm(obj[-1, :2] - c[:2])) <= PLACE_XY_TOL_M
    low = (
        float(z_table) + PLACE_Z_MIN_ABOVE_TABLE_M
        <= float(z[-1])
        <= float(c[2]) + PLACE_Z_MAX_ABOVE_CONTAINER_M
    )
    if not (near and low):
        return False, "not_placed"
    return True, None


def classify_b1(
    episode: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score one episode for the B1 JSONL row extras."""
    ok, mode = b1_success(episode, meta, **kwargs)
    diag = (
        close_diagnostics(episode, dt=kwargs.get("dt", 1.0 / 30.0))
        if episode is not None
        else {"n_close_entries": 0}
    )
    return {
        "success": bool(ok),
        "failure_mode": None if ok else (mode or "other"),
        "first_attempt": bool(ok and int(diag.get("n_close_entries", 0)) == 1),
        "n_close_entries": int(diag.get("n_close_entries", 0)),
        "diag_closed_above": bool(diag.get("closed_above", False)),
        "diag_closed_empty_xy": bool(diag.get("closed_empty_xy", False)),
        "diag_knocked": bool(diag.get("knocked", False)),
    }
