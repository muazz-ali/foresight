"""Timestamped action buffer (Job 4): never act on old information.

Each action has a time it should run. A new chunk overwrites every future slot
from its start time onward. Slots whose time has passed are dropped.
"""

from __future__ import annotations

import numpy as np


class ActionBuffer:
    def __init__(self, max_age: float = 0.1):
        """max_age: a due action older than this many seconds is too stale; skip it."""
        self.max_age = float(max_age)
        self.t = np.empty(0)  # (N,) slot times, sorted
        self.a = np.empty((0, 0))  # (N, D) actions

    def __len__(self) -> int:
        return len(self.t)

    def clear(self) -> None:
        self.t, self.a = np.empty(0), np.empty((0, 0))

    def push_chunk(self, actions, t0: float, dt: float, now: float | None = None,
                   blend: int = 0) -> None:
        """Add chunk: action i runs at t0 + i*dt. Overwrites old slots with time >= t0.

        now: if given, slots (new or old) earlier than now are dropped.
        blend: first `blend` new slots are mixed with the old plan (old fades out).
        """
        acts = np.asarray(actions, dtype=np.float64)
        if acts.ndim != 2 or len(acts) == 0:
            raise ValueError(f"actions must be (N, D), got {acts.shape}")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if len(self.t) and self.a.shape[1] != acts.shape[1]:
            raise ValueError(f"action dim {acts.shape[1]} != buffer dim {self.a.shape[1]}")
        times = t0 + dt * np.arange(len(acts))
        n_old_keep = np.searchsorted(self.t, t0, side="left")  # old slots strictly before t0
        # Blend: where the old plan still covers a new slot's time, mix them.
        if blend > 0 and len(self.t) > 1:
            for i in range(min(blend, len(acts))):
                if times[i] > self.t[-1]:
                    break
                old = np.array([np.interp(times[i], self.t, self.a[:, d])
                                for d in range(acts.shape[1])])
                w = (i + 1) / (blend + 1)  # weight of the new action
                acts[i] = w * acts[i] + (1 - w) * old
        t_all = np.concatenate([self.t[:n_old_keep], times])
        a_all = np.concatenate([self.a[:n_old_keep].reshape(n_old_keep, acts.shape[1]), acts])
        if now is not None:
            keep = t_all >= now
            t_all, a_all = t_all[keep], a_all[keep]
        self.t, self.a = t_all, a_all

    def pop_due(self, now: float):
        """Return newest action whose time <= now, drop it and every older slot.

        Returns None if nothing is due, or the due action is older than max_age.
        """
        n_due = int(np.searchsorted(self.t, now, side="right"))
        if n_due == 0:
            return None
        t_due, act = self.t[n_due - 1], self.a[n_due - 1].copy()
        self.t, self.a = self.t[n_due:], self.a[n_due:]
        return None if now - t_due > self.max_age else act
