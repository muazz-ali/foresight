"""Timestamped action buffer (Job 4): never act on old information.

Each action has a time it should run. A new chunk overwrites every future slot
from its start time onward. Slots whose time has passed are dropped, except the
one that is due right now. A chunk planned from an older camera frame than one
already in the buffer is ignored (results can arrive out of order).
"""

from __future__ import annotations

import numpy as np


class ActionBuffer:
    def __init__(self, max_age: float = 0.1):
        """max_age: a due action older than this many seconds is too stale; skip it."""
        self.max_age = float(max_age)
        self.t = np.empty(0)  # (N,) slot times, sorted
        self.a = np.empty((0, 0))  # (N, D) actions
        self.last_obs_t = -np.inf  # camera time of the newest plan pushed so far

    def __len__(self) -> int:
        return len(self.t)

    def clear(self) -> None:
        self.t, self.a = np.empty(0), np.empty((0, 0))
        self.last_obs_t = -np.inf

    def push_chunk(self, actions, t0: float, dt: float, now: float | None = None,
                   blend: int = 0, obs_t: float | None = None) -> bool:
        """Add chunk: action i runs at t0 + i*dt. Overwrites old slots with time >= t0.

        now: if given, slots (new or old) older than one step before now are dropped.
            The newest slot at or before now is kept: it is the one due this tick.
        blend: first `blend` new slots are mixed with the old plan (old fades out).
        obs_t: camera time the chunk was planned from. If it is older than a plan
            already pushed, the chunk is ignored (returns False). None = no check.
        """
        if obs_t is not None:
            if float(obs_t) < self.last_obs_t - 1e-9:
                return False
            self.last_obs_t = float(obs_t)
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
            # Keep the slot due this tick (time in (now - dt, now]); drop older ones.
            keep = t_all > now - dt + 1e-9
            t_all, a_all = t_all[keep], a_all[keep]
        self.t, self.a = t_all, a_all
        return True

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
