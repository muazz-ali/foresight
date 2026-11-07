"""A6: the async executor executes no stale motions.

Isaac-free stress test of the two executors on the claim path:

  buffer   control.ActionBuffer (+ control.exec_chunk.push_forecast_chunk), the executor behind
           eval_policy --action-buffer and the claim chunk path
  collect  the expert's delay hold in sim/collect.py (claim_timing.enforce_apply_at): pending
           commands are released when now ≥ apply_at. That loop is inline in run_episode and cannot
           be imported, so it is replicated line for line in `collect_queue` below (keep in sync).

Every action carries a tag (chunk id, slot time, observation time t0) in action dims the conversion
leaves untouched, so each executed motion can be traced. Rules (any violation = stale motion):

  R1 early      executed before its slot time            (slot_t > now)
  R2 late       executed more than one control step late (now − slot_t > h)
  R3 superseded a newer-observation plan for that same moment had already arrived
  R4 regress    its observation is older than the motion executed just before it
  R5 forecast   the forecast time baked into the target is more than h + |jitter| from the moment
                it is executed (forecast path only: object moving at 1 m/s along x, so target x
                = forecast time in seconds)

Scenarios cover sync replan (as eval_policy does today), one inference in flight with fixed and
variable Δ (DelayInjection train mode, 0–200 ms per cycle), two inferences in flight (results can
arrive out of order), and control-loop jitter / stalls.

Usage: python eval/a6_stale_motions.py [--episodes 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.buffer import ActionBuffer  # noqa: E402
from control.delay_injection import DelayInjection  # noqa: E402
from control.exec_chunk import push_forecast_chunk  # noqa: E402
from interfaces.state import ObjectState  # noqa: E402

H = 1.0 / 30.0      # control step (sim.dt)
EPS = 1e-6
CHUNK = 25
EVAL_CHUNK = 50   # SmolVLA chunk_size; eval_policy pushes the whole chunk when the buffer is on
EP_S = 4.0          # episode length simulated


def tagged_chunk(cid, t0, slot0, n, h):
    """(n, 8): xyz = 0 (relative), dims 3..5 = chunk id, slot time, obs time; gripper open."""
    a = np.zeros((n, 8))
    a[:, 3] = cid
    a[:, 4] = slot0 + h * np.arange(n)
    a[:, 5] = t0
    a[:, 7] = 1.0
    return a


class Checker:
    def __init__(self, h):
        self.h = h
        self.v = {k: 0 for k in ("R1_early", "R2_late", "R3_superseded", "R4_regress", "R5_forecast")}
        self.n_exec = 0
        self.n_hold = 0
        self.last_obs = -np.inf
        self.arrived = []  # (t0, slot_first, slot_last) of chunks pushed so far
        self.examples = []
        self.dropped_due = 0
        self.count_late = True

    def arrive(self, t0, s0, s1):
        self.arrived.append((t0, s0, s1))

    def check(self, now, act, jitter_tol=None):
        if act is None:
            self.n_hold += 1
            return
        self.n_exec += 1
        slot_t, obs_t0 = float(act[4]), float(act[5])
        bad = []
        if slot_t > now + EPS:
            bad.append("R1_early")
        if now - slot_t > self.h + EPS:
            bad.append("R2_late") if self.count_late else None
            if not self.count_late:
                self.info_late = getattr(self, "info_late", 0) + 1
        if any(t0 > obs_t0 + EPS and s0 - EPS <= slot_t <= s1 + EPS for t0, s0, s1 in self.arrived):
            bad.append("R3_superseded")
        if obs_t0 < self.last_obs - EPS:
            bad.append("R4_regress")
        if jitter_tol is not None:
            fc_time = float(act[0])  # x = 1 m/s · forecast lead from obs → forecast time − t0
            if abs((obs_t0 + fc_time) - now) > self.h + jitter_tol + EPS:
                bad.append("R5_forecast")
        for b in bad:
            self.v[b] += 1
        if bad and len(self.examples) < 3:
            self.examples.append({"now": round(now, 4), "slot_t": round(slot_t, 4), "obs_t0": round(obs_t0, 4), "rules": bad})
        self.last_obs = max(self.last_obs, obs_t0)


def run_buffer(scn, rng, delay):
    """Simulate one episode of the ActionBuffer executor under scenario `scn`."""
    buf = ActionBuffer(max_age=scn.get("max_age", 0.20))
    chk = Checker(H)
    # control ticks (optionally jittered / with stalls)
    ticks, t = [], 0.0
    while t < EP_S:
        ticks.append(t)
        step = H + (rng.normal(0, scn["tick_jitter"]) if scn["tick_jitter"] else 0.0)
        if scn.get("stall_p") and rng.random() < scn["stall_p"]:
            step += scn["stall_s"]
        t += max(step, 0.002)
    in_flight = []  # (ready_time, t0, sample, cid)
    cid = 0
    next_start = 0.0
    obj = ObjectState(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros((6, 6)), 0.0, True)
    for k, now in enumerate(ticks):
        # launch inferences
        if scn["mode"] == "sync":
            if k % scn["n_exec"] == 0:
                lat = scn["latency"]
                in_flight.append((now, now - lat, None, cid)); cid += 1  # plans from an obs lat old, ready now
        else:
            while next_start <= now + EPS and len([f for f in in_flight]) < scn["max_in_flight"]:
                t0 = now
                s = delay.sample(t0, mode=scn["delay_mode"])
                ready = t0 + s.applied_delta_s   # compute time = applied Δ: result usable at apply_at
                in_flight.append((ready, t0, s, cid)); cid += 1
                next_start = now + scn["launch_every"] if scn["max_in_flight"] > 1 else np.inf
            if scn["max_in_flight"] == 1 and not in_flight:
                next_start = now
        # deliver results that are ready (arrival order = ready time)
        in_flight.sort(key=lambda f: f[0])
        still = []
        for ready, t0, s, c in in_flight:
            if ready <= now + EPS:
                if s is None:  # sync path, as eval_policy: slot i at obs time + i·h
                    acts = tagged_chunk(c, t0, t0, EVAL_CHUNK, H)
                    slots = t0 + H * np.arange(EVAL_CHUNK)
                    buf.push_chunk(acts, t0=t0, dt=H, now=now - EPS, obs_t=t0)
                    due_now = slots[(slots <= now + EPS) & (slots > now - H + EPS)]
                    if len(due_now) and not np.any(np.isclose(buf.t, due_now[-1])):
                        chk.dropped_due += 1
                    chk.arrive(t0, t0, t0 + H * (EVAL_CHUNK - 1))
                else:  # claim path: slot 0 at apply_at, forecast at reported Δ
                    rel = tagged_chunk(c, t0, s.apply_at, CHUNK, H)
                    st = ObjectState(obj.position, obj.velocity, obj.covariance, t0, True)
                    slots = s.apply_at + H * np.arange(CHUNK)
                    push_forecast_chunk(buf, rel_actions=rel, state=st, delta_s=s.reported_delta_s,
                                        apply_at=s.apply_at, now=now - EPS, dt=H)
                    due_now = slots[(slots <= now + EPS) & (slots > now - H + EPS)]
                    if len(due_now) and not np.any(np.isclose(buf.t, due_now[-1])):
                        chk.dropped_due += 1
                    chk.arrive(t0, s.apply_at, s.apply_at + H * (CHUNK - 1))
            else:
                still.append((ready, t0, s, c))
        in_flight = still
        act = buf.pop_due(now + EPS)
        tol = None if scn["mode"] == "sync" else 4 * float(delay.jitter_std_s)
        chk.check(now, act, jitter_tol=tol)
    return chk


def collect_queue(rng, delay, n_ticks=int(EP_S / H)):
    """Line-for-line replica of sim/collect.py enforce_apply_at release (one expert cmd per tick)."""
    chk = Checker(H)
    chk.count_late = False  # holding a released cmd between releases is by design here
    pending, held, held_t0 = [], None, -np.inf
    for t in range(n_ticks):
        now = t * H
        t0 = now
        s = delay.sample(t0, mode="train")
        cmd = np.array([0, 0, 0, t, s.apply_at, t0, 0, 1.0])  # tag: id, release time, obs time
        pending.append((float(s.apply_at), cmd, s))
        still = []
        for apply_at, cmd_i, _ in pending:            # collect.py: for apply_at, cmd_i, samp_i in pending
            if now + 1e-9 >= float(apply_at):
                if float(_.t0) >= held_t0 - 1e-9:      # collect.py: never step back to an older frame
                    held, held_t0 = cmd_i, float(_.t0)
            else:
                still.append((apply_at, cmd_i, _))
        pending = still
        chk.arrive(t0, s.apply_at, np.inf)   # a newer obs's command, once released, should hold from then on
        if held is not None:
            # R2 here = executing a command whose release time is > h in the past while a fresher one exists
            chk.check(now, held)
    return chk


SCENARIOS = {
    "sync, L=0 (eval today)":            dict(mode="sync", n_exec=8, latency=0.0, tick_jitter=0.0),
    "sync, L=100 ms":                    dict(mode="sync", n_exec=8, latency=0.10, tick_jitter=0.0),
    "sync, L=300 ms":                    dict(mode="sync", n_exec=8, latency=0.30, tick_jitter=0.0),
    "async 1 in flight, Δ=100 ms":       dict(mode="async", max_in_flight=1, delay_mode="eval", eval_delta=0.10, tick_jitter=0.0),
    "async 1 in flight, Δ=300 ms":       dict(mode="async", max_in_flight=1, delay_mode="eval", eval_delta=0.30, tick_jitter=0.0),
    "async 1 in flight, Δ~U(0,200) ms":  dict(mode="async", max_in_flight=1, delay_mode="train", tick_jitter=0.0),
    "async 2 in flight, Δ~U(0,200) ms":  dict(mode="async", max_in_flight=2, launch_every=0.05, delay_mode="train", tick_jitter=0.0),
    "async 1 in flight, tick jitter+stalls": dict(mode="async", max_in_flight=1, delay_mode="train", tick_jitter=0.004,
                                                  stall_p=0.02, stall_s=0.06),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "A6_stale"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, scn in list(SCENARIOS.items()) + [("collect enforce_apply_at, Δ~U(0,200) ms", {"collect": True})]:
        tot = {k: 0 for k in ("R1_early", "R2_late", "R3_superseded", "R4_regress", "R5_forecast")}
        n_exec = n_hold = dropped = info_late = 0
        ex = []
        for e in range(a.episodes):
            rng = np.random.default_rng(a.seed * 100003 + e)
            delay = DelayInjection(jitter_std_s=0.010, train_delta_min_s=0.0, train_delta_max_s=0.200,
                                   eval_delta_s=scn.get("eval_delta", 0.0), seed=a.seed * 100003 + e)
            chk = collect_queue(rng, delay) if scn.get("collect") else run_buffer(scn, rng, delay)
            for k in tot:
                tot[k] += chk.v[k]
            n_exec += chk.n_exec; n_hold += chk.n_hold; dropped += chk.dropped_due
            info_late += getattr(chk, "info_late", 0)
            ex += chk.examples[: max(0, 3 - len(ex))]
        bad = sum(tot.values())
        rows.append({"scenario": name, "executed": n_exec, "held": n_hold, "dropped_due_at_push": dropped,
                     "held_cmd_info": info_late, **tot,
                     "stale_frac": bad / max(n_exec, 1), "pass": bad == 0, "examples": ex})
    (out / "a6_report.json").write_text(json.dumps(rows, indent=1))
    L = ["# A6: async executor executes no stale motions", "",
         f"{a.episodes} simulated episodes per scenario, {EP_S:.0f} s each, control step {H*1000:.1f} ms, chunk {CHUNK}.",
         "R1 early · R2 > 1 step late · R3 newer plan for that moment already in · R4 older obs than previous motion · "
         "R5 forecast time ≠ execution time (> h + 4σ_jitter).", "",
         "Held = nothing due, so the loop re-sends the previous command (eval_policy does this): that re-sent motion is "
         "≥ 1 step old. 'Due slot dropped' = a chunk arrived between ticks and push_chunk(now=…) threw away the slot "
         "that was due right now.", "",
         "| scenario | executed | held (re-sent old cmd) | due slot dropped at push | R1 | R2 | R3 | R4 | R5 | pass |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scenario']} | {r['executed']} | {r['held']} | {r['dropped_due_at_push']} | {r['R1_early']} | {r['R2_late']} | "
                 f"{r['R3_superseded']} | {r['R4_regress']} | {r['R5_forecast']} | {'yes' if r['pass'] else '**no**'} |")
    for r in rows:
        if r["examples"]:
            L.append(f"\n{r['scenario']} — first violations: {r['examples']}")
    L += ["", "**Overall: " + ("PASS" if all(r["pass"] for r in rows) else "FAIL") + "**"]
    (out / "REPORT_auto.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
