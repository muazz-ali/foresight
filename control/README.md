# control/

Timing tools for the claim path. No learning, no Isaac.

| File | Role |
|---|---|
| `buffer.py` | Timestamped action queue (drop stale slots) |
| `delay_injection.py` | Applied vs reported Δ for demos / pack — train range in `sim/cfg/training.yaml` |
| `exec_chunk.py` | Relative chunk → world targets into the buffer |
| `close_guard.py` | Hand–object speed match while closing |
| `reflex.py` / `capture.py` | Last-cm close helpers (G3 overlays) |
| `reflex_overlay.py` | Wire reflex onto a state-machine command |

## `ActionBuffer` — never act on old info
- `push_chunk(actions, t0, dt, now=None, blend=0)`: action `i` runs at `t0 + i*dt`.
  A new chunk wipes old plans from `t0` onward. `blend=k` mixes the first `k` new
  actions with the old plan so the hand does not jerk.
- `pop_due(now)`: gives the newest action whose time has come, and throws away
  older ones. Gives `None` if nothing is due, or the action is older than `max_age`.

## `DelayInjection`
- Train: draw applied Δ in `[min_delta_delay, max_delta_delay]` (yaml ms → seconds).
- Reported Δ = applied (+ optional jitter; default **0** in shipped training yaml).
- Robot waits only if `apply_delta_delay_to_robot: true` (default **false** — numbers only).

## `ReflexClose` — last ~10 cm
- `step(obj, ee_pos, now, ee_vel=None) -> ReflexOut(active, target, vel_cmd, close, tta)`
- `obj` is the frozen `ObjectState` (position, velocity, covariance, timestamp, valid).
- Far away or `valid=False`: `active=False` (policy keeps driving).
- Near: `target` is where the object will be; `vel_cmd` steers the hand there.
- `close=True` when time-to-arrival < gripper close time (or hand is almost touching).
  It stays True until `reset()`.

## Where they run
- **Collect** (state-machine demos): reflex only when `state_machine_params.use_reflex: true`
  (default **false**; G3 demos are recorded with the reflex off).
- **Policy eval** (`scripts/eval_policy.py`): eval-time overlays, explicit flags only
  (the config's `use_reflex` is ignored there). All default off = old behaviour.
  - `--reflex on`: ReflexClose with `robot.gripper_close_time_s` from the sim config.
  - `--action-buffer on`: chunks go into `ActionBuffer`; slots already in the past are dropped.
  - `--inject-latency-s L`: the policy plans from an observation L s old.
- G3 baseline = same model, same seeds, `--reflex off`. Score: `eval/gate_g3.py`.
