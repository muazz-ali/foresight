# How the expert works (simple guide)

Code: [`state_machine.py`](state_machine.py).  
Knobs: [`phase0_cfg.yaml`](phase0_cfg.yaml) → `state_machine_params`.  
Loop: [`collect.py`](collect.py).

This file uses short, everyday words. Hard ideas get a plain sentence first, then a link into the code.

---

## Big picture

We teach a robot arm to **pick up a sliding object** on a table and **put it in a box**.

The **state machine** (also called the **expert**) is a hand-written checklist. It is **not** a neural net.

Every small time step it:

1. Looks at where the object is and how fast it moves
2. Looks at where the hand is
3. Picks the next aim for the hand, and whether fingers are open or closed
4. Says how the object should move: slide, stick to the hand, or stay still

Then the scene does that, and we repeat until **done** or time runs out.

---

## What is a “state machine”?

Think of a short play with numbered scenes. Only **one scene is active** at a time.

- The scene number is `stage` (0 through 9).
- Spoken names live here: [`STAGE_PLAIN`](state_machine_logs.py#L12).
- The checklist that runs each step is [`PickPlaceStateMachine.step`](state_machine.py#L360).

When a gate passes (“hand close enough”, “fingers closed long enough”, …), we move to the next scene.

---

## One episode, start to finish

The loop that drives one demo is in [`run_episode`](collect.py#L124):

```
reset scene + expert
        │
        ▼
┌────────────────────────────────┐
│  for each sim step (≤ ~10 s)   │
│                                │
│  1. read object + hand         │
│  2. expert.step(...)           │  ← decide stage + command
│  3. move hand / fingers        │
│  4. slide / freeze / attach    │  ← flags on the command
│  5. log frames                 │
│  6. stop if stage == done      │
└────────────────────────────────┘
        │
        ▼
  success check (object in box, arm home)
```

Key lines in the loop:

- Ask the expert: [`cmd = expert.step(...)`](collect.py#L188)
- Move the hand: [`scene.set_ee_target(...)`](collect.py#L190)
- Apply object flags: [`scene.step(...)`](collect.py#L191)

Default step size: `sim.dt = 0.04` s (25 steps per second).  
Max length: `inference_window_s = 10` → about 250 steps. Running out of time = fail.

---

## What the expert outputs each step

Type: [`ExpertCommand`](state_machine.py#L84)

| Field | Plain meaning |
|---|---|
| `position` | Where the hand should go next (world XYZ) |
| `gripper_open` | `1` = open, `0` = closed |
| `stage` | Current scene number (0–9) |
| `kinematic_object` | Object **slides** on the table at fixed speed |
| `attach_object` | Object is **glued** to the hand (held) |
| `freeze_object` | Object **stays put** (no slide) |

Rule of thumb:

- **Before grasp:** object slides.
- **Close → open:** object is glued to the hand.
- **After release / home / done:** object is often frozen so it does not bounce out of the box.

---

## Intercept-servo (the hard idea, said simply)

**Intercept-servo** is how the hand chases a *moving* object in stages 1–2.

Two ideas glued together:

1. **Intercept** — Aim where the object *will be when the hand can get there*, not a fixed “always 0.25 s ahead” point. Far away → lead more. Close → lead shrinks so the hand locks onto the object.
2. **Servo** — Every step, recompute that aim and steer toward it (a closed loop). Also add a bit of **lag compensation** so the hand settles *on* the object, not forever behind the command.

### Lag compensation (why we need `tau`)

The arm does not jump to a command instantly. There is a short delay from “tell the hand to go here” to “hand actually gets there”.

We call that delay **`tau`** ≈ `servo_lag_s + dt` ([property](state_machine.py#L170)).

If we aim at the object *now*, the hand arrives late and trails behind.  
If we aim a little *ahead by `tau`*, the hand can ride **on** the object in steady state.

### Meeting time (the lead)

The lead time is **not** always 0.25 s. It is roughly:

> how long the hand needs to reach the meeting point  
> (capped at `lookahead_s` = 0.25 s)  
> **plus** `tau`

Code: [`_intercept_lead_s`](state_machine.py#L231).

- Far away → lead near the 0.25 s cap.
- On top of the object → travel time → 0 → lead collapses to `tau` → hand rides the object.

### Bounce-aware guess

The object bounces off the play-band walls. A straight-line “fly 0.25 s ahead” guess can leave the table.

We roll the object forward with the **same** bounce rules the scene uses: [`_forecast`](state_machine.py#L220) → [`kinematic_forecast`](motion.py#L34).

### Object-frame gates

“Are we ready to go down / grasp?” compares the hand to the **object now**, not to the moving aim point.

With lag compensation, that error can go to zero. Without it, the aim stays ahead forever and the gate never closes.

### Grasp gate that grows with speed

At high speed, one control step moves the object several centimeters. A fixed 1–2 cm gate is too tight.

Effective XY gate: `max(grasp_xy_tol, k × speed × dt)` — [`_grasp_tol_xy`](state_machine.py#L258).

### Bounce guard

Do not commit to “go down” if the object is about to reverse off a wall before the descend can finish — [`_bounce_clear`](state_machine.py#L300).

---

## The stages (0 → 9)

Four jobs, split into small steps:

1. Meet the moving object (ride on it)
2. Grasp and lift
3. Carry to the box and release
4. Go home

Spoken names: [`STAGE_PLAIN`](state_machine_logs.py#L12).  
Schema used by success/logging: [`STAGE_SCHEMA`](state_machine.py#L48) (engaged=1, lift=4, placed=7, done=9).

### Stage 0 — start

[`STAGE_RESET` branch](state_machine.py#L381)

Arm sits at home. Wait `settle_s` so the scene is ready.

→ Then **intercept**.

---

### Stage 1 — intercept

[`STAGE_APPROACH` branch](state_machine.py#L388)

**Goal:** put the hand over the object and keep it there briefly.

- Compute lead with [`_intercept_lead_s`](state_machine.py#L231).
- Aim with [`_forecast`](state_machine.py#L220).
- Lower height as XY gets closer (`z_release_radius`) so we do not waste time on a flat hover then a late drop.
- “Locked” = hand near object *now* (`track_xy_tol` / `approach_z_tol`) for `track_hold_s`.
- Bounce guard must pass before we commit.

→ **go down**.  
If stuck too long → retry intercept ([`_restart_approach`](state_machine.py#L346)).

---

### Stage 2 — go down

[`STAGE_DESCEND` branch](state_machine.py#L436)

Keep riding the object in XY (aim with lag `tau` only). Drive Z onto it.

When XY/Z gates pass, record honest grasp error ([`grasp_err_xy` / `grasp_err_z`](state_machine.py#L445)) so a loose gate cannot hide a bad grab.

→ **close fingers**. Stall → re-intercept.

---

### Stage 3 — close fingers

[`STAGE_CLOSE` branch](state_machine.py#L455)

Fingers close. Object **attaches** (glued to the hand).

We do **not** freeze here: freezing a fast object would teleport-stop it.

Wait `close_gripper_s` → **lift**.

---

### Stage 4 — lift

[`STAGE_LIFT` branch](state_machine.py#L467)

Keep holding. Raise by `lift_height` at the grasp XY.

→ **carry to box**. Stall → home.

---

### Stage 5 — carry to box

[`STAGE_CARRY` branch](state_machine.py#L483)

Fly high to the box XY. Height clears the rim ([`_retract_z`](state_machine.py#L311)).

→ **lower into box**. Stall → home.

---

### Stage 6 — lower into box

[`STAGE_LOWER` branch](state_machine.py#L498)

Move down to the drop pose inside the container ([`_drop_target`](state_machine.py#L316)).

→ **open fingers**. Stall → home.

---

### Stage 7 — open fingers

[`STAGE_OPEN` branch](state_machine.py#L512)

Release (unglue). Freeze so the object cannot bounce out.

Require object XY near box center (`place_xy_tol`) — [`_in_container`](state_machine.py#L343).

→ **go home**.

---

### Stage 8 — go home

[`STAGE_HOME` branch](state_machine.py#L524)

Object stays in the box (frozen). Arm climbs clear if needed, then returns home — [`_home_target`](state_machine.py#L332).

Done when hand is home **and** object is still in the box.

→ **done**.

---

### Stage 9 — done

[`DONE` branch](state_machine.py#L532)

Success latch. Arm home, fingers open, object frozen in the box. The episode loop can stop early when `stage == stage_done` (9).

---

## Flow diagram

```
  [0 start]
      │  settle_s
      ▼
  [1 intercept]  ←── meet the object (lag + shrinking lead)
      │  locked over object + bounce clear
      ▼
  [2 go down]    ←── ride object; grasp gate
      │
      ▼
  [3 close]      ←── glue to hand (no freeze)
      │  close_gripper_s
      ▼
  [4 lift]
      │
      ▼
  [5 carry to box]
      │
      ▼
  [6 lower into box]
      │
      ▼
  [7 open]       ←── release + freeze
      │
      ▼
  [8 go home]
      │
      ▼
  [9 done]
```

---

## Knob cheat sheet

All under `state_machine_params` in [`phase0_cfg.yaml`](phase0_cfg.yaml#L77). Loaded in [`__init__`](state_machine.py#L102).

| Key | Default | Plain meaning |
|---|---|---|
| `lookahead_s` | 0.25 | Max look-ahead for the meeting point (s) |
| `conditioning_delta_s` | 0.25 | How far ahead the **policy** numbers look (s); not used for aiming |
| `servo_lag_s` | 0.16 | Hand delay we compensate for (s). Calibrate — do not guess |
| `ee_speed_max` | 0.90 | How fast we assume the hand can travel (m/s) |
| `hover_z` | 0.08 | Start height above the object when far away (m) |
| `track_xy_tol` | 0.02 | How close in XY counts as “riding” the object (m) |
| `track_hold_s` | 0.08 | Must stay locked this long before go-down (s) |
| `bounce_guard_s` | 0.10 | Need at least this much clear time before a wall hit (s) |
| `z_release_radius` | 0.12 | Start lowering when this far in XY (m) |
| `grasp_xy_tol` / `grasp_z_tol` | 0.02 | Base grasp gates (m) |
| `grasp_tol_step_k` | 0.6 | Grasp gate grows with speed |
| `servo_xy_tol` | 0.03 | “Close enough” in XY for carry / lower / home (m) |
| `place_xy_tol` | 0.12 | “In the box” XY gate (m) |
| `lift_height` | 0.12 | Raise after grasp (m) |
| `stage_timeout_s` | 1.5 | Stall cap after grasp (s) |
| `grasp_stage_timeout_s` | 3.0 | Longer stall cap for intercept / go-down (s) |
| `max_retries` | 3 | Re-intercept tries before giving up |
| `inference_window_s` | 10 | Max episode time; timeout = fail |

Calibrate lag / hand speed with `tools/diag/calibrate_servo_lag.py`.

---

## Did we succeed?

After the loop, [`episode_success`](success.py#L43) / [`classify_failure`](success.py#L90) check lift height, box placement, and farthest stage.

Failure labels (plain): [`WHY_FAILED`](state_machine_logs.py#L39) — never reached object, too late, grasp slip, off table, timeout, …

---

## How this ties to the rest of `sim/`

| Piece | Role |
|---|---|
| [`scene.py`](scene.py) | Table, arm, object, box, cameras; applies slide / freeze / glue |
| [`motion.py`](motion.py) | On-table bounce + forecast |
| [`state_machine.py`](state_machine.py) | Stage logic above |
| [`collect.py`](collect.py) | Step loop: expert → scene → log |
| [`success.py`](success.py) | Pass / fail label |
| [`video.py`](video.py) | Debug MP4 with stage text |
| [`state_machine_logs.py`](state_machine_logs.py) | Human-readable stage names |
| `../interfaces/state.py` | Shared object message (`position, velocity, …`) |
