# Intercept-servo flow diagrams

Plain guide to how the expert chases a moving object. Code: [`sim/state_machine.py`](../sim/state_machine.py).

---

## One sim step (stages 0–9)

```mermaid
flowchart TD
    A[Read object pos + vel + hand pos] --> B{Which stage?}

    B -->|0 start| S0[Wait settle_s]
    S0 --> S1

    B -->|1 intercept| S1

    S1 --> L1["Compute lead time lead_s<br/>_intercept_lead_s()"]
    L1 --> L2["Forecast object at lead_s<br/>(with wall bounces)"]
    L2 --> L3["Aim hand at forecast XY + blended Z"]
    L3 --> L4{Hand over object now?<br/>track_xy + approach_z<br/>held for track_hold_s}
    L4 -->|no| L5{Stuck too long?}
    L5 -->|yes| RET["Re-intercept (retry)"]
    L5 -->|no| L3
    L4 -->|yes| BG{Bounce clear?<br/>enough time before wall hit}
    BG -->|no| L3
    BG -->|yes| S2

    B -->|2 go down| S2
    S2 --> D1["Aim at object + tau lead in XY<br/>drive Z onto object"]
    D1 --> D2{Grasp gate OK?<br/>XY + Z + speed-aware tol}
    D2 -->|no| D3{Stalled?}
    D3 -->|yes| RET
    D3 -->|no| D1
    D2 -->|yes| S3["Log grasp_err, grasp_speed"]

    B -->|3 close| S3
    S3 --> S4[lift → carry → lower → open → home → done]

    RET --> S1
```

---

## Inside `_intercept_lead_s` (meet-up solver)

Code: [`state_machine.py` lines 230–256](../sim/state_machine.py#L230).

```mermaid
flowchart TD
    START[Hand ee, object obj, velocity vel] --> GZ["Pick target height gz<br/>(hover above object)"]
    GZ --> T0["t_travel = 0"]

    T0 --> LOOP{{"Repeat intercept_iters (≈3)"}}

    LOOP --> F1["Where will object be in t_travel?<br/>aim = forecast(obj, vel, min(t_travel, 0.25s))"]
    F1 --> G1["goal = aim XY + gz"]
    G1 --> F2["t_travel = distance(ee, goal) / ee_speed_max<br/>how long hand needs to get there"]

    F2 --> LOOP

    LOOP --> OUT["lead_s = min(t_travel, 0.25s) + tau"]
    OUT --> USE["Forecast object at lead_s → that's the hand command"]
```

---

## What `tau` does (lag compensation)

`tau` ≈ `servo_lag_s + dt` — arm delay from command to motion. We aim **ahead** so lateness puts the hand **on** the object, not behind it. The object speed is **not** slowed down.

```mermaid
flowchart LR
    subgraph without ["Without compensation"]
        O1[Object moving →] --> A1[Aim at object NOW]
        A1 --> H1[Hand lags behind command]
        H1 --> B1[Hand arrives BEHIND object]
    end

    subgraph with ["With compensation (+ tau)"]
        O2[Object moving →] --> A2["Aim at object + v×tau<br/>(a little ahead)"]
        A2 --> H2[Hand still lags...]
        H2 --> B2[Hand lands ON object]
    end
```

---

## Far vs close (why lead shrinks)

```mermaid
flowchart TD
    FAR[Hand far from object] --> FT["t_travel large"]
    FT --> FC["Cap → 0.25s + tau"]
    FC --> FA["Aim well ahead<br/>(full intercept)"]

    CLOSE[Hand on object] --> CT["t_travel → 0"]
    CT --> CC["lead ≈ tau only"]
    CC --> CA["Hand rides WITH object<br/>(not 0.25s in front forever)"]
```

---

## Legend

| Term | Plain meaning |
|------|----------------|
| **forecast** | Roll object forward in time (same bounce rules as sim) |
| **lead_s** | How many seconds ahead we look when aiming |
| **tau** | Arm delay — command arrives before motion catches up |
| **track gate** | Is hand over object *now*? (not over a moving aim point) |
| **bounce guard** | Don't go down if object will hit a wall before grasp finishes |

---

## Related docs

- **`step()` function (all stages):** [`step-function-flow.md`](step-function-flow.md)
- Full stage walkthrough: [`sim/STATE_MACHINE.md`](../sim/STATE_MACHINE.md)
- Knobs: [`sim/phase0_cfg.yaml`](../sim/phase0_cfg.yaml) → `state_machine_params`
