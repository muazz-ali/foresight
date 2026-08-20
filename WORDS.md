# Plain words (glossary)

We keep short code names for compatibility. Prefer these plain phrases in docs and talk.

| Code / old word | Plain meaning |
|---|---|
| **oracle** | Perfect object pose/speed from the **simulator** (not from cameras) |
| **oracle_state** | That perfect state saved each frame |
| **conditioning / conditioning vector** | The **future numbers** we give the policy. Phase-1 live pack = **5**: where the object will be soon in XY, how fast in XY, and **Δ** |
| **Δ / delta** | How far ahead we look (seconds), e.g. 0.25 s. Must be **in** the vector, not only used to build p̂ |
| **filter / estimator** | Same 12 numbers, but from **camera tracking** instead of the simulator |
| **proprio** | Robot arm joint / hand pose readings |
| **gate (G0, G1, …)** | A **pass/fail checkpoint** before we build the next piece |
| **Model A** | Policy that sees **images + robot pose + text only** |
| **Model B** | Same as A, plus the **5 future numbers** (8 robot + 5 = 13). Full ~12 later when the filter is live |
| **expert** (old: SM) | Our **scripted pick-and-place** — a hand-written list of stages, not a learned policy |
| **holding** | We **snap the object to the hand**. Not a real physics grab. Logged as `holding` 0/1. |
| **hover ahead** | Hand goes **where the object will be after the look-ahead**, a little above it — not where it is now |
| **wait still** | Pause table motion so a moving object can be caught |
| **ee / EE** | The **hand** (the gripper). Array name `ee_pos` stays for the robot pose. |
| **stage** | Expert step number (0–11). Human logs print the name too. |
| **max_stage** | **Farthest stage** reached |
| **failure** | **Why it failed** (or “success”) |
| **carry pose** | Hand pose **over the box** (clear of the rim) |
| **drop target** | Hand pose **inside the box** |
| **kinematic** | Object motion we **set by code**, not full physics bounce |
| **reflex** | Fast last-centimeter grasp helper (later phase) |
| **VLA** | Vision–language–action policy (e.g. SmolVLA) |
| **LeRobot** | Dataset/training library we use for SmolVLA |

## Expert stages (one name each)

The JSON stores the number (`stage`). The human log and video use the name.

| # | Name | What it means |
|---|---|---|
| 0 | start | Arm at rest. Object on the table. |
| 1 | hover ahead | Hand goes where the object will be in 0.25 s, a little above it. |
| 2 | go down | Hand drops onto that same future point. |
| 3 | wait still | Pause the object. Wait until it is almost stopped and the hand is on it. |
| 4 | close fingers | Fingers shut. Object is held (snapped to the hand). |
| 5 | lift | Raise the object. |
| 6 | carry to box | Move high, over the box. |
| 7 | lower into box | Come down inside the box. |
| 8 | hold in box | Stay there until the object is still. |
| 9 | open fingers | Let go. Object should stay in the box. |
| 10 | go home | Arm returns to rest. Object stays in the box. |
| 11 | done | Finished. |

**Do not mix with policy eval:** policy logs `holding` 0/1. Expert `4` means **close fingers**. Never write holding into `stage`.

**“Drop” in the plan** means the object *fell*. Stages 7–8 are putting it in the box, so we say **lower into box** / **hold in box**.

Frozen message (do not rename fields): `position, velocity, covariance, timestamp, valid`.
