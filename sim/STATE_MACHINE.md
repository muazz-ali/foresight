# State machine

Code: [`state_machine.py`](state_machine.py) · loop: [`collect.py`](collect.py)  
Knobs: [`cfg/state_machine.yaml`](cfg/state_machine.yaml) (via `phase0_cfg` / `mode2_cfg`)

Scripted controller that records demos for the **policy**. Not a neural net.  
Each step: read object + robot → pick aim / gripper / object flags → scene runs it.  
One **stage** at a time (`0`–`8`). Names: [`STAGE_PLAIN`](state_machine_logs.py).

---

## Episode loop

[`run_episode`](collect.py): reset → each tick (`sim.dt` ≈ 30 Hz, max **8 s**)  
state machine → move robot → log → stop on **done** or timeout → [`success.py`](success.py).

**Command** (`StateMachineCommand`): robot XYZ, gripper open/closed, stage,  
object flags — slide / glue / freeze.

| When | Object |
|---|---|
| Before grasp | slides |
| Close → open | glued to robot |
| After release / home | often frozen |

---

## Chase (stage 1)

Aim where the object **will be** when the robot can get there (lead capped at **0.40 s**).  
Recompute every step. Joint speed feed keeps the robot from lagging forever.

Close when the robot is inside `grasp_xyz_ok_m` on the **object now** (GT).  
Aim Z = **object centre**. Gripper then closes only inside `close_xy_ok_m`.

---

## Stages

```
0 start → 1 intercept → 2 close → 3 lift → 4 carry → 5 lower → 6 open → 7 home → 8 done
```

| Stage | What happens |
|---|---|
| 0 | Same step → chase (no home pause) |
| 1 | Lead + forecast; gate `grasp_xyz_ok_m` |
| 2 | Close gripper, glue; wait `SM_hold_to_close_fingers_s` |
| 3 | Lift by `lift_height_m` |
| 4 | Fly to container (clear rim) |
| 5 | Lower to drop pose |
| 6 | Open gripper; object in container |
| 7 | Rise above rim, then start pose (`home_radius_m`) |
| 8 | Done latch |

Miss / slip / timeout → fail labels in [`WHY_FAILED`](state_machine_logs.py).  
No per-stage stall clock; only `episode_max_s`.

---

## Knobs

Edit [`cfg/state_machine.yaml`](cfg/state_machine.yaml). Values are SI (m / s / m/s).

| Yaml | Default | Meaning |
|---|---|---|
| `max_aim_ahead_s` | 0.4 | Lead cap |
| `robot_speed_max_m_s` | 2.5 | Meet-time robot speed |
| `robot_base_safe_zone_m` | 0.1 | Base keep-out circle |
| `grasp_xyz_ok_m` | [0.01, 0.01, 0.012] | Chase gate |
| `close_xy_ok_m` | 0.002 | Close XY box |
| `SM_hold_to_close_fingers_s` | 0.3 | Hold after close, then lift |
| `arrived_xyz_m` | 0.03 | “Arrived” for later stages |
| `release_below_rim_m` | 0.01 | Open under rim |
| `lift_height_m` | 0.12 | Lift (mode2: 0.05) |
| `home_radius_m` | 0.1 | Home done ball |
| `move_up_above_container_m` | 0.05 | Rise before home |
| `episode_max_s` | 8.0 | Only clock |
| `z_slack_m` | 0.005 | Height close-band (± half) |

Also: [`cfg/collect.yaml`](cfg/collect.yaml), [`cfg/training.yaml`](cfg/training.yaml), [`cfg/evaluation.yaml`](cfg/evaluation.yaml).  
Scene / motion / video: see [`README.md`](README.md).
