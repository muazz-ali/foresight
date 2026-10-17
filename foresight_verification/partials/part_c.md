# Part C — Controller ID + oracle eval offsets

Seed=0. Demo root=`data/p3_throw/*/speed_*/*.h5`. h=`ee_pos`, c=`action_pos`. dt=0.04s.
Conveyor axis: **y** (object motion).

## C1 — First-order α fit

Episodes used: 6529 (success preferred; all loaded demos were success).

| Axis | α | R² | n | second-order? |
|---|---:|---:|---:|---|
| x | 0.1051 | 0.2987 | 763074 | yes R²=0.447 (a=0.456, b=0.048, c=0.009) |
| y | 0.1787 | 0.8956 | 759111 | yes R²=0.930 (a=0.466, b=0.104, c=-0.006) |
| z | 0.1626 | 0.8697 | 763309 | yes R²=0.916 (a=0.702, b=0.111, c=-0.069) |

Per-phase / per-speed tables: see `part_c.json` keys `C1.per_phase`, `C1.per_speed`.

## C2 — Simulate with α vs logged hand

Simulated 200 episodes.
- Best lag d (xy): mean **-0.45 frames** (-0.018 s), std 0.50.
- Measured tracking lag d = mean(c−h) before close (cm):
  - x: measured 2.40±1.22 cm; α-ramp predict 1.34±2.33 cm
  - y: measured 0.84±1.34 cm; α-ramp predict 1.06±1.56 cm
  - z: measured -10.39±0.91 cm; α-ramp predict -8.93±1.02 cm

_Measured d = mean(c−h) cm during approach/close before first gripper<0.5. Predicted d = v_cmd·dt·(1−α)/α (ramp-tracking lag of discrete first-order). d_frames = best shift aligning simulated h to logged h (xy RMSE)._

## C3 — Oracle eval hand−object offsets

Oracle episodes: 80 (checkpoint `runs/p3_throw/pretrained_model`).

### Inventory
- `data/eval/p3_throw_oracle_n20`: {'exists': True, 'n_metrics_rows': 80, 'n_oracle': 80, 'n_filter': 0, 'n_cond_h5': 80, 'metrics_file': 'metrics.jsonl'}
- `data/eval/claims/base/oracle`: {'exists': True, 'n_metrics_rows': 80, 'n_oracle': 80, 'n_filter': 0, 'n_cond_h5': 80, 'metrics_file': 'metrics.jsonl'}
- `data/eval/p3_throw_n100`: {'exists': True, 'n_metrics_rows': 400, 'n_oracle': 0, 'n_filter': 400, 'n_cond_h5': 400, 'metrics_file': 'metrics.jsonl'}
- `data/eval/p3_throw_n100_diverse`: {'exists': True, 'n_metrics_rows': 400, 'n_oracle': 0, 'n_filter': 400, 'n_cond_h5': 400, 'metrics_file': 'metrics_all.jsonl'}

### Hand − object xy at close (cm), per speed

| speed | n | mean x | std x | mean y | std y | mean ‖·‖ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 20 | 0.96 | 1.85 | 0.00 | 0.37 | 1.84 |
| 0.10 | 20 | 2.07 | 2.58 | -1.40 | 2.19 | 3.26 |
| 0.15 | 20 | 1.64 | 3.84 | -1.78 | 2.45 | 4.17 |
| 0.20 | 20 | 4.51 | 3.20 | -5.27 | 7.36 | 7.58 |

### Hand − object xy at min-distance frame (cm), per speed

| speed | n | mean x | std x | mean y | std y | mean ‖·‖ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 20 | 0.76 | 1.73 | 0.06 | 0.25 | 1.68 |
| 0.10 | 20 | 0.79 | 1.64 | -0.18 | 0.45 | 1.50 |
| 0.15 | 20 | 0.51 | 2.21 | -0.46 | 0.67 | 2.11 |
| 0.20 | 20 | 1.06 | 1.25 | -0.59 | 1.09 | 1.60 |

### Last-50 hand − true future object (truth_pack)

- n=80
- late-window mean offset (cm): {'x': 1.0534396548971432, 'y': -2.9156924703739002}
- end-frame mean (cm): {'x': 2.591351941227913, 'y': -4.082876828906592}
- early→late Δmean (cm): {'x': 4.524937104294676, 'y': -3.5420297222114554}
- settles to constant offset? **False**
- figures: ['foresight_verification/figures/c3_hand_minus_future_last50.png', 'foresight_verification/figures/c3_hand_minus_future_mean.png']

_Offset = hand_xy − true_future_xy (truth_pack if present else oracle conditioning) in cm, ≤50 frames ending at close. settles_to_constant_offset=True if mean |early−late| < 1 cm on both axes._

## C4 — α·(1 − w_h)

Status: **OK**
- α: {'x': 0.10511328590078355, 'y': 0.17865277642533178, 'z': 0.16260380781861977}
- w_h: {'x': 0.5850301528172548, 'y': 0.5926688124397346}
- α·(1−w_h): {'x': 0.043618844187124355, 'y': 0.07277084758226898}
- stable (0<·<2): {'x': True, 'y': True}

Stability requires 0 < α·(1−w_h) < 2 per axis. w_h from summary_pooled_k1_8.{axis}.w_h.


## Files / method

- Script: `foresight_verification/scripts/part_c_controller_eval.py`
- Demo files: `data/p3_throw/*/speed_*/*.h5` (ee_pos, action_pos, stage, gripper, success)
- Oracle eval: `*_cond.h5` + `metrics.jsonl` under oracle dirs listed above
