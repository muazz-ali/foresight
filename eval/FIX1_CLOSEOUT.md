# Fix 1 closeout (before Fix 2)

## 1. Predictive close gate (settle_ticks reverted)

- `close_settle_ticks: 1` (was 15).
- Close when **both** `|e_now| ≤ 2 mm` and `|e + Δv_rel·t_contact| ≤ 2 mm`, and
  `|Δv| ≤ 1.39 cm/s`.
- `e = hand − object` (+ = hand ahead). `Δv_rel = v_hand − v_obj`.
- `t_contact = t_cmd + s/v_finger = 130 ms` (`sim.gripper_actuator.nominal_t_contact_s`).

### B5 static recheck (`b5_recheck_pred2/`)

Scored at **pad contact** (CLOSE + 130 ms). Bar for Fix1 closeout: mean
`d_x`, `d_y` within **±0.3 mm**.

| object | n | mean d_x mm (contact) | mean d_y mm | ±0.3 mm |
|---|---|---|---|---|
| egg | 4 | −0.31 | +0.08 | **miss** (dx) |
| lemon | 4 | −0.75 | −0.34 | miss |
| onion | 4 | −0.57 | −0.06 | miss |

B5’s own ±2 mm gate: **PASS**. The ±0.3 mm closeout bar is **not met** with
predictive@2 mm alone (egg is closest). Residual ~0.3–0.8 mm at contact is the
DiffIK approach leftover — Fix 2 / lag territory.

### C30 first-attempt (egg, v=30 cm/s, Δ=0)

| seed | success | failure |
|---|---|---|
| 1000 | **yes** | — |
| 1001 | no | never_closed |
| 1003 | no | never_closed |

**1 / 3** first-attempt successes. 1001/1003 never satisfy the predictive +
speed gate (along-track lag ≈ Fix 2).

## 2. Pinch frame for Z

- `robot.tcp_offset_mm: [0, 0, 103.4]` = URDF pad midpoint (was 107).
- `z_grasp` still aims at object centre; DiffIK `ee` is now the **pinch**, so
  the pinch lands on the centre.
- `closed_above` in `eval/b1_success.py` compares `ee_z − obj_z` in that same
  pinch frame.

## 3. Lag fit from B1 v7 (matched-speed window)

Metric: **mean** `e = hand − object` over frames with `|Δv| ≤ 1.39 cm/s` in
intercept/close. `+` = hand ahead.

| | |
|---|---|
| `L_x` (e_x ≈ L_x·v_x) | **−6.4 ms** |
| `L_y` (e_y ≈ L_y·v_y) | **−8.9 ms** |
| matched frames | 617 |
| RMS residual e_x / e_y | 2.06 / 1.04 mm |

### C30 predicted cross from (L_x, L_y)

| seed | along meas mm | cross meas mm | along pred mm | cross pred mm |
|---|---|---|---|---|
| 1001 | −1.81 | −1.91 | −2.04 | **−0.27** |
| 1003 | −5.29 | +0.27 | −2.28 | **+0.37** |

Lag axes explain little of s1001’s −1.9 mm cross (pred −0.27 mm). That leftover
is not a pure `L·v` effect.

## 4. Standard metric + τ_c

- **Error:** `e = hand − object`, mean over the matched-speed window, `+` = hand ahead.
- **τ_c** (v=0 DiffIK approach, AB_recordings_v5): **0.24 ± 0.017 s** (n=24 traces).
- Fix1 settle wall time (108 fully settled): mean **1.95 s** (includes move-in).

Raw fit: `data/eval/fix1_static_offset/closeout_lag_fit.json`.

## Files

- `control/close_guard.py` — `predictive_close_check`
- `sim/gripper_actuator.py` — `nominal_t_contact_s` (130 ms)
- `sim/state_machine.py` — predictive gate; pinch tcp default
- `sim/phase0_cfg.yaml` — `tcp_offset_mm: 103.4`, `close_settle_ticks: 1`
- `eval/b1_success.py`, `eval/b5_hand_trail_bias.py` — pinch / contact scoring
- `interfaces/grasp_height.py` — DEFAULT tcp 103.4 mm
- `tests/test_predictive_close.py`
