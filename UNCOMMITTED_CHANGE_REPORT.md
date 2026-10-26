# Uncommitted change report

**Branch:** `fo/v1/dev` @ `6264413`  
**Scope:** working tree only (not committed)  
**Date:** 2026-10-25  
**Scale:** ~35 modified files, ~18 new files, 1 deleted · **+1021 / −972** lines in tracked diffs

---

## Verdict (one line)

This batch drops **floor / bounce mode**, locks the world to **house + coast**, moves the control loop to **30 Hz**, grows the policy pack from **4 → 5** numbers (`p̂z` = grasp height), and lands the **claim timing path** (delay injection, forecast-relative actions, close-speed helpers).

> **Compat warning:** old 4-D checkpoints and floor-mode datasets are **not** bit-compatible. Retrain / recollect after this lands.

---

## How to read this report

Each change is marked:

| Tag | Meaning |
|---|---|
| **HIGHLIGHT** | Core behavior change — read first |
| **NEW** | New file |
| **REMOVED** | Deleted API / file / mode |
| **EFFECT** | What breaks or improves if you run it |

---

# 1. Sim world — house-only, no bounce

## **HIGHLIGHT** Floor / bounce mode removed

| Item | Before | After |
|---|---|---|
| `scene.use_house` | Flag: house vs floor slab | **Gone** — house is always on |
| Object motion | Bounce off XY walls in floor mode | **Coast straight**; miss leaves the table |
| Weld default | Weld on floor, PhysX in house | **PhysX by default**; weld only if `use_weld_grasp: true` |
| Bounce guard in SM | Wait for a straight run before close | **Removed** (`bounce_guard_s`, `_bounce_clear`, `_time_to_wall_s`) |

**Files:** `sim/scene.py`, `sim/motion.py`, `sim/state_machine.py`, `sim/phase0_cfg.yaml`, `sim/video.py`, `perception/kalman.py`, `scripts/g2_perception_report.py`

**EFFECT**
- One world only: house USD + throw toward bowl.
- Kalman no longer has bounce-NIS / `bounce_accel_std`.
- G2a report no longer splits “near wall hit” vs calm windows.
- Old floor-mode YAML keys (`table_height_mm`, `table_size_mm`, `object_still_cm_s`, `min_wait_after_bounce_ms`) are dropped from unit normalize.

---

## **HIGHLIGHT** Control rate 25 Hz → 30 Hz

```yaml
# sim/phase0_cfg.yaml
sim.dt_ms: 33.333333   # was 40 → 25 Hz
static_camera.fps / wrist_camera.fps: 30  # labels only; real rate = 1/dt
```

**EFFECT**
- Expert, cameras, logs, and claim timing all tick at **30 Hz**.
- With `physics_substeps: 4` → PhysX at **120 Hz** inside each control step.
- Camera `fps` in yaml is a **label** for videos/logs, not Isaac `CameraCfg` rate.

---

## **HIGHLIGHT** Bowl / container layout defaults moved

| Key | Before (mm) | After (mm) |
|---|---|---|
| `container_xy_mm` | `[500, 300]` | `[350, 420]` |
| `container_xy_clip_mm` | `[[450,580],[260,340]]` | `[[300,400],[410,450]]` (seed; layout overwrites) |
| `house_layout` defaults | `container_x=0.50`, `y=0.35` wait — was `y=0.30` | `container_x=0.43`, `container_y=0.35` |

**Files:** `sim/phase0_cfg.yaml`, `sim/house_layout.py`, `sim/scene.py` (passes yaml `container_xy` into `compute_play_layout`)

**EFFECT**
- Bowl sits farther on the table Y side; layout may still clip to the table inner half.
- Episode meta always includes house fields; `use_house` / `bounce` flags removed from meta.

---

## **HIGHLIGHT** Grasp height (`p̂z`) + close at object centre

**NEW** `interfaces/grasp_height.py` (~300 lines)

- `z_grasp = max(z_top − g, z_table + c)`
- Collect samples while hand ≥ 20 cm above top; **lock median** when hand &lt; 10 cm
- Dual-cam helpers: wrist depth + static depth → `fuse_z_top`
- Expert default: **`close_at_object_center: true`** → close / pack Z = object centre (fixes mid-air close from bad depth latch ~18 cm vs fruit ~3.6 cm)

**Wired in**
- `PickPlaceStateMachine._update_z_grasp` / `_grasp_gate_ok(..., goal_z=...)`
- Z gate tol = `min(grasp_z_tol, s_z/2)` (floor 2 mm)
- `set_object_half_height(half_h, z_table=...)` each episode from scene meta
- Collect passes wrist + static depth into `expert.step`
- Static cam now requests **`depth`** in yaml

**EFFECT**
- Approach Z is latched grasp height, not raw GT centre alone (unless centre mode).
- Close gate compares hand Z to `z_grasp`, not only object centre.
- Depth path kept for later perception when `close_at_object_center: false`.

---

## Scene timing stamp (Step 1)

| Addition | Role |
|---|---|
| `Phase0Scene._last_capture_t0` | Sim time of last camera update |
| `last_capture_t0` / `get_image_packet()` | Frames + stamp together |
| `get_static_depth()` / `static_cam_z()` | Static RGB-D for Z fuse |
| Meta `table_z` | Expert latch uses real table height |

**EFFECT**
- Delay injection can stamp each plan with capture time `t0`.

---

# 2. Claim timing & control stack

## **HIGHLIGHT** New yaml blocks

```yaml
claim_timing:
  enabled: true
  delta_jitter_ms: 10.0          # reported − applied ~ N(0, 10 ms)
  train_delta_min_ms: 0.0
  train_delta_max_ms: 200.0
  eval_delta_ms: 0.0
  enforce_apply_at: false        # hold motions until t0+applied (opt-in)
  use_noisy_gt: false
  close_speed_guard: false

claim_height:
  close_at_object_center: true
  finger_insert_mm / table_clear_mm / z_slack_mm / latch_lo|hi_mm / ...
```

**EFFECT**
- Timing is config-driven; unit normalize converts ms → s in `interfaces/cfg_units.py`.

---

## **NEW** Control modules

| File | Role | Effect |
|---|---|---|
| `control/delay.py` | `DelayInjection` / `DelaySample` — applied vs reported Δ | Train draws Δ ∈ [0, 200] ms; eval fixes Δ; jitter ~10 ms |
| `control/exec_chunk.py` | Relative chunk → world via forecast into `ActionBuffer` | Step 5 executor glue |
| `control/close_guard.py` | `|v_hand − v_obj| ≲ 3.1 cm/s` during close | Optional ride-object XY if speeds diverge |

`control/__init__.py` + `control/README.md` export the new types.

---

## **HIGHLIGHT** Collect loop (`sim/collect.py`)

When `claim_timing.enabled`:

1. Sample delay each step → log `capture_t0`, `applied_delta_s`, `reported_delta_s`
2. Pack uses **reported** Δ (and optional `p_hat_z` from expert)
3. Optional **hold** cmd until `apply_at` (`enforce_apply_at`)
4. Optional **noisy_gt** filter pack
5. Optional **close_speed_guard** during CLOSE
6. Episode sets object half-height + table Z on the expert
7. Task language seeded per episode (`default_language(..., seed=...)`)
8. Logged **action** = planned expert cmd (not the delayed hold)

**EFFECT**
- Demos can teach variable delay without packing Δ as its own feature.
- Holding motions is off by default until proven not to kill expert success.

---

# 3. Policy pack & training

## **HIGHLIGHT** Live pack 4 → 5 numbers

| | Before | After |
|---|---|---|
| Dim | 4 | **5** |
| Layout | `p̂x, p̂y, vx, vy` | **`p̂x, p̂y, p̂z, vx, vy`** |
| State dim | 12 (8 proprio + 4) | **13** |
| Δ in pack? | No (fixed yaml) | Still **no** — Δ baked into XY `p̂` |

**Files:** `interfaces/state.py`, `eval/cond_source.py`, `policy/features.py`, `policy/transforms.py`, `.cursor/rules/40-interfaces-code-budget.mdc`

**EFFECT**
- Height differences show up in `p̂z`, not as a relative `d_z` change.
- Zero-conditioning / sanity shifts must use indices **3:5** for velocity (was 2:4).
- **Must retrain** after this change.

---

## **NEW** Forecast + relative actions

| File | Role |
|---|---|
| `interfaces/forecast.py` | `p̂_k = p + v·(Δ + k·dt)`; Z held at grasp height |
| `policy/relabel.py` | `d_k = target_k − p̂_k` and inverse; delay-aware demo relabel |

**Transforms / train**
- New action frame: **`forecast`** (with `abs` / `delta`)
- Claim chunk defaults documented: `chunk_size=25`, `n_action_steps=10`
- Train CLI: `--n-action-steps`; yaml defaults wired through `interfaces/config.py`
- Train collate can **paraphrase** task text each batch (`paraphrase_task`)

**EFFECT**
- Model can learn offsets to the object forecast instead of absolute world XYZ.
- Eval: `--forecast-rel on` + `--claim-timing on`.

---

## **NEW** Language templates

**NEW** `policy/configs/instructions.yaml`  
**Changed** `policy/features.default_language` / `paraphrase_task`

**EFFECT**
- Object noun stays in the sentence (apple ≠ tangerine); many wordings so the net cannot memorize one frozen phrase.

---

# 4. Perception & eval

## Kalman — bounce path removed

**REMOVED** from `perception/kalman.py`: `bounce_accel_std`, `bounce_nis`, `n_bounces`, bounce re-predict.

**EFFECT**
- Filter assumes coast / constant velocity only — matches house throws.

## **NEW** `perception/noisy_gt.py`

Noise on GT XY (static 4 mm / wrist 2 mm when close) + short windowed CV filter.

**EFFECT**
- Optional train-time “camera-like” pack without running the real detector.

## Eval policy (`scripts/eval_policy.py`)

| Flag | Effect |
|---|---|
| `--claim-timing on` | DelayInjection reported Δ into pack |
| `--forecast-rel on` | Decode XYZ relative to forecast `p̂` |
| ZLatch | Same latch idea as expert for `p̂z` |

## G2 report cleanup

Bounce / wall-hit splits and bounce Kalman knobs removed from `scripts/g2_perception_report.py`.

## Scripts defaults

| Script | Change |
|---|---|
| `run_phase0.py` | Default out → `data/smoke`; seed 41; `--debug` default **true**; no bounce-guard log |
| `run_phase1_collect.py` | Default out → `data/smoke` |
| **REMOVED** `scripts/select_box_episodes.py` | Deleted (231 lines) |

## **NEW** helper / smoke scripts

| File | Purpose |
|---|---|
| `scripts/smoke_claim_grid.py` | Claim timing grid smoke |
| `scripts/smoke_pad_conditioning_4to5.py` | Pad old 4-D → 5-D |
| `scripts/adjust_table_layout.py` | Interactive / offline table layout tweak |
| `tools/diag/depth_z_error.py` | Depth vs GT Z diagnostics |
| `tools/diag/verify_camera_hz.py` | Camera rate check |
| `tools/diag/verify_z_latch.py` | Z latch verify |

---

# 5. Plan, rules, tests

## `foresight_plan.md` — rewritten

Replaced stub with:

- Claim success grid (speed × delay) vs no-compensation / fixed 0.25 s / measured-Δ forecast
- Done checklist Steps 1–5 + Z height
- Remaining / landed evening notes (eval flags, enforce delay, noisy_gt, dual-cam Z, language)

## Rules

| File | Change |
|---|---|
| `.cursor/rules/40-interfaces-code-budget.mdc` | Live pack = **5** numbers; claim Δ path |
| **NEW** `.cursor/rules/25-doc-new-code.mdc` | Require purpose + usage comments on new callables |

## Tests updated / added

**Modified:** `tests/test_cfg_units.py`, `test_house_layout.py`, `test_perception.py`, `test_policy_frames.py`, `test_sim_substeps.py`, `test_state_machine.py`, `scripts/test_state.py`

**NEW:** `tests/test_claim_steps.py`, `test_delay_injection.py`, `test_grasp_height.py`, `test_instructions.py`

**EFFECT**
- SM tests drop bounce / floor assumptions; substeps no longer force floor → 1.

---

# 6. Change map (quick index)

### Modified (tracked)

```
.cursor/rules/40-interfaces-code-budget.mdc
README.md
control/README.md, control/__init__.py
eval/cond_source.py
foresight_plan.md
interfaces/cfg_units.py, config.py, state.py
perception/kalman.py
policy/configs/smolvla.yaml, features.py, train_smolvla.py, transforms.py
scripts/eval_policy.py, g2_perception_report.py, run_phase0.py, run_phase1_collect.py, test_state.py
sim/README.md, STATE_MACHINE.md, collect.py, house_layout.py, motion.py,
    phase0_cfg.yaml, scene.py, state_machine.py, video.py
tests/test_*.py (cfg_units, house_layout, perception, policy_frames, sim_substeps, state_machine)
```

### Deleted

```
scripts/select_box_episodes.py
```

### Untracked (new)

```
.cursor/rules/25-doc-new-code.mdc
control/close_guard.py, delay.py, exec_chunk.py
interfaces/forecast.py, grasp_height.py
perception/noisy_gt.py
policy/configs/instructions.yaml, relabel.py
scripts/adjust_table_layout.py, smoke_claim_grid.py, smoke_pad_conditioning_4to5.py
tests/test_claim_steps.py, test_delay_injection.py, test_grasp_height.py, test_instructions.py
tools/diag/depth_z_error.py, verify_camera_hz.py, verify_z_latch.py
```

---

# 7. Downstream effects checklist

| If you… | Then… |
|---|---|
| Load an old 4-D checkpoint | **Mismatch** — state is 13-D now; retrain or pad (see `smoke_pad_conditioning_4to5.py`) |
| Expect floor bounce demos | **Gone** — only house coast |
| Run Phase-0 with old yaml | Drop `use_house` / table slab / bounce keys; use new `claim_*` blocks |
| Collect with defaults | 30 Hz, PhysX grasp, centre close, claim timing **logged** but `enforce_apply_at` **off** |
| Train claim path | Prefer `chunk_size=25`, `n_action_steps=10`, optional `action_frame=forecast`, language paraphrase on |
| Eval claim path | `--claim-timing on` and/or `--forecast-rel on`; zero-out still proves the pack is used |
| Tune Kalman for walls | Bounce knobs removed — coast-only |

---

# 8. Suggested verification (not run by this report)

1. Unit: `pytest tests/test_delay_injection.py tests/test_grasp_height.py tests/test_claim_steps.py tests/test_state_machine.py`
2. Isaac smoke: short `run_phase0.py` → confirm close at fruit centre, no mid-air latch
3. Collect one episode → check log keys `capture_t0`, `applied_delta_s`, `reported_delta_s`, conditioning shape `(5,)`
4. Train/eval only after confirming 5-D features in the LeRobot dataset meta

---

*Generated from `git status` / `git diff` against `origin/fo/v1/dev` (`6264413`). Uncommitted only.*
