# Part E — DART design inputs

Evidence-only. No production edits. No Isaac run.

## E1 — Grasp tolerance τ

**Formula.**  
`τ_geom = (opening − width) / 2`

| Knob | Value | Source |
|---|---|---|
| Opening | **80 mm** | `robot.gripper_open_mm: 40` → Franka open ≈ `2 × gripper_open` (`sim/phase0_cfg.yaml` L52–95; `sim/scene.py` L407–410) |
| Width gate | **≤ 75 mm** | `scene.max_object_width_mm` |
| Width used | `max(sx, sy)` of USD bbox | same as `sim/asset_filters.horizontal_width_m` |
| Finger / throw axis | world **Y** | cfg L99–101; house throws often aim at the bowl (±Y) |

`sim/physics_grasp.py` has no size or τ — only grip material and wrist depth.  
`muazzam/objects/metadata.json` is tags only. Sizes come from existing Isaac logs (`Skip object` / `Object mesh` lines under `data/**/*.log`).

**Related but not τ:** expert close gate `grasp_xyz_ok_mm: [10, 10, 15]` (`phase0_cfg.yaml` L155).

### Geometric τ by category (fitting meshes only)

| Category | n | width mm (med) | τ_geom mm (min / med / max) |
|---|---:|---:|---|
| apple | 7 | 67 | 3.0 / 6.5 / 9.5 |
| avocado | 6 | 51.5 | 11.5 / 14.2 / 18.0 |
| can | 10 | 58.5 | 2.5 / 10.8 / 14.5 |
| egg | 11 | 41 | 17.5 / 19.5 / 22.0 |
| kiwi | 2 | 49 | 12.5 / 15.5 / 18.5 |
| lemon | 11 | 45 | 12.0 / 17.5 / 23.5 |
| lime | 3 | 59 | 9.5 / 10.5 / 12.5 |
| onion | 7 | 47 | 2.5 / 16.5 / 24.5 |
| orange | 5 | 70 | 4.0 / 5.0 / 6.5 |
| peach | 5 | 70 | 2.5 / 5.0 / 10.0 |
| tangerine | 2 | 70 | 5.0 / 5.0 / 5.0 |

**Catalog:** 69 fitting meshes. Global τ_geom **min 2.5 / median 13.0 / max 24.5 mm**.  
Tightest: `fcan15`, `onion07`, `peach02` (τ ≈ 2.5 mm).

### Empirical sweep

**Status: `SKIPPED_NEED_APPROVAL` / FOR_HUMAN**  
Do not run Isaac here. Proposed later: n=20 × 6 expert-offset bins (likely >10 min).

---

## E2 — Patch proposal (not applied)

Split **executed** command vs **logged** label in `sim/collect.py` (not inside the state machine).

| Step | File:line | Today |
|---|---|---|
| Expert target | `sim/state_machine.py:709` | returns clean `ExpertCommand` |
| Expert call | `sim/collect.py:294` | `cmd = expert.step(...)` |
| Execute | `sim/collect.py:300` | `scene.set_ee_target(cmd.position, ...)` |
| Label | `sim/collect.py:326–328` | logs same `cmd.position` |

**Proposal:** keep `label_pos = cmd.position.copy()`, add zero-mean noise to a copy for `set_ee_target`, log the clean label. Full text: `foresight_verification/scripts/part_e_dart_notes.md`.  
Callers already go through `run_episode`: `scripts/run_phase0.py`, `scripts/run_phase1_collect.py`.

---

## E3 — White noise ↔ command noise (α from part_c)

`part_c.json` is **missing** → **α = UNKNOWN**.

White per-frame link:

\[
\sigma_\delta^2 = \sigma_c^2 \cdot \frac{\alpha}{2-\alpha}
\quad\Rightarrow\quad
\sigma_c = \sigma_\delta \sqrt{\frac{2-\alpha}{\alpha}}
\]

| Target σ_δ (cm) | σ_c needed |
|---|---|
| 1.5 | `1.5 × √((2−α)/α)` |
| 3.0 | `3.0 × √((2−α)/α)` |
| 5.0 | `5.0 × √((2−α)/α)` |

**OU note:** without α, do not pick a white σ_c. Prefer OU / colored noise that matches smooth filter error, or wait for part_c α. Train pack noise today is independent white **±1.5 cm** pos (`policy/configs/smolvla.yaml`).

Illustrative only (not measured α):

| α | σ_c for σ_δ=1.5 | 3.0 | 5.0 cm |
|---|---:|---:|---:|
| 0.05 | 9.37 | 18.74 | 31.23 |
| 0.10 | 6.54 | 13.08 | 21.79 |
| 0.20 | 4.50 | 9.00 | 15.00 |
| 0.50 | 2.60 | 5.20 | 8.66 |

---

## E4 — FOR_HUMAN

1. Target τ — catalog min ~2.5 mm, orange/tangerine ~5 mm, or category-conditional?
2. Keep p̂ noise ±1.5 cm after DART, or retune?
3. Max acceptable executed-command perturbation (σ_δ or hard clip)?
4. Approve E1 empirical sweep (n=20×6, Isaac, >10 min)?
5. When part_c lands, plug α into the σ_c formula and choose white vs OU.

## E3 updated from Part C α

- α = (x=0.1051, y=0.1787, z=0.1626)

- **x**: σ_δ=1.5→σ_c=6.369 cm, σ_δ=3.0→σ_c=12.737 cm, σ_δ=5.0→σ_c=21.229 cm
- **y**: σ_δ=1.5→σ_c=4.789 cm, σ_δ=3.0→σ_c=9.579 cm, σ_δ=5.0→σ_c=15.965 cm
- **z**: σ_δ=1.5→σ_c=5.042 cm, σ_δ=3.0→σ_c=10.085 cm, σ_δ=5.0→σ_c=16.808 cm

x-axis first-order R² is weak (~0.3); treat x σ_c as approximate.

