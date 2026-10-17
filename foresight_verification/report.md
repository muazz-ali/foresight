# Foresight shortcut-model verification

Evidence only. No training. No long Isaac runs.
Checkpoint: `runs/p3_throw/pretrained_model`. Seed=0. Units: **cm** / **cm/s**.

Agents: [A](c8d3cc59-b732-42ba-bad9-65b7713eb911) · [B](47af6406-bb68-4edf-a35f-4d70ea951655) · [C](03dd0967-ae24-45b5-b889-5c65c79bc2b8) · [D](e69fa12a-85b5-4814-a228-a12389bb8a5e) · [E](c1c1430c-db0f-4346-bfae-3ba06d209aa5)

## 1. Summary table

| Quantity | Predicted (B8 / model) | Measured |
|---|---|---|
| d_x / d_y (pooled k=1..8) | — | **0.793** / **0.821** |
| d_x (k=8 pooled / speed0) | ~2.1 (story) | **1.528** / **2.156** |
| σ_u_x / σ_u_y | — | 1.318 / 1.315 |
| σ_n_x / σ_n_y | ~1.5 (train) | 1.565 / 1.586 |
| w_h_x / w_n_x | **0.585** / **0.415** | S below |
| e*_x / e*_y | 1.119 / 1.195 | C3 close miss grows with speed |
| α_x / α_y (C1) | — | 0.105 / 0.179 |
| α(1−w_h) x/y | ∈ (0,2) | 0.044 / 0.073 |
| S_h_x @δ=1.5 cm, chunk 0 | w_h≈0.585 | **1.081±0.175** |
| S_p_x @δ=1.5 cm, chunk 0 | w_n≈0.415 | **0.045±0.034** |
| S_h_y / S_p_y | 0.593 / 0.407 | 1.050±0.081 / 0.058±0.094 |

### C3 oracle ‖hand−object‖ at close (cm)

| speed (m/s) | n | mean ‖·‖ | mean x | mean y |
|---:|---:|---:|---:|---:|
| 0.00 | 20 | 1.84 | 0.96 | 0.00 |
| 0.10 | 20 | 3.26 | 2.07 | -1.40 |
| 0.15 | 20 | 4.17 | 1.64 | -1.78 |
| 0.20 | 20 | 7.58 | 4.51 | -5.27 |

Late-window hand−future mean (cm): **(1.05, -2.92)**. End-at-close: **(2.59, -4.08)**. Settles to constant? **False**.

**Verdict:** The net is a near-pure hand copycat (**S_h≈1**, **S_p≈0**). B8’s optimal mixture (w_h≈0.59) under-states the shortcut. Fix class = change the data (DART / hand noise), not the architecture.

## 2. UNKNOWN and FOR_HUMAN

### UNKNOWN
- A10: Full XY AABB along vs across throw direction per mesh (not in episode meta/H5; needs usd_bbox_size).
- A11: Exact object×commanded speed_m_s crosstab (no source_h5.jsonl under data/lerobot/p3_throw/meta/).

### FOR_HUMAN
- E1.empirical_sweep: SKIPPED_NEED_APPROVAL — Geometric estimate only in this part. No sim executed.
- E4: ['What target τ (mm or cm) should DART use — catalog min ~2.5 mm, orange/tangerine ~5 mm, or a softer category-conditional τ?', 'Keep p̂ / pack train noise at ±1.5 cm (smolvla.yaml), or retune once DART command noise is on?', 'What is the max acceptable executed-command perturbation (σ_δ or hard clip) so expert success / demos stay clean?', 'Approve E1 empirical sweep (n=20×6 offsets, Isaac, likely >10 min) before claiming measured τ?', 'When part_c.json lands, plug α into σ_c = σ_δ·√((2−α)/α) and choose white vs OU for collect perturbation.'] — 
- E1 empirical τ sweep (20×6 still-object offsets in Isaac): needs approval (>10 min).
- E4: choose target τ; keep p̂ noise ±1.5 cm?; max executed perturbation; white vs OU.

## 3. Contradictions / caveats vs the mathematical model

- d depends on k: pooled k=1..8 d_x≈0.79 cm vs k=8 pooled d_x≈1.53 cm vs speed0 k=8 d_x≈2.16 cm — single-d model is k-sensitive.
- Measured S_h_x≈1.081 ≫ w_h≈0.585; S_p_x≈0.045 ≪ w_n≈0.415 — optimal mixture ≠ learned policy (near-pure copycat).
- C3 last-50 hand−future does not settle to constant offset (settles=False).
- C1 x-axis first-order R²≈0.299 — weak P-gain model on x.
- No conveyor belt (Part A): throw/slide in XY; fingers open along world Y.
- A6 confirms labels are commanded EE targets (matches model y:=action).
- A5 confirms independent Gaussian p̂/v̂ noise after lookahead (matches η_n; not rebuilt from noisy v̂).
- C3 close ‖hand−object‖ grows with speed (1.8→7.6 cm at 0→20 cm/s) — overshoot worsens when motion is hard.

## Partials & scripts

| Part | JSON | Script |
|---|---|---|
| A | `partials/part_a.json` | code cites |
| B | `partials/part_b.json` | `scripts/part_b_dataset_stats.py` |
| C | `partials/part_c.json` | `scripts/part_c_controller_eval.py` |
| D | `partials/part_d.json` | `scripts/part_d_sensitivity.py` |
| E | `partials/part_e.json` | `scripts/part_e_dart_notes.md` |

`report.json` omits D per-frame rows and bulky D tables (see `partials/part_d.json`).
