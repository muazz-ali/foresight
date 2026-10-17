# Part D — Inference sensitivity (D1–D5)

- Checkpoint: `runs/p3_throw/pretrained_model`
- Frames probed: **512** (target ≥512); seed=0
- Device: `cuda:0`; micro-batch=16
- Status: `DONE`
- Δ lead = 0.25 s; primary δ=[0.5, 1.0, 1.5] cm; extra δ=3.0 cm
- Chunk indices: [0, 10, 25, 49]; k=1..16 before close
- Frames config: `{'cond_frame': 'abs', 'action_frame': 'abs'}`

## Headline (δ=1.5 cm, chunk 0, pooled)

- **S_h_x_c0**: 1.0808±0.1748 (95% CI [1.0657, 1.0960], n=512)
- **S_h_y_c0**: 1.0502±0.0809 (95% CI [1.0432, 1.0572], n=512)
- **S_p_x_c0**: 0.0447±0.0338 (95% CI [0.0418, 0.0477], n=512)
- **S_p_y_c0**: 0.0583±0.0941 (95% CI [0.0501, 0.0664], n=512)
- **S_both_x_c0**: 1.1267±0.1700 (95% CI [1.1119, 1.1414], n=512)
- **S_v_x_c0**: 0.1435±0.1217 (95% CI [0.1330, 0.1541], n=512)

## Compare to B8 (w_h / w_n)

- status: ok (from `summary_pooled_k1_8`)
- predicted w_h_x / w_n_x = 0.5850301528172548 / 0.4149698471827452
- predicted w_h_y / w_n_y = 0.5926688124397346 / 0.4073311875602655
- measured S_h_x @c0 δ=1.5cm = 1.0808±0.1748 CI [1.0657, 1.0960]
- measured S_p_x @c0 δ=1.5cm = 0.0447±0.0338 CI [0.0418, 0.0477]
- S_h − w_h (x) = 0.49581212064779256
- S_p − w_n (x) = -0.3702390952523196
- Expectation: S_h ≈ w_h, S_p ≈ w_n.

## Noise spread (unperturbed, 5 seeds)

- Mean std a_x @c0: 0.2651557864286699 cm
- Mean std a_y @c0: 0.41699405499994213 cm

## Notes

- Perturbations applied in raw metres (and m/s for v̂) before policy normalization.
- Flow-matching noise fixed per frame across all D1–D5 cases; 5 seeds only for unperturbed spread.
- S_v uses lead-equivalent denominator δ (v̂ shifted by ±δ/Δ).
- δ=3.0 cm table is under `tables.delta_3.0cm` (outside train noise support).
- Full per-k / per-speed grids: see `part_d.json`.
