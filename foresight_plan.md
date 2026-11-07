Move on to the real robot only when all four conditions hold: (2026-10-24)

| Speed / delay      | No compensation | Fixed look-ahead 0.25 s | Forecast at measured Δ |
|--------------------|-----------------|-------------------------|------------------------|
| 0.10 m/s, 100 ms   | 0%              | 0%                      | 100%                   |
| 0.10 m/s, 300 ms   | 0%              | 50%                     | 98%                    |
| 0.20 m/s, 100 ms   | 0%              | 0%                      | 97%                    |
| 0.20 m/s, 300 ms (unseen)  | 0%              | 3%                      | 93%                    |
| 0.30 m/s, 200 ms (unseen)  | 0%              | 0%                      | 88%                    |
| 0.30 m/s, 300 ms (unseen)  | 0%              | 0%                      | 84%                    |


The Foresight policy reaches at least 80% success in every cell of the grid, including the unseen cells (30 cm/s or 300 ms).
Its success in unseen cells is within 10 percentage points of its success in seen cells.
The no-compensation and fixed look-ahead policies fall off.
In the shift test, the hand follows a deliberate shift of p̂ with a ratio between 0.8 and 1.2.


## Claim scope (2026-10-29)

Objects: **egg, kiwi, lemon, lime, onion** (low fruit). Tall fruit (apple,
avocado, orange, peach, tangerine; half-height ≥ ~35 mm) are **out of scope**
for the timing claim: during the ride their top touches the gripper underside,
pushing the hand up 5–8 mm and dragging it 1–4 mm behind (C30 v5,
`data/eval/C1_c30_all_objects_v5_joint_vel_ff`; contact probe: +12 mm hand
height removes it). Fixing that is grasp geometry, not timing.

Throws: a spawn filter (`sim/ride_ik_check.py`, `scene.spawn_j7_margin_rad`)
re-draws any throw whose planned ride brings the wrist (joint 7) within 0.3 rad
of its limit; re-draws are logged per episode (`spawn_redraws`). Avocado seed
1009 (C30 v5) is dropped with the tall fruit.

Expert tracking accepted at ≤ ~1 mm (2 mm close gate, 5 mm pre-shape slack).
Success = lifted + held (1 s from LIFT) + placed; close-time checks are
diagnostics only (`eval/b1_success.py`).


Hypothesis: explicit future-state from a physics filter beats implicit/latent
predictors for dynamic grasping — smaller, cheaper, easier to debug.



### Done (libs + unit / Isaac smoke)
- Step 1: 30 Hz + DelayInjection (applied vs reported Δ); collect logs t0 / Δ
- Step 2: noisy_gt + WindowedCVFilter (library)
- Step 3: forecast + 5-D pack `p̂x,p̂y,p̂z,vx,vy` (Δ baked into XY p̂)
- Step 4: relative actions `d = target − p̂`
- Step 5: ActionBuffer, exec_chunk, close_speed_check helpers
- Z height: collect ≥20 cm (median), lock & lt;10 cm; Isaac verify PASS

### Remaining / landed (2026-10-24 evening)
1. **Wire into eval** — landed: `--claim-timing`, `--forecast-rel`, ZLatch p̂z,
   5-D `choose_pack` (`scripts/eval_policy.py`).
2. **Enforce delay on motion** — landed: `apply_delta_delay_to_robot`
   (`sim/cfg/training.yaml`; folded into `claim_timing`) holds the cmd until
   `apply_at` in `sim/collect.py` (default off).
3. **Optional noise + close-speed** — landed: `use_noisy_gt`,
   `close_speed_guard` on folded `claim_timing` (default off).
4. **Dual cameras for Z** — landed: static depth in yaml + `fuse_z_top` /
   `z_top_from_static_depth`; expert fuses wrist+static.
5. **Language + vision** — landed: `policy/configs/instructions.yaml` with
   apple vs tangerine templates; `default_language()` loads them.


