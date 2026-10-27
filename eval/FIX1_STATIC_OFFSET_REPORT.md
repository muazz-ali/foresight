# Fix 1 — static hand–object offset

**All expert demos recorded before this fix are invalid as training labels.**

## Verdict

Settled hand (before CLOSE) already sits on the object: geometric
`b_world` / `b_hand` / `g(pose)` are **negligible** (≤0.1 mm).

The B5 “static −1.6 mm in x at close” was **not** a wrong TCP or object
centre. With `close_settle_ticks: 1` the expert closed on the **first**
tick inside the 2 mm box while DiffIK was still approaching (~1.65 mm
short). Fix: keep `close_xy_ok = 2 mm` and the speed gate unchanged; set
`close_settle_ticks: 15` so CLOSE waits ~0.5 s inside that box.

Also: pinch point = URDF pad midpoint (Z=103.4 mm). Do **not** midpoint
`panda_leftfinger` / `panda_rightfinger` (those are joint origins at
58.4 mm).

## Hypothesis fit (108 settles, v=0, Δ=0)

| Term | Value (mm) |
|---|---|
| `b_world` XY | **[0.066, 0.007]** |
| `b_hand` XY | **[0.008, −0.005]** |
| Residual RMS | **0.023** |
| Residual std \|e\| | 0.014 |
| Slope residual vs r (mm/m) | −0.062 (no `g(pose)`) |
| Dominant | **negligible** |
| `tcp − pinch` (hand frame, mm) | **[0, 0, 3.60]** |

Yaw / pose trends: slope of \|e\| vs yaw is tiny; 95% CI for slope vs r
includes 0 → no pose term.

## Chosen fix

1. **`state_machine_params.close_settle_ticks: 15`** — operational fix for
   B5 early-close on the DiffIK approach (2 mm / speed gate untouched).
2. **`sim/pinch_point.py` + `Phase0Scene.get_pinch_point`** — pad midpoint
   from URDF / fingertip bodies only (not finger-joint links).

No TCP Z retarget (107→103.4): palm-down that offset is pure world −Z, not
the XY bias B5 saw. No object-reference shift (`b_world` ~0). No joint
integral (`g` not needed).

## Pass bar (before → after)

Settled measure is the same before/after settle-ticks (script never
closes). “Before” B5 = AB_recordings_v5; “after” = this folder’s
`b5_recheck/`.

| Metric | Before | After | Bar |
|---|---|---|---|
| Mean e_x / e_y per object (108 settles) | — | egg 0.07/0.01, lemon 0.07/0.01, onion 0.07/0.01 mm | \|mean\| ≤ 0.3 mm |
| RMS e (all 108) | — | **0.076 mm** | ≤ 0.5 mm |
| Max \|e\| | — | **0.096 mm** | ≤ 1.0 mm |
| Trend vs ψ / pose | — | slope vs r CI includes 0 | CI includes 0 |
| B5 static mean d_x at close | **−1.61…−1.68 mm** (AB v5) | **−0.18…−0.22 mm** | within ±0.3 mm |

B5 recheck detail (`b5_recheck/`): egg −0.18, lemon −0.22, onion −0.18 mm
(n=4–5 each). **PASS.**

## C30 cross-check (diagnostic only, Δ=0)

Hand−object at matched speed (min \|Δv\| in intercept). Sign: +along =
hand ahead of object.

| Seed | Along mm | Cross mm | \|Δv\| cm/s |
|---|---|---|---|
| 1001 | −2.10 | −1.88 | 0.05 |
| 1003 | −7.77 | −1.65 | 0.43 |

Same numbers as B1_grid_v7 H5 (along unchanged to ≪0.5 mm → no Fix1/Fix2
coupling). Cross stays ~1.9 mm — **not** removed by this fix: settled
static e≈0, so that cross-track is not the B5 early-close bias. Leave for
Fix 2 / tip lag.

## Files changed

- `sim/phase0_cfg.yaml` — `close_settle_ticks: 15`
- `sim/pinch_point.py` — URDF pad midpoint helpers
- `sim/scene.py` — `get_hand_pose` / `get_pinch_point` / `get_tcp_minus_pinch_hand_m`
- `scripts/fix1_static_offset.py` — 108-settle measure + fit
- `eval/run_fix1_measure.sh`, `eval/launch_fix1_static_offset.sh`, `eval/run_fix1_b5_recheck.sh`
- `tests/test_pinch_point.py`, `tests/test_state_machine.py` (settle-ticks test)

## Commit

`0db0c8fb34786087bdf2278bc7090f91aefef684`

Tracked copy of this report: `eval/FIX1_STATIC_OFFSET_REPORT.md`
(data/ is gitignored).
