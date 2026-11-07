# C33 cross-cut: park / delay / aim-lead

**Workstream:** eval
**Data:** `/home/gpuadmin/Desktop/foresight/data/eval/C1_c30_c31_c33/cells/C33` (5 egg `.h5`)
**Hypothesis** (`C1_grid_v4/FAILURE_REPORT.md` §3): under delay the hand parks just outside the 10 mm ride box because chase aim residual = `v·dt` ≈ 10 mm at 30 cm/s.

## Verdict: **PROVED**

All 5 seeds arrive ahead; park-band ticks median 25 (report ~22); late chase aim median 12.4 mm ≈ v·dt=10.0 mm (err +2.4 mm), never ≈ v·(Δ+dt)=100 mm; edge gap median 13.6 mm sits on the 10–16 mm park line. In-box ticks stay low (median 5) except s1003 (19) which still never closes.

## Mechanism (code)

- `base_tau = Δ_meas + tip_lag_pad + dt` with `tip_lag_pad_ms=0` (`sim/state_machine.py`).
- Far (outside ride box): `lead = intercept_travel + tau + Δ_meas`. As travel → 0, live-equivalent aim lead → `v·(tip_lag+dt) = v·dt`.
- Inside box: ride with `base_tau`, then soft pull `ride_k=0.55` (or hold XY inside 2 mm close box).
- Parking line and residual ride aim are the same **~+10 mm** line at 30 cm/s — hand approaching from ahead settles on that line and stays out of the box.

## 10 facts

1. C33 slice: 5 egg seeds 1000–1004 @ 30 cm/s Δ=300 ms; 0/5 success (never_closed×4, unreachable×1).
2. Cfg: tip_lag_pad_ms=0, ride_correction_gain=0.55, ride box 10 mm/axis, dt=0.0333s → v·dt=10.0 mm; v·(Δ+dt)=100.0 mm.
3. Arrive ahead: 5/5 (hand ahead when first within 5 cm).
4. Park ticks (stage1, max-axis gap 10–16 mm, ahead, |Δv_along|≤5 cm/s): [24, 25, 25, 29, 7], median 25 (FAILURE_REPORT C33 never_closed median parked=22).
5. In-box ticks (gap≤10 mm): [4, 5, 5, 19, 0], median 5 (report never_closed median in-box=6; s1003 is the outlier with 19).
6. Late intercept aim_along median=12.39 mm vs v·dt=10.00 mm (Δ=+2.39 mm); |aim−v·dt|<5 mm on 5/5 seeds.
7. Late aim never matches v·(Δ_meas+dt)≈100 mm (0/5) — delay cancels in the forecast; leftover lead is the +dt ZOH pad.
8. Edge window (post-catch gap 8–20 mm ahead): median gap_max=13.62 mm; settles_on_park_band=5/5.
9. Per-seed: s1000: park=24, in_box=4, late_aim=10.9, edge_gap=12.7, min_gap=3.6, mode=never_closed | s1001: park=25, in_box=5, late_aim=12.4, edge_gap=13.6, min_gap=2.7, mode=never_closed | s1002: park=25, in_box=5, late_aim=12.1, edge_gap=14.4, min_gap=2.3, mode=never_closed | s1003: park=29, in_box=19, late_aim=12.8, edge_gap=10.7, min_gap=0.2, mode=never_closed | s1004: park=7, in_box=0, late_aim=12.8, edge_gap=15.7, min_gap=12.5, mode=unreachable
10. Hypothesis verdict: PROVED — All 5 seeds arrive ahead; park-band ticks median 25 (report ~22); late chase aim median 12.4 mm ≈ v·dt=10.0 mm (err +2.4 mm), never ≈ v·(Δ+dt)=100 mm; edge gap median 13.6 mm sits on the 10–16 mm park line. In-box ticks stay low (median 5) except s1003 (19) which still never closes.

## Per-seed table

| seed | mode | ahead | park | in_box | late_aim_mm | v·dt_mm | aim−v·dt | edge_gap_mm | min_gap_mm | settle |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | never_closed | True | 24 | 4 | 10.94 | 10.00 | +0.94 | 12.71 | 3.63 | True |
| 1001 | never_closed | True | 25 | 5 | 12.39 | 10.00 | +2.39 | 13.62 | 2.71 | True |
| 1002 | never_closed | True | 25 | 5 | 12.06 | 10.00 | +2.06 | 14.37 | 2.31 | True |
| 1003 | never_closed | True | 29 | 19 | 12.78 | 10.00 | +2.78 | 10.65 | 0.19 | True |
| 1004 | unreachable | True | 7 | 0 | 12.78 | 10.00 | +2.78 | 15.69 | 12.53 | True |

## Predicted lead

| quantity | formula | mm @ 30 cm/s |
|---|---|---|
| one-tick leftover | `v · dt` | 10.00 |
| full `base_tau` from delayed p | `v · (Δ + tip_lag + dt)` | 100.00 |
| observed late aim_along (median) | `(action−object)·û` | 12.39 |

**Reading:** observed aim ≈ `v·dt`, not `v·(Δ+dt)`. Delay cancels when forecasting from delayed state; the `+dt` ZOH pad remains as a ~10–14 mm parking line on the ride-box edge.

## Open questions

1. Why does s1003 get 19 in-box ticks (ride_k working) yet still never_closed — pure close-rule starve (RC1)?
2. Does dropping `+dt` from `tau` / `base_tau` move the park line inside the 10 mm box (FAILURE_REPORT fixes 1–2)?
3. s1004 unreachable: park only 7 ticks and min_gap 12.5 mm — reach-stop before a long park?

Numbers JSON: `/home/gpuadmin/Desktop/foresight/debug_agents/C33/agent_cross_park_aim.json`

