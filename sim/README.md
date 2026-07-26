# Phase 0 — Environment + oracle (Franka, in-repo)

## Status
**G0 PASS (2026-07-26):** Franka in-repo expert, 99% overall @ 0–40 cm/s, ~2790 eps/h.
DynamicVLA `PickStateMachine` wrap abandoned (~40% flat). Runtime lives here.

## Gate G0
Expert ≥70% success at each of 0/10/20/30/40 cm/s; collection ≥100 episodes/hour.

| speed | succ | rate | eps/h |
|---|---|---|---|
| 0.00 | 20/20 | 100% | 3690 |
| 0.10 | 20/20 | 100% | 3694 |
| 0.20 | 20/20 | 100% | 3187 |
| 0.30 | 20/20 | 100% | 2546 |
| 0.40 | 19/20 | 95% | 1842 |

## Layout
| Path | Role |
|---|---|
| `interfaces/state.py` | Frozen `position, velocity, covariance, timestamp, valid` |
| `sim/phase0_cfg.yaml` | Franka-only scene + expert + asset roots |
| `sim/scene.py` | Isaac Lab Franka + table + object + cameras + DiffIK |
| `sim/motion.py` | Lead clamp + kinematic on-table motion |
| `sim/expert.py` | Four-stage pick expert (plan §4) |
| `sim/success.py` | Lift+hold success + taxonomy |
| `sim/collect.py` | Episode loop + oracle/conditioning log |
| `scripts/run_phase0.py` | G0 harness (no DynamicVLA) |

## Run
```bash
conda activate dynamicVLA_isaac
bash scripts/run_g0.sh smoke   # one static episode
bash scripts/run_g0.sh         # full G0 sweep
```

Assets (read-only): `~/Desktop/muazzam/{objects,scenes}`. Default object is a primitive sphere (`use_primitive: true`); set false for USD household assets. IsaacLab: `~/Desktop/IsaacLab`. Connect facts only: `ppt.md`.
