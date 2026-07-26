# AGENTS.md — Foresight lab contract

Multi-agent operating manual for this repo. Humans and Cursor agents both follow it.

## Project in one line
Explicit-future dynamic manipulation: Kalman-predicted object state → small VLA → reflex grasp. Plan: `foresight_plan.md`. Isaac connect: `ppt.md` (launch/cameras/step loop only).

## Ownership (post-G0)
**Sim scene, scripted expert SM, success criterion, and collection run from this repo (`sim/`, `scripts/`).** DynamicVLA is not a runtime dependency. Wrapping their SM for G0 produced a flat ~40% ceiling — do not resume that path.

## External dependencies
| Path | Role |
|---|---|
| `~/Desktop/IsaacLab` | Symlink → `muazzam/IsaacLab`. Editable Isaac Lab install. Verify APIs here. |
| `ppt.md` | Headless AppLauncher / camera / step-loop facts only. |
| `~/Desktop/muazzam/{objects,scenes}` | USD assets (read). |
| `~/Desktop/muazzam/DynamicVLA` | Optional *read-only* reference for launch patterns. **Not** expert SM / termination / latch. |
| `~/Desktop/muazzam/IsaacLab` | Canonical Isaac Lab tree. |

Conda: `dynamicVLA_isaac` for sim; separate env for LeRobot/training. Never mix.

**Import rule:** smoke-test any external module once before wiring it into Foresight. Fix breaks first.

## Suggested layout (create only when needed)
```
interfaces/     # frozen state + conditioning schemas
sim/            # Isaac scene, OUR expert SM, collection
perception/     # detector, tracker, Kalman, 3D lift
policy/         # SmolVLA / LeRobot glue
control/        # executor buffer + reflex
eval/           # sweeps, gates, taxonomy
```
Do not invent extra top-level packages.

## Workstreams
Tag every subagent Task with `WORKSTREAM: <tag>`.

| Tag | Deliverable | Gate focus |
|---|---|---|
| `sim-scene` | In-repo scene + Foresight expert SM + logging | G0 |
| `perception` | Track → filter → predict | G2 |
| `policy` | A vs B training + Δ/noise aug | G1 |
| `control` | Buffer + reflex + coast/safety | G3 |
| `eval` | Speed curve, RMSE, taxonomy | all gates |
| `explore` | Read-only investigation | — |

### Handoff checklist
1. Own only your directory (+ agreed interface files).
2. Preserve `position, velocity, covariance, timestamp, valid`.
3. Return: what changed, how verified, residual risks.
4. Do not start the next phase's infrastructure because "we'll need it."
5. Do not reintroduce DynamicVLA as the Phase-0 controller.

## Parallelism
- Independent workstreams → launch in parallel.
- Shared interface change → one owner, others wait or read-only.
- Parent merges; parent runs the gate checklist.

## Non-negotiables
- No hallucination of Isaac APIs — verify in `ppt.md` / IsaacLab source.
- Prefer few lines + off-the-shelf libs over new frameworks.
- Never commit unless a human asks.
- Stop on red gates; diagnose before adding complexity.
- Runtime code path must work from this repo without DynamicVLA on `PYTHONPATH` for the expert.
