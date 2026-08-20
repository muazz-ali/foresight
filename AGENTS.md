# AGENTS.md — Foresight lab contract

How humans and Cursor agents work in this repo. Plain words: `WORDS.md`.

## Project in one line
Tell a small policy where a moving object will be (physics prediction) → grasp. Plan: `foresight_plan.md`. Isaac tips: `ppt.md` (launch / cameras / step loop only).

## Ownership (after Gate G0)
**Scene, scripted expert, success rules, and demo recording live in this repo (`sim/`, `scripts/`).** Do not depend on DynamicVLA at runtime. Wrapping their pick state machine for G0 topped out ~40% — do not go back to that.

## External dependencies
| Path | Role |
|---|---|
| `~/Desktop/IsaacLab` | Symlink → `muazzam/IsaacLab`. Editable Isaac Lab install. Verify APIs here. |
| `ppt.md` | Headless AppLauncher / camera / step-loop facts only. |
| `~/Desktop/muazzam/{objects,scenes}` | USD meshes (read). |
| `~/Desktop/muazzam/DynamicVLA` | Optional *read-only* tips for launch. **Not** our expert or success latch. |
| `~/Desktop/muazzam/IsaacLab` | Canonical Isaac Lab tree. |

Conda: `dynamicVLA_isaac` for sim; separate env for training. Never mix.

**Import rule:** try any outside module once before wiring it in. Fix breaks first.

## Suggested layout (create only when needed)
```
interfaces/     # shared object message + future-number helper
sim/            # Isaac scene, OUR scripted expert, recording
perception/     # detect, track, predict (later)
policy/         # SmolVLA / LeRobot glue
control/        # action queue + fast grasp helper (later)
eval/           # speed tests and pass/fail scores
```
Do not invent extra top-level packages.

## Workstreams
Tag every subagent Task with `WORKSTREAM: <tag>`.

| Tag | Deliverable | Gate focus |
|---|---|---|
| `sim-scene` | In-repo scene + Foresight expert + logging | G0 |
| `perception` | Track → filter → predict | G2 |
| `policy` | A vs B training + look-ahead / noise aug. Pack **Δ**. No latent WM. | G1 |
| `control` | Buffer + reflex + coast/safety | G3 |
| `eval` | Speed curve, RMSE, **policy** labels (`never-glued / glued-no-lift / …`). n=20 scout; n≥80 to claim G1. | all gates |
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
