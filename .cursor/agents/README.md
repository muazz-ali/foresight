---
name: README
model: inherit
---

# Foresight Cursor agents — quick roles

Use with Task tool. Always include `WORKSTREAM: <tag>` in the prompt.

## sim-scene
In-repo Isaac scene, cameras, **Foresight** four-stage scripted expert (plan §4), episode logging. Gate G0.
Read: `foresight_plan.md` §4, `ppt.md` (connect only), `.cursor/rules/15-own-the-stack.mdc`.
Do **not** drive episodes through DynamicVLA `PickPlaceStateMachine` / `simulate`.

## perception
Detector → tracker → Kalman → 3D lift. Same interface as oracle. Gate G2.
Budget: ~430 lines total across detector/tracker/lift/filter.

## policy
SmolVLA/LeRobot: Model A vs B; fixed Δ from yaml + filter noise. Gate G1.
Do not rewrite the backbone — append the **4-number** pack (XY p̂, XY v̂).
Δ stays in yaml (`conditioning_delta_s`), not in the vector, while it is constant.
Do not add a latent world model. G1 scout holes: unglue-on-open, `{0,4}` labels, n=20.

## control
Timestamped action buffer, reflex PD, coast/safety. Gate G3.

## eval
Speed sweeps, prediction RMSE, failure taxonomy, sample-efficiency curves.
Never claim a gate without numbers. Log `max_sm` / lift stats with taxonomy.
Policy eval: do not run the expert classifier on `{0,4}` glue. n=20 is scout; n≥80 to claim G1.

## explore
Read-only. Paths + citations only. DynamicVLA skims are fine; do not propose re-wrapping their SM for G0.
