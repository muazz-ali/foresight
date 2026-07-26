---
name: README
model: inherit
---

# Foresight Cursor agents — quick roles

Use with Task tool. Always include `WORKSTREAM: <tag>` in the prompt.

## sim-scene
In-repo Isaac scene, cameras, **Foresight** four-stage scripted expert (plan §4), episode logging. Gate G0.
Read: `foresight_plan.md` §4, `ppt.md` (connect only), `.cursor/rules/15-own-the-stack.mdc`.
Do **not** drive episodes through DynamicVLA `PickStateMachine` / `simulate`.

## perception
Detector → tracker → Kalman → 3D lift. Same interface as oracle. Gate G2.
Budget: ~430 lines total across detector/tracker/lift/filter.

## policy
SmolVLA/LeRobot: Model A vs B; Δ randomization + filter noise. Gate G1.
Do not rewrite the backbone — append the conditioning vector.

## control
Timestamped action buffer, reflex PD, coast/safety. Gate G3.

## eval
Speed sweeps, prediction RMSE, failure taxonomy, sample-efficiency curves.
Never claim a gate without numbers. Log `max_sm` / lift stats with taxonomy.

## explore
Read-only. Paths + citations only. DynamicVLA skims are fine; do not propose re-wrapping their SM for G0.
