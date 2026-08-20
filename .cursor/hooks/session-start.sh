#!/usr/bin/env bash
# sessionStart: set lab env vars + inject brief context for new agent sessions.
set -uo pipefail

root="/home/gpuadmin/Desktop/foresight"
ctx="Foresight lab. Source of truth: foresight_plan.md. G1 write-up: foresight_revised_plan2.md. Contract: AGENTS.md + .cursor/rules/15-own-the-stack.mdc. ppt.md = Isaac launch/connect ONLY — not DynamicVLA architecture. Own sim+expert SM in this repo (sim/). Do NOT wrap DynamicVLA PickPlaceStateMachine/simulate for G0 (failed ~40% flat). G1 scout (8-step) is not a dead hypothesis: Δ was not packed, eval unglues on open, labels {0,4} vs expert stages, n=20 cannot pass a 20-pt bar. 50-step still was 18/20 — latch/chunk, not a world model. Do not add AHEAD/DynaWM/PhysMani. Live pack = 5 numbers including Δ. Verify external modules with a smoke call before import. Tag Tasks: WORKSTREAM: sim-scene|perception|policy|control|eval|explore. Always explain to the user in plain English — short words, no jargon unless they ask."

if command -v jq >/dev/null 2>&1; then
  jq -n \
    --arg root "$root" \
    --arg plan "$root/foresight_plan.md" \
    --arg agents "$root/AGENTS.md" \
    --arg isaac "/home/gpuadmin/Desktop/IsaacLab" \
    --arg ctx "$ctx" \
    '{
      env: {
        FORESIGHT_ROOT: $root,
        FORESIGHT_PLAN: $plan,
        FORESIGHT_AGENTS: $agents,
        ISAACLAB_PATH: $isaac
      },
      additional_context: $ctx
    }'
else
  printf '{"env":{"FORESIGHT_ROOT":"%s","ISAACLAB_PATH":"/home/gpuadmin/Desktop/IsaacLab"},"additional_context":"%s"}\n' "$root" "$ctx"
fi
exit 0
