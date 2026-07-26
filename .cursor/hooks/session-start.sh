#!/usr/bin/env bash
# sessionStart: set lab env vars + inject brief context for new agent sessions.
set -uo pipefail

root="/home/gpuadmin/Desktop/foresight"
ctx="Foresight lab. Source of truth: foresight_plan.md. Contract: AGENTS.md + .cursor/rules/15-own-the-stack.mdc. ppt.md = Isaac launch/connect ONLY — not DynamicVLA architecture. Own sim+expert SM in this repo (sim/). Do NOT wrap DynamicVLA PickStateMachine/simulate for G0 (failed ~40% flat). Verify external modules with a smoke call before import. Tag Tasks: WORKSTREAM: sim-scene|perception|policy|control|eval|explore."

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
