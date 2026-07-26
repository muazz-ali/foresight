#!/usr/bin/env bash
# preToolUse (Task): enforce WORKSTREAM tags for parallel research subagents.
# Read-only explore/shell may proceed without a tag; others must declare ownership.
set -uo pipefail

input="$(cat 2>/dev/null || true)"

if ! command -v jq >/dev/null 2>&1; then
  echo '{"permission":"allow"}'
  exit 0
fi

tool="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null || true)"
# Some runtimes nest fields under tool_input; support both.
prompt="$(printf '%s' "$input" | jq -r '.tool_input.prompt // .prompt // empty' 2>/dev/null || true)"
stype="$(printf '%s' "$input" | jq -r '.tool_input.subagent_type // .subagent_type // empty' 2>/dev/null || true)"
desc="$(printf '%s' "$input" | jq -r '.tool_input.description // .description // empty' 2>/dev/null || true)"

# Non-Task (shouldn't match) → allow.
if [ -n "$tool" ] && [ "$tool" != "Task" ]; then
  echo '{"permission":"allow"}'
  exit 0
fi

# Hard block: proposals to re-wrap DynamicVLA SM as the Phase-0 expert.
if printf '%s\n%s' "$prompt" "$desc" | grep -Eiq \
  'wrap.*(DynamicVLA|PickStateMachine)|PickStateMachine.*(expert|G0)|sys\.path.*DynamicVLA.*(expert|simulate)|vendor.*(pick_sm|PlaceStateMachine)'; then
  msg="Blocked: DynamicVLA SM is not the Phase-0 expert."
  agent="G0 lesson: DynamicVLA PickStateMachine/simulate is a dead end (~40% flat). Implement the four-stage expert in foresight/sim/ per foresight_plan.md §4. ppt.md is Isaac connect only. See .cursor/rules/15-own-the-stack.mdc."
  jq -n --arg m "$msg" --arg a "$agent" \
    '{permission:"deny", user_message:$m, agent_message:$a}'
  exit 0
fi

# explore / shell / ci-investigator / cursor-guide: inject a light read-mostly brief if missing.
case "$stype" in
  explore|shell|ci-investigator|cursor-guide)
    if printf '%s' "$prompt" | grep -Eq 'WORKSTREAM:[[:space:]]*(explore|shell)'; then
      echo '{"permission":"allow"}'
      exit 0
    fi
    brief="WORKSTREAM: explore
Read-only unless the parent explicitly asked for writes. Cite file paths. ppt.md = Isaac connect only. Do not propose wrapping DynamicVLA SM for G0. Verify Isaac APIs in ppt.md or ~/Desktop/IsaacLab. Return: findings / evidence paths / open questions.

---
"
    new_prompt="${brief}${prompt}"
    printf '%s' "$input" | jq -c --arg p "$new_prompt" '
      (if .tool_input then (.tool_input.prompt = $p) else (.prompt = $p) end)
      | {permission:"allow", updated_input: (if .tool_input then .tool_input else del(.tool_name) end)}
    '
    exit 0
    ;;
esac

# generalPurpose / best-of-n / unspecified: require WORKSTREAM ownership tag.
if printf '%s\n%s' "$prompt" "$desc" | grep -Eq 'WORKSTREAM:[[:space:]]*(sim-scene|perception|policy|control|eval|explore)'; then
  # Ensure a short anti-complexity footer is present once.
  if printf '%s' "$prompt" | grep -Fq 'FORESIGHT_LAB_BRIEF'; then
    echo '{"permission":"allow"}'
    exit 0
  fi
  footer="

---
FORESIGHT_LAB_BRIEF: Stay in your WORKSTREAM directories. Preserve state interface (position, velocity, covariance, timestamp, valid). Prefer few lines. Own runtime in this repo — no DynamicVLA expert SM. ppt.md = connect only. Smoke-test imports before wiring. No Phase-skipping. Return: result / evidence / open questions.
"
  new_prompt="${prompt}${footer}"
  printf '%s' "$input" | jq -c --arg p "$new_prompt" '
    (if .tool_input then (.tool_input.prompt = $p) else (.prompt = $p) end)
    | {permission:"allow", updated_input: (if .tool_input then .tool_input else del(.tool_name) end)}
  '
  exit 0
fi

msg="Subagent blocked: missing WORKSTREAM tag."
agent="Re-launch Task with an explicit line like: WORKSTREAM: perception
Valid tags: sim-scene | perception | policy | control | eval | explore
See AGENTS.md. Parallel agents must own disjoint paths."
jq -n --arg m "$msg" --arg a "$agent" \
  '{permission:"deny", user_message:$m, agent_message:$a}'
exit 0
