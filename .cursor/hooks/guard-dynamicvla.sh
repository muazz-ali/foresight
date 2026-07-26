#!/usr/bin/env bash
# beforeShellExecution: ask before wiring DynamicVLA as Phase-0 runtime.
# Allows read-only inspection (cat/less/rg/grep/find/ls/head/tail) of that tree.
set -uo pipefail

input="$(cat 2>/dev/null || true)"

cmd=""
if command -v jq >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | jq -r '.command // empty' 2>/dev/null || true)"
fi
[ -z "$cmd" ] && cmd="$input"

# Read-only inspection of DynamicVLA is fine.
if printf '%s' "$cmd" | grep -Eq '(^|[[:space:]])(ls|cat|less|more|head|tail|rg|grep|find|wc|file|stat|bat|sed[[:space:]]+-n)([[:space:]]|$)'; then
  echo '{"permission":"allow"}'
  exit 0
fi

# Flag: putting DynamicVLA on PYTHONPATH / importing their simulate/SM for a foresight run.
if printf '%s' "$cmd" | grep -Eqi \
  'PYTHONPATH=.*DynamicVLA|sys\.path.*(DynamicVLA|dynamicvla)|from[[:space:]]+state_machines|import[[:space:]]+simulate|run_phase0\.py.*DynamicVLA|muazzam/DynamicVLA.*(simulate|pick_sm|state_machines)'; then
  msg="Command touches DynamicVLA as runtime. Confirm — G0 wrapping that stack failed."
  agent="Hook: this looks like wiring DynamicVLA simulate/SM into Foresight. Post-G0 policy: own expert in foresight/sim/; ppt.md is Isaac connect only; DynamicVLA is read-only reference. Allow only if intentionally inspecting or migrating assets — not for re-wrapping the expert."
  if command -v jq >/dev/null 2>&1; then
    jq -n --arg m "$msg" --arg a "$agent" \
      '{permission:"ask", user_message:$m, agent_message:$a}'
  else
    printf '{"permission":"ask","user_message":"%s","agent_message":"%s"}\n' "$msg" "$agent"
  fi
  exit 0
fi

echo '{"permission":"allow"}'
exit 0
