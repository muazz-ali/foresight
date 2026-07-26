#!/usr/bin/env bash
# subagentStop: nudge parent to integrate evidence against gates, not expand scope.
set -uo pipefail

input="$(cat 2>/dev/null || true)"

status=""
stype=""
if command -v jq >/dev/null 2>&1; then
  status="$(printf '%s' "$input" | jq -r '.status // .error // empty' 2>/dev/null || true)"
  stype="$(printf '%s' "$input" | jq -r '.subagent_type // empty' 2>/dev/null || true)"
fi

follow="Subagent ($stype) finished ($status). Integrate only verified results. Check: shapes/seeds ok? Still inside WORKSTREAM? Runtime still in-repo (no DynamicVLA expert SM)? Any gate metric moved (G0–G4)? Do not start the next phase unless the current gate evidence is logged. Prefer the simplest merge."

if command -v jq >/dev/null 2>&1; then
  jq -n --arg m "$follow" '{followup_message:$m}'
else
  echo '{}'
fi
exit 0
