#!/usr/bin/env bash
# beforeShellExecution: ask before destructive / irreversible commands.
set -uo pipefail

input="$(cat 2>/dev/null || true)"

cmd=""
if command -v jq >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | jq -r '.command // empty' 2>/dev/null || true)"
fi
[ -z "$cmd" ] && cmd="$input"

DANGER='(^|[^a-zA-Z])rm[[:space:]]+(-[a-zA-Z]*[rf][a-zA-Z]*[[:space:]]+)+|[[:space:]]mkfs|[[:space:]]dd[[:space:]]+if=|:\(\)\{[[:space:]]*:\|:|git[[:space:]]+push[[:space:]].*--force|git[[:space:]]+reset[[:space:]]+--hard|git[[:space:]]+clean[[:space:]]+-[a-zA-Z]*f|git[[:space:]]+config|chmod[[:space:]]+-R[[:space:]]+777|[[:space:]]killall|kill[[:space:]]+-9|>[[:space:]]*/dev/sd|shutdown|reboot'

if printf '%s' "$cmd" | grep -Eq "$DANGER"; then
  msg="Destructive or irreversible command flagged. Review before running."
  agent="Project hook: this shell command looks destructive (rm -rf, force-push, hard reset, git clean -f, git config, kill -9, disk ops). Confirm intent. Prefer safer alternatives."
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
