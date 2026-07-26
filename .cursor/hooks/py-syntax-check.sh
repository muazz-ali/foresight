#!/usr/bin/env bash
# afterFileEdit: byte-compile edited Python files (non-destructive, fail-open).
set -uo pipefail

input="$(cat 2>/dev/null || true)"

f=""
if command -v jq >/dev/null 2>&1; then
  f="$(printf '%s' "$input" | jq -r '.file_path // .path // .filePath // empty' 2>/dev/null || true)"
fi

case "$f" in
  *.py) : ;;
  *) echo '{}'; exit 0 ;;
esac
[ -f "$f" ] || { echo '{}'; exit 0; }

py="$(command -v python3 || command -v python || true)"
[ -z "$py" ] && { echo '{}'; exit 0; }

err="$("$py" -m py_compile "$f" 2>&1)"
if [ $? -ne 0 ]; then
  if command -v jq >/dev/null 2>&1; then
    jq -n --arg c "Python syntax error in $f:
$err
Fix before continuing." '{additional_context:$c}'
  else
    echo '{}'
  fi
  exit 0
fi

echo '{}'
exit 0
