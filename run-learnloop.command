#!/bin/zsh
set -e
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ ! -x .venv/bin/learnloop-start ]]; then
  echo "First-time setup: installing LearnLoop…"
  uv sync --no-editable
fi

exec .venv/bin/learnloop-start "$@"
