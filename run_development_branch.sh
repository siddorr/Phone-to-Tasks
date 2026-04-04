#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$SHARED_PYTHON" ]]; then
  echo "Missing virtualenv interpreter at $SHARED_PYTHON"
  exit 1
fi

cd "$ROOT_DIR"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "development" ]]; then
  echo "This worktree is on branch '$CURRENT_BRANCH', not 'development'."
  echo "Development launcher refuses to switch branches automatically to protect uncommitted changes."
  exit 1
fi

mkdir -p data/logs

PIDS="$(pgrep -f "main.py" || true)"
if [[ -n "$PIDS" ]]; then
  echo "Stopping existing app process(es): $PIDS"
  while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid"
  done <<< "$PIDS"
  sleep 2
fi

echo "Starting app from development worktree..."
nohup "$SHARED_PYTHON" main.py > data/logs/server_stdout.log 2>&1 &
NEW_PID="$!"

sleep 2

if ps -p "$NEW_PID" > /dev/null 2>&1; then
  echo "Started app with PID $NEW_PID"
  echo "Branch: development"
  echo "Worktree: $ROOT_DIR"
  echo "UI: http://127.0.0.1:8081"
  echo "Log: $ROOT_DIR/data/logs/server_stdout.log"
else
  echo "App failed to stay running. Check $ROOT_DIR/data/logs/server_stdout.log"
  exit 1
fi
