#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$ROOT_DIR")"
MAIN_WORKTREE="$PARENT_DIR/Phone-to-Tasks-main"
SHARED_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$SHARED_PYTHON" ]]; then
  echo "Missing virtualenv interpreter at $SHARED_PYTHON"
  exit 1
fi

if [[ ! -d "$MAIN_WORKTREE/.git" && ! -f "$MAIN_WORKTREE/.git" ]]; then
  echo "Creating main worktree at $MAIN_WORKTREE"
  git -C "$ROOT_DIR" worktree add "$MAIN_WORKTREE" main
fi

cd "$MAIN_WORKTREE"
mkdir -p data/logs

PIDS="$(pgrep -f "main.py" || true)"
if [[ -n "$PIDS" ]]; then
  echo "Stopping existing app process(es): $PIDS"
  while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid"
  done <<< "$PIDS"
  sleep 2
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "Expected main worktree, found branch: $CURRENT_BRANCH"
  exit 1
fi

echo "Starting app from main worktree..."
nohup "$SHARED_PYTHON" main.py > data/logs/server_stdout.log 2>&1 &
NEW_PID="$!"

sleep 2

if ps -p "$NEW_PID" > /dev/null 2>&1; then
  echo "Started app with PID $NEW_PID"
  echo "Branch: main"
  echo "Worktree: $MAIN_WORKTREE"
  echo "UI: http://127.0.0.1:8081"
  echo "Log: $MAIN_WORKTREE/data/logs/server_stdout.log"
else
  echo "App failed to stay running. Check $MAIN_WORKTREE/data/logs/server_stdout.log"
  exit 1
fi
