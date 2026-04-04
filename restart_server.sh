#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing virtualenv interpreter at .venv/bin/python"
  exit 1
fi

mkdir -p data/logs

PIDS="$(pgrep -f "(.*/)?\\.venv/bin/python main.py" || true)"
if [[ -n "$PIDS" ]]; then
  echo "Stopping existing app process(es): $PIDS"
  while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid"
  done <<< "$PIDS"
  sleep 2
fi

echo "Starting app..."
nohup .venv/bin/python main.py > data/logs/server_stdout.log 2>&1 &
NEW_PID="$!"

sleep 2

if ps -p "$NEW_PID" > /dev/null 2>&1; then
  echo "Started app with PID $NEW_PID"
  echo "UI: http://127.0.0.1:8081"
  echo "Log: $ROOT_DIR/data/logs/server_stdout.log"
else
  echo "App failed to stay running. Check $ROOT_DIR/data/logs/server_stdout.log"
  exit 1
fi
