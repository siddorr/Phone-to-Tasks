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

  DEADLINE=$((SECONDS + 15))
  while [[ $SECONDS -lt $DEADLINE ]]; do
    STILL_RUNNING=0
    while read -r pid; do
      if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
        STILL_RUNNING=1
        break
      fi
    done <<< "$PIDS"
    [[ $STILL_RUNNING -eq 0 ]] && break
    sleep 1
  done

  FORCE_KILL=""
  while read -r pid; do
    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
      FORCE_KILL+="${FORCE_KILL:+ }$pid"
    fi
  done <<< "$PIDS"

  if [[ -n "$FORCE_KILL" ]]; then
    echo "Force stopping unresponsive app process(es): $FORCE_KILL"
    while read -r pid; do
      [[ -n "$pid" ]] && kill -9 "$pid"
    done <<< "$(tr ' ' '\n' <<< "$FORCE_KILL")"
    sleep 1
  fi
fi

PORT_DEADLINE=$((SECONDS + 15))
while [[ $SECONDS -lt $PORT_DEADLINE ]]; do
  if ! ss -ltn "sport = :8081" | grep -q LISTEN; then
    break
  fi
  sleep 1
done

if ss -ltn "sport = :8081" | grep -q LISTEN; then
  echo "Port 8081 is still busy; aborting restart."
  exit 1
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
