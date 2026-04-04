#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/garik/Documents/git/Phone-to-Tasks"
SOURCE_FILE="${1:-$ROOT/data/incoming/Call recording милый Наташон_250216_100928.m4a}"
ATTEMPTS="${2:-5}"
SLEEP_SECONDS="${3:-60}"
LOG_FILE="${4:-$ROOT/data/logs/openai_connectivity_checks.log}"

cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing virtualenv Python at $ROOT/.venv/bin/python" >&2
  exit 1
fi

mkdir -p "$ROOT/data/logs"

for ((i=1; i<=ATTEMPTS; i++)); do
  {
    echo "===== attempt=$i at $(date -Is) ====="
    ./check_openai_audio.sh "$SOURCE_FILE"
    echo
  } | tee -a "$LOG_FILE"

  if [ "$i" -lt "$ATTEMPTS" ]; then
    sleep "$SLEEP_SECONDS"
  fi
done

echo "Log written to: $LOG_FILE"
