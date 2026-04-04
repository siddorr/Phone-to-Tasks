#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/garik/Documents/git/Phone-to-Tasks"

cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing virtualenv Python at $ROOT/.venv/bin/python" >&2
  exit 1
fi

".venv/bin/python" scripts/generate_transcription_comparison.py "$@"

echo
echo "Report written to:"
echo "  $ROOT/transcription_model_comparison.txt"
