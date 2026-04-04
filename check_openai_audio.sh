#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/garik/Documents/git/Phone-to-Tasks"
SOURCE_FILE="${1:-$ROOT/data/incoming/Call recording милый Наташон_250216_100928.m4a}"

cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing virtualenv Python at $ROOT/.venv/bin/python" >&2
  exit 1
fi

if [ ! -f "$SOURCE_FILE" ]; then
  echo "Audio file not found: $SOURCE_FILE" >&2
  exit 1
fi

".venv/bin/python" - <<'PY' "$SOURCE_FILE"
import os
import sys
import tempfile
from pathlib import Path

from openai import OpenAI
from pydub import AudioSegment

source = Path(sys.argv[1])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print("== OpenAI model-list connectivity test ==")
try:
    models = client.models.list()
    first = next(iter(models.data), None)
    print("status=ok")
    print("first_model=" + (first.id if first else "none"))
except Exception as e:
    print("status=failed")
    print("type=" + type(e).__name__)
    print("error=" + str(e))

print()
print("== OpenAI audio transcription test ==")
audio = AudioSegment.from_file(source).set_channels(1).set_frame_rate(16000)
with tempfile.NamedTemporaryFile(prefix="openai_smoke_", suffix=".wav", delete=False) as tmp:
    tmp_path = Path(tmp.name)

audio.export(tmp_path, format="wav")
print("source=" + str(source))
print("normalized=" + str(tmp_path))

try:
    with tmp_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    payload = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    print("status=ok")
    print("language=" + str(payload.get("language")))
    print("text=" + (payload.get("text") or "")[:200])
except Exception as e:
    print("status=failed")
    print("type=" + type(e).__name__)
    print("error=" + str(e))
finally:
    if tmp_path.exists():
        tmp_path.unlink()
PY
