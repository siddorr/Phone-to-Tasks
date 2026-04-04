# Phone-to-Tasks

Local-first call processing pipeline for:

`audio -> transcript -> summary -> action items`

## What Changed

The project now uses a modular archive-based pipeline instead of a single SQLite-only prototype.

- Incoming recordings are scanned from `data/incoming`
- Calls are archived under `data/calls/YYYY/MM/DD/call_<id>/`
- Files on disk are the source of truth
- SQLite in `data/index/call_assistant.db` is the searchable index
- A local FastAPI UI exposes calls, task review, and queue state

## Main Entry Points

- `python3 main.py`
  Starts the local web UI and background worker
- `python3 process_downloads.py`
  Runs a one-shot scan/import and drains the queue
- `python3 process_downloads.py --list`
  Lists eligible incoming files without processing
- `python3 show_transcript.py [call_id]`
  Prints the clean transcript for the latest or selected call

## Configuration

`config.yaml` controls:

- incoming/archive/index paths
- queue retry limits
- local-first transcription settings
- optional cloud fallback
- local UI host/port

## Artifact Layout

Each imported call creates:

```text
data/calls/YYYY/MM/DD/call_<id>/
  audio_original.ext
  audio_normalized.wav
  metadata.json
  transcript_raw.json
  transcript_segments.json
  transcript_clean.txt
  summary.json
  tasks.json
  tasks_reviewed.json
  processing_log.json
```

## Dependencies

Install the runtime dependencies before using the full pipeline:

```bash
pip install -r requirements.txt
```

Local transcription uses `openai-whisper`. Cloud fallback and LLM analysis require `OPENAI_API_KEY`.

Default transcription models:

- local: `large-v3-turbo`
- cloud: `whisper-1`

## Tests

```bash
python3 -m unittest discover -s tests
```
