# Phone-to-Tasks

Local-first call processing pipeline for:

`audio -> transcript -> summary -> action items`

## Overview

The project uses an archive-based pipeline instead of the original SQLite-only prototype.

- Incoming recordings are scanned from `data/incoming`
- Archived calls are stored under the configured `archive_root`
- Files on disk are the source of truth
- SQLite in `data/index/call_assistant.db` is the searchable index
- A local FastAPI UI exposes calls, task review, retranscription, speaker mapping, and queue state
- A background worker scans for new recordings continuously and processes queued jobs stage by stage

## Implemented Features

Current app capabilities:

- Local web UI at `/calls` and `/queue`
- Incoming-folder ingest from `data/incoming`
- Browser upload of supported audio files
- Duplicate detection by SHA-256
- Fast duplicate short-circuiting by known source path during incoming scans
- Per-call archive folders with canonical artifacts on disk
- SQLite indexing for calls, tasks, artifacts, and queue state
- Audio normalization to mono 16 kHz WAV
- Local transcription with selectable Whisper models
- Optional OpenAI transcription fallback with `whisper-1`
- Real-time worker loop that rescans incoming files while processing backlog
- Stage-aware queue recovery for interrupted jobs
- In-progress states in the UI such as `transcribing`, `diarizing`, `analyzing`
- Transcript cleaning and normalized segment output
- Diarization with pyannote when available, plus safe fallback behavior
- Manual speaker mapping in the UI (`me` / `other` / `unknown`)
- Summary and task extraction
- Task review/edit flow in the UI
- Queue inspection and retry from the UI
- Reindex action from the UI
- Retranscribe action from the UI with model selection
- Batch retranscribe CLI for all archived calls
- Recorded-time extraction from filename/media metadata with backfill support
- Background worker that continues processing after import
- One-shot CLI processing for batch runs and debugging

## Main Entry Points

- `python3 main.py`
  Starts the local web UI and background worker
- `python3 process_downloads.py`
  Runs a one-shot scan/import and drains the queue
- `python3 process_downloads.py --list`
  Lists eligible incoming files without processing
- `python3 show_transcript.py [call_id]`
  Prints the clean transcript for the latest or selected call
- `python3 retranscribe_all.py --model local:large-v3-turbo`
  Resets all archived calls from `transcription` onward and reprocesses them
- `python3 backfill_recorded_times.py`
  Recomputes `recorded_at` for existing calls without retranscribing audio
- `./restart_server.sh`
  Restarts the local web UI and background worker

## Configuration

`config.yaml` controls:

- incoming/archive/index paths
- queue stale-job logic
- queue retry limits
- local-first transcription settings
- diarization settings
- optional cloud fallback
- local UI host/port

Important defaults in the current config:

- archive root: `/home/garik/CallAssistantData/calls`
- local transcription model: `large-v3-turbo`
- cloud transcription model: `whisper-1`
- scan interval: `5` seconds

### Queue timeout behavior

The worker uses stage-aware stale-job handling.

- non-transcription jobs use a fixed stale timeout
- transcription jobs use a duration-based timeout:

`max(stale_job_seconds, transcription_stale_min_seconds, duration_seconds * transcription_stale_multiplier + transcription_stale_buffer_seconds)`

Current transcription timeout settings:

- `transcription_stale_multiplier: 3`
- `transcription_stale_buffer_seconds: 600`
- `transcription_stale_min_seconds: 1800`

## Artifact Layout

Each imported call creates:

```text
<archive_root>/YYYY/MM/DD/call_<id>/
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
.venv/bin/pip install -r requirements.txt
```

Local transcription uses `openai-whisper`. Cloud fallback and LLM analysis require `OPENAI_API_KEY`.
Real diarization requires `pyannote.audio` plus model access in your environment.

Default transcription models:

- local: `large-v3-turbo`
- cloud: `whisper-1`

## Typical Usage

Start the app:

```bash
cd /home/garik/Documents/git/Phone-to-Tasks
./restart_server.sh
```

Open:

```text
http://127.0.0.1:8081/calls
```

Then either:

- upload audio from the `/calls` page
- or place recordings into `data/incoming`

The worker will:

1. import new stable files
2. normalize audio
3. transcribe
4. diarize
5. clean transcript
6. analyze
7. index results in SQLite

## UI Features

The web UI currently supports:

- archive list with recorded time, respondent, state, tasks, and short description
- queue view
- call detail page
- retranscribe with model selection
- speaker mapping
- task review/editing
- reindex

## Recorded Time

`recorded_at` is resolved in this order:

1. timestamp parsed from filename
2. media metadata
3. filesystem `mtime`

Backfill existing rows with:

```bash
cd /home/garik/Documents/git/Phone-to-Tasks
.venv/bin/python backfill_recorded_times.py
```

## Notes

- Long local CPU transcriptions can take many minutes for 6-7 minute calls.
- Incoming scans now skip already imported files cheaply via `known_source_path`.
- Calls may temporarily show states such as `audio_prepared`, `transcribing`, `diarizing`, `analyzing`, and `indexing`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
