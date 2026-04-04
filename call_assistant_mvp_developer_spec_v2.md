# Call Assistant MVP — Developer Specification v2

## 1. Document purpose

This document defines the **MVP scope**, **project goals**, **functional and non-functional requirements**, **architecture**, **data model**, **storage model**, **processing rules**, **UI**, **configuration**, **testing**, **acceptance criteria**, and **deferred items** for the private **Call Assistant** project.

It is intended to be the primary implementation document for developers building the first working version.

---

## 2. Project overview

The project is a **private personal tool** designed to ensure that important information from calls is not lost.

The system automatically turns recorded calls into structured, searchable, reviewable outputs:

**audio -> transcript -> summary -> action items**

The intent is to:
1. reduce information loss,
2. eliminate manual note-taking,
3. preserve verbal commitments and decisions,
4. extract follow-up tasks,
5. prepare reviewed tasks for later Telegram-based task handling.

---

## 3. Project goals

### 3.1 Primary goals
1. Capture all calls relevant to the user workflow.
2. Produce a full transcript shortly after the call ends.
3. Generate useful summaries in a consistent format.
4. Extract action items, decisions, commitments, and unresolved points.
5. Save all outputs in a local searchable archive.
6. Support later use of Telegram as a task-list interface.

### 3.2 MVP success criteria
The MVP is successful if it can:
1. detect a new recording automatically,
2. process it within **several minutes**,
3. store a usable transcript with timestamps and diarization,
4. generate usable summaries and tasks,
5. allow manual review and correction of tasks,
6. keep everything locally and searchable,
7. support a **real local transcription path** without mandatory cloud dependency.

---

## 4. Scope

### 4.1 In scope
1. Single-user private workflow.
2. Local-first storage and processing.
3. One synced incoming folder.
4. Background queue-based processing.
5. Local review UI.
6. Mobile call recordings already present on the phone.
7. Search by keyword, date, and source.
8. Local transcription option in MVP.

### 4.2 Out of scope for MVP
1. Automatic WhatsApp recording acquisition.
2. Automatic Teams recording acquisition.
3. Telegram task sync.
4. Telegram routing logic.
5. Semantic search.
6. Multi-user support.
7. Cloud-first deployment.
8. Complex collaboration workflows.
9. Advanced analytics dashboards.

### 4.3 Supported source vision
The full product direction is intended to support:
1. mobile phone calls,
2. WhatsApp calls,
3. Teams calls.

### 4.4 MVP source reality
For MVP:
1. mobile recordings already exist on the phone,
2. WhatsApp and Teams acquisition are not required,
3. architecture must leave room for these sources later.

---

## 5. Business problem and rationale

Important information from calls is often:
1. forgotten,
2. remembered inaccurately,
3. not turned into tasks,
4. scattered across memory, notes, and chats.

Manual call note-taking is inconsistent and inconvenient.

The system exists to:
1. preserve what was said,
2. make it searchable,
3. identify what needs to happen next,
4. support later operational follow-up,
5. create a reliable personal archive of commitments, decisions, and open issues.

---

## 6. Confirmed discovery requirements

### 6.1 User and scope
1. Private tool.
2. Single-user only.
3. Main purpose: **not to lose information**.
4. Save all artifacts.

### 6.2 Languages
The system must support:
1. Russian,
2. Hebrew,
3. English,
4. mixed-language calls.

### 6.3 Timing
1. Processing target: within **several minutes** after call end.

### 6.4 Transcript requirements
The system must produce:
1. full transcript,
2. timestamps,
3. speaker diarization,
4. mapping to **me / other**,
5. raw/literal version,
6. clean/readable version.

### 6.5 Summary outputs
The system must produce:
1. short summary,
2. detailed summary,
3. key points,
4. action items,
5. decisions made,
6. open questions,
7. promises / commitments.

### 6.6 Task fields
Tasks must support:
1. text,
2. owner,
3. type,
4. source timestamp,
5. source quote,
6. status,
7. optional deadline.

### 6.7 Search and review
MVP must support:
1. keyword search,
2. date filter,
3. source filter,
4. local review UI,
5. manual task correction.

### 6.8 Low-confidence behavior
If transcript quality is poor:
1. outputs should still be saved,
2. summaries and tasks should still be generated,
3. the system should mark them as low confidence.

---

## 7. Glossary

1. **Raw transcript** — timestamped transcription output that stays close to the speech recognition result.
2. **Clean transcript** — readable transcript produced from raw transcript plus diarization, with improved readability but preserved meaning.
3. **Extracted task** — machine-generated task in `tasks.json`.
4. **Reviewed task** — user-confirmed or user-edited task in `tasks_reviewed.json`.
5. **Decision** — conclusion or agreed outcome from the call; not necessarily a task.
6. **Open question** — unresolved issue explicitly or implicitly left unanswered.
7. **Commitment** — promise or stated obligation made by a speaker.
8. **Waiting item** — item that depends on action or information from the other side.
9. **Follow-up** — future action to revisit, check, or continue progress later.
10. **Reminder** — note for self without clear external dependency.

---

## 8. Core workflow

1. A call is recorded.
2. The recording is synced from phone to PC using **Syncthing-Fork**.
3. A local watcher detects a new file in the incoming folder.
4. The system checks that the file is stable and fully synced.
5. The recording is imported into the archive.
6. Audio is normalized.
7. Speech-to-text runs.
8. Diarization runs.
9. A raw/literal transcript is produced.
10. A clean/readable transcript is produced.
11. AI generates:
   - short summary,
   - detailed summary,
   - key points,
   - decisions,
   - open questions,
   - commitments,
   - action items.
12. Tasks are normalized into structured task objects.
13. All artifacts are stored locally.
14. SQLite index is updated.
15. User reviews the call in the local UI.
16. Reviewed tasks are saved separately for later Telegram export or sync.

---

## 9. High-level architecture

### 9.1 Architecture style
A **local modular pipeline** with:
1. file-based artifact storage,
2. SQLite indexing,
3. background queue worker,
4. local review UI.

### 9.2 Major layers
1. Capture / sync
2. Ingest
3. Audio preparation
4. Transcription
5. Diarization
6. Transcript cleaning
7. Analysis
8. Task normalization
9. Indexing
10. UI

### 9.3 Platform assumptions
1. Incoming recordings sync from phone to PC using **Syncthing-Fork**.
2. PC environment may be **Windows** and/or **Ubuntu/Linux**.
3. UI is local, not cloud-hosted.

---

## 10. Source of truth policy

1. **Files on disk** are the canonical artifacts.
2. **SQLite** is the searchable index and UI state store.
3. **Reviewed tasks** override extracted tasks in UI display when available.
4. **Original machine outputs** are never deleted automatically during ordinary processing.
5. If SQLite is corrupted, it must be rebuildable from files on disk.

---

## 11. Functional requirements

### 11.1 Ingestion
The system shall:
1. watch one synced incoming folder,
2. detect new recordings,
3. determine whether a file is fully synced and stable,
4. avoid duplicate imports,
5. create call archive folders,
6. create initial metadata,
7. enqueue background processing.

### 11.2 Audio processing
The system shall:
1. validate audio format,
2. normalize audio to a working format,
3. extract technical metadata,
4. store duration and format info.

### 11.3 Transcription
The system shall:
1. run multilingual speech-to-text,
2. preserve timestamps,
3. support mixed-language content,
4. store transcription confidence,
5. save raw machine output,
6. provide a **local transcription option** in MVP,
7. allow provider selection via configuration.

### 11.4 Diarization
The system shall:
1. detect speaker turns,
2. assign speaker labels,
3. resolve labels as **me / other / unknown**,
4. preserve timing alignment.

### 11.5 Transcript cleaning
The system shall:
1. produce a readable transcript,
2. preserve meaning,
3. preserve speaker boundaries,
4. avoid hallucinated rewrites,
5. use deterministic cleanup first and AI polish second.

### 11.6 Analysis
The system shall:
1. generate a short summary,
2. generate a detailed summary,
3. extract key points,
4. extract decisions,
5. extract open questions,
6. extract commitments,
7. extract candidate tasks,
8. assign analysis confidence.

### 11.7 Task normalization
The system shall:
1. convert extracted tasks into normalized records,
2. attach source quote and timestamp,
3. preserve optional deadline fields,
4. support reviewed version separately from extracted version.

### 11.8 Indexing and search
The system shall:
1. index call metadata,
2. index transcript text,
3. index summary text,
4. index task counts and statuses,
5. support archive search and filtering.

### 11.9 UI
The system shall provide a local interface to:
1. browse calls,
2. search and filter calls,
3. open call detail,
4. inspect transcript and summaries,
5. review and edit tasks,
6. inspect queue / processing state.

---

## 12. Non-functional requirements

### 12.1 Privacy
1. Local-first.
2. Local storage only for MVP.
3. No mandatory cloud dependency in the critical path.

### 12.2 Reliability
1. Background processing.
2. Safe stage retries.
3. Preserve artifacts from successful earlier stages.

### 12.3 Rebuildability
1. SQLite must be rebuildable from files on disk.

### 12.4 Traceability
1. Every task must trace back to quote and timestamp.
2. Original extracted tasks must be preserved until explicit reprocess behavior applies.

### 12.5 Idempotency
1. Reprocessing must not create duplicate calls.
2. Queue handling must avoid duplicate terminal outcomes for the same stage run.

### 12.6 Performance
1. Designed for single-user personal volume.
2. One worker is sufficient for MVP.

### 12.7 Observability
1. Processing failures must be logged.
2. Latest errors must be visible to the user in the UI or queue view.

---

## 13. Configuration

### 13.1 Settings edit model
1. Settings must be editable in a config file.
2. Some settings may also be editable in the UI.

### 13.2 Required MVP config fields
1. `incoming_folder`
2. `archive_root`
3. `transcription_provider`
4. `analysis_provider`
5. `retry_limits`
6. `supported_extensions`
7. `ui_port`

### 13.3 Recommended transcription config fields
1. `local_transcription_model`
2. `local_transcription_device`
3. `cloud_transcription_enabled`
4. `transcription_language_hints`

### 13.4 Recommended analysis config fields
1. `analysis_model`
2. `analysis_timeout_sec`
3. `analysis_max_retries`

---

## 14. Storage model

### 14.1 Project structure
```text
call-assistant/
  pyproject.toml
  README.md
  .env
  /data
    /incoming
    /calls
    /index
    /logs
    /temp
  /src
    /call_assistant
      /ingest
      /audio
      /transcription
      /diarization
      /transcript_cleaner
      /analysis
      /tasks
      /indexing
      /orchestrator
      /ui
      /common
  /tests
```

### 14.2 Per-call folder structure
```text
/data/calls/YYYY/MM/DD/call_<id>/
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

### 14.3 File responsibilities
1. `audio_original.ext` — original synced recording, never modified.
2. `audio_normalized.wav` — working audio for downstream processing.
3. `metadata.json` — operational call metadata and processing state.
4. `transcript_raw.json` — raw timestamped transcription.
5. `transcript_segments.json` — speaker-aware transcript segments.
6. `transcript_clean.txt` — readable transcript.
7. `summary.json` — structured AI output.
8. `tasks.json` — extracted machine tasks.
9. `tasks_reviewed.json` — user-reviewed tasks.
10. `processing_log.json` — per-call processing history.

### 14.4 Retention policy for MVP
1. No automatic deletion.
2. Manual cleanup only.
3. UI should allow easy folder access.
4. Future retention policy may be added later.

---

## 15. Processing states and enums

### 15.1 Call processing states
1. `detected`
2. `imported`
3. `audio_prepared`
4. `transcribed`
5. `diarized`
6. `analyzed`
7. `indexed`
8. `reviewed`
9. `failed`

### 15.2 Review state
Only one explicit completed review state exists:
1. `reviewed`

A call becomes `reviewed` when the user **saves reviewed tasks**.

### 15.3 Queue job statuses
1. `queued`
2. `running`
3. `done`
4. `failed`

### 15.4 Processing run statuses
1. `started`
2. `succeeded`
3. `failed`

### 15.5 Task statuses
1. `new`
2. `confirmed`
3. `dismissed`
4. `edited`

### 15.6 Task types
1. `task` — explicit action to perform.
2. `waiting` — awaiting action or information from the other side.
3. `follow_up` — action to revisit later.
4. `reminder` — note for self without clear external dependency.

### 15.7 Confidence enums
Recommended textual confidence levels:
1. `low`
2. `medium`
3. `high`

---

## 16. Artifact JSON schemas

### 16.1 Common versioning rule
Every relevant JSON artifact should include:
1. `schema_version`
2. `app_version`
3. `provider_version` or `model_version` where relevant

All version fields may use simple strings.

### 16.2 `metadata.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "call_id": "2026-04-03_183012_ab12",
  "source": "mobile",
  "original_file_name": "call_2026_04_03_183012.m4a",
  "call_dir": "/data/calls/2026/04/03/call_2026-04-03_183012_ab12",
  "imported_at": "2026-04-03T18:35:10",
  "call_datetime": "2026-04-03T18:30:12",
  "duration_sec": 842,
  "file_hash": "abc123",
  "status": "indexed",
  "review_status": "reviewed",
  "languages": ["ru", "he", "en"],
  "transcript_confidence": "medium",
  "analysis_confidence": "medium",
  "participant_hint": null
}
```

### 16.3 `transcript_raw.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "provider_version": "local-whisper-x",
  "call_id": "2026-04-03_183012_ab12",
  "language_detected": ["ru", "he"],
  "segments": [
    {
      "start": "00:00:02",
      "end": "00:00:07",
      "text": "Hello, I wanted to discuss the updated pricing.",
      "confidence": 0.91
    }
  ],
  "transcript_confidence": "medium"
}
```

### 16.4 `transcript_segments.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "provider_version": "local-diarization-v1",
  "call_id": "2026-04-03_183012_ab12",
  "segments": [
    {
      "start": "00:00:02",
      "end": "00:00:07",
      "speaker": "me",
      "text": "Hello, I wanted to discuss the updated pricing.",
      "confidence": 0.88
    }
  ],
  "diarization_confidence": "medium"
}
```

### 16.5 `summary.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "model_version": "analysis-model-v1",
  "call_id": "2026-04-03_183012_ab12",
  "short_summary": "Short summary text.",
  "detailed_summary": "Detailed summary text.",
  "key_points": [
    "Point 1",
    "Point 2"
  ],
  "decisions": [
    "Decision 1"
  ],
  "open_questions": [
    "Question 1"
  ],
  "commitments": [
    {
      "owner": "me",
      "text": "Send updated file tomorrow"
    }
  ],
  "action_items": [
    {
      "text": "Send updated file tomorrow",
      "owner": "me",
      "type": "task",
      "source_timestamp": "00:12:43",
      "source_quote": "I'll send the updated file tomorrow."
    }
  ],
  "analysis_confidence": "medium"
}
```

### 16.6 `tasks.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "model_version": "analysis-model-v1",
  "call_id": "2026-04-03_183012_ab12",
  "tasks": [
    {
      "task_id": "t1",
      "text": "Send updated pricing file",
      "owner": "me",
      "type": "task",
      "deadline_text": "tomorrow",
      "deadline_iso": null,
      "source_timestamp": "00:12:43",
      "source_quote": "I'll send you the updated pricing file tomorrow.",
      "status": "new",
      "confidence": 0.88,
      "manually_edited": false
    }
  ]
}
```

### 16.7 `tasks_reviewed.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "call_id": "2026-04-03_183012_ab12",
  "reviewed_at": "2026-04-03T19:10:00",
  "tasks": [
    {
      "task_id": "t1",
      "text": "Send updated pricing file",
      "owner": "me",
      "type": "task",
      "deadline_text": "tomorrow",
      "deadline_iso": null,
      "source_timestamp": "00:12:43",
      "source_quote": "I'll send you the updated pricing file tomorrow.",
      "status": "confirmed",
      "confidence": 0.88,
      "manually_edited": true
    }
  ]
}
```

### 16.8 `processing_log.json`
```json
{
  "schema_version": "1.0",
  "app_version": "1.0",
  "call_id": "2026-04-03_183012_ab12",
  "runs": [
    {
      "stage": "transcription",
      "status": "succeeded",
      "started_at": "2026-04-03T18:36:00",
      "finished_at": "2026-04-03T18:36:45",
      "duration_ms": 45000,
      "error_text": null
    }
  ]
}
```

---

## 17. Module responsibilities and contracts

### 17.1 `ingest`
**Purpose:** detect and register new audio files.

**Responsibilities:**
1. detect stable fully synced files,
2. hash files,
3. avoid duplicates,
4. create call folder,
5. initialize metadata,
6. enqueue next stage.

**Must not do:**
1. transcription,
2. diarization,
3. AI analysis.

**Public contract**
```python
class IngestService:
    def scan_incoming() -> list["DetectedFile"]: ...
    def import_file(file_path: str) -> "CallRecord": ...
```

### 17.2 `audio`
**Purpose:** normalize audio and extract technical metadata.

**Public contract**
```python
class AudioPreparationService:
    def prepare(call_id: str) -> "AudioPreparationResult": ...
```

### 17.3 `transcription`
**Purpose:** produce timestamped multilingual transcription.

**Provider rule:**
1. MVP must implement a **local provider**.
2. Cloud provider may exist later.
3. Provider selection must be configurable.

**Public contract**
```python
class TranscriptionService:
    def transcribe(call_id: str) -> "TranscriptionResult": ...
```

### 17.4 `diarization`
**Purpose:** assign speaker turns and labels.

**Public contract**
```python
class DiarizationService:
    def diarize(call_id: str) -> "DiarizationResult": ...
```

### 17.5 `transcript_cleaner`
**Purpose:** produce readable transcript.

**Public contract**
```python
class TranscriptCleaner:
    def build_clean_transcript(call_id: str) -> "CleanTranscriptResult": ...
```

### 17.6 `analysis`
**Purpose:** generate structured interpretation of the call.

**Public contract**
```python
class AnalysisService:
    def analyze(call_id: str) -> "AnalysisResult": ...
```

### 17.7 `tasks`
**Purpose:** normalize extracted action items.

**Public contract**
```python
class TaskExtractionService:
    def build_tasks(call_id: str) -> list["TaskRecord"]: ...
```

### 17.8 `indexing`
**Purpose:** maintain SQLite search/index layer.

**Public contract**
```python
class IndexingService:
    def upsert_call(call_id: str) -> None: ...
    def search(query: str, filters: dict) -> list["CallSearchHit"]: ...
```

### 17.9 `orchestrator`
**Purpose:** run stages, retries, and job transitions.

**Public contract**
```python
class PipelineOrchestrator:
    def enqueue(call_id: str, stage: str) -> None: ...
    def run_next_job() -> None: ...
```

### 17.10 `ui`
**Purpose:** local review interface for archive, detail, tasks, and queue.

---

## 18. Acceptance criteria per module

### 18.1 Ingest acceptance criteria
Complete when:
1. stable synced file is detected,
2. duplicate import is prevented using hash,
3. call folder is created,
4. `metadata.json` exists,
5. queue job for next stage exists.

### 18.2 Audio acceptance criteria
Complete when:
1. original audio exists,
2. normalized audio exists,
3. duration and technical metadata are recorded,
4. next stage is queued.

### 18.3 Transcription acceptance criteria
Complete when:
1. `transcript_raw.json` exists,
2. timestamps exist,
3. transcription confidence exists,
4. multilingual content is accepted,
5. next stage is queued.

### 18.4 Diarization acceptance criteria
Complete when:
1. `transcript_segments.json` exists,
2. speaker labels exist,
3. timestamps remain aligned,
4. diarization confidence exists,
5. next stage is queued.

### 18.5 Transcript cleaning acceptance criteria
Complete when:
1. `transcript_clean.txt` exists,
2. speaker boundaries remain readable,
3. readable transcript preserves meaning,
4. next stage is queued.

### 18.6 Analysis acceptance criteria
Complete when:
1. `summary.json` exists,
2. short and detailed summaries exist,
3. key points, decisions, open questions, and commitments exist,
4. action items are present or explicitly empty,
5. analysis confidence exists.

### 18.7 Tasks acceptance criteria
Complete when:
1. `tasks.json` exists,
2. every task has text, owner, type, quote, timestamp, and status,
3. deadline fields are present where inferable,
4. reviewed version can be saved separately.

### 18.8 Indexing acceptance criteria
Complete when:
1. SQLite record exists for the call,
2. searchable fields are populated,
3. task counts are available,
4. archive UI can display the call.

### 18.9 UI acceptance criteria
Complete when:
1. archive list is visible,
2. call detail opens,
3. summaries and transcripts are shown,
4. tasks can be edited,
5. reviewed tasks can be saved,
6. queue status can be inspected.

---

## 19. File stability detection rules

1. File must remain unchanged for a configurable period.
2. File size must stabilize across at least two scans.
3. Partial or in-progress sync artifacts must be ignored.
4. Import must only start after the file is considered stable.

---

## 20. Reprocessing rules

1. Reprocess is allowed **per stage**, mainly for debug.
2. Reprocess **overwrites old generated files**.
3. If `tasks_reviewed.json` exists and reprocess runs, reviewed tasks are **discarded** and regenerated.
4. Reprocess must update processing logs and SQLite state.
5. Reprocess must not create a duplicate call record.

---

## 21. Stage transition rules

1. `detected -> imported` only if file is stable and hash is new.
2. `imported -> audio_prepared` only if original audio exists.
3. `audio_prepared -> transcribed` only if normalized audio exists.
4. `transcribed -> diarized` only if transcript exists.
5. `diarized -> analyzed` only if speaker-segment data exists.
6. `analyzed -> indexed` only if summary and tasks exist.
7. `indexed -> reviewed` after user saves reviewed tasks.

---

## 22. Error handling rules

1. If a stage fails, keep all prior artifacts.
2. Write failure to:
   - `processing_log.json`,
   - SQLite status fields.
3. Retry per stage, not by rerunning the full pipeline blindly.
4. Low-confidence calls still proceed.
5. Low confidence is shown in UI as a **badge only**.

---

## 23. Core data objects

### 23.1 `CallRecord`
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class CallRecord:
    call_id: str
    source: str
    original_file: Path
    call_dir: Path
    imported_at: str
    status: str
    duration_sec: Optional[int] = None
```

### 23.2 `TaskRecord`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TaskRecord:
    task_id: str
    call_id: str
    text: str
    owner: str
    type: str
    source_timestamp: str
    source_quote: str
    status: str
    deadline_text: Optional[str] = None
    deadline_iso: Optional[str] = None
    confidence: float = 0.0
    manually_edited: bool = False
```

---

## 24. SQLite schema

### 24.1 Design principle
SQLite is the index/search layer, not the master artifact store.

### 24.2 Tables

#### `calls`
```sql
CREATE TABLE calls (
    call_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    original_file_name TEXT,
    call_dir TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    call_datetime TEXT,
    duration_sec INTEGER,
    file_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    review_status TEXT,
    transcript_confidence TEXT,
    analysis_confidence TEXT,
    languages TEXT,
    participant_hint TEXT,
    tasks_count INTEGER NOT NULL DEFAULT 0,
    confirmed_tasks_count INTEGER NOT NULL DEFAULT 0,
    dismissed_tasks_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

#### `call_text_index`
```sql
CREATE TABLE call_text_index (
    call_id TEXT PRIMARY KEY,
    transcript_clean TEXT,
    transcript_raw TEXT,
    short_summary TEXT,
    detailed_summary TEXT,
    key_points_text TEXT,
    decisions_text TEXT,
    open_questions_text TEXT,
    commitments_text TEXT,
    FOREIGN KEY (call_id) REFERENCES calls(call_id) ON DELETE CASCADE
);
```

#### `tasks`
```sql
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    text TEXT NOT NULL,
    owner TEXT NOT NULL,
    type TEXT NOT NULL,
    deadline_text TEXT,
    deadline_iso TEXT,
    source_timestamp TEXT,
    source_quote TEXT,
    status TEXT NOT NULL,
    confidence REAL,
    manually_edited INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (call_id) REFERENCES calls(call_id) ON DELETE CASCADE
);
```

#### `processing_runs`
```sql
CREATE TABLE processing_runs (
    run_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    error_text TEXT,
    worker_id TEXT,
    FOREIGN KEY (call_id) REFERENCES calls(call_id) ON DELETE CASCADE
);
```

#### `queue_jobs`
```sql
CREATE TABLE queue_jobs (
    job_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    not_before TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_text TEXT,
    FOREIGN KEY (call_id) REFERENCES calls(call_id) ON DELETE CASCADE
);
```

#### `app_settings`
```sql
CREATE TABLE app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### 24.3 Indexes
```sql
CREATE INDEX idx_calls_imported_at ON calls(imported_at);
CREATE INDEX idx_calls_status ON calls(status);
CREATE INDEX idx_calls_source ON calls(source);
CREATE INDEX idx_tasks_call_id ON tasks(call_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_owner ON tasks(owner);
CREATE INDEX idx_queue_jobs_status_priority ON queue_jobs(status, priority, created_at);
CREATE INDEX idx_processing_runs_call_stage ON processing_runs(call_id, stage);
```

### 24.4 Conventions
1. `languages` stored as JSON string, e.g. `["ru","he","en"]`
2. `source_version` = `extracted` or `reviewed`

---

## 25. UI specification

### 25.1 Screen A — Archive
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ Call Assistant                                      [Rescan] [Queue] [Settings]             │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Search: [________________________________________]                                          │
│ Filters: Source [All▼]  Status [All▼]  Confidence [All▼]  Date [____][____]                │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Date/Time        │ Source   │ Dur. │ Langs    │ Status     │ Review     │ Tasks │ Conf.   │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2026-04-03 18:30 │ mobile   │ 14m  │ ru/he    │ indexed    │ reviewed   │ 3     │ medium  │
│ 2026-04-03 16:05 │ mobile   │  6m  │ en       │ indexed    │            │ 1     │ high    │
│ 2026-04-02 21:11 │ unknown  │ 22m  │ ru/en    │ failed     │            │ 0     │ low     │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Open Selected] [Retry Failed]                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 25.2 Screen B — Call Detail
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ Call Detail                                                           [Back to Archive]     │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Call ID: 2026-04-03_183012_ab12                                                              │
│ Source: mobile   Date: 2026-04-03 18:30   Duration: 14m 02s   Status: indexed   Conf: med  │
│ Files: [Open Folder] [Play Audio] [Reprocess] [Mark Reviewed]                               │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ SHORT SUMMARY                                                                                │
│ Short summary text...                                                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ KEY POINTS                                                                                   │
│ • Point 1                                                                                    │
│ • Point 2                                                                                    │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ DECISIONS          │ OPEN QUESTIONS                │ COMMITMENTS                             │
│ • Decision 1       │ • Question 1                  │ • [me] Commitment 1                    │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ DETAILED SUMMARY                                                                             │
│ [scrollable text area...................................................................]   │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ TABS: [Tasks] [Clean Transcript] [Raw Transcript] [Processing Log]                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tab content area                                                                             │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 25.3 Screen C — Tasks tab
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ TASKS                                                                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Ver: [Extracted▼]  [Load Reviewed] [Save Reviewed]                                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Status   │ Owner   │ Type       │ Deadline      │ Text                                       │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ new      │ me      │ task       │ tomorrow      │ Send revised pricing file                  │
│ new      │ other   │ waiting    │               │ Confirm updated quantities                 │
│ new      │ me      │ follow_up  │ next week     │ Check final delivery date                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Selected Task                                                                                │
│ Text:        [______________________________________________________________]               │
│ Owner:       [me▼]   Type: [task▼]   Status: [confirmed▼]                                    │
│ Deadline:    [________________]   ISO: [________________]                                    │
│ Confidence:  0.88                                                                            │
│ Timestamp:   00:12:43                                                                        │
│ Quote:       "I'll send you the revised pricing file tomorrow."                              │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Confirm] [Dismiss] [Add Task] [Delete Task] [Apply Changes]                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 25.4 Screen D — Clean Transcript tab
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ CLEAN TRANSCRIPT                                                                             │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ 00:00:02  me:    Hello, I wanted to discuss the updated pricing and delivery date.          │
│ 00:00:07  other: Yes, I saw the previous file. Some quantities still need correction.       │
│ 00:00:15  me:    Fine, I'll send you a revised version tomorrow.                            │
│ ...                                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Copy Text] [Open Raw]                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 25.5 Screen E — Raw Transcript tab
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ RAW TRANSCRIPT                                                                               │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ JSON / raw segment view                                                                      │
│ speaker, start, end, text, engine_confidence                                                 │
│ ...                                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Copy JSON] [Open Clean]                                                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 25.6 Screen F — Queue / Processing view
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ QUEUE                                                                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Job ID     │ Call ID                  │ Stage          │ Status   │ Attempts │ Updated       │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ j_001      │ 2026-04-03_183012_ab12   │ analysis       │ running  │ 1        │ 18:36:10      │
│ j_002      │ 2026-04-03_160501_ff09   │ indexing       │ queued   │ 0        │ 18:36:02      │
│ j_003      │ 2026-04-02_211155_aa31   │ transcription  │ failed   │ 3        │ 18:35:45      │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Retry Selected] [Clear Done] [Open Call]                                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 25.7 UI behavior rules
1. Archive default sort: newest first.
2. Search checks transcript, summaries, and tasks.
3. Failed calls show warning marker.
4. Low confidence is shown as badge only.
5. Summary blocks remain near top of call detail.
6. Tasks are a dedicated tab.
7. Processing log is visible but not dominant.
8. User edits affect reviewed version only.
9. Reviewed tasks override extracted tasks for display counts when available.

### 25.8 Empty, error, and loading states
The UI should support:
1. empty archive,
2. empty queue,
3. missing summary,
4. failed transcript,
5. failed analysis,
6. low-confidence badge display,
7. loading state while call detail opens.

---

## 26. API route layout

```text
GET  /archive
GET  /call/{call_id}
GET  /call/{call_id}/tasks
POST /call/{call_id}/tasks/reviewed
POST /call/{call_id}/reprocess
GET  /queue
POST /queue/retry/{job_id}
GET  /api/search
```

---

## 27. Test strategy

### 27.1 Unit tests
1. ingest
2. audio
3. transcription
4. diarization
5. transcript cleaning
6. analysis
7. task normalization
8. indexing
9. UI route basics

### 27.2 Integration tests
1. end-to-end pipeline test,
2. rebuild SQLite from files test,
3. duplicate-ingest prevention test,
4. reviewed-task preservation behavior test,
5. queue retry behavior test.

### 27.3 Fixtures
1. short sample audio,
2. mixed-language sample audio,
3. low-confidence sample audio,
4. duplicate file fixture,
5. reviewed task fixture.

---

## 28. Recommended implementation order

### 28.1 Sprint 1
1. folder structure
2. core models
3. ingest
4. metadata lifecycle
5. queue skeleton

### 28.2 Sprint 2
1. audio prep
2. transcription
3. diarization
4. raw/clean transcript outputs

### 28.3 Sprint 3
1. analysis
2. task normalization
3. confidence handling

### 28.4 Sprint 4
1. SQLite indexing
2. archive UI
3. call detail UI
4. task review UI

### 28.5 Later
1. queue view
2. Telegram integration
3. richer search

---

## 29. Open questions intentionally deferred

1. Exact acquisition method for WhatsApp recordings.
2. Exact acquisition method for Teams recordings.
3. Exact source classification logic for synced files.
4. Exact algorithm for reliable **me / other** speaker resolution.
5. Final local transcription engine choice.
6. Optional cloud or hybrid provider strategy.
7. Telegram task commands and routing rules.
8. Semantic search design.

---

## 30. Telegram future phase

Telegram is expected to become a **task-list interface**, not just a message dump.

Likely future behavior:
1. publish confirmed tasks,
2. show task status,
3. allow simple status changes,
4. avoid sending full transcripts by default.

Telegram remains outside MVP critical processing path.
