# Manual QA Checklist

This document lists the implemented product features in the current `Phone-to-Tasks` codebase and gives a manual test checklist for each one.

It is intended for feature-by-feature validation of:
- ingest
- processing pipeline
- transcription
- diarization
- speaker identity
- UI workflows
- queue behavior
- maintenance scripts

## Feature Inventory

### Core system
- Local FastAPI UI on `/calls`, `/queue`, `/speakers`
- Archive-first processing model with SQLite index
- Manual-step mode and automatic worker mode
- Status line with running/queued/failed counts and current task info
- Restart script for local server

### Ingest
- Incoming-folder ingest from `data/incoming`
- Browser upload of supported audio files
- Duplicate detection by SHA-256
- Fast duplicate skip by known source path
- Per-call archive folder creation

### Pipeline
- `audio_prepare`
- `transcription`
- `diarization`
- `speaker_identity`
- `transcript_clean`
- `analysis`
- `indexing`

### Transcription
- Local Whisper transcription with selectable models
- OpenAI cloud transcription with `whisper-1`
- Retranscribe from UI with model selection
- Language selection on retranscribe
- Quality-aware retry with forced Hebrew for suspicious transcripts

### Diarization and speakers
- Diarization with pyannote when available
- Fallback single-speaker diarization mode
- Manual speaker mapping (`me` / `other` / `unknown`)
- Cross-call speaker identity stage
- Suggested speaker identity acceptance/rejection
- Create / assign / clear speaker identity
- Speaker profile rename
- Speaker profile archive

### Analysis and tasks
- Summary generation
- Short description in archive list
- Task extraction
- Task review/edit flow

### Queue and recovery
- Queue view and retry
- Manual `Process Next Call`
- Manual `Scan Incoming`
- Stage-aware stale job handling
- Startup recovery of stale running jobs
- In-progress states in UI

### Maintenance and CLI
- Reindex from UI
- `process_downloads.py`
- `show_transcript.py`
- `retranscribe_all.py`
- `backfill_recorded_times.py`

### UI/UX
- Compact archive table
- Collapsible long blocks on major pages
- Synced transcript segments with audio click-to-seek

## Manual Test Checklist

## 1. Server startup and page availability

### Steps
1. Run:
   ```bash
   cd /home/garik/Documents/git/Phone-to-Tasks
   ./restart_server.sh
   ```
2. Open:
   - `http://127.0.0.1:8081/calls`
   - `http://127.0.0.1:8081/queue`
   - `http://127.0.0.1:8081/speakers`

### Expected
- all pages load
- no 500 page
- status line is visible

## 2. Status line

### Steps
1. Open `/calls`
2. Observe status with no active work
3. Trigger a manual action or queue work
4. Wait for polling update

### Expected
- counts update
- current task updates when work runs
- app status is truthful:
  - `Idle`
  - `Paused`
  - `Running`
  - `Needs attention`

## 3. Incoming folder import

### Steps
1. Copy a new supported audio file into [data/incoming](/home/garik/Documents/git/Phone-to-Tasks/data/incoming)
2. If in manual mode, click `Scan Incoming`
3. Open `/calls`

### Expected
- a new call appears
- archive folder is created
- queue entry for processing exists

## 4. Browser upload

### Steps
1. Open `/calls`
2. Upload a supported audio file

### Expected
- redirect back to `/calls`
- success notice shown
- imported call appears in archive list

## 5. Duplicate detection

### Steps
1. Import one audio file
2. Re-upload or re-place the same file

### Expected
- no duplicate call is created
- import is skipped

## 6. Archive list view

### Steps
1. Open `/calls`
2. Verify visible columns:
   - recorded date/time
   - duration
   - respondent
   - state
   - tasks
   - short description

### Expected
- rows are compact
- short description is clamped
- each row links to call detail

## 7. Recorded-time extraction

### Steps
1. Check a call with filename timestamp like `_YYMMDD_HHMMSS`
2. Verify displayed recorded time matches filename
3. Run:
   ```bash
   cd /home/garik/Documents/git/Phone-to-Tasks
   .venv/bin/python backfill_recorded_times.py
   ```

### Expected
- recorded time comes from filename if available
- non-filename sources show badges:
  - `metadata`
  - `fallback`

## 8. Call detail page

### Steps
1. Open a processed call
2. Check:
   - pipeline status
   - transcription quality
   - audio
   - summary
   - transcript
   - speakers
   - tasks
   - processing log

### Expected
- page renders fully
- large blocks are collapsible
- key controls stay visible

## 9. Audio playback and synced transcript

### Steps
1. Open a call with audio and transcript segments
2. Play audio
3. Click a transcript segment

### Expected
- active segment highlights during playback
- clicking a segment seeks audio to that point

## 10. Audio normalization

### Steps
1. Import a new file
2. Open its archive directory

### Expected
- both files exist:
  - `audio_original.*`
  - `audio_normalized.wav`

## 11. Local transcription

### Steps
1. Process a call with local model
2. Check artifacts and metadata

### Expected
- `transcript_raw.json` exists
- provider is `local`
- local model is recorded in metadata

## 12. Cloud transcription

### Steps
1. Retranscribe a call with `whisper-1`
2. Check artifacts and metadata

### Expected
- provider is `cloud`
- model is `whisper-1`
- transcript is updated

## 13. Retranscribe from UI

### Steps
1. Open a call
2. Use `Retranscribe`
3. Choose a different model
4. Optionally choose explicit language

### Expected
- downstream artifacts rebuild
- call re-enters processing stages
- new transcript replaces old one

## 14. Transcription quality recovery

### Steps
1. Use a short ambiguous Hebrew-like call
2. Retranscribe with language mode `Auto`
3. Open `Transcription Quality` section

### Expected
- quality metadata appears
- strategy may show:
  - `baseline`
  - `forced_hebrew`
  - `merged_segments`
- retry flags and languages appear when triggered

## 15. Diarization

### Steps
1. Process a multi-speaker call
2. Open call detail
3. Check diarization mode and transcript segments

### Expected
- clustered speakers when backend succeeds
- explicit fallback notice when single-speaker fallback is used

## 16. Speaker mapping

### Steps
1. Open a call with speaker clusters
2. Assign clusters to:
   - `me`
   - `other`
   - `unknown`
3. Save mapping

### Expected
- mapping persists
- transcript display updates

## 17. Speaker identity stage

### Steps
1. Process a call through `speaker_identity`
2. Open call detail

### Expected
- speaker identity mode is visible:
  - `full`
  - `degraded`
  - `skipped`
- summary text explains the outcome

## 18. Assign existing speaker identity

### Steps
1. Open a call with speaker clusters
2. In `Speaker Identities`, choose an existing profile
3. Click `Assign existing`

### Expected
- assignment persists
- call display updates

## 19. Create and assign speaker identity

### Steps
1. Open a call with speaker clusters
2. Enter a new profile name
3. Click `Create and assign`

### Expected
- new profile appears in `/speakers`
- current call cluster is assigned to it

## 20. Clear speaker identity

### Steps
1. Open a call with assigned identity
2. Click `Clear identity`

### Expected
- assignment is removed
- transcript outputs refresh

## 21. Suggested identity review

### Steps
1. Find a call with `Suggested` identity
2. Click `Accept suggestion`
3. Repeat on another suggested cluster with `Reject suggestion`

### Expected
- accepted suggestion becomes confirmed
- rejected suggestion is removed

## 22. Speaker profiles list

### Steps
1. Open `/speakers`

### Expected
- table shows:
  - name
  - status
  - linked calls
  - last seen

## 23. Speaker profile detail

### Steps
1. Open `/speakers/{speaker_identity_id}`

### Expected
- rename form visible
- archive action visible
- assignments table visible
- raw profile block collapsible

## 24. Rename speaker profile

### Steps
1. Rename a profile on its detail page

### Expected
- name updates on:
  - profile page
  - speaker list
  - linked call pages

## 25. Archive speaker profile

### Steps
1. Archive a non-hidden speaker profile

### Expected
- status becomes `hidden`
- hidden profile is excluded from normal assignment dropdowns
- historical assignments still show

## 26. Transcript cleaning

### Steps
1. Process a call through transcript cleaning
2. Open detail page and archive directory

### Expected
- `transcript_clean.txt` exists
- clean transcript is visible in UI

## 27. Summary generation

### Steps
1. Process a call through analysis
2. Open call detail and archive list

### Expected
- `summary.json` exists
- summary block shows content
- archive short description reflects summary

## 28. Task extraction

### Steps
1. Process a call likely to contain action items
2. Open `Review Tasks`

### Expected
- extracted tasks are present
- editable in UI

## 29. Task review/edit

### Steps
1. Edit extracted task fields:
   - text
   - owner
   - type
   - status
   - deadline
2. Save reviewed tasks

### Expected
- `tasks_reviewed.json` updates
- review state changes
- UI reflects reviewed data

## 30. Queue page and retry

### Steps
1. Open `/queue`
2. Check jobs with and without errors
3. Retry a failed job

### Expected
- queue page renders compactly
- long errors are expandable
- retry returns failed job to queued state

## 31. Manual processing mode

### Steps
1. Confirm app is in manual mode
2. Use `Scan Incoming`
3. Use `Process Next Call`

### Expected
- `Scan Incoming` only scans/imports
- `Process Next Call` processes without blocking the page
- action returns immediately and work continues in background

## 32. Automatic mode

### Steps
1. Switch config to automatic mode
2. Restart app
3. Add new file to incoming folder

### Expected
- no button click required
- worker scans and processes automatically

## 33. Startup stale-job recovery

### Steps
1. Leave a job in running state
2. Restart the app

### Expected
- stale running job is requeued
- status line does not show stale work as live running

## 34. In-progress states

### Steps
1. Observe a call while it moves through processing

### Expected
- states such as these appear appropriately:
  - `preparing_audio`
  - `transcribing`
  - `diarizing`
  - `resolving_speaker_identity`
  - `cleaning_transcript`
  - `analyzing`
  - `indexing`

## 35. Reindex action

### Steps
1. Open a processed call
2. Click `Reindex`

### Expected
- request succeeds
- call remains searchable and intact

## 36. `process_downloads.py`

### Steps
1. Run:
   ```bash
   cd /home/garik/Documents/git/Phone-to-Tasks
   .venv/bin/python process_downloads.py --list
   ```
2. Then run:
   ```bash
   .venv/bin/python process_downloads.py
   ```

### Expected
- `--list` shows eligible incoming files
- one-shot processing imports and drains the queue

## 37. `show_transcript.py`

### Steps
1. Run:
   ```bash
   cd /home/garik/Documents/git/Phone-to-Tasks
   .venv/bin/python show_transcript.py
   ```
2. Run again with a specific `call_id`

### Expected
- clean transcript is printed for latest or selected call

## 38. `retranscribe_all.py`

### Steps
1. Run help:
   ```bash
   cd /home/garik/Documents/git/Phone-to-Tasks
   .venv/bin/python retranscribe_all.py --help
   ```
2. Run a targeted call reset:
   ```bash
   .venv/bin/python retranscribe_all.py --call-id <call_id> --model local:large-v3-turbo --no-process
   ```

### Expected
- call reset occurs from transcription onward
- queue entries are created

## 39. `backfill_recorded_times.py`

### Steps
1. Run:
   ```bash
   cd /home/garik/Documents/git/Phone-to-Tasks
   .venv/bin/python backfill_recorded_times.py
   ```
2. Refresh `/calls`

### Expected
- existing recorded times are corrected
- recorded-time badges are updated

## 40. Compact/collapsible UI

### Steps
1. Open:
   - `/calls`
   - one large `/calls/{call_id}`
   - `/queue`
   - `/speakers`
   - `/speakers/{speaker_identity_id}`
2. Expand and collapse long sections

### Expected
- call detail long blocks are collapsed by default
- queue long errors expand inline
- speaker raw profile is collapsible
- archive list remains compact

## Recommended full manual QA scenarios

### Scenario A: New incoming call end-to-end
1. Drop a new audio file into `data/incoming`
2. Import/process it
3. Verify full pipeline completion and archive UI visibility

### Scenario B: Upload then retranscribe
1. Upload a file from browser
2. Open call detail
3. Retranscribe with another model/provider
4. Verify transcript and downstream artifacts changed

### Scenario C: Speaker workflow
1. Process a multi-speaker call
2. Map `me/other`
3. Create speaker profile
4. Assign it
5. Rename it
6. Archive it

### Scenario D: Queue and recovery
1. Trigger queued work
2. Retry a failed job
3. Restart during processing
4. Verify stale-job recovery

### Scenario E: Multilingual transcript recovery
1. Use a short ambiguous Hebrew-like call
2. Retranscribe in auto mode
3. Verify transcription quality strategy and recovery metadata

## Acceptance criteria

The manual QA pass is complete when:
- all major UI pages render
- at least one call completes end-to-end
- at least one retranscribe succeeds
- at least one speaker workflow succeeds
- at least one degraded/fallback case is visible and understandable
- restart does not leave stale work shown as active
