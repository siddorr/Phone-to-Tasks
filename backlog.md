# Backlog

## Cross-Call Speaker Recognition

Status: planned
Priority: high

### Goal

Recognize the same real speaker across multiple calls, not just within a single call, and allow persistent naming such as `Natasha`, `Mom`, or `Client A`.

### Outcome

The app should be able to:

- assign stable speaker identities across calls
- keep per-call diarization clusters separate from global identities
- let the user confirm or correct speaker identity in the UI
- use confirmed speaker identities in future calls when confidence is high
- fall back to `unknown` when confidence is low

### Scope

In scope:

- speaker embedding extraction
- persistent speaker profile storage
- cross-call matching
- manual labeling and correction UI
- confidence thresholds and conservative auto-assignment
- reindex support for identity updates

Out of scope for first version:

- full biometric-grade identification
- automatic speaker recognition without user confirmation history
- merging different people automatically with no review path
- multi-user profile sharing

### Architecture

Add a new speaker identity layer on top of existing diarization.

Current model:

- diarization creates per-call speaker clusters such as `speaker_1`, `speaker_2`

Target model:

- per-call cluster: `speaker_cluster_id`
- optional global identity: `speaker_identity_id`
- user-facing label: `speaker_display_name`

This preserves the distinction between:

- who the model thinks is a consistent voice across calls
- what the user has confirmed that voice to be

### Data Model

Add SQLite tables:

#### `speaker_profiles`

Columns:

- `speaker_identity_id` TEXT PRIMARY KEY
- `display_name` TEXT NOT NULL
- `status` TEXT NOT NULL
  - allowed: `confirmed`, `candidate`, `hidden`
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL
- `notes` TEXT NULL

#### `speaker_embeddings`

Columns:

- `embedding_id` TEXT PRIMARY KEY
- `speaker_identity_id` TEXT NULL
- `call_id` TEXT NOT NULL
- `speaker_cluster_id` TEXT NOT NULL
- `segment_count` INTEGER NOT NULL
- `duration_seconds` REAL NOT NULL
- `embedding_vector_json` TEXT NOT NULL
- `model_name` TEXT NOT NULL
- `confidence` REAL NULL
- `created_at` TEXT NOT NULL

#### `speaker_assignments`

Columns:

- `assignment_id` TEXT PRIMARY KEY
- `call_id` TEXT NOT NULL
- `speaker_cluster_id` TEXT NOT NULL
- `speaker_identity_id` TEXT NULL
- `assignment_source` TEXT NOT NULL
  - allowed: `auto`, `user`
- `match_score` REAL NULL
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL

### Artifact Changes

Extend `transcript_segments.json` entries with:

- `speaker_cluster_id`
- `speaker_identity_id`
- `speaker_display_name`
- `diarization_confidence`
- `identity_confidence`

Rules:

- `speaker_cluster_id` is always present
- `speaker_identity_id` is present only when matched or assigned
- `speaker_display_name` is:
  - confirmed profile name when available
  - otherwise `speaker_1`, `speaker_2`, etc.

### Embedding Pipeline

Add a new stage after `diarization` and before `transcript_clean`:

- `speaker_identity`

Responsibilities:

1. group transcript segments by `speaker_cluster_id`
2. extract audio spans for each cluster from `audio_normalized.wav`
3. compute a voice embedding for each cluster
4. compare embedding against known confirmed speaker profiles
5. assign:
   - confirmed identity if score >= high threshold
   - no identity if score < threshold
6. persist embedding and assignment
7. update `transcript_segments.json`

### Embedding Backend

Recommended backend:

- `pyannote.audio` embeddings if diarization already depends on pyannote

Fallback option:

- SpeechBrain ECAPA speaker embeddings

Decision for first implementation:

- use pyannote-based embeddings if pyannote is installed
- do not add a second backend in the first version

### Matching Logic

For each cluster embedding:

1. compare against all `confirmed` speaker profiles
2. compute cosine similarity
3. choose the highest scoring profile
4. assign only if score >= threshold

Thresholds:

- `>= 0.75`: auto-assign
- `0.60 - 0.74`: show as suggested match, but keep unassigned by default
- `< 0.60`: unknown

These values should live in config.

### Configuration

Add config section:

```yaml
speaker_identity:
  enabled: true
  provider: "pyannote"
  auto_assign_threshold: 0.75
  suggest_threshold: 0.60
  min_cluster_duration_seconds: 6
  max_segments_per_cluster: 20
```

### UI Changes

Update call detail page:

- show each `speaker_cluster_id`
- show current identity assignment if any
- allow user to:
  - assign to existing profile
  - create new profile name
  - clear assignment

Add new speaker profile page:

- `/speakers`

Show:

- confirmed speaker profiles
- number of linked calls
- most recent activity
- rename option
- hide/delete option

Add speaker detail page:

- `/speakers/{speaker_identity_id}`

Show:

- linked calls
- linked clusters
- assignment confidence examples

### User Workflow

1. call is diarized into `speaker_1`, `speaker_2`
2. system computes cluster embeddings
3. if strong match exists, assign known identity
4. user reviews call page
5. user confirms:
   - `speaker_2 = Natasha`
6. future calls with strong voice similarity can reuse `Natasha`

### Safety Rules

- never silently overwrite a user assignment with an automatic one
- user assignment always wins
- auto-assign only above threshold
- otherwise leave unknown
- keep full cluster history for traceability

### Worker / Queue Changes

Add new stage ordering:

- `audio_prepare`
- `transcription`
- `diarization`
- `speaker_identity`
- `transcript_clean`
- `analysis`
- `indexing`

If `speaker_identity` fails:

- log error
- continue with transcript cleaning only if configured
- default behavior for first version: continue with unknown identities

### API / UI Form Interfaces

Add:

- `POST /calls/{call_id}/speakers/assign`
- `POST /calls/{call_id}/speakers/create-and-assign`

## Calls Archive Sorting and Recent Filters

Status: done
Priority: medium

### Goal

Improve review speed on the Calls archive page with faster sorting and recent-date filtering.

### Scope

In scope:

- sortable columns for all visible call table fields
- relative date filters for:
  - last day
  - last 3 days
  - last week
  - last month
- keep the existing exact date picker for precise lookup
- preserve current sort/filter state in the URL

Out of scope for first version:

- multi-column sorting
- custom date ranges
- saved filter presets

### UI Changes

Update `/calls`:

- make each visible column header sortable
- add sort direction indicator
- add a `Recent` dropdown with:
  - `All`
  - `Last day`
  - `Last 3 days`
  - `Last week`
  - `Last month`
- keep keyword, source, and exact date controls
- add a `Clear` action to reset filters

Sortable columns:

- `Recorded Date/Time`
- `Respondent`
- `State`
- `Tasks`
- `Short Description`

### Route and Query Parameters

Extend `GET /calls` with:

- `recent`
- `sort_by`
- `sort_dir`

Allowed `recent` values:

- `1d`
- `3d`
- `7d`
- `30d`

Allowed `sort_by` values:

- `recorded_at`
- `respondent`
- `state`
- `tasks`
- `short_description`

Allowed `sort_dir` values:

- `asc`
- `desc`

### Sorting Rules

Default sorting:

- `recorded_at desc`

Per-column default directions:

- `recorded_at`: `desc`
- `respondent`: `asc`
- `state`: `asc`
- `tasks`: `desc`
- `short_description`: `asc`

Tie-breaking:

1. effective timestamp descending
2. `call_id` ascending

### Filtering Rules

Use `COALESCE(recorded_at, imported_at)` as the effective call timestamp.

Rules:

- if `date` is set, exact date filtering wins
- if `date` is empty and `recent` is set, apply relative time filtering
- `Last week` means rolling last 7 days
- `Last month` means rolling last 30 days

### Implementation Notes

- derive respondent, task count, and short description before sorting
- use Python-side sorting for derived fields instead of adding schema changes
- keep current DB filtering for keyword, source, and exact date

### Acceptance Criteria

- user can sort by every visible column
- user can filter to last day / 3 days / week / month
- current filter and sort state stays in the URL
- active sort direction is visible in the header
- exact date and recent filter controls can coexist without ambiguous behavior

## Audio Playback with Transcript Highlighting

Status: planned
Priority: high

### Goal

Allow the user to play call audio directly from the GUI and see the relevant transcript lines highlighted in sync with playback.

### Scope

In scope:

- audio playback from the call detail page
- playback controls in the browser
- transcript line highlighting based on playback position
- click-to-seek from transcript line to audio timestamp
- support for both raw segment timing and cleaned transcript display

Out of scope for first version:

- waveform visualization
- word-level highlighting
- variable-speed transcription re-alignment
- live streaming from a remote source

### UI Changes

Update `/calls/{call_id}`:

- add an audio player section
- add a `Play audio` control for the archived call audio
- render transcript segments as individually addressable lines
- highlight the active segment while audio is playing
- auto-scroll active segment into view when needed

Playback behavior:

- if the user clicks a transcript segment, seek audio to that segment start
- if audio is paused, keep the last active segment highlighted
- if timestamps are missing, disable synced highlighting but keep playback available

### Audio Source

Use archived call audio from the call folder.

Preferred source order:

1. `audio_original.*`
2. `audio_normalized.wav`

Expose a backend route for the browser to fetch the audio file safely instead of linking directly to arbitrary filesystem paths.

### Backend Changes

Add route:

- `GET /calls/{call_id}/audio`

Behavior:

- resolve the call from `calls.archive_path`
- serve the preferred audio file for that call
- return `404` if no playable audio file exists
- support browser range requests if the serving mechanism requires it

Template context changes:

- pass a playback-safe audio URL
- pass transcript segments with stable DOM IDs and `start_sec` / `end_sec`

### Frontend Behavior

Add lightweight JavaScript on the call detail page:

1. listen to audio `timeupdate`
2. find the active transcript segment whose time range contains the current playback time
3. add an active CSS class to that segment
4. remove the class from previously active segments
5. scroll the active segment into view if it is outside the visible transcript area

Click-to-seek:

- clicking a segment seeks the player to `start_sec`
- optionally starts playback if the audio was already playing

### Transcript Rendering Rules

Primary sync source:

- `transcript_segments.json`

Rules:

- highlighting is based on segment timestamps, not cleaned transcript paragraphs
- cleaned transcript can remain visible, but segment list is the synced view
- if diarization exists, keep speaker labels visible in the synced segment list

### Failure Handling

- if audio file is missing, show transcript with no player
- if transcript segments are missing, show player with no synced highlight
- if timestamps are malformed, disable sync logic and log the issue

### Performance Rules

- do not poll the server during playback
- load all needed segment timing data when rendering the page
- use minimal DOM updates on each `timeupdate`

### Acceptance Criteria

- user can play a call recording from the call detail page
- active transcript segment highlights while audio is playing
- clicking a transcript segment seeks playback to that point
- player and transcript remain usable when sync data is missing
- implementation works on current desktop browser target without a frontend framework

## Call Duration in Archive List

Status: planned
Priority: medium

### Goal

Show each call duration directly in the Calls archive list so the user can quickly assess call length without opening the call detail page.

### Scope

In scope:

- display call duration on `/calls`
- use existing `duration_seconds` from the indexed call record
- format duration into a readable display value

Out of scope for first version:

- duration-based filtering
- duration-based sorting beyond adding it later as a separate enhancement

### UI Changes

Update the Calls archive table to include a `Duration` column.

Formatting rules:

- under 1 hour: `MM:SS`
- 1 hour or longer: `H:MM:SS`
- missing duration: `-`

Recommended placement:

- place `Duration` near `Recorded Date/Time`

### Backend Changes

No schema changes required.

Use existing:

- `calls.duration_seconds`

Add a small display formatter in the calls-list route or UI helper layer.

### Acceptance Criteria

- every indexed call with known duration shows a readable duration in the archive list
- calls without known duration show `-`
- duration display is consistent across the list

## Store Call Summaries in SQLite

Status: planned
Priority: medium

### Goal

Make generated call summaries queryable from SQL while keeping artifact files as the canonical source of truth.

### Scope

In scope:

- project summary fields from `summary.json` into SQLite during indexing
- keep files canonical and SQLite rebuildable
- support SQL inspection and future filtering/reporting on summary content

Out of scope for first version:

- full text-search redesign
- semantic indexing
- replacing artifact files as the source of truth

### Problem

Current design stores canonical summary output in:

- `summary.json`

But SQLite currently stores only limited summary-related information:

- `language_summary`
- `search_text`

This makes it hard to query exact summary fields directly from SQL.

### Design

Keep:

- artifact files as canonical

Add:

- normalized summary projection into SQLite

Recommended schema option:

- create a `call_summaries` table instead of widening `calls`

#### `call_summaries`

Columns:

- `call_id` TEXT PRIMARY KEY
- `short_summary` TEXT NULL
- `detailed_summary` TEXT NULL
- `key_points_json` TEXT NULL
- `decisions_json` TEXT NULL
- `open_questions_json` TEXT NULL
- `commitments_json` TEXT NULL
- `analysis_confidence` REAL NULL
- `analysis_language` TEXT NULL
- `low_confidence_reason` TEXT NULL
- `updated_at` TEXT NOT NULL

### Indexing Changes

Update indexing so `index_call()`:

- reads `summary.json`
- upserts the corresponding `call_summaries` row
- keeps `search_text` behavior as-is

Update rebuild behavior so deleting SQLite and rebuilding from files restores:

- `calls`
- `tasks`
- `artifacts`
- `call_summaries`

### Query and UI Benefits

This enables:

- direct SQL inspection of summary fields
- easier future filters such as:
  - calls with open questions
  - calls with decisions
  - low-confidence summaries
- easier admin/debug tooling

### Safety Rules

- artifact files remain canonical
- SQLite summary rows are projections only
- rebuild from files must remain lossless for indexed summary data

### Acceptance Criteria

- each indexed call with `summary.json` has a matching `call_summaries` row
- SQLite can be rebuilt from artifact files alone
- summary text and structured arrays are available in SQL
- no change to the source-of-truth rule for artifact files
- `POST /calls/{call_id}/speakers/clear`
- `GET /speakers`
- `GET /speakers/{speaker_identity_id}`

### Migration

Add DB migration/initializer updates for:

- `speaker_profiles`
- `speaker_embeddings`
- `speaker_assignments`

Existing calls:

- no destructive migration needed
- speaker identity can be backfilled by re-running from `speaker_identity` onward

### Tests

Unit tests:

- embedding matching respects thresholds
- user assignment overrides automatic assignment
- transcript segment rendering uses confirmed display names
- low-duration clusters do not produce auto-assignment

Integration tests:

- reprocessing one call creates speaker embeddings
- assigning a name in UI persists across reindex
- future call with same speaker auto-matches confirmed profile
- clearing assignment reverts transcript labels to cluster names

Manual acceptance:

- label one speaker in call A
- process call B with same person
- verify identity is auto-suggested or auto-assigned according to threshold

### Rollout Order

1. add DB tables and config
2. add embedding extraction and matching stage
3. persist identity assignments into transcript segments
4. add UI assignment controls
5. add speaker profile pages
6. add backfill/reprocess path

### Acceptance Criteria

- user can name a speaker once and reuse that identity in later calls
- per-call speaker clusters remain traceable
- incorrect auto-assignments can be corrected in UI
- user corrections persist and override future automatic matches
- low-confidence speakers remain unknown instead of being mislabeled
