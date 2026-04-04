# Backlog

## Cross-Call Speaker Recognition

Status: partially implemented
Priority: high

### Summary

The codebase already has a working first version of cross-call speaker recognition:

- per-call speaker clusters from diarization
- a `speaker_identity` worker stage
- speaker embeddings stored in SQLite
- profile creation and assignment UI
- speaker list/detail pages

What remains is mainly product hardening and profile-management UX.

### Already implemented

#### Pipeline

- `speaker_identity` stage exists in the worker pipeline
- current stage order:
  - `audio_prepare`
  - `transcription`
  - `diarization`
  - `speaker_identity`
  - `transcript_clean`
  - `analysis`
  - `indexing`

#### Data model

SQLite tables already exist:

- `speaker_profiles`
- `speaker_embeddings`
- `speaker_assignments`

Transcript segment records already support:

- `speaker_cluster_id`
- `speaker_identity_id`
- `speaker_display_name`
- `diarization_confidence`
- `identity_confidence`

#### Matching implementation

Current backend:

- pyannote-based embedding extraction
- cosine similarity matching against confirmed profiles

Current config section exists:

```yaml
speaker_identity:
  enabled: true
  provider: "pyannote"
  auto_assign_threshold: 0.75
  suggest_threshold: 0.60
  min_cluster_duration_seconds: 6
  continue_on_error: true
```

Current behavior:

- user assignment wins
- high-confidence matches can auto-assign
- midrange matches are stored as `suggested`
- low-confidence matches stay unassigned

#### UI

Implemented UI surfaces:

- call detail page supports:
  - assign existing speaker profile
  - create and assign new speaker profile
  - clear assignment
- speaker profile list:
  - `/speakers`
- speaker profile detail:
  - `/speakers/{speaker_identity_id}`

#### Tests

Implemented tests:

- `tests/test_speaker_identity.py`

### Remaining work

#### 1. Speaker profile management

Still needed:

- rename speaker profile
- hide speaker profile
- delete or archive speaker profile safely

Current limitation:

- profiles can be created and assigned
- they cannot yet be managed after creation from the UI

#### 2. Suggested-match review UX

Still needed:

- explicit UI for `suggested` matches
- user-facing indication of:
  - suggested identity
  - confidence score
  - accept/reject action

Current limitation:

- the backend can distinguish suggested matches
- the UI does not yet expose that flow clearly

#### 3. Embedding input control

Still needed:

- cap or sample segments per cluster before embedding
- add and use `max_segments_per_cluster` in config if needed

Reason:

- long clusters can become unnecessarily expensive
- current backlog spec mentioned this, but the implementation does not yet enforce it

#### 4. Operational validation

Still needed:

- validate speaker identity quality on real calls
- confirm behavior when diarization falls back to single-speaker mode
- measure whether auto-assignment thresholds are appropriate in practice

Reason:

- the infrastructure exists
- production behavior still needs tuning and evaluation

#### 5. Optional backend fallback

Nice-to-have, not required now:

- alternative embedding backend such as SpeechBrain ECAPA

Current state:

- only pyannote is implemented

### Current gaps versus the original design note

These items from the old design note are not fully finished:

- rename/hide/delete profile UX
- richer suggested-match review
- explicit `max_segments_per_cluster` handling
- a second embedding backend

Also, one detail changed from the original note:

- assignment sources are currently:
  - `auto`
  - `user`
  - `suggested`

So the original note listing only `auto` and `user` is outdated.

### Recommended next milestone

#### Milestone: speaker identity UX completion

Goal:

- make the existing speaker identity backend fully usable in the UI

Scope:

- rename profile
- hide profile
- surface suggested matches
- allow accept/reject of suggested identity
- show assignment confidence more clearly on call detail and speaker pages

### Acceptance criteria for this backlog item to move from partial to complete

- user can create, rename, hide, and review speaker profiles
- suggested matches are visible and actionable in the UI
- auto-assignment behavior is understandable and traceable
- speaker identity results remain stable across multiple calls for confirmed profiles
- backlog note matches actual implementation state
