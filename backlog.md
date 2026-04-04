# Backlog

This file tracks remaining product and engineering tasks.

Done items are listed briefly for context.
Open items are the actual backlog to be implemented.

## Done

### Speaker identity foundation

- added `speaker_identity` stage to the worker pipeline
- added SQLite tables:
  - `speaker_profiles`
  - `speaker_embeddings`
  - `speaker_assignments`
- added speaker embedding and matching service
- added call-detail speaker assignment UI
- added `/speakers` and `/speakers/{speaker_identity_id}` pages
- added tests for speaker identity flow

### Retranscription and processing controls

- added retranscribe support with model selection
- added batch retranscribe CLI
- added manual processing controls in the UI
- added full-call processing in manual-step mode

### Archive and ingest improvements

- added recorded-time resolution from filename / metadata / fallback
- added recorded-time backfill CLI
- added live incoming-folder scanning improvements
- added duplicate-source-path short-circuiting
- added stage-specific runtime states such as:
  - `transcribing`
  - `diarizing`
  - `analyzing`

## Open

### High priority

#### Speaker profiles: rename / hide / delete

Status: not done

Tasks:

- add rename action for speaker profiles
- add hide action for speaker profiles
- add safe delete or archive flow for speaker profiles
- define how deletes affect existing assignments and transcript display names

#### Suggested speaker matches in UI

Status: partially done in backend, not done in UI

Tasks:

- surface `suggested` speaker matches on the call detail page
- show suggested identity name and match score
- add accept / reject actions for suggested matches
- make suggested-vs-confirmed assignment visually clear

#### Real-world validation of speaker identity quality

Status: not done

Tasks:

- evaluate matching quality on real calls
- validate current thresholds:
  - `auto_assign_threshold`
  - `suggest_threshold`
- document recommended threshold values
- identify common false-match and missed-match cases

### Medium priority

#### Limit embedding input size

Status: not done

Tasks:

- add `max_segments_per_cluster` behavior to speaker identity processing
- decide whether to cap by:
  - segment count
  - total duration
  - both
- prefer representative segments instead of all segments for long speakers

#### Better diarization observability

Status: partially done

Tasks:

- make fallback diarization mode more visible in the UI
- show whether diarization was:
  - real clustered diarization
  - single-speaker fallback
- expose diarization confidence more clearly on the call page

#### Speaker identity management on speaker pages

Status: partially done

Tasks:

- add profile notes editing
- show more useful linked-call summaries
- show assignment examples with timestamps or transcript excerpts

### Low priority

#### Alternative embedding backend

Status: not done

Tasks:

- evaluate a non-pyannote embedding backend
- add backend abstraction if needed
- keep pyannote as the default unless a better fallback is proven

#### Backlog cleanup and structure

Status: ongoing

Tasks:

- keep this file as a task list, not a design document
- move large design notes into separate docs if needed
- keep each backlog item phrased as implementable work

## Next recommended milestone

### Speaker identity UX completion

Goal:

- make the existing speaker identity backend fully usable and reviewable in the UI

Recommended scope:

- rename / hide speaker profiles
- suggested-match review flow
- clearer confidence display
- better fallback-diarization visibility
