from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AudioMetadata:
    duration_seconds: float | None
    sample_rate: int
    channels: int
    audio_format: str


@dataclass
class RawSegment:
    start_sec: float
    end_sec: float
    text: str
    confidence: float | None = None
    speaker: str | None = None


@dataclass
class RawTranscript:
    provider: str
    model: str
    language: str
    confidence: float | None
    segments: list[RawSegment]
    text: str


@dataclass
class TranscriptSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    speaker_label: str
    speaker_channel_label: str | None
    text: str
    confidence: float | None
    speaker_cluster_id: str = "speaker_1"
    diarization_confidence: str | None = None


@dataclass
class CleanTranscript:
    text: str
    segments: list[TranscriptSegment]


@dataclass
class TaskRecord:
    task_id: str
    text: str
    owner: str | None
    type: str
    source_timestamp: float | None
    source_quote: str | None
    status: str = "new"
    deadline: str | None = None
    confidence: float | None = None
    notes: str | None = None
    reviewed_at: str | None = None
    review_source: str | None = None
    original_task_id: str | None = None


@dataclass
class AnalysisBundle:
    short_summary: str
    detailed_summary: str
    key_points: list[str]
    decisions: list[str]
    open_questions: list[str]
    commitments: list[str]
    tasks: list[TaskRecord]
    analysis_confidence: float | None
    analysis_language: str
    low_confidence_reason: str | None = None


@dataclass
class CallMetadata:
    schema_version: str
    app_version: str
    call_id: str
    source_filename: str
    source_path: str
    imported_at: str
    recorded_at: str | None
    file_size_bytes: int
    sha256: str
    audio_format: str | None = None
    duration_seconds: float | None = None
    language_hints: list[str] = field(default_factory=list)
    current_state: str = "detected"
    review_state: str = "pending"
    low_confidence: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def archive_relative_parts(self) -> list[str]:
        timestamp = datetime.fromisoformat(self.imported_at.replace("Z", "+00:00"))
        return [f"{timestamp.year:04d}", f"{timestamp.month:02d}", f"{timestamp.day:02d}", f"call_{self.call_id}"]


@dataclass
class QueueJob:
    job_id: str
    call_id: str
    stage: str
    status: str
    priority: int
    attempt_count: int
    max_attempts: int
    available_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None


def artifact_paths(call_dir: Path) -> dict[str, Path]:
    return {
        "metadata": call_dir / "metadata.json",
        "audio_normalized": call_dir / "audio_normalized.wav",
        "transcript_raw": call_dir / "transcript_raw.json",
        "transcript_segments": call_dir / "transcript_segments.json",
        "transcript_clean": call_dir / "transcript_clean.txt",
        "summary": call_dir / "summary.json",
        "tasks": call_dir / "tasks.json",
        "tasks_reviewed": call_dir / "tasks_reviewed.json",
        "processing_log": call_dir / "processing_log.json",
    }


def call_dir_from_metadata(archive_root: Path, metadata: CallMetadata) -> Path:
    return archive_root.joinpath(*metadata.archive_relative_parts)
