from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import read_json, write_json
from call_assistant.common.models import TranscriptSegment, utc_now

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    speaker_identity_id: str | None
    match_score: float | None
    assignment_source: str | None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _clustered_segments(segments: list[TranscriptSegment]) -> dict[str, list[TranscriptSegment]]:
    grouped: dict[str, list[TranscriptSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.speaker_cluster_id].append(segment)
    return grouped


def _cluster_duration(segments: list[TranscriptSegment]) -> float:
    return sum(max(0.0, segment.end_sec - segment.start_sec) for segment in segments)


def _compute_pyannote_embedding(audio_path: Path, spans: list[tuple[float, float]]) -> list[float] | None:
    from pydub import AudioSegment
    from pyannote.audio import Inference, Model

    audio = AudioSegment.from_file(audio_path)
    if not spans:
        return None

    combined = AudioSegment.empty()
    for start_sec, end_sec in spans:
        start_ms = max(0, int(start_sec * 1000))
        end_ms = max(start_ms, int(end_sec * 1000))
        combined += audio[start_ms:end_ms]
    if len(combined) == 0:
        return None

    combined = combined.set_channels(1).set_frame_rate(16000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        combined.export(temp_path, format="wav")
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model, window="whole")
        vector = inference(str(temp_path))
        if hasattr(vector, "tolist"):
            return [float(item) for item in vector.tolist()]
        return [float(item) for item in vector]
    finally:
        temp_path.unlink(missing_ok=True)


def _compute_embedding(
    audio_path: Path,
    cluster_segments: list[TranscriptSegment],
    config: AppConfig,
) -> list[float] | None:
    provider = config.section("speaker_identity").get("provider", "pyannote")
    spans = [(segment.start_sec, segment.end_sec) for segment in cluster_segments]
    if provider != "pyannote":
        raise ValueError(f"Unsupported speaker identity provider: {provider}")
    return _compute_pyannote_embedding(audio_path, spans)


def _existing_user_assignment(db: sqlite3.Connection, call_id: str, cluster_id: str) -> sqlite3.Row | None:
    return db.execute(
        """
        SELECT *
        FROM speaker_assignments
        WHERE call_id = ? AND speaker_cluster_id = ? AND assignment_source = 'user'
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (call_id, cluster_id),
    ).fetchone()


def _profile_name(db: sqlite3.Connection, speaker_identity_id: str | None) -> str | None:
    if not speaker_identity_id:
        return None
    row = db.execute(
        "SELECT display_name FROM speaker_profiles WHERE speaker_identity_id = ?",
        (speaker_identity_id,),
    ).fetchone()
    return row["display_name"] if row else None


def _best_profile_match(db: sqlite3.Connection, embedding: list[float]) -> MatchResult:
    rows = db.execute(
        """
        SELECT e.speaker_identity_id, e.embedding_vector_json
        FROM speaker_embeddings e
        JOIN speaker_profiles p ON p.speaker_identity_id = e.speaker_identity_id
        WHERE p.status = 'confirmed' AND e.speaker_identity_id IS NOT NULL
        """
    ).fetchall()
    best_identity_id: str | None = None
    best_score: float | None = None
    for row in rows:
        candidate = [float(item) for item in json.loads(row["embedding_vector_json"])]
        score = _cosine_similarity(embedding, candidate)
        if best_score is None or score > best_score:
            best_identity_id = row["speaker_identity_id"]
            best_score = score
    return MatchResult(speaker_identity_id=best_identity_id, match_score=best_score, assignment_source=None)


def _upsert_embedding(
    db: sqlite3.Connection,
    *,
    call_id: str,
    cluster_id: str,
    duration_seconds: float,
    segment_count: int,
    embedding: list[float],
    speaker_identity_id: str | None,
    confidence: float | None,
    model_name: str,
) -> None:
    db.execute(
        """
        INSERT INTO speaker_embeddings (
            embedding_id, speaker_identity_id, call_id, speaker_cluster_id, segment_count,
            duration_seconds, embedding_vector_json, model_name, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            speaker_identity_id,
            call_id,
            cluster_id,
            segment_count,
            duration_seconds,
            json.dumps(embedding),
            model_name,
            confidence,
            utc_now(),
        ),
    )


def _upsert_assignment(
    db: sqlite3.Connection,
    *,
    call_id: str,
    cluster_id: str,
    speaker_identity_id: str | None,
    assignment_source: str,
    match_score: float | None,
) -> None:
    now = utc_now()
    existing = db.execute(
        "SELECT assignment_id FROM speaker_assignments WHERE call_id = ? AND speaker_cluster_id = ?",
        (call_id, cluster_id),
    ).fetchone()
    if existing:
        db.execute(
            """
            UPDATE speaker_assignments
            SET speaker_identity_id = ?, assignment_source = ?, match_score = ?, updated_at = ?
            WHERE assignment_id = ?
            """,
            (speaker_identity_id, assignment_source, match_score, now, existing["assignment_id"]),
        )
    else:
        db.execute(
            """
            INSERT INTO speaker_assignments (
                assignment_id, call_id, speaker_cluster_id, speaker_identity_id,
                assignment_source, match_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), call_id, cluster_id, speaker_identity_id, assignment_source, match_score, now, now),
        )


def apply_identity_assignments(
    segments: list[TranscriptSegment],
    db: sqlite3.Connection,
    call_id: str,
) -> list[TranscriptSegment]:
    assignments = {
        row["speaker_cluster_id"]: row
        for row in db.execute(
            """
            SELECT speaker_cluster_id, speaker_identity_id, assignment_source, match_score
            FROM speaker_assignments
            WHERE call_id = ?
            """,
            (call_id,),
        ).fetchall()
    }
    updated: list[TranscriptSegment] = []
    for segment in segments:
        assignment = assignments.get(segment.speaker_cluster_id)
        segment.speaker_identity_id = assignment["speaker_identity_id"] if assignment else None
        segment.identity_confidence = assignment["match_score"] if assignment else None
        profile_name = _profile_name(db, segment.speaker_identity_id)
        segment.speaker_display_name = profile_name or segment.speaker_label
        updated.append(segment)
    return updated


def run_speaker_identity_stage(config: AppConfig, call_dir: Path, db: sqlite3.Connection | None = None) -> list[dict]:
    connection = db or connect(config.sqlite_path)
    metadata = read_json(call_dir / "metadata.json", default={})
    call_id = metadata["call_id"]
    payload = read_json(call_dir / "transcript_segments.json", default=[])
    segments = [TranscriptSegment(**item) for item in payload]
    if not segments:
        return []

    if not config.section("speaker_identity").get("enabled", True):
        updated = apply_identity_assignments(segments, connection, call_id)
        write_json(call_dir / "transcript_segments.json", updated)
        if db is None:
            connection.close()
        return [segment.__dict__ for segment in updated]

    min_duration = float(config.section("speaker_identity").get("min_cluster_duration_seconds", 6))
    auto_threshold = float(config.section("speaker_identity").get("auto_assign_threshold", 0.75))
    suggest_threshold = float(config.section("speaker_identity").get("suggest_threshold", 0.60))
    model_name = f"{config.section('speaker_identity').get('provider', 'pyannote')}:embedding"
    grouped = _clustered_segments(segments)
    audio_path = call_dir / "audio_normalized.wav"

    for cluster_id, cluster_segments in grouped.items():
        user_assignment = _existing_user_assignment(connection, call_id, cluster_id)
        if user_assignment:
            continue
        duration = _cluster_duration(cluster_segments)
        if duration < min_duration:
            continue
        try:
            embedding = _compute_embedding(audio_path, cluster_segments, config)
        except Exception:
            logger.exception("Speaker identity embedding failed call_id=%s cluster=%s", call_id, cluster_id)
            if not config.section("speaker_identity").get("continue_on_error", True):
                raise
            continue
        if not embedding:
            continue
        match = _best_profile_match(connection, embedding)
        assigned_identity_id: str | None = None
        assignment_source = "auto"
        match_score = match.match_score
        if match.match_score is not None and match.match_score >= auto_threshold:
            assigned_identity_id = match.speaker_identity_id
        elif match.match_score is not None and match.match_score >= suggest_threshold:
            assigned_identity_id = None
            assignment_source = "suggested"
        else:
            assigned_identity_id = None
        _upsert_embedding(
            connection,
            call_id=call_id,
            cluster_id=cluster_id,
            duration_seconds=duration,
            segment_count=len(cluster_segments),
            embedding=embedding,
            speaker_identity_id=assigned_identity_id,
            confidence=match_score,
            model_name=model_name,
        )
        _upsert_assignment(
            connection,
            call_id=call_id,
            cluster_id=cluster_id,
            speaker_identity_id=assigned_identity_id,
            assignment_source=assignment_source,
            match_score=match_score,
        )
    updated = apply_identity_assignments(segments, connection, call_id)
    write_json(call_dir / "transcript_segments.json", updated)
    connection.commit()
    if db is None:
        connection.close()
    return [segment.__dict__ for segment in updated]


def refresh_segments_with_assignments(config: AppConfig, call_dir: Path) -> list[dict]:
    db = connect(config.sqlite_path)
    metadata = read_json(call_dir / "metadata.json", default={})
    segments = [TranscriptSegment(**item) for item in read_json(call_dir / "transcript_segments.json", default=[])]
    updated = apply_identity_assignments(segments, db, metadata["call_id"])
    write_json(call_dir / "transcript_segments.json", updated)
    db.close()
    return [segment.__dict__ for segment in updated]


def create_speaker_profile(db: sqlite3.Connection, display_name: str, notes: str | None = None) -> str:
    now = utc_now()
    speaker_identity_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO speaker_profiles (speaker_identity_id, display_name, status, created_at, updated_at, notes)
        VALUES (?, ?, 'confirmed', ?, ?, ?)
        """,
        (speaker_identity_id, display_name.strip(), now, now, notes),
    )
    db.commit()
    return speaker_identity_id


def assign_speaker_identity(
    config: AppConfig,
    call_dir: Path,
    cluster_id: str,
    speaker_identity_id: str,
    assignment_source: str = "user",
) -> None:
    db = connect(config.sqlite_path)
    metadata = read_json(call_dir / "metadata.json", default={})
    _upsert_assignment(
        db,
        call_id=metadata["call_id"],
        cluster_id=cluster_id,
        speaker_identity_id=speaker_identity_id,
        assignment_source=assignment_source,
        match_score=1.0 if assignment_source == "user" else None,
    )
    db.commit()
    db.close()


def clear_speaker_identity(config: AppConfig, call_dir: Path, cluster_id: str) -> None:
    db = connect(config.sqlite_path)
    metadata = read_json(call_dir / "metadata.json", default={})
    db.execute(
        "DELETE FROM speaker_assignments WHERE call_id = ? AND speaker_cluster_id = ?",
        (metadata["call_id"], cluster_id),
    )
    db.commit()
    db.close()


def list_speaker_profiles(db: sqlite3.Connection) -> list[dict]:
    rows = db.execute(
        """
        SELECT p.*, COUNT(DISTINCT a.call_id) AS linked_calls, MAX(a.updated_at) AS last_seen
        FROM speaker_profiles p
        LEFT JOIN speaker_assignments a ON a.speaker_identity_id = p.speaker_identity_id
        GROUP BY p.speaker_identity_id
        ORDER BY p.display_name COLLATE NOCASE ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def speaker_profile_detail(db: sqlite3.Connection, speaker_identity_id: str) -> dict[str, object] | None:
    profile = db.execute(
        "SELECT * FROM speaker_profiles WHERE speaker_identity_id = ?",
        (speaker_identity_id,),
    ).fetchone()
    if not profile:
        return None
    assignments = db.execute(
        """
        SELECT a.*, c.source_filename, c.recorded_at, c.imported_at
        FROM speaker_assignments a
        JOIN calls c ON c.call_id = a.call_id
        WHERE a.speaker_identity_id = ?
        ORDER BY COALESCE(c.recorded_at, c.imported_at) DESC
        """,
        (speaker_identity_id,),
    ).fetchall()
    return {"profile": dict(profile), "assignments": [dict(row) for row in assignments]}
