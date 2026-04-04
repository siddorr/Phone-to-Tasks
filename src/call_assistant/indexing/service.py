from __future__ import annotations

import uuid
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import read_json
from call_assistant.common.models import artifact_paths


def _stored_task_id(call_id: str, source_kind: str, task: dict, seen_ids: set[str]) -> str:
    base_task_id = task.get("task_id") or str(uuid.uuid4())
    candidate = f"{call_id}:{source_kind}:{base_task_id}"
    suffix = 1
    while candidate in seen_ids:
        suffix += 1
        candidate = f"{call_id}:{source_kind}:{base_task_id}:{suffix}"
    seen_ids.add(candidate)
    return candidate


def index_call(call_dir: Path, config: AppConfig) -> None:
    db = connect(config.sqlite_path)
    paths = artifact_paths(call_dir)
    metadata = read_json(paths["metadata"], default={})
    summary = read_json(paths["summary"], default={})
    tasks = read_json(paths["tasks"], default=[])
    reviewed_tasks = read_json(paths["tasks_reviewed"], default=[])
    transcript_text = paths["transcript_clean"].read_text(encoding="utf-8") if paths["transcript_clean"].exists() else ""
    search_text = "\n".join(
        [
            transcript_text,
            summary.get("short_summary", ""),
            summary.get("detailed_summary", ""),
            " ".join(item.get("text", "") for item in tasks),
            " ".join(item.get("text", "") for item in reviewed_tasks),
        ]
    ).strip()
    db.execute(
        """
        INSERT INTO calls (
            call_id, archive_path, source_filename, sha256, recorded_at, imported_at,
            duration_seconds, audio_format, language_summary, current_state, review_state,
            low_confidence, last_error, search_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(call_id) DO UPDATE SET
            archive_path=excluded.archive_path,
            source_filename=excluded.source_filename,
            recorded_at=COALESCE(excluded.recorded_at, calls.recorded_at),
            imported_at=COALESCE(calls.imported_at, excluded.imported_at),
            duration_seconds=excluded.duration_seconds,
            audio_format=excluded.audio_format,
            language_summary=excluded.language_summary,
            current_state=excluded.current_state,
            review_state=excluded.review_state,
            low_confidence=excluded.low_confidence,
            last_error=excluded.last_error,
            search_text=excluded.search_text
        """,
        (
            metadata["call_id"],
            str(call_dir),
            metadata["source_filename"],
            metadata["sha256"],
            metadata.get("recorded_at"),
            metadata["imported_at"],
            metadata.get("duration_seconds"),
            metadata.get("audio_format"),
            summary.get("analysis_language"),
            metadata.get("current_state", "indexed"),
            metadata.get("review_state", "pending"),
            int(bool(metadata.get("low_confidence", False))),
            metadata.get("errors", [None])[-1] if metadata.get("errors") else None,
            search_text,
        ),
    )
    db.execute("DELETE FROM tasks WHERE call_id = ?", (metadata["call_id"],))
    seen_task_ids: set[str] = set()
    for source_kind, entries, is_reviewed in (("extracted", tasks, 0), ("reviewed", reviewed_tasks, 1)):
        for task in entries:
            db.execute(
                """
                INSERT INTO tasks (
                    task_id, call_id, source_kind, text, owner, type, source_timestamp,
                    source_quote, status, deadline, confidence, is_reviewed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _stored_task_id(metadata["call_id"], source_kind, task, seen_task_ids),
                    metadata["call_id"],
                    source_kind,
                    task["text"],
                    task.get("owner"),
                    task.get("type", "task"),
                    task.get("source_timestamp"),
                    task.get("source_quote"),
                    task.get("status", "new"),
                    task.get("deadline"),
                    task.get("confidence"),
                    is_reviewed,
                ),
            )
    db.execute("DELETE FROM artifacts WHERE call_id = ?", (metadata["call_id"],))
    for artifact_type, artifact_path in paths.items():
        if artifact_path.exists():
            db.execute(
                "INSERT INTO artifacts (artifact_id, call_id, artifact_type, path, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), metadata["call_id"], artifact_type, str(artifact_path), metadata["imported_at"]),
            )
    db.commit()


def rebuild_index(archive_root: Path, config: AppConfig) -> None:
    for metadata_path in archive_root.rglob("metadata.json"):
        index_call(metadata_path.parent, config)
