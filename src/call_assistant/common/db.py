from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    initialize(connection)
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS calls (
            call_id TEXT PRIMARY KEY,
            archive_path TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            recorded_at TEXT NULL,
            imported_at TEXT NOT NULL,
            duration_seconds REAL NULL,
            audio_format TEXT NULL,
            language_summary TEXT NULL,
            current_state TEXT NOT NULL,
            review_state TEXT NOT NULL,
            low_confidence INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NULL,
            search_text TEXT NULL
        );

        CREATE TABLE IF NOT EXISTS queue_jobs (
            job_id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL,
            available_at TEXT NOT NULL,
            started_at TEXT NULL,
            finished_at TEXT NULL,
            error_message TEXT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            text TEXT NOT NULL,
            owner TEXT NULL,
            type TEXT NOT NULL,
            source_timestamp REAL NULL,
            source_quote TEXT NULL,
            status TEXT NOT NULL,
            deadline TEXT NULL,
            confidence REAL NULL,
            is_reviewed INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS processing_runs (
            run_id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NULL,
            details_json TEXT NULL
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_calls_state ON calls(current_state, review_state);
        CREATE INDEX IF NOT EXISTS idx_calls_imported_at ON calls(imported_at);
        CREATE INDEX IF NOT EXISTS idx_tasks_call_id ON tasks(call_id, status, is_reviewed);
        CREATE INDEX IF NOT EXISTS idx_queue_jobs_status ON queue_jobs(status, priority, available_at);
        """
    )
    connection.commit()
