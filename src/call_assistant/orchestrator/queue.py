from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import uuid
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.models import QueueJob, utc_now

logger = logging.getLogger(__name__)


def reset_running_jobs_on_startup(config: AppConfig) -> int:
    db = connect(config.sqlite_path)
    rows = db.execute("SELECT job_id, call_id, stage FROM queue_jobs WHERE status = 'running'").fetchall()
    if not rows:
        return 0
    for row in rows:
        db.execute(
            """
            UPDATE queue_jobs
            SET status = 'queued', finished_at = ?, error_message = ?
            WHERE job_id = ?
            """,
            (utc_now(), "Recovered running job during worker startup", row["job_id"]),
        )
        logger.warning(
            "Queue reset running job on startup job_id=%s call_id=%s stage=%s",
            row["job_id"],
            row["call_id"],
            row["stage"],
        )
    db.commit()
    return len(rows)


def _stale_after_seconds_for_job(config: AppConfig, row) -> int:
    base_stale_seconds = int(config.section("queue").get("stale_job_seconds", 300))
    if row["stage"] != "transcription":
        return base_stale_seconds

    db = connect(config.sqlite_path)
    call_row = db.execute(
        "SELECT archive_path, duration_seconds FROM calls WHERE call_id = ?",
        (row["call_id"],),
    ).fetchone()

    duration_seconds = 0.0
    if call_row is not None and call_row["duration_seconds"] is not None:
        duration_seconds = float(call_row["duration_seconds"])
    elif call_row is not None:
        metadata_path = Path(call_row["archive_path"]) / "metadata.json"
        if metadata_path.exists():
            import json

            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                duration_seconds = float(payload.get("duration_seconds") or 0.0)
            except (ValueError, TypeError, json.JSONDecodeError):
                duration_seconds = 0.0

    multiplier = float(config.section("queue").get("transcription_stale_multiplier", 20))
    buffer_seconds = int(config.section("queue").get("transcription_stale_buffer_seconds", 600))
    minimum_seconds = int(config.section("queue").get("transcription_stale_min_seconds", 1800))
    scaled_seconds = int(duration_seconds * multiplier) + buffer_seconds
    return max(base_stale_seconds, minimum_seconds, scaled_seconds)


def is_job_stale(config: AppConfig, row, now: datetime | None = None) -> bool:
    started_at_raw = row["started_at"] if isinstance(row, dict) else row["started_at"]
    if not started_at_raw:
        return False
    started_at = datetime.fromisoformat(started_at_raw)
    reference = now or datetime.now(timezone.utc)
    stale_after_seconds = _stale_after_seconds_for_job(config, row)
    cutoff = reference - timedelta(seconds=stale_after_seconds)
    return started_at <= cutoff


def _reclaim_stale_running_jobs(config: AppConfig) -> None:
    db = connect(config.sqlite_path)
    running_jobs = db.execute(
        """
        SELECT * FROM queue_jobs
        WHERE status = 'running' AND started_at IS NOT NULL
        """,
    ).fetchall()
    now = datetime.now(timezone.utc)
    for row in running_jobs:
        stale_after_seconds = _stale_after_seconds_for_job(config, row)
        if not is_job_stale(config, row, now=now):
            continue
        next_status = "queued" if row["attempt_count"] < row["max_attempts"] else "failed"
        error_message = "Reclaimed stale running job after worker interruption"
        db.execute(
            """
            UPDATE queue_jobs
            SET status = ?, finished_at = ?, error_message = ?
            WHERE job_id = ?
            """,
            (next_status, utc_now(), error_message, row["job_id"]),
        )
        logger.warning(
            "Queue reclaimed stale job_id=%s call_id=%s stage=%s next_status=%s stale_after_seconds=%s",
            row["job_id"],
            row["call_id"],
            row["stage"],
            next_status,
            stale_after_seconds,
        )
    db.commit()


def enqueue(config: AppConfig, call_id: str, stage: str, priority: int = 0) -> None:
    db = connect(config.sqlite_path)
    existing = db.execute(
        "SELECT job_id FROM queue_jobs WHERE call_id = ? AND stage = ? AND status IN ('queued', 'running')",
        (call_id, stage),
    ).fetchone()
    if existing:
        logger.info("Queue enqueue skipped call_id=%s stage=%s reason=existing_active_job", call_id, stage)
        return
    job = QueueJob(
        job_id=str(uuid.uuid4()),
        call_id=call_id,
        stage=stage,
        status="queued",
        priority=priority,
        attempt_count=0,
        max_attempts=config.retry_limit(stage),
        available_at=utc_now(),
    )
    db.execute(
        """
        INSERT INTO queue_jobs (
            job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (job.job_id, job.call_id, job.stage, job.status, job.priority, job.attempt_count, job.max_attempts, job.available_at),
    )
    db.commit()
    logger.info("Queue enqueue call_id=%s stage=%s job_id=%s", call_id, stage, job.job_id)


def claim_next_job(config: AppConfig) -> QueueJob | None:
    _reclaim_stale_running_jobs(config)
    db = connect(config.sqlite_path)
    row = db.execute(
        """
        SELECT * FROM queue_jobs
        WHERE status = 'queued'
        ORDER BY priority DESC, available_at ASC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    db.execute(
        "UPDATE queue_jobs SET status = 'running', attempt_count = attempt_count + 1, started_at = ? WHERE job_id = ?",
        (utc_now(), row["job_id"]),
    )
    db.commit()
    row = db.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
    return QueueJob(**dict(row))


def claim_next_job_for_call(config: AppConfig, call_id: str) -> QueueJob | None:
    _reclaim_stale_running_jobs(config)
    db = connect(config.sqlite_path)
    row = db.execute(
        """
        SELECT * FROM queue_jobs
        WHERE status = 'queued' AND call_id = ?
        ORDER BY priority DESC, available_at ASC
        LIMIT 1
        """,
        (call_id,),
    ).fetchone()
    if not row:
        return None
    db.execute(
        "UPDATE queue_jobs SET status = 'running', attempt_count = attempt_count + 1, started_at = ? WHERE job_id = ?",
        (utc_now(), row["job_id"]),
    )
    db.commit()
    row = db.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
    return QueueJob(**dict(row))


def complete_job(config: AppConfig, job_id: str) -> None:
    db = connect(config.sqlite_path)
    db.execute(
        "UPDATE queue_jobs SET status = 'done', finished_at = ?, error_message = NULL WHERE job_id = ?",
        (utc_now(), job_id),
    )
    db.commit()
    logger.info("Queue complete job_id=%s", job_id)


def fail_job(config: AppConfig, job: QueueJob, error: str, retryable: bool) -> None:
    db = connect(config.sqlite_path)
    next_status = "queued" if retryable and job.attempt_count < job.max_attempts else "failed"
    db.execute(
        "UPDATE queue_jobs SET status = ?, finished_at = ?, error_message = ? WHERE job_id = ?",
        (next_status, utc_now(), error[:2000], job.job_id),
    )
    db.commit()
    logger.error("Queue fail job_id=%s call_id=%s stage=%s retryable=%s error=%s", job.job_id, job.call_id, job.stage, retryable, error)


def list_jobs(config: AppConfig, statuses: tuple[str, ...] | None = None) -> list[dict]:
    db = connect(config.sqlite_path)
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        query = f"SELECT * FROM queue_jobs WHERE status IN ({placeholders}) ORDER BY available_at DESC"
        rows = db.execute(query, statuses).fetchall()
    else:
        rows = db.execute("SELECT * FROM queue_jobs ORDER BY available_at DESC").fetchall()
    return [dict(row) for row in rows]
