from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import uuid

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.models import QueueJob, utc_now

logger = logging.getLogger(__name__)


def _reclaim_stale_running_jobs(config: AppConfig) -> None:
    stale_after_seconds = int(config.section("queue").get("stale_job_seconds", 300))
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
    db = connect(config.sqlite_path)
    stale_jobs = db.execute(
        """
        SELECT * FROM queue_jobs
        WHERE status = 'running' AND started_at IS NOT NULL AND started_at <= ?
        """,
        (cutoff,),
    ).fetchall()
    for row in stale_jobs:
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
            "Queue reclaimed stale job_id=%s call_id=%s stage=%s next_status=%s",
            row["job_id"],
            row["call_id"],
            row["stage"],
            next_status,
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
