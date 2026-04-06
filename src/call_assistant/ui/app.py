from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import cmp_to_key
import logging
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import append_log, read_json, write_json
from call_assistant.common.models import TranscriptSegment
from call_assistant.common.models import utc_now
from call_assistant.common.progress import get_stage_progress
from call_assistant.diarization.service import apply_speaker_mapping
from call_assistant.ingest.watcher import detect_new_calls, import_file
from call_assistant.indexing.service import index_call
from call_assistant.orchestrator.queue import enqueue, is_job_stale, list_jobs
from call_assistant.orchestrator.worker import processing_mode, run_manual_step
from call_assistant.reprocess import reset_call_for_retranscription, transcription_choices
from call_assistant.reprocess import transcription_language_choices
from call_assistant.speaker_identity.service import (
    accept_suggested_speaker_identity,
    archive_speaker_profile,
    assign_speaker_identity,
    call_speaker_assignments,
    clear_speaker_identity,
    create_speaker_profile,
    list_assignable_speaker_profiles,
    list_speaker_profiles,
    rename_speaker_profile,
    reject_suggested_speaker_identity,
    refresh_segments_with_assignments,
    speaker_profile_detail,
)
from call_assistant.transcript_cleaner.service import clean_transcript

logger = logging.getLogger(__name__)
STATUS_LINE_LOG_INTERVAL_SECONDS = 15.0
_LAST_STATUS_LINE_LOG: dict[str, object] = {"signature": None, "at": 0.0}

DEFAULT_CALL_SORT_BY = "recorded_at"
DEFAULT_CALL_SORT_DIRECTIONS = {
    "recorded_at": "desc",
    "respondent": "asc",
    "state": "asc",
    "tasks": "desc",
    "short_description": "asc",
}
ALLOWED_CALL_SORTS = set(DEFAULT_CALL_SORT_DIRECTIONS)
ALLOWED_RECENT_FILTERS = {
    "": None,
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}

STATUS_POLL_INTERVAL_SECONDS = 5
ETA_RECALC_MIN_SECONDS = 15.0
ETA_RECALC_MAX_SECONDS = 300.0
STAGE_DEFAULT_RUNTIME_SECONDS = {
    "audio_prepare": 30.0,
    "transcription": 180.0,
    "diarization": 300.0,
    "speaker_identity": 45.0,
    "transcript_clean": 15.0,
    "analysis": 45.0,
    "indexing": 15.0,
}
PROGRESS_STEP_LABELS = {
    "loading_model": "loading model",
    "model_loaded": "model ready",
    "model_cached": "reusing model",
    "preparing_audio": "preparing audio",
    "detecting_language": "detecting language",
    "transcribing": "transcribing",
    "cloud_upload": "uploading audio",
}


def _guess_direction(call: dict) -> str:
    metadata = read_json(Path(call["archive_path"]) / "metadata.json", default={})
    for key in ("direction", "call_direction", "in_out"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"in", "incoming"}:
                return "In"
            if normalized in {"out", "outgoing"}:
                return "Out"
            return value.strip()
    source = call.get("source_filename", "").lower()
    if "incoming" in source:
        return "In"
    if "outgoing" in source:
        return "Out"
    return "-"


def _guess_respondent(call: dict) -> str:
    metadata = read_json(Path(call["archive_path"]) / "metadata.json", default={})
    for key in ("respondent", "contact_name", "caller_name", "other_party"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    source = call.get("source_filename", "")
    stem = Path(source).stem
    prefix = "Call recording "
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    parts = stem.rsplit("_", 1)
    candidate = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
    candidate = candidate.strip(" _-")
    return candidate or "-"


def _short_description(call: dict) -> str:
    summary = read_json(Path(call["archive_path"]) / "summary.json", default={})
    short_summary = summary.get("short_summary")
    if isinstance(short_summary, str) and short_summary.strip():
        return short_summary.strip()
    return "-"


def _recorded_at_badge(recorded_at_source: str | None, recorded_at: str | None) -> str | None:
    if not recorded_at:
        return None
    if recorded_at_source == "media_metadata":
        return "metadata"
    if recorded_at_source == "filesystem_mtime":
        return "fallback"
    return None


def _format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "-"
    total_seconds = max(0, int(round(duration_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_local_timestamp(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    if not parsed:
        return "-"
    local = parsed.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S")


def _compact_preview(text: str | None, limit: int = 120) -> str:
    if not text:
        return "-"
    normalized = " ".join(str(text).split())
    if not normalized:
        return "-"
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 1)].rstrip() + "…"


def _summary_preview(summary: dict) -> str:
    short_summary = summary.get("short_summary") if isinstance(summary, dict) else None
    if isinstance(short_summary, str) and short_summary.strip():
        return _compact_preview(short_summary, limit=140)
    return "No short summary."


def _clean_transcript_preview(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return _compact_preview(stripped, limit=140)
    return "No clean transcript yet."


def _processing_log_preview(processing_log: list[dict]) -> str:
    if not processing_log:
        return "No processing log entries."
    latest = processing_log[-1]
    event = latest.get("event") or "unknown"
    timestamp = latest.get("at") or latest.get("reviewed_at")
    if timestamp:
        return f"Latest: {event} at {_format_local_timestamp(timestamp)}"
    return f"Latest: {event}"


def _task_list_preview(tasks: list[dict]) -> str:
    if not tasks:
        return "No tasks."
    first_text = tasks[0].get("text") if isinstance(tasks[0], dict) else None
    if isinstance(first_text, str) and first_text.strip():
        return _compact_preview(first_text, limit=140)
    return f"{len(tasks)} task(s)"


def _error_preview(error_message: str | None, limit: int = 120) -> str:
    return _compact_preview(error_message, limit=limit)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _preferred_audio_path(call_dir: Path) -> Path | None:
    originals = sorted(call_dir.glob("audio_original.*"))
    if originals:
        return originals[0]
    normalized = call_dir / "audio_normalized.wav"
    if normalized.exists():
        return normalized
    return None


def _segment_view(segments: list[dict]) -> list[dict]:
    view: list[dict] = []
    for item in segments:
        start_sec = float(item.get("start_sec", 0.0))
        end_sec = float(item.get("end_sec", start_sec))
        minutes, seconds = divmod(int(start_sec), 60)
        hours, minutes = divmod(minutes, 60)
        timestamp = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
        view.append(
            {
                **item,
                "segment_dom_id": f"segment-{item.get('segment_id', 'unknown')}",
                "display_timestamp": timestamp,
                "display_speaker": item.get("speaker_display_name") or item.get("speaker_label") or item.get("speaker_cluster_id"),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    return view


def _diarization_context(processing_log: list[dict], segments: list[dict]) -> dict[str, str | None]:
    for event in reversed(processing_log):
        if event.get("event") == "fallback_diarization_completed" and event.get("mode") == "single_speaker":
            return {
                "diarization_mode": "single_speaker_fallback",
                "diarization_notice": "Diarization used single-speaker fallback for this call.",
            }
    if segments:
        return {
            "diarization_mode": "clustered",
            "diarization_notice": "Diarization produced speaker clusters for this call.",
        }
    return {"diarization_mode": "unknown", "diarization_notice": None}


def _speaker_identity_context(metadata: dict, assignments: dict[str, dict]) -> dict[str, str | None]:
    mode = metadata.get("speaker_identity_mode")
    summary = metadata.get("speaker_identity_summary")
    if mode or summary:
        return {"speaker_identity_mode": mode, "speaker_identity_notice": summary}
    if assignments:
        return {"speaker_identity_mode": "full", "speaker_identity_notice": "Speaker identity assignments are available."}
    return {"speaker_identity_mode": None, "speaker_identity_notice": None}


def _transcription_quality_context(metadata: dict) -> dict[str, object]:
    return {
        "transcription_selection_strategy": metadata.get("transcription_selection_strategy"),
        "transcription_retried_languages": metadata.get("transcription_retried_languages", []),
        "transcription_quality_flags": metadata.get("transcription_quality_flags", []),
        "transcription_quality_score": metadata.get("transcription_quality_score"),
        "transcription_low_confidence_reason": metadata.get("transcription_low_confidence_reason"),
        "transcription_language_detected": metadata.get("transcription_language_detected"),
    }


def _cluster_status_label(assignment_source: str | None, speaker_identity_id: str | None) -> str:
    if assignment_source == "user" and speaker_identity_id:
        return "Confirmed"
    if assignment_source == "auto" and speaker_identity_id:
        return "Automatic"
    if assignment_source == "suggested":
        return "Suggested"
    return "Unassigned"


def _augment_segments_with_assignments(segments: list[dict], assignments: dict[str, dict]) -> list[dict]:
    updated: list[dict] = []
    for item in segments:
        cluster_id = item.get("speaker_cluster_id") or item.get("speaker_label") or "speaker_1"
        assignment = assignments.get(cluster_id, {})
        profile_name = assignment.get("display_name")
        updated.append(
            {
                **item,
                "speaker_identity_id": assignment.get("speaker_identity_id"),
                "speaker_display_name": profile_name or item.get("speaker_display_name") or item.get("speaker_label"),
                "identity_confidence": assignment.get("match_score"),
                "assignment_source": assignment.get("assignment_source"),
            }
        )
    return updated


def _call_queue_statuses(db, call_id: str) -> list[dict]:
    rows = db.execute(
        """
        SELECT stage, status, started_at, finished_at, error_message
        FROM queue_jobs
        WHERE call_id = ?
        ORDER BY available_at DESC
        """,
        (call_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _queue_counts(config: AppConfig, db) -> dict[str, int]:
    counts = {"running": 0, "queued": 0, "failed": 0}
    rows = db.execute(
        """
        SELECT status, started_at, stage, call_id
        FROM queue_jobs
        WHERE status IN ('running', 'queued', 'failed')
        """
    ).fetchall()
    for row in rows:
        if row["status"] == "running" and is_job_stale(config, row):
            counts["queued"] += 1
            continue
        counts[row["status"]] += 1
    return counts


def _current_running_job(config: AppConfig, db) -> dict | None:
    rows = db.execute(
        """
        SELECT q.*, c.duration_seconds
        FROM queue_jobs q
        LEFT JOIN calls c ON c.call_id = q.call_id
        WHERE q.status = 'running'
        ORDER BY (q.started_at IS NULL), q.started_at ASC, q.available_at ASC
        """
    ).fetchall()
    for row in rows:
        if is_job_stale(config, row):
            continue
        return dict(row)
    return None


def _completed_stage_runtime_samples(db, stage: str, limit: int = 20) -> list[dict]:
    rows = db.execute(
        """
        SELECT q.call_id, c.duration_seconds, q.started_at, q.finished_at
        FROM queue_jobs q
        LEFT JOIN calls c ON c.call_id = q.call_id
        WHERE q.stage = ?
          AND q.status = 'done'
          AND q.started_at IS NOT NULL
          AND q.finished_at IS NOT NULL
        ORDER BY q.finished_at DESC
        LIMIT ?
        """,
        (stage, limit),
    ).fetchall()
    samples: list[dict] = []
    for row in rows:
        started_at = _parse_timestamp(row["started_at"])
        finished_at = _parse_timestamp(row["finished_at"])
        if not started_at or not finished_at:
            continue
        runtime_seconds = max(0.0, (finished_at - started_at).total_seconds())
        samples.append(
            {
                "call_id": row["call_id"],
                "duration_seconds": row["duration_seconds"],
                "runtime_seconds": runtime_seconds,
            }
        )
    return samples


def _stage_default_runtime_seconds(stage: str, audio_duration_seconds: float | None) -> float:
    if stage == "transcription" and audio_duration_seconds:
        return max(60.0, audio_duration_seconds * 0.5)
    if stage == "diarization" and audio_duration_seconds:
        return max(120.0, audio_duration_seconds * 0.8)
    return STAGE_DEFAULT_RUNTIME_SECONDS.get(stage, 60.0)


def _estimate_stage_runtime_seconds(db, running_job: dict) -> float:
    stage = running_job["stage"]
    audio_duration_seconds = (
        float(running_job["duration_seconds"])
        if running_job.get("duration_seconds") is not None
        else None
    )
    samples = _completed_stage_runtime_samples(db, stage)
    if stage in {"transcription", "diarization"} and audio_duration_seconds:
        ratios = [
            sample["runtime_seconds"] / float(sample["duration_seconds"])
            for sample in samples
            if sample.get("duration_seconds")
            and float(sample["duration_seconds"]) > 0
        ]
        median_ratio = _median(ratios)
        if median_ratio is not None:
            return max(1.0, audio_duration_seconds * median_ratio)
    median_runtime = _median([sample["runtime_seconds"] for sample in samples])
    if median_runtime is not None:
        return max(1.0, median_runtime)
    return _stage_default_runtime_seconds(stage, audio_duration_seconds)


def _eta_recalc_window_seconds(estimated_total_seconds: float) -> float:
    return max(ETA_RECALC_MIN_SECONDS, min(ETA_RECALC_MAX_SECONDS, estimated_total_seconds * 0.2))


def _recalculated_total_seconds(estimated_total_seconds: float, elapsed_seconds: float) -> float:
    reference_seconds = max(estimated_total_seconds, elapsed_seconds)
    return elapsed_seconds + _eta_recalc_window_seconds(reference_seconds)


def _app_status_label(processing_mode_value: str, counts: dict[str, int]) -> str:
    if counts["running"]:
        return "Running"
    if processing_mode_value == "manual_step" and counts["queued"]:
        return "Paused"
    if counts["failed"]:
        return "Needs attention"
    if counts["queued"]:
        return "Backlog"
    return "Idle"


def _display_progress_step(step_name: str | None) -> str | None:
    if not step_name:
        return None
    return PROGRESS_STEP_LABELS.get(step_name, step_name.replace("_", " "))


def _task_status_payload(db, running_job: dict | None) -> dict | None:
    if not running_job:
        return None
    started_at = _parse_timestamp(running_job.get("started_at"))
    elapsed_seconds: float | None = None
    if started_at:
        elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
    estimated_total_seconds = _estimate_stage_runtime_seconds(db, running_job)
    progress = get_stage_progress(running_job["call_id"], running_job["stage"]) or {}
    progress_step_name = progress.get("step_name")
    progress_step_display = _display_progress_step(progress_step_name)
    progress_completed = progress.get("completed")
    progress_total = progress.get("total")
    progress_fraction: float | None = None
    progress_percent: float | None = None
    if isinstance(progress_completed, (int, float)) and isinstance(progress_total, (int, float)) and progress_total > 0:
        progress_fraction = max(0.0, min(1.0, float(progress_completed) / float(progress_total)))
        progress_percent = progress_fraction * 100.0
        if elapsed_seconds is not None and progress_fraction > 0.0 and progress_fraction < 1.0:
            estimated_total_seconds = max(elapsed_seconds, elapsed_seconds / progress_fraction)
    remaining_seconds: float | None = None
    eta_status = "unknown"
    estimated_finish_display = "unknown"
    estimated_finish_at_iso = None
    eta_extension_seconds: float | None = None
    if started_at:
        if elapsed_seconds is not None:
            if elapsed_seconds <= estimated_total_seconds:
                remaining_seconds = max(0.0, estimated_total_seconds - elapsed_seconds)
                eta_status = "on_track"
                estimated_finish_at = started_at + timedelta(seconds=estimated_total_seconds)
                estimated_finish_at_iso = estimated_finish_at.isoformat()
                estimated_finish_display = f"{_format_local_timestamp(estimated_finish_at_iso)} (about {_format_duration(remaining_seconds)} left)"
            else:
                eta_status = "recalculating"
                recalculated_total_seconds = _recalculated_total_seconds(estimated_total_seconds, elapsed_seconds)
                remaining_seconds = max(0.0, recalculated_total_seconds - elapsed_seconds)
                eta_extension_seconds = remaining_seconds
                estimated_finish_at = started_at + timedelta(seconds=recalculated_total_seconds)
                estimated_finish_at_iso = estimated_finish_at.isoformat()
                estimated_finish_display = f"{_format_local_timestamp(estimated_finish_at_iso)} (recalculating, about {_format_duration(remaining_seconds)} left)"
        else:
            estimated_finish_at = started_at + timedelta(seconds=estimated_total_seconds)
            estimated_finish_at_iso = estimated_finish_at.isoformat()
            estimated_finish_display = _format_local_timestamp(estimated_finish_at_iso)
    return {
        "call_id": running_job["call_id"],
        "call_url": f"/calls/{running_job['call_id']}",
        "stage": running_job["stage"],
        "started_at": running_job.get("started_at"),
        "started_at_display": _format_local_timestamp(running_job.get("started_at")),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_display": _format_duration(elapsed_seconds),
        "progress_percent": round(progress_percent, 1) if progress_percent is not None else None,
        "progress_completed": progress_completed,
        "progress_total": progress_total,
        "progress_step_name": progress_step_name,
        "progress_step_display": progress_step_display,
        "estimated_total_seconds": estimated_total_seconds,
        "estimated_total_display": _format_duration(estimated_total_seconds),
        "estimated_finish_at": estimated_finish_at_iso,
        "estimated_finish_display": estimated_finish_display,
        "remaining_seconds": remaining_seconds,
        "remaining_display": _format_duration(remaining_seconds),
        "eta_status": eta_status,
        "eta_extension_seconds": eta_extension_seconds,
    }


def _app_status_payload(config: AppConfig, db) -> dict:
    counts = _queue_counts(config, db)
    mode = processing_mode(config)
    running_job = _current_running_job(config, db)
    return {
        "server_now": utc_now(),
        "app_status": _app_status_label(mode, counts),
        "processing_mode": mode,
        "counts": counts,
        "current_task": _task_status_payload(db, running_job),
        "poll_interval_seconds": STATUS_POLL_INTERVAL_SECONDS,
    }


def _log_status_line_snapshot(payload: dict) -> None:
    task = payload.get("current_task") or {}
    signature = (
        payload.get("app_status"),
        payload.get("processing_mode"),
        payload.get("counts", {}).get("running"),
        payload.get("counts", {}).get("queued"),
        payload.get("counts", {}).get("failed"),
        task.get("call_id"),
        task.get("stage"),
        task.get("eta_status"),
        task.get("started_at"),
    )
    now = time.monotonic()
    last_signature = _LAST_STATUS_LINE_LOG.get("signature")
    last_at = float(_LAST_STATUS_LINE_LOG.get("at") or 0.0)
    if signature == last_signature and (now - last_at) < STATUS_LINE_LOG_INTERVAL_SECONDS:
        return
    _LAST_STATUS_LINE_LOG["signature"] = signature
    _LAST_STATUS_LINE_LOG["at"] = now
    logger.info(
        "Status line app_status=%s mode=%s running=%s queued=%s failed=%s call_id=%s stage=%s elapsed=%s progress=%s estimated_total=%s remaining=%s eta_status=%s eta_finish=%s",
        payload.get("app_status"),
        payload.get("processing_mode"),
        payload.get("counts", {}).get("running"),
        payload.get("counts", {}).get("queued"),
        payload.get("counts", {}).get("failed"),
        task.get("call_id"),
        task.get("stage"),
        task.get("elapsed_display"),
        f"{task.get('progress_percent')}%" if task.get("progress_percent") is not None else None,
        task.get("estimated_total_display"),
        task.get("remaining_display"),
        task.get("eta_status"),
        task.get("estimated_finish_display"),
    )


def _refresh_transcript_outputs(call_dir: Path, config: AppConfig) -> list[dict]:
    payload = refresh_segments_with_assignments(config, call_dir)
    clean = clean_transcript([TranscriptSegment(**item) for item in payload])
    (call_dir / "transcript_clean.txt").write_text(clean.text, encoding="utf-8")
    index_call(call_dir, config)
    return payload


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _effective_call_timestamp(call: dict) -> datetime | None:
    return _parse_timestamp(call.get("display_recorded_at") or call.get("recorded_at") or call.get("imported_at"))


def _normalize_call_sort(sort_by: str, sort_dir: str) -> tuple[str, str]:
    normalized_sort_by = sort_by if sort_by in ALLOWED_CALL_SORTS else DEFAULT_CALL_SORT_BY
    default_dir = DEFAULT_CALL_SORT_DIRECTIONS[normalized_sort_by]
    normalized_sort_dir = sort_dir if sort_dir in {"asc", "desc"} else default_dir
    return normalized_sort_by, normalized_sort_dir


def _filter_calls_by_recent(rows: list[dict], recent: str) -> list[dict]:
    delta = ALLOWED_RECENT_FILTERS.get(recent)
    if delta is None:
        return rows
    cutoff = datetime.now(timezone.utc) - delta
    return [row for row in rows if (_effective_call_timestamp(row) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]


def _compare_call_rows(left: dict, right: dict, sort_by: str, sort_dir: str) -> int:
    def compare_values(left_value, right_value) -> int:
        left_missing = left_value in (None, "")
        right_missing = right_value in (None, "")
        if left_missing and right_missing:
            return 0
        if left_missing:
            return 1
        if right_missing:
            return -1
        if left_value < right_value:
            return -1
        if left_value > right_value:
            return 1
        return 0

    sort_left = {
        "recorded_at": _effective_call_timestamp(left),
        "respondent": (left.get("display_respondent") or "").casefold(),
        "state": (left.get("current_state") or "").casefold(),
        "tasks": left.get("display_task_count", 0),
        "short_description": (left.get("display_short_description") or "").casefold(),
    }[sort_by]
    sort_right = {
        "recorded_at": _effective_call_timestamp(right),
        "respondent": (right.get("display_respondent") or "").casefold(),
        "state": (right.get("current_state") or "").casefold(),
        "tasks": right.get("display_task_count", 0),
        "short_description": (right.get("display_short_description") or "").casefold(),
    }[sort_by]
    result = compare_values(sort_left, sort_right)
    if sort_dir == "desc":
        result *= -1
    if result != 0:
        return result

    left_timestamp = _effective_call_timestamp(left)
    right_timestamp = _effective_call_timestamp(right)
    timestamp_tie_break = compare_values(right_timestamp, left_timestamp)
    if timestamp_tie_break != 0:
        return timestamp_tie_break

    return compare_values(left.get("call_id"), right.get("call_id"))


def _calls_sort_link(
    q: str,
    source: str,
    date: str,
    recent: str,
    current_sort_by: str,
    current_sort_dir: str,
    target_sort_by: str,
) -> str:
    target_dir = DEFAULT_CALL_SORT_DIRECTIONS[target_sort_by]
    if current_sort_by == target_sort_by:
        target_dir = "asc" if current_sort_dir == "desc" else "desc"
    params = {
        "q": q,
        "source": source,
        "date": date,
        "recent": recent,
        "sort_by": target_sort_by,
        "sort_dir": target_dir,
    }
    filtered = {key: value for key, value in params.items() if value}
    query_string = urlencode(filtered)
    return f"/calls?{query_string}" if query_string else "/calls"


def _manual_processing_notice(config: AppConfig) -> str:
    if processing_mode(config) != "manual_step":
        return "Manual processing is disabled in automatic mode."
    _, call_id, stage_count = run_manual_step(config, scan_first=False)
    if not call_id:
        return "No queued job was available."
    return f"Completed {stage_count} stage(s) for call {call_id}."


def _manual_scan_notice(config: AppConfig) -> str:
    if processing_mode(config) != "manual_step":
        return "Manual processing is disabled in automatic mode."
    imported_count = len(detect_new_calls(config))
    if not imported_count:
        return "Scan completed; no new calls imported."
    return f"Scan completed; imported {imported_count} new call(s)."


def _with_notice(url: str, notice: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["notice"] = notice
    return urlunparse(parsed._replace(query=urlencode(params)))


def _set_processing_mode_notice(config: AppConfig, requested_mode: str) -> str:
    normalized_mode = requested_mode.strip().lower()
    if normalized_mode not in {"automatic", "manual_step"}:
        return "Unsupported processing mode."
    current_mode = processing_mode(config)
    if current_mode == normalized_mode:
        return f"Processing mode is already {normalized_mode}."
    config.data.setdefault("processing", {})["startup_mode"] = normalized_mode
    config.save()
    logger.info("Processing mode changed mode=%s", normalized_mode)
    return f"Switched processing mode to {normalized_mode}."


def _set_archive_root_notice(config: AppConfig, requested_path: str) -> str:
    raw_path = requested_path.strip()
    if not raw_path:
        return "Calls folder path cannot be empty."
    normalized = Path(raw_path).expanduser()
    if not normalized.is_absolute():
        normalized = (config.root_dir / normalized).resolve()
    else:
        normalized = normalized.resolve()
    config.data.setdefault("paths", {})["archive_root"] = str(normalized)
    config.save()
    config.ensure_directories()
    logger.info("Archive root changed path=%s", normalized)
    return f"Switched calls folder to {normalized}."


def _clear_queue_notice(config: AppConfig) -> str:
    previous_mode = processing_mode(config)
    if previous_mode != "manual_step":
        config.data.setdefault("processing", {})["startup_mode"] = "manual_step"
        config.save()
    db = connect(config.sqlite_path)
    queued_count = db.execute("SELECT COUNT(*) FROM queue_jobs WHERE status = 'queued'").fetchone()[0]
    failed_count = db.execute("SELECT COUNT(*) FROM queue_jobs WHERE status = 'failed'").fetchone()[0]
    running_count = db.execute("SELECT COUNT(*) FROM queue_jobs WHERE status = 'running'").fetchone()[0]
    db.execute("DELETE FROM queue_jobs WHERE status IN ('queued', 'failed')")
    db.commit()
    logger.info(
        "Queue cleared queued=%s failed=%s running=%s mode_before=%s",
        queued_count,
        failed_count,
        running_count,
        previous_mode,
    )
    if running_count:
        return (
            f"Cleared {queued_count + failed_count} queued/failed job(s) and switched to manual_step. "
            f"{running_count} running job(s) may still finish the current stage."
        )
    return f"Cleared {queued_count + failed_count} queued/failed job(s) and switched to manual_step."


def _shared_ui_context(config: AppConfig, db, notice: str = "") -> dict:
    return {
        "processing_mode": processing_mode(config),
        "notice": notice,
        "status_line": _app_status_payload(config, db),
        "archive_root_path": str(config.archive_root),
    }


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Call Assistant")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    db = connect(config.sqlite_path)
    manual_action_lock = threading.Lock()
    manual_action_state: dict[str, threading.Thread | None] = {"thread": None}

    def _start_manual_action(target, started_notice: str) -> str:
        if processing_mode(config) != "manual_step":
            return "Manual processing is disabled in automatic mode."
        with manual_action_lock:
            existing = manual_action_state.get("thread")
            if existing and existing.is_alive():
                return "Manual processing is already running."

            def runner() -> None:
                try:
                    target()
                except Exception:
                    logger.exception("Manual action failed")
                finally:
                    with manual_action_lock:
                        if manual_action_state.get("thread") is threading.current_thread():
                            manual_action_state["thread"] = None

            thread = threading.Thread(target=runner, daemon=True, name="call-assistant-manual-action")
            manual_action_state["thread"] = thread
            thread.start()
        return started_notice

    def _load_call(call_id: str) -> tuple[dict, Path]:
        row = db.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Call not found")
        call = dict(row)
        return call, Path(call["archive_path"])

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request):
        return RedirectResponse(url="/calls")

    @app.get("/ui/status-line")
    def status_line():
        payload = _app_status_payload(config, db)
        _log_status_line_snapshot(payload)
        return payload

    @app.get("/calls", response_class=HTMLResponse)
    def calls(
        request: Request,
        q: str = "",
        source: str = "",
        date: str = "",
        recent: str = "",
        sort_by: str = DEFAULT_CALL_SORT_BY,
        sort_dir: str = "",
        notice: str = "",
    ):
        sort_by, sort_dir = _normalize_call_sort(sort_by, sort_dir)
        recent = recent if recent in ALLOWED_RECENT_FILTERS else ""
        query = "SELECT * FROM calls WHERE 1=1"
        params: list[str] = []
        if q:
            query += " AND search_text LIKE ?"
            params.append(f"%{q}%")
        if source:
            query += " AND source_filename LIKE ?"
            params.append(f"%{source}%")
        if date:
            query += " AND COALESCE(recorded_at, imported_at) LIKE ?"
            params.append(f"{date}%")
        query += " ORDER BY (recorded_at IS NULL), recorded_at DESC, imported_at DESC"
        rows = [dict(row) for row in db.execute(query, params).fetchall()]
        counts = {
            row["call_id"]: db.execute("SELECT COUNT(*) FROM tasks WHERE call_id = ?", (row["call_id"],)).fetchone()[0]
            for row in rows
        }
        for row in rows:
            row["display_recorded_at"] = row.get("recorded_at") or row.get("imported_at")
            row["display_recorded_at_badge"] = _recorded_at_badge(row.get("recorded_at_source"), row.get("recorded_at"))
            row["display_call_short_id"] = str(row.get("call_id", "")).rsplit("_", 1)[-1]
            row["display_duration"] = _format_duration(row.get("duration_seconds"))
            row["display_respondent"] = _guess_respondent(row)
            row["display_short_description"] = _short_description(row)
            row["display_short_description_preview"] = _compact_preview(row["display_short_description"], limit=110)
            row["display_task_count"] = counts[row["call_id"]]
        if not date:
            rows = _filter_calls_by_recent(rows, recent)
        rows = sorted(rows, key=cmp_to_key(lambda left, right: _compare_call_rows(left, right, sort_by, sort_dir)))
        sort_links = {
            column: _calls_sort_link(q, source, date, recent, sort_by, sort_dir, column)
            for column in DEFAULT_CALL_SORT_DIRECTIONS
        }
        return templates.TemplateResponse(
            request,
            "calls.html",
            {
                "calls": rows,
                "q": q,
                "source": source,
                "date": date,
                "recent": recent,
                "sort_by": sort_by,
                "sort_dir": sort_dir,
                "sort_links": sort_links,
                **_shared_ui_context(config, db, notice),
            },
        )

    @app.post("/upload-audio")
    async def upload_audio(audio_file: UploadFile = File(...)):
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file selected")

        suffix = Path(audio_file.filename).suffix.lower()
        allowed_extensions = {item.lower() for item in config.section("ingest")["supported_extensions"]}
        if suffix not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'none'}")

        incoming_path = config.incoming_folder / Path(audio_file.filename).name
        if incoming_path.exists():
            stem = incoming_path.stem
            incoming_path = incoming_path.with_name(f"{stem}_{utc_now().replace(':', '').replace('-', '')}{suffix}")

        with incoming_path.open("wb") as handle:
            shutil.copyfileobj(audio_file.file, handle)
        file_size = incoming_path.stat().st_size
        logger.info("Uploaded audio file name=%s path=%s size_bytes=%s", incoming_path.name, incoming_path, file_size)

        call_id = import_file(incoming_path, config)
        if call_id:
            logger.info("Upload import succeeded file=%s call_id=%s", incoming_path.name, call_id)
            notice = f"Uploaded and imported {incoming_path.name}"
        else:
            logger.info("Upload import skipped file=%s reason=duplicate_or_unstable", incoming_path.name)
            notice = f"Uploaded {incoming_path.name}; import skipped because it is duplicate or unstable"
        return RedirectResponse(url=f"/calls?notice={notice}", status_code=303)

    @app.get("/calls/{call_id}", response_class=HTMLResponse)
    def call_detail(request: Request, call_id: str):
        call, call_dir = _load_call(call_id)
        retranscribe_choices = transcription_choices(config)
        retranscribe_language_choices = transcription_language_choices()
        metadata = read_json(call_dir / "metadata.json", default={})
        raw_segments = read_json(call_dir / "transcript_segments.json", default=[])
        processing_log = read_json(call_dir / "processing_log.json", default=[])
        summary = read_json(call_dir / "summary.json", default={})
        tasks = read_json(call_dir / "tasks.json", default=[])
        reviewed_tasks = read_json(call_dir / "tasks_reviewed.json", default=[])
        transcript_clean = (call_dir / "transcript_clean.txt").read_text(encoding="utf-8") if (call_dir / "transcript_clean.txt").exists() else ""
        assignment_by_cluster = call_speaker_assignments(db, call_id)
        segments = _augment_segments_with_assignments(raw_segments, assignment_by_cluster)
        segment_view = _segment_view(segments)
        speaker_clusters = sorted(
            {
                item.get("speaker_cluster_id") or item.get("speaker_label")
                for item in segments
                if (item.get("speaker_cluster_id") or item.get("speaker_label"))
            }
        )
        selected_choice = f"{metadata.get('transcription_preference', {}).get('provider', 'local')}:{metadata.get('transcription_preference', {}).get('model', config.section('transcription')['local_model'])}"
        selected_language_override = metadata.get("transcription_language_override") or "auto"
        audio_path = _preferred_audio_path(call_dir)
        speaker_profiles = list_assignable_speaker_profiles(db)
        diarization_context = _diarization_context(processing_log, segments)
        speaker_identity_context = _speaker_identity_context(metadata, assignment_by_cluster)
        queue_statuses = _call_queue_statuses(db, call_id)
        segment_cluster_ids = {
            item.get("speaker_cluster_id")
            for item in segments
            if item.get("speaker_cluster_id")
        }
        context = {
            "request": request,
            "call": call,
            "metadata": metadata,
            "summary": summary,
            "summary_preview": _summary_preview(summary),
            "tasks": tasks,
            "task_list_preview": _task_list_preview(tasks),
            "reviewed_tasks": reviewed_tasks,
            "reviewed_tasks_preview": _task_list_preview(reviewed_tasks),
            "segments": segments,
            "transcript_clean": transcript_clean,
            "clean_transcript_preview": _clean_transcript_preview(transcript_clean),
            "processing_log": processing_log,
            "processing_log_preview": _processing_log_preview(processing_log),
            "retranscribe_choices": retranscribe_choices,
            "retranscribe_language_choices": retranscribe_language_choices,
            "selected_retranscribe_choice": selected_choice,
            "selected_retranscribe_language": selected_language_override,
            "speaker_clusters": speaker_clusters,
            "speaker_mapping": metadata.get("speaker_mapping", {}),
            "audio_url": f"/calls/{call_id}/audio" if audio_path else None,
            "segment_view": segment_view,
            "segment_count": len(segment_view),
            "speaker_cluster_count": len(segment_cluster_ids or set(speaker_clusters)),
            "review_task_count": len(tasks),
            "reviewed_task_count": len(reviewed_tasks),
            "processing_log_count": len(processing_log),
            "queue_status_count": len(queue_statuses),
            "speaker_profiles": speaker_profiles,
            "assignment_by_cluster": assignment_by_cluster,
            "cluster_status_label": _cluster_status_label,
            "queue_statuses": queue_statuses,
            **diarization_context,
            **speaker_identity_context,
            **_transcription_quality_context(metadata),
            **_shared_ui_context(config, db),
        }
        return templates.TemplateResponse(request, "call_detail.html", context)

    @app.get("/calls/{call_id}/audio")
    def call_audio(call_id: str):
        _, call_dir = _load_call(call_id)
        audio_path = _preferred_audio_path(call_dir)
        if not audio_path:
            raise HTTPException(status_code=404, detail="No playable audio found")
        return FileResponse(audio_path)

    @app.post("/calls/{call_id}/tasks/review")
    async def review_tasks(
        call_id: str,
        request: Request,
        task_id: list[str] = Form(default=[]),
        text: list[str] = Form(default=[]),
        owner: list[str] = Form(default=[]),
        task_type: list[str] = Form(default=[]),
        status: list[str] = Form(default=[]),
        deadline: list[str] = Form(default=[]),
    ):
        call, call_dir = _load_call(call_id)
        extracted_tasks = {item["task_id"]: item for item in read_json(call_dir / "tasks.json", default=[])}
        reviewed = []
        for index, current_task_id in enumerate(task_id):
            original = extracted_tasks.get(current_task_id, {})
            reviewed.append(
                {
                    "task_id": f"reviewed_{current_task_id}",
                    "text": text[index],
                    "owner": owner[index] or None,
                    "type": task_type[index] or "task",
                    "source_timestamp": original.get("source_timestamp"),
                    "source_quote": original.get("source_quote"),
                    "status": status[index] or "edited",
                    "deadline": deadline[index] or None,
                    "confidence": None,
                    "reviewed_at": utc_now(),
                    "review_source": "ui",
                    "original_task_id": current_task_id,
                }
            )
        write_json(call_dir / "tasks_reviewed.json", reviewed)
        metadata = read_json(call_dir / "metadata.json", default={})
        metadata["review_state"] = "reviewed"
        metadata["current_state"] = "reviewed"
        write_json(call_dir / "metadata.json", metadata)
        index_call(call_dir, config)
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.get("/queue", response_class=HTMLResponse)
    def queue_view(request: Request, notice: str = ""):
        jobs = list_jobs(config)
        for job in jobs:
            error_message = job.get("error_message") or ""
            job["error_preview"] = _error_preview(error_message)
            job["error_is_long"] = bool(error_message) and job["error_preview"] != error_message
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "jobs": jobs,
                **_shared_ui_context(config, db, notice),
            },
        )

    @app.post("/queue/process-next")
    def process_next_queue_job(request: Request):
        def run_processing() -> None:
            imported_count, call_id, stage_count = run_manual_step(config, scan_first=False)
            logger.info(
                "Manual processing finished imported=%s call_id=%s stage_count=%s",
                imported_count,
                call_id,
                stage_count,
            )

        notice = _start_manual_action(run_processing, "Started manual processing.")
        target = request.headers.get("referer") or "/queue"
        return RedirectResponse(url=_with_notice(target, notice), status_code=303)

    @app.post("/queue/scan-incoming")
    def scan_incoming(request: Request):
        def run_scan() -> None:
            imported_count = len(detect_new_calls(config))
            logger.info("Manual incoming scan finished imported=%s", imported_count)

        notice = _start_manual_action(run_scan, "Started incoming scan.")
        target = request.headers.get("referer") or "/queue"
        return RedirectResponse(url=_with_notice(target, notice), status_code=303)

    @app.post("/processing-mode")
    def update_processing_mode(request: Request, mode: str = Form(...)):
        notice = _set_processing_mode_notice(config, mode)
        target = request.headers.get("referer") or "/calls"
        return RedirectResponse(url=_with_notice(target, notice), status_code=303)

    @app.post("/settings/archive-root")
    def update_archive_root(request: Request, archive_root: str = Form(...)):
        notice = _set_archive_root_notice(config, archive_root)
        target = request.headers.get("referer") or "/calls"
        return RedirectResponse(url=_with_notice(target, notice), status_code=303)

    @app.post("/queue/clear")
    def clear_queue(request: Request):
        notice = _clear_queue_notice(config)
        target = request.headers.get("referer") or "/queue"
        return RedirectResponse(url=_with_notice(target, notice), status_code=303)

    @app.post("/queue/retry/{job_id}")
    def retry_job(job_id: str):
        row = db.execute("SELECT * FROM queue_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        db.execute("UPDATE queue_jobs SET status = 'queued', error_message = NULL WHERE job_id = ?", (job_id,))
        db.commit()
        return RedirectResponse(url="/queue", status_code=303)

    @app.post("/calls/{call_id}/reindex")
    def reindex_call(call_id: str):
        _, call_dir = _load_call(call_id)
        index_call(call_dir, config)
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/retranscribe")
    def retranscribe_call(call_id: str, model_choice: str = Form(...), language_override: str = Form("auto")):
        _, call_dir = _load_call(call_id)
        choices = {item["value"] for item in transcription_choices(config)}
        if model_choice not in choices:
            raise HTTPException(status_code=400, detail="Unsupported transcription model selection")
        if language_override not in {"auto", "ru", "he", "en"}:
            raise HTTPException(status_code=400, detail="Unsupported transcription language selection")
        reset_call_for_retranscription(config, call_id, call_dir, model_choice, language_override)
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/speaker-mapping")
    async def save_speaker_mapping(
        call_id: str,
        request: Request,
        speaker_cluster_id: list[str] = Form(default=[]),
        speaker_role: list[str] = Form(default=[]),
    ):
        _, call_dir = _load_call(call_id)
        mapping = {
            cluster_id: role
            for cluster_id, role in zip(speaker_cluster_id, speaker_role)
            if role in {"me", "other", "unknown"}
        }
        metadata = read_json(call_dir / "metadata.json", default={})
        metadata["speaker_mapping"] = mapping
        metadata["speaker_mapping_reviewed_at"] = utc_now()
        metadata["speaker_mapping_source"] = "ui"
        write_json(call_dir / "metadata.json", metadata)

        raw_segments = read_json(call_dir / "transcript_segments.json", default=[])
        mapped_segments = apply_speaker_mapping(raw_segments, mapping)
        write_json(call_dir / "transcript_segments.json", mapped_segments)
        _refresh_transcript_outputs(call_dir, config)
        append_log(call_dir / "processing_log.json", {"event": "speaker_mapping_saved", "at": utc_now(), "mapping": mapping})
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/speakers/assign")
    async def assign_speaker(
        call_id: str,
        speaker_cluster_id: str = Form(...),
        speaker_identity_id: str = Form(...),
    ):
        _, call_dir = _load_call(call_id)
        try:
            assign_speaker_identity(config, call_dir, speaker_cluster_id, speaker_identity_id, assignment_source="user")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _refresh_transcript_outputs(call_dir, config)
        append_log(
            call_dir / "processing_log.json",
            {
                "event": "speaker_identity_assigned",
                "at": utc_now(),
                "speaker_cluster_id": speaker_cluster_id,
                "speaker_identity_id": speaker_identity_id,
            },
        )
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/speakers/create-and-assign")
    async def create_and_assign_speaker(
        call_id: str,
        speaker_cluster_id: str = Form(...),
        display_name: str = Form(...),
    ):
        if not display_name.strip():
            raise HTTPException(status_code=400, detail="Display name is required")
        _, call_dir = _load_call(call_id)
        speaker_identity_id = create_speaker_profile(db, display_name)
        try:
            assign_speaker_identity(config, call_dir, speaker_cluster_id, speaker_identity_id, assignment_source="user")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _refresh_transcript_outputs(call_dir, config)
        append_log(
            call_dir / "processing_log.json",
            {
                "event": "speaker_identity_created_and_assigned",
                "at": utc_now(),
                "speaker_cluster_id": speaker_cluster_id,
                "speaker_identity_id": speaker_identity_id,
                "display_name": display_name.strip(),
            },
        )
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/speakers/clear")
    async def clear_speaker(call_id: str, speaker_cluster_id: str = Form(...)):
        _, call_dir = _load_call(call_id)
        clear_speaker_identity(config, call_dir, speaker_cluster_id)
        _refresh_transcript_outputs(call_dir, config)
        append_log(
            call_dir / "processing_log.json",
            {"event": "speaker_identity_cleared", "at": utc_now(), "speaker_cluster_id": speaker_cluster_id},
        )
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/speakers/accept-suggestion")
    async def accept_speaker_suggestion(call_id: str, speaker_cluster_id: str = Form(...)):
        _, call_dir = _load_call(call_id)
        try:
            speaker_identity_id = accept_suggested_speaker_identity(config, call_dir, speaker_cluster_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _refresh_transcript_outputs(call_dir, config)
        append_log(
            call_dir / "processing_log.json",
            {
                "event": "speaker_identity_suggestion_accepted",
                "at": utc_now(),
                "speaker_cluster_id": speaker_cluster_id,
                "speaker_identity_id": speaker_identity_id,
            },
        )
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/speakers/reject-suggestion")
    async def reject_speaker_suggestion(call_id: str, speaker_cluster_id: str = Form(...)):
        _, call_dir = _load_call(call_id)
        try:
            reject_suggested_speaker_identity(config, call_dir, speaker_cluster_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _refresh_transcript_outputs(call_dir, config)
        append_log(
            call_dir / "processing_log.json",
            {
                "event": "speaker_identity_suggestion_rejected",
                "at": utc_now(),
                "speaker_cluster_id": speaker_cluster_id,
            },
        )
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    @app.get("/speakers", response_class=HTMLResponse)
    def speakers(request: Request):
        profiles = list_speaker_profiles(db)
        for profile in profiles:
            profile["display_name_preview"] = _compact_preview(profile.get("display_name"), limit=48)
        return templates.TemplateResponse(
            request,
            "speakers.html",
            {"request": request, "profiles": profiles, **_shared_ui_context(config, db)},
        )

    @app.get("/speakers/{speaker_identity_id}", response_class=HTMLResponse)
    def speaker_detail(request: Request, speaker_identity_id: str):
        detail = speaker_profile_detail(db, speaker_identity_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Speaker profile not found")
        detail["assignment_count"] = len(detail.get("assignments", []))
        detail["profile_preview"] = f"Status: {detail.get('status', '-')}, linked calls: {detail.get('linked_calls', 0)}"
        return templates.TemplateResponse(
            request,
            "speaker_detail.html",
            {"request": request, **detail, **_shared_ui_context(config, db)},
        )

    @app.post("/speakers/{speaker_identity_id}/rename")
    async def rename_speaker(request: Request, speaker_identity_id: str, display_name: str = Form(...)):
        try:
            rename_speaker_profile(db, speaker_identity_id, display_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url=f"/speakers/{speaker_identity_id}", status_code=303)

    @app.post("/speakers/{speaker_identity_id}/archive")
    async def archive_speaker(request: Request, speaker_identity_id: str):
        archive_speaker_profile(db, speaker_identity_id)
        return RedirectResponse(url="/speakers", status_code=303)

    return app
