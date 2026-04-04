from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import cmp_to_key
import logging
import shutil
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import append_log, read_json, write_json
from call_assistant.common.models import TranscriptSegment
from call_assistant.common.models import utc_now
from call_assistant.diarization.service import apply_speaker_mapping
from call_assistant.ingest.watcher import import_file
from call_assistant.indexing.service import index_call
from call_assistant.orchestrator.queue import enqueue, list_jobs
from call_assistant.orchestrator.worker import processing_mode, run_manual_step
from call_assistant.reprocess import reset_call_for_retranscription, transcription_choices
from call_assistant.reprocess import transcription_language_choices
from call_assistant.speaker_identity.service import (
    assign_speaker_identity,
    clear_speaker_identity,
    create_speaker_profile,
    list_speaker_profiles,
    refresh_segments_with_assignments,
    speaker_profile_detail,
)
from call_assistant.transcript_cleaner.service import clean_transcript

logger = logging.getLogger(__name__)

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
    imported_count, call_id, stage_count = run_manual_step(config)
    if not call_id:
        return f"Imported {imported_count} new calls; no queued job was available."
    return f"Imported {imported_count} new calls; completed {stage_count} stage(s) for call {call_id}."


def _manual_processing_context(config: AppConfig, notice: str = "") -> dict[str, str]:
    return {
        "processing_mode": processing_mode(config),
        "notice": notice,
    }


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Call Assistant")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    db = connect(config.sqlite_path)

    def _load_call(call_id: str) -> tuple[dict, Path]:
        row = db.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Call not found")
        call = dict(row)
        return call, Path(call["archive_path"])

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request):
        return RedirectResponse(url="/calls")

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
            row["display_duration"] = _format_duration(row.get("duration_seconds"))
            row["display_respondent"] = _guess_respondent(row)
            row["display_short_description"] = _short_description(row)
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
                **_manual_processing_context(config, notice),
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
        segments = read_json(call_dir / "transcript_segments.json", default=[])
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
        speaker_profiles = list_speaker_profiles(db)
        context = {
            "request": request,
            "call": call,
            "metadata": metadata,
            "summary": read_json(call_dir / "summary.json", default={}),
            "tasks": read_json(call_dir / "tasks.json", default=[]),
            "reviewed_tasks": read_json(call_dir / "tasks_reviewed.json", default=[]),
            "segments": segments,
            "transcript_clean": (call_dir / "transcript_clean.txt").read_text(encoding="utf-8")
            if (call_dir / "transcript_clean.txt").exists()
            else "",
            "processing_log": read_json(call_dir / "processing_log.json", default=[]),
            "retranscribe_choices": retranscribe_choices,
            "retranscribe_language_choices": retranscribe_language_choices,
            "selected_retranscribe_choice": selected_choice,
            "selected_retranscribe_language": selected_language_override,
            "speaker_clusters": speaker_clusters,
            "speaker_mapping": metadata.get("speaker_mapping", {}),
            "audio_url": f"/calls/{call_id}/audio" if audio_path else None,
            "segment_view": _segment_view(segments),
            "speaker_profiles": speaker_profiles,
            **_manual_processing_context(config),
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
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "jobs": list_jobs(config),
                **_manual_processing_context(config, notice),
            },
        )

    @app.post("/queue/process-next")
    def process_next_queue_job():
        notice = _manual_processing_notice(config)
        return RedirectResponse(url=f"/queue?notice={notice}", status_code=303)

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
        assign_speaker_identity(config, call_dir, speaker_cluster_id, speaker_identity_id, assignment_source="user")
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
        assign_speaker_identity(config, call_dir, speaker_cluster_id, speaker_identity_id, assignment_source="user")
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

    @app.get("/speakers", response_class=HTMLResponse)
    def speakers(request: Request):
        return templates.TemplateResponse(
            request,
            "speakers.html",
            {"request": request, "profiles": list_speaker_profiles(db), **_manual_processing_context(config)},
        )

    @app.get("/speakers/{speaker_identity_id}", response_class=HTMLResponse)
    def speaker_detail(request: Request, speaker_identity_id: str):
        detail = speaker_profile_detail(db, speaker_identity_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Speaker profile not found")
        return templates.TemplateResponse(
            request,
            "speaker_detail.html",
            {"request": request, **detail, **_manual_processing_context(config)},
        )

    return app
