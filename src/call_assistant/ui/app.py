from __future__ import annotations

import logging
import shutil
from pathlib import Path

import whisper
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import append_log, read_json, write_json
from call_assistant.common.models import utc_now
from call_assistant.diarization.service import apply_speaker_mapping
from call_assistant.ingest.watcher import import_file
from call_assistant.indexing.service import index_call
from call_assistant.orchestrator.queue import enqueue, list_jobs
from call_assistant.transcript_cleaner.service import clean_transcript

logger = logging.getLogger(__name__)
RETRANSCRIBE_STAGES = ("transcription", "diarization", "transcript_clean", "analysis", "indexing")


def _transcription_choices(config: AppConfig) -> list[dict[str, str]]:
    choices = [
        {
            "value": f"local:{config.section('transcription')['local_model']}",
            "label": f"Local default ({config.section('transcription')['local_model']})",
        }
    ]
    try:
        for model_name in sorted(whisper.available_models()):
            value = f"local:{model_name}"
            if value not in {item["value"] for item in choices}:
                choices.append({"value": value, "label": f"Local {model_name}"})
    except Exception:
        logger.exception("Unable to enumerate Whisper models for retranscribe choices")
    if config.section("transcription").get("cloud_enabled", False):
        cloud_model = config.section("transcription")["cloud_model"]
        choices.append({"value": f"cloud:{cloud_model}", "label": f"OpenAI {cloud_model}"})
    return choices


def _reset_call_for_retranscription(config: AppConfig, call_id: str, call_dir: Path, model_choice: str) -> None:
    provider, model = model_choice.split(":", 1)
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "audio_prepared"
    metadata["review_state"] = "pending"
    metadata["transcription_preference"] = {"provider": provider, "model": model}
    metadata["speaker_mapping"] = {}
    metadata["speaker_mapping_reviewed_at"] = None
    metadata["speaker_mapping_source"] = "none"
    metadata["errors"] = []
    write_json(call_dir / "metadata.json", metadata)
    write_json(call_dir / "tasks_reviewed.json", [])

    for artifact_name in ("transcript_raw.json", "transcript_segments.json", "transcript_clean.txt", "summary.json", "tasks.json"):
        artifact_path = call_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    append_log(
        call_dir / "processing_log.json",
        {"event": "retranscribe_requested", "at": utc_now(), "provider": provider, "model": model},
    )

    db = connect(config.sqlite_path)
    placeholders = ",".join("?" for _ in RETRANSCRIBE_STAGES)
    db.execute(
        f"DELETE FROM queue_jobs WHERE call_id = ? AND stage IN ({placeholders})",
        (call_id, *RETRANSCRIBE_STAGES),
    )
    db.execute("DELETE FROM tasks WHERE call_id = ?", (call_id,))
    db.execute("DELETE FROM artifacts WHERE call_id = ? AND artifact_type IN ('transcript_raw', 'transcript_segments', 'transcript_clean', 'summary', 'tasks')", (call_id,))
    db.execute(
        """
        UPDATE calls
        SET current_state = ?, review_state = ?, search_text = NULL, last_error = NULL
        WHERE call_id = ?
        """,
        ("audio_prepared", "pending", call_id),
    )
    db.commit()
    enqueue(config, call_id, "transcription")


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
    def calls(request: Request, q: str = "", source: str = "", date: str = "", notice: str = ""):
        query = "SELECT * FROM calls WHERE 1=1"
        params: list[str] = []
        if q:
            query += " AND search_text LIKE ?"
            params.append(f"%{q}%")
        if source:
            query += " AND source_filename LIKE ?"
            params.append(f"%{source}%")
        if date:
            query += " AND imported_at LIKE ?"
            params.append(f"{date}%")
        query += " ORDER BY imported_at DESC"
        rows = [dict(row) for row in db.execute(query, params).fetchall()]
        counts = {
            row["call_id"]: db.execute("SELECT COUNT(*) FROM tasks WHERE call_id = ?", (row["call_id"],)).fetchone()[0]
            for row in rows
        }
        return templates.TemplateResponse(
            request,
            "calls.html",
            {"calls": rows, "task_counts": counts, "q": q, "source": source, "date": date, "notice": notice},
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
        retranscribe_choices = _transcription_choices(config)
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
            "selected_retranscribe_choice": selected_choice,
            "speaker_clusters": speaker_clusters,
            "speaker_mapping": metadata.get("speaker_mapping", {}),
        }
        return templates.TemplateResponse(request, "call_detail.html", context)

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
    def queue_view(request: Request):
        return templates.TemplateResponse(request, "queue.html", {"jobs": list_jobs(config)})

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
    def retranscribe_call(call_id: str, model_choice: str = Form(...)):
        _, call_dir = _load_call(call_id)
        choices = {item["value"] for item in _transcription_choices(config)}
        if model_choice not in choices:
            raise HTTPException(status_code=400, detail="Unsupported transcription model selection")
        _reset_call_for_retranscription(config, call_id, call_dir, model_choice)
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

        from call_assistant.common.models import TranscriptSegment

        clean = clean_transcript([TranscriptSegment(**item) for item in mapped_segments])
        (call_dir / "transcript_clean.txt").write_text(clean.text, encoding="utf-8")
        append_log(call_dir / "processing_log.json", {"event": "speaker_mapping_saved", "at": utc_now(), "mapping": mapping})
        index_call(call_dir, config)
        return RedirectResponse(url=f"/calls/{call_id}", status_code=303)

    return app
