from __future__ import annotations

import logging
from pathlib import Path

import whisper

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import append_log, read_json, write_json
from call_assistant.common.models import utc_now
from call_assistant.orchestrator.queue import enqueue

logger = logging.getLogger(__name__)
RETRANSCRIBE_STAGES = ("transcription", "diarization", "speaker_identity", "transcript_clean", "analysis", "indexing")


def transcription_choices(config: AppConfig) -> list[dict[str, str]]:
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


def transcription_language_choices() -> list[dict[str, str]]:
    return [
        {"value": "auto", "label": "Auto detect"},
        {"value": "ru", "label": "Force Russian"},
        {"value": "he", "label": "Force Hebrew"},
        {"value": "en", "label": "Force English"},
    ]


def reset_call_for_retranscription(
    config: AppConfig,
    call_id: str,
    call_dir: Path,
    model_choice: str,
    language_override: str = "auto",
) -> None:
    provider, model = model_choice.split(":", 1)
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "audio_prepared"
    metadata["review_state"] = "pending"
    metadata["transcription_preference"] = {"provider": provider, "model": model}
    metadata["transcription_language_mode"] = "auto" if language_override == "auto" else "metadata_override"
    metadata["transcription_language_override"] = None if language_override == "auto" else language_override
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
        {
            "event": "retranscribe_requested",
            "at": utc_now(),
            "provider": provider,
            "model": model,
            "language_override": None if language_override == "auto" else language_override,
        },
    )

    db = connect(config.sqlite_path)
    placeholders = ",".join("?" for _ in RETRANSCRIBE_STAGES)
    db.execute(
        f"DELETE FROM queue_jobs WHERE call_id = ? AND stage IN ({placeholders})",
        (call_id, *RETRANSCRIBE_STAGES),
    )
    db.execute("DELETE FROM tasks WHERE call_id = ?", (call_id,))
    db.execute(
        "DELETE FROM artifacts WHERE call_id = ? AND artifact_type IN ('transcript_raw', 'transcript_segments', 'transcript_clean', 'summary', 'tasks')",
        (call_id,),
    )
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


def reset_call_for_speaker_identity(config: AppConfig, call_id: str, call_dir: Path) -> None:
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "diarized"
    metadata["errors"] = []
    write_json(call_dir / "metadata.json", metadata)

    for artifact_name in ("transcript_clean.txt", "summary.json", "tasks.json"):
        artifact_path = call_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    db = connect(config.sqlite_path)
    db.execute(
        "DELETE FROM queue_jobs WHERE call_id = ? AND stage IN ('speaker_identity', 'transcript_clean', 'analysis', 'indexing')",
        (call_id,),
    )
    db.execute("DELETE FROM tasks WHERE call_id = ?", (call_id,))
    db.execute(
        "DELETE FROM artifacts WHERE call_id = ? AND artifact_type IN ('transcript_clean', 'summary', 'tasks')",
        (call_id,),
    )
    db.execute(
        """
        UPDATE calls
        SET current_state = ?, search_text = NULL, last_error = NULL
        WHERE call_id = ?
        """,
        ("diarized", call_id),
    )
    db.commit()
    enqueue(config, call_id, "speaker_identity")


def reset_all_calls_for_retranscription(
    config: AppConfig,
    model_choice: str,
    call_ids: list[str] | None = None,
) -> list[str]:
    db = connect(config.sqlite_path)
    if call_ids:
        placeholders = ",".join("?" for _ in call_ids)
        rows = db.execute(
            f"SELECT call_id, archive_path FROM calls WHERE call_id IN ({placeholders}) ORDER BY imported_at ASC",
            tuple(call_ids),
        ).fetchall()
    else:
        rows = db.execute("SELECT call_id, archive_path FROM calls ORDER BY imported_at ASC").fetchall()

    reset: list[str] = []
    for row in rows:
        archive_path = Path(row["archive_path"])
        if not archive_path.exists():
            logger.warning("Skipping call_id=%s because archive path is missing: %s", row["call_id"], archive_path)
            continue
        reset_call_for_retranscription(config, row["call_id"], archive_path, model_choice)
        reset.append(row["call_id"])
    return reset
