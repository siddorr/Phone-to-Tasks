from __future__ import annotations

import hashlib
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from call_assistant import __version__
from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import append_log, write_json
from call_assistant.common.models import CallMetadata, call_dir_from_metadata
from call_assistant.ingest.recorded_time import resolve_recorded_at
from call_assistant.orchestrator.queue import enqueue

logger = logging.getLogger(__name__)


def scan_incoming(config: AppConfig) -> list[Path]:
    extensions = {item.lower() for item in config.section("ingest")["supported_extensions"]}
    return sorted(
        path for path in config.incoming_folder.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )


def is_file_stable(path: Path, config: AppConfig) -> bool:
    poll_interval = int(config.section("ingest")["stability_poll_interval_seconds"])
    checks = max(2, int(config.section("ingest")["stability_check_seconds"]) // max(poll_interval, 1))
    observations: list[tuple[int, float]] = []
    for _ in range(checks):
        stat = path.stat()
        observations.append((stat.st_size, stat.st_mtime))
        time.sleep(poll_interval)
    return len(set(observations)) == 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def already_imported(config: AppConfig, sha256: str) -> bool:
    db = connect(config.sqlite_path)
    row = db.execute("SELECT call_id FROM calls WHERE sha256 = ?", (sha256,)).fetchone()
    return row is not None


def already_imported_source_path(config: AppConfig, source_path: Path) -> bool:
    db = connect(config.sqlite_path)
    row = db.execute("SELECT call_id FROM calls WHERE source_path = ?", (str(source_path.resolve()),)).fetchone()
    return row is not None


def register_imported_call(config: AppConfig, metadata: CallMetadata, call_dir: Path) -> bool:
    db = connect(config.sqlite_path)
    try:
        db.execute(
            """
            INSERT INTO calls (
                call_id, archive_path, source_filename, source_path, sha256, recorded_at, recorded_at_source, recorded_at_confidence, imported_at,
                duration_seconds, audio_format, language_summary, current_state, review_state,
                low_confidence, last_error, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.call_id,
                str(call_dir),
                metadata.source_filename,
                metadata.source_path,
                metadata.sha256,
                metadata.recorded_at,
                metadata.recorded_at_source,
                metadata.recorded_at_confidence,
                metadata.imported_at,
                metadata.duration_seconds,
                metadata.audio_format,
                None,
                metadata.current_state,
                metadata.review_state,
                int(metadata.low_confidence),
                metadata.errors[-1] if metadata.errors else None,
                None,
            ),
        )
        db.commit()
        return True
    except Exception as exc:
        logger.info("Import skipped file=%s reason=duplicate sha256=%s error=%s", metadata.source_path, metadata.sha256, exc)
        return False


def import_file(path: Path, config: AppConfig) -> str | None:
    if already_imported_source_path(config, path):
        logger.info("Import skipped file=%s reason=known_source_path", path)
        return None
    if not is_file_stable(path, config):
        logger.info("Import skipped file=%s reason=unstable", path)
        return None
    sha256 = file_sha256(path)
    if already_imported(config, sha256):
        logger.info("Import skipped file=%s reason=duplicate sha256=%s", path, sha256)
        return None
    imported_at = datetime.now(timezone.utc).isoformat()
    call_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sha256[:6]}"
    recorded_at, recorded_at_source, recorded_at_confidence = resolve_recorded_at(path)
    metadata = CallMetadata(
        schema_version="1.0",
        app_version=__version__,
        call_id=call_id,
        source_filename=path.name,
        source_path=str(path.resolve()),
        imported_at=imported_at,
        recorded_at=recorded_at,
        recorded_at_source=recorded_at_source,
        recorded_at_confidence=recorded_at_confidence,
        file_size_bytes=path.stat().st_size,
        sha256=sha256,
        language_hints=config.section("transcription").get("language_hints", []),
        current_state="imported",
    )
    call_dir = call_dir_from_metadata(config.archive_root, metadata)
    call_dir.mkdir(parents=True, exist_ok=True)
    if not register_imported_call(config, metadata, call_dir):
        return None
    shutil.copy2(path, call_dir / f"audio_original{path.suffix.lower()}")
    write_json(call_dir / "metadata.json", metadata)
    write_json(call_dir / "tasks_reviewed.json", [])
    write_json(call_dir / "processing_log.json", [])
    append_log(call_dir / "processing_log.json", {"event": "imported", "at": imported_at, "source": str(path)})
    enqueue(config, call_id, "audio_prepare")
    logger.info("Imported file=%s call_id=%s archive_dir=%s", path, call_id, call_dir)
    return call_id


def detect_new_calls(config: AppConfig) -> list[str]:
    call_ids: list[str] = []
    for candidate in scan_incoming(config):
        call_id = import_file(candidate, config)
        if call_id:
            call_ids.append(call_id)
    return call_ids
