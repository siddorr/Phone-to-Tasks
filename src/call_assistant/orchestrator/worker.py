from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from call_assistant.analysis.service import analyze_call
from call_assistant.audio.normalize import normalize_audio
from call_assistant.common.config import AppConfig
from call_assistant.common.io import append_log, read_json, write_json
from call_assistant.common.models import AnalysisBundle, artifact_paths
from call_assistant.diarization.service import diarize
from call_assistant.indexing.service import index_call
from call_assistant.ingest.watcher import detect_new_calls
from call_assistant.orchestrator.queue import claim_next_job, complete_job, enqueue, fail_job
from call_assistant.transcript_cleaner.service import clean_transcript
from call_assistant.transcription.service import transcribe

logger = logging.getLogger(__name__)

STAGE_SEQUENCE = {
    "audio_prepare": "transcription",
    "transcription": "diarization",
    "diarization": "transcript_clean",
    "transcript_clean": "analysis",
    "analysis": "indexing",
}


def _call_dir(config: AppConfig, call_id: str) -> Path:
    matches = list(config.archive_root.rglob(f"call_{call_id}"))
    if not matches:
        raise FileNotFoundError(f"Could not locate archive directory for {call_id}")
    return matches[0]


def _update_metadata(call_dir: Path, **updates: object) -> dict:
    metadata_path = call_dir / "metadata.json"
    metadata = read_json(metadata_path, default={})
    metadata.update(updates)
    write_json(metadata_path, metadata)
    return metadata


def _sync_call_row(config: AppConfig, call_dir: Path) -> None:
    # Reindexing from canonical artifact files keeps SQLite state consistent
    # with the archive even if intermediate stage metadata changes.
    index_call(call_dir, config)


def _process_audio_prepare(config: AppConfig, call_dir: Path) -> None:
    original_audio = next(path for path in call_dir.iterdir() if path.name.startswith("audio_original"))
    logger.info("Stage audio_prepare start call_dir=%s source=%s", call_dir, original_audio.name)
    audio_metadata = normalize_audio(original_audio, call_dir / "audio_normalized.wav")
    metadata = _update_metadata(
        call_dir,
        current_state="audio_prepared",
        audio_format=audio_metadata.audio_format,
        duration_seconds=audio_metadata.duration_seconds,
    )
    _sync_call_row(config, call_dir)
    logger.info("Stage audio_prepare success call_dir=%s duration_seconds=%s format=%s", call_dir, audio_metadata.duration_seconds, audio_metadata.audio_format)


def _process_transcription(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage transcription start call_dir=%s", call_dir)
    raw = transcribe(call_dir / "audio_normalized.wav", config)
    write_json(call_dir / "transcript_raw.json", raw)
    low_confidence = not raw.segments
    metadata = _update_metadata(call_dir, current_state="transcribed", low_confidence=low_confidence)
    _sync_call_row(config, call_dir)
    logger.info("Stage transcription success call_dir=%s provider=%s segments=%s", call_dir, raw.provider, len(raw.segments))


def _process_diarization(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage diarization start call_dir=%s", call_dir)
    raw = read_json(call_dir / "transcript_raw.json", default={})
    normalized = transcribe_data_to_segments(raw)
    write_json(call_dir / "transcript_segments.json", normalized)
    metadata = _update_metadata(call_dir, current_state="diarized")
    _sync_call_row(config, call_dir)
    logger.info("Stage diarization success call_dir=%s segments=%s", call_dir, len(normalized))


def transcribe_data_to_segments(raw_payload: dict) -> list[dict]:
    from call_assistant.common.models import RawSegment, RawTranscript

    raw = RawTranscript(
        provider=raw_payload.get("provider", "unknown"),
        model=raw_payload.get("model", "unknown"),
        language=raw_payload.get("language", "unknown"),
        confidence=raw_payload.get("confidence"),
        text=raw_payload.get("text", ""),
        segments=[
            RawSegment(
                start_sec=float(item.get("start_sec", item.get("start", 0.0))),
                end_sec=float(item.get("end_sec", item.get("end", 0.0))),
                text=item.get("text", ""),
                confidence=item.get("confidence"),
                speaker=item.get("speaker"),
            )
            for item in raw_payload.get("segments", [])
        ],
    )
    return diarize(raw)


def _process_transcript_clean(config: AppConfig, call_dir: Path) -> None:
    from call_assistant.common.models import TranscriptSegment

    logger.info("Stage transcript_clean start call_dir=%s", call_dir)
    payload = read_json(call_dir / "transcript_segments.json", default=[])
    segments = [TranscriptSegment(**item) for item in payload]
    clean = clean_transcript(segments)
    (call_dir / "transcript_clean.txt").write_text(clean.text, encoding="utf-8")
    metadata = _update_metadata(call_dir, current_state="transcript_clean")
    _sync_call_row(config, call_dir)
    logger.info("Stage transcript_clean success call_dir=%s segment_count=%s", call_dir, len(segments))


def _process_analysis(config: AppConfig, call_dir: Path) -> None:
    from call_assistant.common.models import TranscriptSegment, CleanTranscript

    logger.info("Stage analysis start call_dir=%s", call_dir)
    payload = read_json(call_dir / "transcript_segments.json", default=[])
    segments = [TranscriptSegment(**item) for item in payload]
    text = (call_dir / "transcript_clean.txt").read_text(encoding="utf-8")
    clean = CleanTranscript(text=text, segments=segments)
    analysis = analyze_call(clean, segments, config)
    write_json(call_dir / "summary.json", analysis)
    write_json(call_dir / "tasks.json", analysis.tasks)
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "analyzed"
    metadata["low_confidence"] = bool(metadata.get("low_confidence")) or bool(analysis.low_confidence_reason)
    write_json(call_dir / "metadata.json", metadata)
    _sync_call_row(config, call_dir)
    logger.info("Stage analysis success call_dir=%s task_count=%s confidence=%s", call_dir, len(analysis.tasks), analysis.analysis_confidence)


def _process_indexing(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage indexing start call_dir=%s", call_dir)
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "indexed"
    write_json(call_dir / "metadata.json", metadata)
    _sync_call_row(config, call_dir)
    index_call(call_dir, config)
    logger.info("Stage indexing success call_dir=%s", call_dir)


def process_job(config: AppConfig, job) -> None:
    call_dir = _call_dir(config, job.call_id)
    append_log(call_dir / "processing_log.json", {"event": f"{job.stage}_started", "at": job.started_at})
    if job.stage == "audio_prepare":
        _process_audio_prepare(config, call_dir)
    elif job.stage == "transcription":
        _process_transcription(config, call_dir)
    elif job.stage == "diarization":
        _process_diarization(config, call_dir)
    elif job.stage == "transcript_clean":
        _process_transcript_clean(config, call_dir)
    elif job.stage == "analysis":
        _process_analysis(config, call_dir)
    elif job.stage == "indexing":
        _process_indexing(config, call_dir)
    else:
        raise ValueError(f"Unknown stage {job.stage}")
    append_log(call_dir / "processing_log.json", {"event": f"{job.stage}_finished"})
    next_stage = STAGE_SEQUENCE.get(job.stage)
    if next_stage:
        enqueue(config, job.call_id, next_stage)


def run_once(config: AppConfig, scan_first: bool = False) -> bool:
    if scan_first:
        detect_new_calls(config)
    job = claim_next_job(config)
    if not job:
        return False
    logger.info("Worker claimed job_id=%s call_id=%s stage=%s attempt=%s", job.job_id, job.call_id, job.stage, job.attempt_count)
    try:
        process_job(config, job)
        complete_job(config, job.job_id)
        return True
    except Exception as exc:
        logger.exception("Job %s failed", job.job_id)
        call_dir = _call_dir(config, job.call_id)
        metadata = read_json(call_dir / "metadata.json", default={})
        metadata.setdefault("errors", []).append(str(exc))
        metadata["current_state"] = "failed"
        write_json(call_dir / "metadata.json", metadata)
        _sync_call_row(config, call_dir)
        append_log(call_dir / "processing_log.json", {"event": f"{job.stage}_failed", "error": str(exc)})
        fail_job(config, job, str(exc), retryable=True)
        return True


def process_pending(config: AppConfig, scan_first: bool = True) -> int:
    processed = 0
    if scan_first:
        detect_new_calls(config)
    while run_once(config, scan_first=False):
        processed += 1
    return processed


def drain_queue(config: AppConfig) -> int:
    processed = 0
    while run_once(config, scan_first=False):
        processed += 1
    return processed


class WorkerThread:
    def __init__(self, config: AppConfig):
        self.config = config
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            imported = detect_new_calls(self.config)
            if imported:
                logger.info("Worker scan imported_count=%s", len(imported))
            processed = drain_queue(self.config)
            if processed:
                logger.info("Worker drain processed_jobs=%s", processed)
                continue
            time.sleep(int(self.config.section("ingest")["scan_interval_seconds"]))
