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
from call_assistant.orchestrator.queue import claim_next_job, complete_job, enqueue, fail_job, reset_running_jobs_on_startup
from call_assistant.speaker_identity.service import run_speaker_identity_stage
from call_assistant.transcript_cleaner.service import clean_transcript
from call_assistant.transcription.service import _looks_mixed_language_problem, _transcription_preferences, transcribe

logger = logging.getLogger(__name__)

STAGE_SEQUENCE = {
    "audio_prepare": "transcription",
    "transcription": "diarization",
    "diarization": "speaker_identity",
    "speaker_identity": "transcript_clean",
    "transcript_clean": "analysis",
    "analysis": "indexing",
}

IN_PROGRESS_STATE = {
    "audio_prepare": "preparing_audio",
    "transcription": "transcribing",
    "diarization": "diarizing",
    "speaker_identity": "resolving_speaker_identity",
    "transcript_clean": "cleaning_transcript",
    "analysis": "analyzing",
    "indexing": "indexing",
}


def processing_mode(config: AppConfig) -> str:
    mode = str(config.section("processing").get("startup_mode", "automatic")).strip().lower()
    return mode if mode in {"automatic", "manual_step"} else "automatic"


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
    audio_path = call_dir / "audio_normalized.wav"
    default_provider, model_override, language_override, language_mode = _transcription_preferences(audio_path, config)
    raw = transcribe(audio_path, config)
    write_json(call_dir / "transcript_raw.json", raw)
    low_confidence = not raw.segments
    metadata = _update_metadata(
        call_dir,
        current_state="transcribed",
        low_confidence=low_confidence,
        transcription_provider=raw.provider,
        transcription_model=raw.model,
        transcription_language_detected=raw.language,
        transcription_language_override=language_override,
        transcription_language_mode=language_mode,
        mixed_language_suspected=_looks_mixed_language_problem(raw, audio_path),
    )
    _sync_call_row(config, call_dir)
    logger.info("Stage transcription success call_dir=%s provider=%s segments=%s", call_dir, raw.provider, len(raw.segments))


def _process_diarization(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage diarization start call_dir=%s", call_dir)
    raw = read_json(call_dir / "transcript_raw.json", default={})
    metadata = read_json(call_dir / "metadata.json", default={})
    normalized = transcribe_data_to_segments(
        raw,
        config,
        call_dir / "audio_normalized.wav",
        metadata.get("speaker_mapping"),
    )
    write_json(call_dir / "transcript_segments.json", normalized)
    metadata = _update_metadata(call_dir, current_state="diarized")
    _sync_call_row(config, call_dir)
    logger.info("Stage diarization success call_dir=%s segments=%s", call_dir, len(normalized))


def transcribe_data_to_segments(
    raw_payload: dict,
    config: AppConfig,
    audio_path: Path,
    speaker_mapping: dict[str, str] | None = None,
) -> list[dict]:
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
    return diarize(raw, config, audio_path=audio_path, speaker_mapping=speaker_mapping)


def _process_transcript_clean(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage transcript_clean start call_dir=%s", call_dir)
    payload = read_json(call_dir / "transcript_segments.json", default=[])
    segments = [TranscriptSegment(**item) for item in payload]
    clean = clean_transcript(segments)
    (call_dir / "transcript_clean.txt").write_text(clean.text, encoding="utf-8")
    metadata = _update_metadata(call_dir, current_state="transcript_clean")
    _sync_call_row(config, call_dir)
    logger.info("Stage transcript_clean success call_dir=%s segment_count=%s", call_dir, len(segments))


def _process_speaker_identity(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage speaker_identity start call_dir=%s", call_dir)
    updated = run_speaker_identity_stage(config, call_dir)
    _update_metadata(call_dir, current_state="speaker_identity")
    _sync_call_row(config, call_dir)
    logger.info("Stage speaker_identity success call_dir=%s segments=%s", call_dir, len(updated))


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
    current_state = IN_PROGRESS_STATE.get(job.stage)
    if current_state:
        _update_metadata(call_dir, current_state=current_state)
        _sync_call_row(config, call_dir)
    append_log(call_dir / "processing_log.json", {"event": f"{job.stage}_started", "at": job.started_at})
    if job.stage == "audio_prepare":
        _process_audio_prepare(config, call_dir)
    elif job.stage == "transcription":
        _process_transcription(config, call_dir)
    elif job.stage == "diarization":
        _process_diarization(config, call_dir)
    elif job.stage == "speaker_identity":
        _process_speaker_identity(config, call_dir)
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


def run_manual_step(config: AppConfig) -> tuple[int, bool]:
    imported = detect_new_calls(config)
    processed = run_once(config, scan_first=False)
    return len(imported), processed


class WorkerThread:
    def __init__(self, config: AppConfig):
        self.config = config
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        if processing_mode(self.config) == "manual_step":
            logger.info("Worker startup_mode=manual_step automatic background processing disabled")
            return
        recovered = reset_running_jobs_on_startup(self.config)
        if recovered:
            logger.warning("Worker startup recovered_running_jobs=%s", recovered)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        scan_interval = max(1, int(self.config.section("ingest")["scan_interval_seconds"]))
        next_scan_at = 0.0
        processed_since_scan = 0

        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_scan_at:
                imported = detect_new_calls(self.config)
                if imported:
                    logger.info("Worker scan imported_count=%s", len(imported))
                if processed_since_scan:
                    logger.info("Worker processed_jobs_since_last_scan=%s", processed_since_scan)
                    processed_since_scan = 0
                next_scan_at = now + scan_interval

            if run_once(self.config, scan_first=False):
                processed_since_scan += 1
                continue

            sleep_for = max(0.2, min(1.0, next_scan_at - time.monotonic()))
            self._stop.wait(sleep_for)
