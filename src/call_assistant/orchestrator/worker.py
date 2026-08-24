from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from shutil import copyfile

from call_assistant.analysis.service import analyze_call
from call_assistant.audio.normalize import normalize_audio
from call_assistant.common.config import AppConfig
from call_assistant.common.io import append_log, read_json, write_json
from call_assistant.common.models import AnalysisBundle, TranscriptSegment, artifact_paths, utc_now
from call_assistant.diarization.service import diarize
from call_assistant.indexing.service import index_call
from call_assistant.ingest.watcher import detect_new_calls
from call_assistant.orchestrator.queue import claim_next_job, claim_next_job_for_call, complete_job, enqueue, fail_job, reset_running_jobs_on_startup
from call_assistant.speaker_identity.service import run_speaker_identity_stage
from call_assistant.transcript_cleaner.service import clean_transcript
from call_assistant.transcription.service import (
    _looks_mixed_language_problem,
    _segment_is_uncertain,
    _transcription_preferences,
    build_transcription_vad_chunk_artifact,
    load_transcription_candidate_selection,
    transcribe,
)

logger = logging.getLogger(__name__)

STAGE_SEQUENCE = {
    "audio_prepare": "transcription",
    "transcription": "diarization",
    "diarization": "speaker_identity",
    "speaker_identity": "transcript_clean",
    "transcript_clean": "analysis",
    "analysis": "indexing",
}

LAST_STABLE_STATE = {
    "audio_prepare": "imported",
    "transcription": "audio_prepared",
    "diarization": "transcribed",
    "speaker_identity": "diarized",
    "transcript_clean": "speaker_identity",
    "analysis": "transcript_clean",
    "indexing": "analyzed",
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


def _record_stage_outcome(call_dir: Path, stage: str, status: str, detail: str) -> dict:
    metadata = read_json(call_dir / "metadata.json", default={})
    outcomes = metadata.setdefault("stage_outcomes", {})
    outcomes[stage] = {"status": status, "detail": detail, "at": utc_now()}
    metadata["stage_outcomes"] = outcomes
    write_json(call_dir / "metadata.json", metadata)
    return metadata


def _set_blocking_state(call_dir: Path, stage: str | None, error: str | None) -> dict:
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["last_blocking_stage"] = stage
    metadata["last_blocking_error"] = error
    write_json(call_dir / "metadata.json", metadata)
    return metadata


def _sync_call_row(config: AppConfig, call_dir: Path) -> None:
    # Reindexing from canonical artifact files keeps SQLite state consistent
    # with the archive even if intermediate stage metadata changes.
    index_call(call_dir, config)


def _require_artifact(path: Path, stage: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{stage} requires artifact {path.name}")


def _process_audio_prepare(config: AppConfig, call_dir: Path) -> None:
    original_audio = next(path for path in call_dir.iterdir() if path.name.startswith("audio_original"))
    logger.info("Stage audio_prepare start call_dir=%s source=%s", call_dir, original_audio.name)
    processing_subdir = Path(config.section("audio_processing").get("temp_subdir", "./audio_processing"))
    processing_dir = call_dir / processing_subdir
    normalized_dest = processing_dir / "audio_normalized.wav"
    audio_metadata, chunk_info = normalize_audio(original_audio, normalized_dest, config)
    copyfile(normalized_dest, call_dir / "audio_normalized.wav")
    metadata = _update_metadata(
        call_dir,
        current_state="audio_prepared",
        audio_format=audio_metadata.audio_format,
        duration_seconds=audio_metadata.duration_seconds,
        audio_chunk_count=len(chunk_info),
        audio_chunks_available=bool(chunk_info),
    )
    if chunk_info:
        manifest_entries: list[dict] = []
        for entry in chunk_info:
            chunk_path = Path(entry["path"])
            relative_path = str(chunk_path.relative_to(call_dir))
            manifest_entries.append(
                {
                    "chunk_id": entry["chunk_id"],
                    "path": relative_path,
                    "start_sec": entry["start_sec"],
                    "end_sec": entry["end_sec"],
                    "duration_seconds": entry["duration_seconds"],
                }
            )
        write_json(call_dir / CHUNK_MANIFEST_FILENAME, manifest_entries)
    _record_stage_outcome(call_dir, "audio_prepare", "success", "Audio normalization completed")
    _sync_call_row(config, call_dir)
    logger.info("Stage audio_prepare success call_dir=%s duration_seconds=%s format=%s", call_dir, audio_metadata.duration_seconds, audio_metadata.audio_format)


def _process_transcription(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage transcription start call_dir=%s", call_dir)
    audio_path = call_dir / "audio_normalized.wav"
    _require_artifact(audio_path, "transcription")
    default_provider, model_override, language_override, language_mode = _transcription_preferences(audio_path, config)
    raw = transcribe(audio_path, config)
    write_json(call_dir / "transcript_raw.json", raw)
    vad_chunk_artifact = build_transcription_vad_chunk_artifact(audio_path, raw, config)
    write_json(call_dir / "transcript_vad_chunks.json", vad_chunk_artifact)
    candidate_selection = load_transcription_candidate_selection(call_dir)
    selection = candidate_selection.get("selection", {})
    segment_detection = candidate_selection.get("segment_detection", {})
    segment_retries = candidate_selection.get("segment_retries", [])
    clause_detection = candidate_selection.get("clause_detection", {})
    clause_retries = candidate_selection.get("clause_retries", [])
    word_span_detection = candidate_selection.get("word_span_detection", {})
    word_span_retries = candidate_selection.get("word_span_retries", [])
    word_span_validation = candidate_selection.get("word_span_validation_llm_response", [])
    low_confidence = not raw.segments
    metadata = _update_metadata(
        call_dir,
        current_state="transcribed",
        low_confidence=low_confidence,
        transcription_provider=raw.provider,
        transcription_model=raw.model,
        transcription_backend=raw.backend_selected or raw.backend_attempted or config.section("transcription").get("backend", "whisper_legacy"),
        transcription_backend_attempted=raw.backend_attempted or config.section("transcription").get("backend", "whisper_legacy"),
        transcription_backend_selected=raw.backend_selected or raw.backend_attempted or config.section("transcription").get("backend", "whisper_legacy"),
        transcription_backend_fallback_reason=raw.backend_fallback_reason,
        transcription_chunk_count=vad_chunk_artifact.get("chunk_count"),
        transcription_segment_language_counts=vad_chunk_artifact.get("segment_language_counts", {}),
        transcription_language_smoothing_applied=bool(config.section("transcription").get("language_smoothing_enabled", True)),
        transcription_uncertain_segment_count=sum(1 for item in raw.segments if _segment_is_uncertain(item)),
        transcription_chunked_plausibility_score=raw.chunked_plausibility_score,
        transcription_chunked_plausibility_flags=raw.chunked_plausibility_flags or [],
        transcription_language_detected=raw.language,
        transcription_language_override=language_override,
        transcription_language_mode=language_mode,
        mixed_language_suspected=_looks_mixed_language_problem(raw, audio_path),
        transcription_selection_strategy=selection.get("strategy"),
        transcription_retried_languages=selection.get("retried_languages", []),
        transcription_quality_flags=selection.get("quality_flags", []),
        transcription_quality_score=selection.get("quality_score"),
        transcription_low_confidence_reason=selection.get("low_confidence_reason"),
        segment_detection_provider=segment_detection.get("provider"),
        segment_detection_model=segment_detection.get("model"),
        segment_detection_suspicious_count=segment_detection.get("suspicious_count"),
        segment_detection_below_threshold_count=segment_detection.get("below_threshold_count"),
        segment_detection_threshold_used=segment_detection.get("threshold_used"),
        segment_detection_retry_count=len(segment_retries),
        segment_detection_strategy=selection.get("strategy"),
        clause_retry_candidate_count=clause_detection.get("clause_candidates_detected"),
        clause_retry_selected_count=clause_detection.get("selected_for_retry"),
        clause_retry_applied_count=sum(1 for item in clause_retries if item.get("replacement_applied")),
        word_span_detection_provider=word_span_detection.get("provider"),
        word_span_detection_model=word_span_detection.get("model"),
        word_span_detection_count=word_span_detection.get("spans_detected"),
        word_span_retry_count=len(word_span_retries),
        word_span_below_threshold_count=word_span_detection.get("below_threshold_count"),
        word_span_threshold_used=word_span_detection.get("threshold_used"),
        word_span_replacement_count=sum(1 for item in word_span_retries if item.get("replacement_applied")),
        word_span_validation_count=len(word_span_validation),
    )
    _record_stage_outcome(call_dir, "transcription", "success", f"Transcription produced {len(raw.segments)} segment(s)")
    _sync_call_row(config, call_dir)
    logger.info("Stage transcription success call_dir=%s provider=%s segments=%s", call_dir, raw.provider, len(raw.segments))


def _process_diarization(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage diarization start call_dir=%s", call_dir)
    transcript_raw_path = call_dir / "transcript_raw.json"
    _require_artifact(transcript_raw_path, "diarization")
    raw = read_json(transcript_raw_path, default={})
    metadata = read_json(call_dir / "metadata.json", default={})
    normalized = transcribe_data_to_segments(
        raw,
        config,
        call_dir / "audio_normalized.wav",
        metadata.get("speaker_mapping"),
    )
    write_json(call_dir / "transcript_segments.json", normalized)
    diarization_mode = (
        "single_speaker_fallback"
        if len({item.speaker_cluster_id for item in normalized}) <= 1
        else "clustered"
    )
    metadata = _update_metadata(call_dir, current_state="diarized", diarization_mode=diarization_mode)
    _record_stage_outcome(
        call_dir,
        "diarization",
        "degraded" if diarization_mode == "single_speaker_fallback" else "success",
        "Diarization used single-speaker fallback" if diarization_mode == "single_speaker_fallback" else "Diarization produced speaker clusters",
    )
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
        language_distribution=raw_payload.get("language_distribution"),
        primary_language_confidence=raw_payload.get("primary_language_confidence"),
        backend_attempted=raw_payload.get("backend_attempted"),
        backend_selected=raw_payload.get("backend_selected"),
        backend_fallback_reason=raw_payload.get("backend_fallback_reason"),
        chunked_plausibility_score=raw_payload.get("chunked_plausibility_score"),
        chunked_plausibility_flags=raw_payload.get("chunked_plausibility_flags"),
        segments=[
            RawSegment(
                start_sec=float(item.get("start_sec", item.get("start", 0.0))),
                end_sec=float(item.get("end_sec", item.get("end", 0.0))),
                text=item.get("text", ""),
                confidence=item.get("confidence"),
                speaker=item.get("speaker"),
                language=item.get("language"),
                language_confidence=item.get("language_confidence"),
                chunk_id=item.get("chunk_id"),
                chunk_start_sec=item.get("chunk_start_sec"),
                chunk_end_sec=item.get("chunk_end_sec"),
                smoothed_language=item.get("smoothed_language"),
                smoothed_language_reason=item.get("smoothed_language_reason"),
            )
            for item in raw_payload.get("segments", [])
        ],
    )
    return diarize(raw, config, audio_path=audio_path, speaker_mapping=speaker_mapping)


def _process_transcript_clean(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage transcript_clean start call_dir=%s", call_dir)
    segments_path = call_dir / "transcript_segments.json"
    _require_artifact(segments_path, "transcript_clean")
    payload = read_json(segments_path, default=[])
    segments = [TranscriptSegment(**item) for item in payload]
    clean = clean_transcript(segments)
    (call_dir / "transcript_clean.txt").write_text(clean.text, encoding="utf-8")
    metadata = _update_metadata(call_dir, current_state="transcript_clean")
    _record_stage_outcome(call_dir, "transcript_clean", "success", f"Transcript cleaned with {len(segments)} segment(s)")
    _sync_call_row(config, call_dir)
    logger.info("Stage transcript_clean success call_dir=%s segment_count=%s", call_dir, len(segments))


def _process_speaker_identity(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage speaker_identity start call_dir=%s", call_dir)
    _require_artifact(call_dir / "transcript_segments.json", "speaker_identity")
    result = run_speaker_identity_stage(config, call_dir)
    identity_mode = {
        "success": "full",
        "degraded": "degraded",
        "skipped": "skipped",
    }.get(result.outcome_status, "degraded")
    _update_metadata(
        call_dir,
        current_state="speaker_identity",
        speaker_identity_mode=identity_mode,
        speaker_identity_summary=result.outcome_detail,
    )
    _record_stage_outcome(call_dir, "speaker_identity", result.outcome_status, result.outcome_detail)
    _sync_call_row(config, call_dir)
    logger.info(
        "Stage speaker_identity success call_dir=%s segments=%s outcome=%s detail=%s",
        call_dir,
        len(result.segments),
        result.outcome_status,
        result.outcome_detail,
    )


def _process_analysis(config: AppConfig, call_dir: Path) -> None:
    from call_assistant.common.models import CleanTranscript

    logger.info("Stage analysis start call_dir=%s", call_dir)
    segments_path = call_dir / "transcript_segments.json"
    clean_path = call_dir / "transcript_clean.txt"
    _require_artifact(segments_path, "analysis")
    _require_artifact(clean_path, "analysis")
    payload = read_json(segments_path, default=[])
    segments = [TranscriptSegment(**item) for item in payload]
    text = clean_path.read_text(encoding="utf-8")
    clean = CleanTranscript(text=text, segments=segments)
    analysis = analyze_call(clean, segments, config)
    write_json(call_dir / "summary.json", analysis)
    write_json(call_dir / "tasks.json", analysis.tasks)
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "analyzed"
    metadata["low_confidence"] = bool(metadata.get("low_confidence")) or bool(analysis.low_confidence_reason)
    write_json(call_dir / "metadata.json", metadata)
    _record_stage_outcome(
        call_dir,
        "analysis",
        "degraded" if analysis.low_confidence_reason else "success",
        analysis.low_confidence_reason or "Analysis completed",
    )
    _sync_call_row(config, call_dir)
    logger.info("Stage analysis success call_dir=%s task_count=%s confidence=%s", call_dir, len(analysis.tasks), analysis.analysis_confidence)


def _process_indexing(config: AppConfig, call_dir: Path) -> None:
    logger.info("Stage indexing start call_dir=%s", call_dir)
    metadata = read_json(call_dir / "metadata.json", default={})
    metadata["current_state"] = "indexed"
    write_json(call_dir / "metadata.json", metadata)
    _record_stage_outcome(call_dir, "indexing", "success", "Indexing completed")
    _sync_call_row(config, call_dir)
    index_call(call_dir, config)
    logger.info("Stage indexing success call_dir=%s", call_dir)


def process_job(config: AppConfig, job) -> None:
    call_dir = _call_dir(config, job.call_id)
    current_state = IN_PROGRESS_STATE.get(job.stage)
    if current_state:
        _update_metadata(call_dir, current_state=current_state)
        _set_blocking_state(call_dir, None, None)
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


def _run_claimed_job(config: AppConfig, job) -> bool:
    logger.info("Worker claimed job_id=%s call_id=%s stage=%s attempt=%s", job.job_id, job.call_id, job.stage, job.attempt_count)
    try:
        process_job(config, job)
        call_dir = _call_dir(config, job.call_id)
        metadata = read_json(call_dir / "metadata.json", default={})
        if metadata.get("errors"):
            metadata["errors"] = []
            write_json(call_dir / "metadata.json", metadata)
            _sync_call_row(config, call_dir)
        complete_job(config, job.job_id)
        return True
    except Exception as exc:
        logger.exception("Job %s failed", job.job_id)
        call_dir = _call_dir(config, job.call_id)
        metadata = read_json(call_dir / "metadata.json", default={})
        metadata.setdefault("errors", []).append(str(exc))
        next_status = "queued" if job.attempt_count < job.max_attempts else "failed"
        metadata["current_state"] = "failed" if next_status == "failed" else LAST_STABLE_STATE.get(job.stage, metadata.get("current_state", "failed"))
        metadata["last_blocking_stage"] = job.stage
        metadata["last_blocking_error"] = str(exc)
        outcomes = metadata.setdefault("stage_outcomes", {})
        outcomes[job.stage] = {
            "status": "failed_terminal" if next_status == "failed" else "failed_retryable",
            "detail": str(exc),
            "at": utc_now(),
        }
        write_json(call_dir / "metadata.json", metadata)
        _sync_call_row(config, call_dir)
        append_log(call_dir / "processing_log.json", {"event": f"{job.stage}_failed", "error": str(exc)})
        fail_job(config, job, str(exc), retryable=True)
        return True


def run_once(config: AppConfig, scan_first: bool = False) -> bool:
    if scan_first:
        detect_new_calls(config)
    job = claim_next_job(config)
    if not job:
        return False
    target_call_id = job.call_id
    processed = _run_claimed_job(config, job)
    while True:
        next_job = claim_next_job_for_call(config, target_call_id)
        if not next_job:
            break
        _run_claimed_job(config, next_job)
    return processed


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


def run_manual_step(config: AppConfig, scan_first: bool = False) -> tuple[int, str | None, int]:
    imported = detect_new_calls(config) if scan_first else []
    job = claim_next_job(config)
    if not job:
        return len(imported), None, 0

    target_call_id = job.call_id
    processed_stages = 0
    _run_claimed_job(config, job)
    processed_stages += 1

    while True:
        next_job = claim_next_job_for_call(config, target_call_id)
        if not next_job:
            break
        _run_claimed_job(config, next_job)
        processed_stages += 1

    return len(imported), target_call_id, processed_stages


class WorkerThread:
    def __init__(self, config: AppConfig):
        self.config = config
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        recovered = reset_running_jobs_on_startup(self.config)
        if recovered:
            logger.warning("Worker startup recovered_running_jobs=%s", recovered)
        if processing_mode(self.config) == "manual_step":
            logger.info("Worker startup_mode=manual_step automatic background processing disabled")
            return
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
CHUNK_MANIFEST_FILENAME = "audio_chunks_manifest.json"
