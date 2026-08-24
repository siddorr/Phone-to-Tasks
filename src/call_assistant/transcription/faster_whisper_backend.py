from __future__ import annotations

import logging
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawSegment, RawTranscript

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: Path,
    config: AppConfig,
    model_override: str | None = None,
    language_hint: str | None = None,
) -> RawTranscript:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    model_name = model_override or config.section("transcription")["local_model"]
    device = config.section("transcription").get("local_device", "auto")
    compute_type = config.section("transcription").get("faster_whisper_compute_type", "auto")
    logger.info(
        "Transcription faster-whisper start audio=%s model=%s device=%s language_hint=%s",
        audio_path,
        model_name,
        device,
        language_hint,
    )
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), language=language_hint)
    normalized_segments: list[RawSegment] = []
    text_parts: list[str] = []
    detected_language = getattr(info, "language", None) or "unknown"
    probability = getattr(info, "language_probability", None)
    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        text_parts.append(text)
        normalized_segments.append(
            RawSegment(
                start_sec=float(getattr(segment, "start", 0.0)),
                end_sec=float(getattr(segment, "end", 0.0)),
                text=text,
                confidence=float(getattr(segment, "avg_logprob", 0.0)) if getattr(segment, "avg_logprob", None) is not None else None,
                speaker=None,
                language=detected_language,
                language_confidence=float(probability) if probability is not None else None,
                smoothed_language=detected_language,
                smoothed_language_reason="asr_language",
            )
        )
    return RawTranscript(
        provider="local",
        model=model_name,
        language=detected_language,
        confidence=float(probability) if probability is not None else None,
        segments=normalized_segments,
        text=" ".join(text_parts).strip(),
    )
