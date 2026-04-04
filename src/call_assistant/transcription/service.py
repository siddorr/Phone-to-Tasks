from __future__ import annotations

import logging
import os
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.io import read_json
from call_assistant.common.models import RawSegment, RawTranscript

logger = logging.getLogger(__name__)


def _language_hint(config: AppConfig) -> str | None:
    hints = config.section("transcription").get("language_hints", [])
    return hints[0] if hints else None


def _transcription_preferences(audio_path: Path, config: AppConfig) -> tuple[str, str | None]:
    metadata = read_json(audio_path.parent / "metadata.json", default={})
    preference = metadata.get("transcription_preference", {})
    provider = preference.get("provider") or config.section("transcription")["provider_default"]
    model = preference.get("model")
    return provider, model


def _to_segments(raw_segments: list[dict]) -> list[RawSegment]:
    return [
        RawSegment(
            start_sec=float(item.get("start", 0.0)),
            end_sec=float(item.get("end", 0.0)),
            text=item.get("text", "").strip(),
            confidence=item.get("avg_logprob"),
            speaker=item.get("speaker"),
        )
        for item in raw_segments
        if item.get("text", "").strip()
    ]


def _transcribe_local(audio_path: Path, config: AppConfig, model_override: str | None = None) -> RawTranscript:
    import whisper

    model_name = model_override or config.section("transcription")["local_model"]
    language = _language_hint(config)
    logger.info("Transcription local start audio=%s model=%s language_hint=%s", audio_path, model_name, language)
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language, verbose=False)
    segments = _to_segments(result.get("segments", []))
    text = result.get("text", "").strip()
    if not segments and text:
        segments = [RawSegment(start_sec=0.0, end_sec=0.0, text=text, confidence=None, speaker=None)]
    logger.info("Transcription local success audio=%s segments=%s language=%s", audio_path, len(segments), result.get("language", language or "unknown"))
    return RawTranscript(
        provider="local",
        model=model_name,
        language=result.get("language", language or "unknown"),
        confidence=None,
        segments=segments,
        text=text,
    )


def _transcribe_cloud(audio_path: Path, config: AppConfig, model_override: str | None = None) -> RawTranscript:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for cloud transcription fallback")
    model_name = model_override or config.section("transcription")["cloud_model"]
    logger.info("Transcription cloud start audio=%s model=%s", audio_path, model_name)
    client = OpenAI(api_key=api_key)
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            response_format="verbose_json",
        )
    transcript_dict = transcript.model_dump() if hasattr(transcript, "model_dump") else dict(transcript)
    segments = _to_segments(transcript_dict.get("segments", []))
    text = transcript_dict.get("text", "").strip()
    if not segments and text:
        segments = [RawSegment(start_sec=0.0, end_sec=0.0, text=text, confidence=None, speaker=None)]
    logger.info("Transcription cloud success audio=%s segments=%s language=%s", audio_path, len(segments), transcript_dict.get("language", "unknown"))
    return RawTranscript(
        provider="cloud",
        model=model_name,
        language=transcript_dict.get("language", "unknown"),
        confidence=None,
        segments=segments,
        text=text,
    )


def transcribe(audio_path: Path, config: AppConfig) -> RawTranscript:
    errors: list[str] = []
    default_provider, model_override = _transcription_preferences(audio_path, config)
    fallback_provider = config.section("transcription").get("provider_fallback")
    providers = [default_provider]
    if fallback_provider and fallback_provider != default_provider:
        providers.append(fallback_provider)
    for provider in providers:
        try:
            if provider == "local":
                return _transcribe_local(audio_path, config, model_override if default_provider == "local" else None)
            if provider == "cloud" and config.section("transcription").get("cloud_enabled", False):
                logger.info("Transcription falling back to cloud audio=%s", audio_path)
                cloud_override = model_override if default_provider == "cloud" else None
                return _transcribe_cloud(audio_path, config, cloud_override)
        except Exception as exc:
            logger.exception("Transcription provider %s failed", provider)
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("Transcription failed; " + "; ".join(errors))
