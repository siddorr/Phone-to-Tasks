from __future__ import annotations

import logging
import os
from pathlib import Path
import re

from call_assistant.common.config import AppConfig
from call_assistant.common.io import read_json
from call_assistant.common.models import RawSegment, RawTranscript

logger = logging.getLogger(__name__)
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def _transcription_preferences(audio_path: Path, config: AppConfig) -> tuple[str, str | None, str | None, str]:
    metadata = read_json(audio_path.parent / "metadata.json", default={})
    preference = metadata.get("transcription_preference", {})
    provider = preference.get("provider") or config.section("transcription")["provider_default"]
    model = preference.get("model")
    language_override = metadata.get("transcription_language_override")
    language_mode = metadata.get("transcription_language_mode") or config.section("transcription").get("language_mode", "auto")
    return provider, model, language_override, language_mode


def _transcription_language(audio_path: Path, config: AppConfig, language_mode: str, language_override: str | None) -> str | None:
    if language_override:
        return language_override
    hints = config.section("transcription").get("language_hints", [])
    if language_mode == "force_first_hint":
        return hints[0] if hints else None
    if language_mode == "metadata_override":
        return None
    if config.section("transcription").get("force_language_for_single_language_calls", False) and len(hints) == 1:
        return hints[0]
    return None


def _looks_mixed_language_problem(raw: RawTranscript, audio_path: Path) -> bool:
    text = raw.text or " ".join(segment.text for segment in raw.segments)
    hebrew_chars = len(HEBREW_RE.findall(text))
    total_chars = len(text.strip())
    if hebrew_chars == 0:
        return False
    if raw.language in {"he", "iw"}:
        return False
    metadata = read_json(audio_path.parent / "metadata.json", default={})
    duration = float(metadata.get("duration_seconds") or 0.0)
    low_segment_density = duration >= 30 and len(raw.segments) <= 2
    suspiciously_short_text = duration >= 20 and total_chars < max(20, int(duration * 0.8))
    return low_segment_density or suspiciously_short_text


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


def _transcribe_local(
    audio_path: Path,
    config: AppConfig,
    model_override: str | None = None,
    language_override: str | None = None,
    language_mode: str = "auto",
) -> RawTranscript:
    import whisper

    model_name = model_override or config.section("transcription")["local_model"]
    language = _transcription_language(audio_path, config, language_mode, language_override)
    logger.info(
        "Transcription local start audio=%s model=%s language_mode=%s language_override=%s language_hint=%s",
        audio_path,
        model_name,
        language_mode,
        language_override,
        language,
    )
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language, verbose=False)
    segments = _to_segments(result.get("segments", []))
    text = result.get("text", "").strip()
    if not segments and text:
        segments = [RawSegment(start_sec=0.0, end_sec=0.0, text=text, confidence=None, speaker=None)]
    transcript = RawTranscript(
        provider="local",
        model=model_name,
        language=result.get("language", language or "unknown"),
        confidence=None,
        segments=segments,
        text=text,
    )
    mixed_language_suspected = _looks_mixed_language_problem(transcript, audio_path)
    logger.info(
        "Transcription local success audio=%s segments=%s language=%s mixed_language_suspected=%s",
        audio_path,
        len(segments),
        transcript.language,
        mixed_language_suspected,
    )
    return transcript


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
    default_provider, model_override, language_override, language_mode = _transcription_preferences(audio_path, config)
    fallback_provider = config.section("transcription").get("provider_fallback")
    providers = [default_provider]
    if fallback_provider and fallback_provider != default_provider:
        providers.append(fallback_provider)
    for provider in providers:
        try:
            if provider == "local":
                transcript = _transcribe_local(
                    audio_path,
                    config,
                    model_override if default_provider == "local" else None,
                    language_override,
                    language_mode,
                )
                if (
                    provider == default_provider
                    and fallback_provider == "cloud"
                    and config.section("transcription").get("cloud_enabled", False)
                    and _looks_mixed_language_problem(transcript, audio_path)
                ):
                    logger.info("Transcription falling back to cloud audio=%s fallback_reason=mixed_language_suspected", audio_path)
                    continue
                return transcript
            if provider == "cloud" and config.section("transcription").get("cloud_enabled", False):
                logger.info("Transcription falling back to cloud audio=%s", audio_path)
                cloud_override = model_override if default_provider == "cloud" else None
                return _transcribe_cloud(audio_path, config, cloud_override)
        except Exception as exc:
            logger.exception("Transcription provider %s failed", provider)
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("Transcription failed; " + "; ".join(errors))
