from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from unittest.mock import patch

from call_assistant.common.config import AppConfig
from call_assistant.common.io import read_json, write_json
from call_assistant.common.models import RawSegment, RawTranscript
from call_assistant.common.progress import clear_stage_progress, set_stage_progress
from call_assistant.transcription.faster_whisper_backend import transcribe_audio as transcribe_faster_whisper_audio
from call_assistant.transcription.vad import detect_speech_chunks

logger = logging.getLogger(__name__)
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
HEBREW_TRANSLIT_RE = re.compile(r"\b(шалом|шэм|шем|барух|баруха|йом|йомтов|сабаба)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[\u0400-\u04FF\u0590-\u05FFA-Za-z]+", re.UNICODE)
SUSPICIOUS_HEBREW_TRANSLIT_TOKENS = {
    "сиха",
    "хивра",
    "бэдик",
    "меткан",
    "микроним",
    "технун",
    "такцив",
    "такцивы",
    "мульбецоа",
    "тахан",
    "баллим",
    "мидхабер",
    "мидхаббер",
    "сиюр",
    "арацот",
    "рацот",
    "монам",
    "бедюк",
}
SUSPICIOUS_HEBREW_TRANSLIT_PHRASES = (
    "мид хаббер",
    "мид хабер",
    "мид хабб",
    "урод рабин",
)
SUSPICIOUS_HEBREW_PREFIXES = ("бе", "ми", "си", "та")
SUSPICIOUS_HEBREW_SUFFIXES = ("им", "от", "ах")
TRANSCRIPTION_CANDIDATES_FILENAME = "transcription_candidates.json"
TRANSCRIPTION_VAD_CHUNKS_FILENAME = "transcript_vad_chunks.json"
_MODEL_CACHE_LOCK = Lock()
_MODEL_CACHE: dict[str, object] = {}
ALLOWED_EXPLICIT_LANGUAGES = {"ru", "he", "en"}
CANONICAL_LANGUAGE_LABELS = ALLOWED_EXPLICIT_LANGUAGES | {"mixed", "unknown"}
LANGUAGE_ALIASES = {
    "iw": "he",
    "hebrew": "he",
    "russian": "ru",
    "english": "en",
}


@dataclass
class TranscriptQualityAssessment:
    score: float
    flags: list[str]
    suspect_language_confusion: bool
    suggested_retry_languages: list[str]
    suspicious_token_count: int
    suspicious_token_examples: list[str]
    suspicious_segment_count: int
    suspicious_segment_examples: list[str]
    low_confidence_reason: str | None = None
    heuristic_score: float | None = None
    llm_score: float | None = None
    llm_flags: list[str] = field(default_factory=list)
    llm_reason: str | None = None


@dataclass
class TranscriptCandidateComparison:
    baseline: RawTranscript
    forced_hebrew: RawTranscript | None
    merged: RawTranscript | None
    winner: str
    strategy: str
    quality_notes: list[str]
    selected: RawTranscript
    baseline_assessment: TranscriptQualityAssessment
    forced_hebrew_assessment: TranscriptQualityAssessment | None
    merged_assessment: TranscriptQualityAssessment | None
    low_confidence_reason: str | None = None


@dataclass
class ChunkedTranscriptPlausibility:
    score: float
    flags: list[str]
    should_fallback: bool
    reason: str | None = None


@dataclass
class SegmentDetectionDecision:
    segment_index: int
    label: str
    confidence: float
    reason: str
    suspicious: bool
    retry_language: str | None


@dataclass
class SegmentRetryCandidate:
    segment_index: int
    rank: int
    score: float
    start_sec: float
    end_sec: float
    duration_seconds: float
    baseline_text: str
    context_before: str
    context_after: str
    llm_label: str
    llm_reason: str
    llm_confidence: float


@dataclass
class SegmentRetryResult:
    candidate: SegmentRetryCandidate
    retry_text: str
    replacement_applied: bool
    replacement_reason: str


@dataclass
class SegmentDetectionSummary:
    provider: str
    model: str
    run_on_every_call: bool
    segments_evaluated: int
    suspicious_count: int
    selected_for_retry: int
    below_threshold_count: int
    threshold_used: float


@dataclass
class WordToken:
    text: str
    start_offset: int
    end_offset: int


@dataclass
class WordSpanCandidate:
    segment_index: int
    segment_start_sec: float
    segment_end_sec: float
    segment_text: str
    span_text: str
    start_token_index: int
    end_token_index: int
    start_char_offset: int
    end_char_offset: int
    context_before: str
    context_after: str
    source_segment_label: str
    source_segment_reason: str
    source_segment_confidence: float


@dataclass
class WordSpanDecision:
    segment_index: int
    span_text: str
    start_token_index: int
    end_token_index: int
    label: str
    confidence: float
    suspicious: bool
    retry_language: str | None
    reason: str


@dataclass
class WordRetryCandidate:
    segment_index: int
    rank: int
    score: float
    start_sec: float
    end_sec: float
    baseline_segment_text: str
    span_text: str
    start_token_index: int
    end_token_index: int
    context_before: str
    context_after: str
    llm_label: str
    llm_reason: str
    llm_confidence: float


@dataclass
class ClauseRetryCandidate:
    segment_index: int
    rank: int
    score: float
    start_sec: float
    end_sec: float
    baseline_segment_text: str
    clause_text: str
    start_token_index: int
    end_token_index: int
    context_before: str
    context_after: str
    source_reason: str


@dataclass
class WordSpanSemanticValidation:
    decision: str
    confidence: float
    reason: str
    normalized_hebrew: str | None


@dataclass
class PostBaselineRecoveryResult:
    selected: RawTranscript
    strategy: str
    comparison: TranscriptCandidateComparison
    baseline_assessment: TranscriptQualityAssessment
    segment_decisions: list[SegmentDetectionDecision]
    segment_detection_summary: dict[str, object]
    segment_detection_llm_response: dict[str, object] | None
    word_span_candidates: list[WordSpanCandidate]
    word_span_decisions: list[WordSpanDecision]
    word_span_detection_summary: dict[str, object]
    word_span_detection_llm_response: dict[str, object] | None
    word_retry_candidates: list[WordRetryCandidate]
    word_span_retry_results: list[dict[str, object]]
    clause_retry_candidates: list[ClauseRetryCandidate]
    clause_detection_summary: dict[str, object]
    clause_retry_results: list[dict[str, object]]
    word_span_validation_llm_response: list[dict[str, object]]
    full_call_hebrew_escalated: bool
    full_call_hebrew_selected: bool
    full_call_hebrew_decision: dict[str, object]
    retried_languages: list[str] = field(default_factory=list)


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


def _duration_seconds(audio_path: Path) -> float:
    metadata = read_json(audio_path.parent / "metadata.json", default={})
    return float(metadata.get("duration_seconds") or 0.0)


def _call_id(audio_path: Path) -> str:
    metadata = read_json(audio_path.parent / "metadata.json", default={})
    call_id = metadata.get("call_id")
    if isinstance(call_id, str) and call_id.strip():
        return call_id.strip()
    name = audio_path.parent.name
    return name[len("call_") :] if name.startswith("call_") else name


def _script_counts(text: str) -> dict[str, int]:
    return {
        "hebrew": len(HEBREW_RE.findall(text)),
        "cyrillic": len(CYRILLIC_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
    }


def _adjacent_repeat_count(lines: list[str]) -> int:
    repeats = 0
    previous = None
    for line in lines:
        if previous and line == previous:
            repeats += 1
        previous = line
    return repeats


def _tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def _segment_tokens(text: str) -> list[WordToken]:
    return [
        WordToken(text=match.group(0), start_offset=match.start(), end_offset=match.end())
        for match in WORD_RE.finditer(text)
    ]


def _count_cyrillic_hebrew_like_tokens(tokens: list[str]) -> tuple[int, list[str]]:
    suspicious: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not CYRILLIC_RE.search(token) or HEBREW_RE.search(token) or LATIN_RE.search(token):
            continue
        token_compact = token.replace(" ", "")
        if token_compact in SUSPICIOUS_HEBREW_TRANSLIT_TOKENS or HEBREW_TRANSLIT_RE.search(token):
            if token_compact not in seen:
                suspicious.append(token_compact)
                seen.add(token_compact)
            continue
        has_suspicious_prefix = token_compact.startswith(SUSPICIOUS_HEBREW_PREFIXES)
        has_suspicious_suffix = token_compact.endswith(SUSPICIOUS_HEBREW_SUFFIXES)
        looks_hebrewish = has_suspicious_prefix and has_suspicious_suffix
        if looks_hebrewish and len(token_compact) >= 5:
            if token_compact not in seen:
                suspicious.append(token_compact)
                seen.add(token_compact)
    return len(suspicious), suspicious


def _has_mixed_hebrew_domain_terms(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in SUSPICIOUS_HEBREW_TRANSLIT_PHRASES)


def _mixed_script_token_count(tokens: list[str]) -> int:
    count = 0
    for token in tokens:
        scripts = 0
        if HEBREW_RE.search(token):
            scripts += 1
        if CYRILLIC_RE.search(token):
            scripts += 1
        if LATIN_RE.search(token):
            scripts += 1
        if scripts >= 2:
            count += 1
    return count


def _script_transition_ratio(text: str) -> float:
    script_sequence: list[str] = []
    for char in text:
        if HEBREW_RE.match(char):
            script_sequence.append("hebrew")
        elif CYRILLIC_RE.match(char):
            script_sequence.append("cyrillic")
        elif LATIN_RE.match(char):
            script_sequence.append("latin")
    if len(script_sequence) < 2:
        return 0.0
    transitions = sum(1 for prev, curr in zip(script_sequence, script_sequence[1:]) if prev != curr)
    return transitions / len(script_sequence)


def _count_suspicious_segments(raw: RawTranscript) -> tuple[int, list[str]]:
    examples: list[str] = []
    for segment in getattr(raw, "segments", []) or []:
        text = getattr(segment, "text", None)
        if not isinstance(text, str):
            continue
        value = text.strip()
        if value and _segment_has_transliterated_hebrew(value):
            examples.append(value)
    return len(examples), examples[:5]


def _normalize_language_label(
    label: str | None,
    text: str = "",
    fallback_language: str | None = None,
) -> tuple[str | None, str]:
    raw = (label or "").strip().lower()
    fallback = LANGUAGE_ALIASES.get((fallback_language or "").strip().lower(), (fallback_language or "").strip().lower())
    canonical = LANGUAGE_ALIASES.get(raw, raw)
    if canonical in ALLOWED_EXPLICIT_LANGUAGES:
        return canonical, "allowed_label"
    if canonical in {"mixed", "unknown"}:
        return canonical, "allowed_special"
    value = (text or "").strip()
    if not value:
        if fallback in ALLOWED_EXPLICIT_LANGUAGES:
            return fallback, "fallback_allowed_language"
        return "unknown", "unsupported_label_unknown"
    scripts = _script_counts(value)
    tokens = _tokenize_words(value)
    suspicious_token_count, _ = _count_cyrillic_hebrew_like_tokens(tokens)
    if scripts["hebrew"] > 0 and scripts["cyrillic"] == 0:
        return "he", "unsupported_label_hebrew_script"
    if scripts["cyrillic"] > 0 and scripts["hebrew"] == 0:
        if suspicious_token_count >= 2 or _has_mixed_hebrew_domain_terms(value) or HEBREW_TRANSLIT_RE.search(value):
            return "mixed", "unsupported_label_mixed"
        return "ru", "unsupported_label_cyrillic_script"
    if scripts["latin"] > 0 and scripts["hebrew"] == 0 and scripts["cyrillic"] == 0:
        return "en", "unsupported_label_latin_script"
    if (scripts["hebrew"] > 0 and scripts["cyrillic"] > 0) or (
        sum(1 for count in scripts.values() if count > 0) >= 2
    ):
        return "mixed", "unsupported_label_mixed"
    return "unknown", "unsupported_label_unknown"


def _detect_segment_language(text: str, fallback_language: str | None = None) -> tuple[str | None, float | None, str]:
    value = (text or "").strip()
    if not value:
        normalized_fallback, _ = _normalize_language_label(fallback_language, "", None)
        return normalized_fallback, None, "empty"
    scripts = _script_counts(value)
    tokens = _tokenize_words(value)
    suspicious_token_count, _ = _count_cyrillic_hebrew_like_tokens(tokens)
    if scripts["hebrew"] > 0 and scripts["cyrillic"] == 0:
        return "he", min(1.0, 0.7 + (scripts["hebrew"] / max(len(value), 1))), "hebrew_script"
    if scripts["cyrillic"] > 0 and scripts["hebrew"] == 0:
        if suspicious_token_count >= 2 or _has_mixed_hebrew_domain_terms(value) or HEBREW_TRANSLIT_RE.search(value):
            return "mixed", 0.45, "cyrillic_hebrew_transliteration"
        return "ru", min(1.0, 0.65 + (scripts["cyrillic"] / max(len(value), 1))), "cyrillic_script"
    if scripts["hebrew"] > 0 and scripts["cyrillic"] > 0:
        return "mixed", 0.4, "mixed_script"
    if scripts["latin"] > 0 and scripts["hebrew"] == 0 and scripts["cyrillic"] == 0:
        return "en", 0.55, "latin_script"
    normalized_fallback, _ = _normalize_language_label(fallback_language, value, None)
    return normalized_fallback, 0.2 if normalized_fallback else None, "fallback"


def _segment_duration(segment: RawSegment) -> float:
    return max(0.0, float(getattr(segment, "end_sec", 0.0)) - float(getattr(segment, "start_sec", 0.0)))


def _segment_text_supports_language(text: str, label: str | None) -> bool:
    value = (text or "").strip()
    if not value or not label:
        return False
    if label == "unknown":
        return True
    scripts = _script_counts(value)
    if label in {"he", "iw"}:
        return scripts["hebrew"] > 0
    if label == "ru":
        return scripts["cyrillic"] > 0 and not _segment_has_transliterated_hebrew(value)
    if label == "en":
        return scripts["latin"] > 0 and scripts["hebrew"] == 0 and scripts["cyrillic"] == 0
    if label == "mixed":
        return True
    return False


def _language_retry_history(transcript: RawTranscript) -> list[str]:
    history = getattr(transcript, "_language_retries", []) or []
    return [language for language in history if language]


def _record_language_retry(transcript: RawTranscript, language: str | None) -> None:
    if not language:
        return
    history = _language_retry_history(transcript)
    if language in history:
        transcript._language_retries = history
        return
    transcript._language_retries = history + [language]


def _merge_language_histories(*transcripts: RawTranscript) -> list[str]:
    seen: list[str] = []
    for candidate in transcripts:
        for language in _language_retry_history(candidate):
            if language not in seen:
                seen.append(language)
    return seen


def _merge_retry_languages(*language_groups: list[str]) -> list[str]:
    seen: list[str] = []
    for group in language_groups:
        for language in group:
            if language and language not in seen:
                seen.append(language)
    return seen


def _language_script_weight(text: str | None, language: str) -> int:
    if not text:
        return 0
    counts = _script_counts(text)
    mapping = {"he": "hebrew", "ru": "cyrillic", "en": "latin"}
    key = mapping.get(language)
    if key:
        return counts.get(key, 0)
    return sum(counts.values())


def _transcript_log_preview(raw: RawTranscript | None, limit: int = 160) -> str:
    if raw is None:
        return "-"
    text = (raw.text or " ".join(segment.text for segment in getattr(raw, "segments", []) or [])).strip()
    if not text:
        return "-"
    compact = re.sub(r"\s+", " ", text)
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)]}..."


def _compare_retry_transcripts(
    primary: RawTranscript,
    candidate: RawTranscript,
    audio_path: Path,
    preferred_language: str,
    config: AppConfig | None = None,
) -> RawTranscript:
    primary_assessment = assess_transcript_quality(primary, audio_path, config=config)
    candidate_assessment = assess_transcript_quality(candidate, audio_path, config=config)
    primary_weight = _language_script_weight(primary.text, preferred_language)
    candidate_weight = _language_script_weight(candidate.text, preferred_language)
    winner = primary.language or "unknown"
    reason = "baseline_stronger"
    if candidate_assessment.score >= primary_assessment.score + 0.05:
        winner = candidate.language or preferred_language
        reason = "candidate_score_margin"
        logger.info(
            "Transcription language retry comparison audio=%s preferred_language=%s primary_language=%s primary_score=%.2f primary_weight=%s primary_preview=%s candidate_language=%s candidate_score=%.2f candidate_weight=%s candidate_preview=%s winner=%s reason=%s",
            audio_path,
            preferred_language,
            primary.language,
            primary_assessment.score,
            primary_weight,
            _transcript_log_preview(primary),
            candidate.language,
            candidate_assessment.score,
            candidate_weight,
            _transcript_log_preview(candidate),
            winner,
            reason,
        )
        return candidate
    if candidate_assessment.score >= primary_assessment.score:
        if candidate_weight >= primary_weight:
            winner = candidate.language or preferred_language
            reason = "candidate_script_weight_tiebreak"
            logger.info(
                "Transcription language retry comparison audio=%s preferred_language=%s primary_language=%s primary_score=%.2f primary_weight=%s primary_preview=%s candidate_language=%s candidate_score=%.2f candidate_weight=%s candidate_preview=%s winner=%s reason=%s",
                audio_path,
                preferred_language,
                primary.language,
                primary_assessment.score,
                primary_weight,
                _transcript_log_preview(primary),
                candidate.language,
                candidate_assessment.score,
                candidate_weight,
                _transcript_log_preview(candidate),
                winner,
                reason,
            )
            return candidate
        reason = "baseline_script_weight_tiebreak"
    logger.info(
        "Transcription language retry comparison audio=%s preferred_language=%s primary_language=%s primary_score=%.2f primary_weight=%s primary_preview=%s candidate_language=%s candidate_score=%.2f candidate_weight=%s candidate_preview=%s winner=%s reason=%s",
        audio_path,
        preferred_language,
        primary.language,
        primary_assessment.score,
        primary_weight,
        _transcript_log_preview(primary),
        candidate.language,
        candidate_assessment.score,
        candidate_weight,
        _transcript_log_preview(candidate),
        winner,
        reason,
    )
    return primary




def _run_language_retry_sequence(
    transcript: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    provider: str,
    model_override: str | None,
    language_mode: str,
) -> RawTranscript:
    order = list(config.section("transcription").get("language_retry_order", ["he", "ru"]))
    current = transcript
    _track_language_usage(current)
    for language in order:
        processed = set(_language_retry_history(current))
        if language in processed:
            continue
        if not _language_retry_needed(current, config):
            break
        candidate = _transcribe_force_language(audio_path, config, provider, model_override, language_mode, language)
        _record_language_retry(candidate, language)
        history = _merge_language_histories(current, candidate)
        candidate._language_retries = history
        selected = _compare_retry_transcripts(current, candidate, audio_path, language, config=config)
        selected._language_retries = history
        current = selected
    return current


def _maybe_run_language_retries(
    transcript: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    provider: str,
    model_override: str | None,
    language_mode: str,
) -> RawTranscript:
    if not config.section("transcription").get("language_retry_enabled", True):
        return transcript
    if not _has_language_retry_evidence(transcript):
        return transcript
    if not _language_retry_needed(transcript, config):
        return transcript
    return _run_language_retry_sequence(transcript, audio_path, config, provider, model_override, language_mode)


def _track_language_usage(transcript: RawTranscript) -> None:
    language = getattr(transcript, "language", None)
    _record_language_retry(transcript, language)


def _uncertain_segment_ratio(transcript: RawTranscript) -> float:
    segments = getattr(transcript, "segments", []) or []
    if not segments:
        return 0.0
    uncertain = sum(1 for segment in segments if _segment_is_uncertain(segment))
    return uncertain / len(segments)


def _language_retry_needed(transcript: RawTranscript, config: AppConfig) -> bool:
    section = config.section("transcription")
    threshold = float(section.get("language_retry_confidence_threshold", 0.6))
    uncertain_ratio_threshold = float(section.get("language_retry_uncertain_segment_ratio", 0.5))
    confidence = getattr(transcript, "primary_language_confidence", None)
    distribution = getattr(transcript, "language_distribution", None)
    primary_language = getattr(transcript, "language", None)
    if (
        confidence is not None
        and confidence >= threshold
        and isinstance(distribution, dict)
        and primary_language in ALLOWED_EXPLICIT_LANGUAGES
        and float(distribution.get(primary_language, 0.0) or 0.0) >= 0.85
        and float(distribution.get("mixed", 0.0) or 0.0) < 0.15
    ):
        return False
    if confidence is None or confidence < threshold:
        return True
    return _uncertain_segment_ratio(transcript) >= uncertain_ratio_threshold


def _has_language_retry_evidence(transcript: RawTranscript) -> bool:
    confidence = getattr(transcript, "primary_language_confidence", None)
    if confidence is not None:
        return True
    distribution = getattr(transcript, "language_distribution", None)
    return isinstance(distribution, dict) and bool(distribution)


def _should_force_hebrew_transcript(transcript: RawTranscript) -> bool:
    if not transcript.text:
        return False
    if "he" in _language_retry_history(transcript):
        return False
    if transcript.language == "he":
        return False
    scripts = _script_counts(transcript.text)
    hebrew_chars = scripts.get("hebrew", 0)
    total = sum(scripts.values())
    if not hebrew_chars or total == 0:
        return False
    ratio = hebrew_chars / max(total, 1)
    confidence = getattr(transcript, "primary_language_confidence", None)
    if ratio >= 0.4 and (confidence is None or confidence < 0.65):
        return True
    return False


def _compare_transcripts_for_hebrew(
    primary: RawTranscript,
    candidate: RawTranscript,
    audio_path: Path,
    config: AppConfig | None = None,
) -> RawTranscript:
    primary_assessment = assess_transcript_quality(primary, audio_path, config=config)
    candidate_assessment = assess_transcript_quality(candidate, audio_path, config=config)
    if candidate_assessment.score >= primary_assessment.score + 0.05:
        return candidate
    if candidate_assessment.score >= primary_assessment.score:
        primary_hebrew = _script_counts(primary.text).get("hebrew", 0)
        candidate_hebrew = _script_counts(candidate.text).get("hebrew", 0)
        if candidate_hebrew >= primary_hebrew:
            return candidate
    return primary


def _transcribe_force_language(
    audio_path: Path,
    config: AppConfig,
    provider: str,
    model_override: str | None,
    language_mode: str,
    forced_language: str,
) -> RawTranscript:
    target_mode = "metadata_override"
    if provider == "local":
        return _transcribe_local(
            audio_path,
            config,
            model_override,
            forced_language,
            target_mode,
        )
    return _transcribe_cloud(
        audio_path,
        config,
        model_override,
        forced_language,
        target_mode,
    )


def _transcribe_force_hebrew(
    audio_path: Path,
    config: AppConfig,
    provider: str,
    model_override: str | None,
    language_mode: str,
) -> RawTranscript:
    return _transcribe_force_language(audio_path, config, provider, model_override, language_mode, "he")


def _maybe_force_hebrew_transcript(
    transcript: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    provider: str,
    model_override: str | None,
    language_mode: str,
) -> RawTranscript:
    if getattr(transcript, "_hebrew_forced", False):
        return transcript
    transcript._hebrew_forced = True
    if not _should_force_hebrew_transcript(transcript):
        return transcript
    forced = _transcribe_force_hebrew(audio_path, config, provider, model_override, language_mode)
    forced._hebrew_forced = True
    baseline_retries = list(getattr(transcript, "_language_retries", []))
    forced_retries = list(dict.fromkeys(baseline_retries + ["he"]))
    forced._language_retries = forced_retries
    selected = _compare_transcripts_for_hebrew(transcript, forced, audio_path, config=config)
    selected._language_retries = forced_retries
    return selected


def _segment_is_uncertain(segment: RawSegment) -> bool:
    label = getattr(segment, "smoothed_language", None) or getattr(segment, "language", None)
    confidence = getattr(segment, "language_confidence", None)
    if label in {None, "", "mixed", "unknown"}:
        return True
    if confidence is not None and confidence < 0.6:
        return True
    text = str(getattr(segment, "text", "") or "").strip()
    if len(_tokenize_words(text)) <= 2:
        return True
    if _segment_duration(segment) <= 1.2:
        return True
    reason = str(getattr(segment, "smoothed_language_reason", "") or "")
    if reason.startswith("unsupported_label_"):
        return True
    if not _segment_text_supports_language(text, label):
        return True
    if label == "ru" and _segment_has_transliterated_hebrew(text):
        return True
    return False


def label_segment_languages(raw: RawTranscript, config: AppConfig) -> RawTranscript:
    section = config.section("transcription")
    tiny_chunk_seconds = float(section.get("chunked_tiny_chunk_seconds", 1.2))
    raw.language, _ = _normalize_language_label(raw.language, raw.text, None)
    fallback_language = raw.language if raw.language in ALLOWED_EXPLICIT_LANGUAGES else None
    for segment in raw.segments:
        existing_language, existing_reason = _normalize_language_label(segment.language, segment.text, fallback_language)
        detected_language, detected_confidence, detected_reason = _detect_segment_language(segment.text, fallback_language)
        segment.language = existing_language if existing_language not in {None, ""} else detected_language
        reason = existing_reason if existing_reason.startswith("unsupported_label_") else detected_reason
        if segment.language_confidence is None:
            segment.language_confidence = detected_confidence
        duration = _segment_duration(segment)
        token_count = len(_tokenize_words(segment.text))
        if duration <= tiny_chunk_seconds or token_count <= 2:
            if segment.language_confidence is None or segment.language_confidence > 0.45:
                segment.language_confidence = 0.35
            if not _segment_text_supports_language(segment.text, segment.language):
                segment.language = "unknown"
                reason = "tiny_chunk_low_evidence"
        current_smoothed = segment.smoothed_language
        if current_smoothed:
            current_smoothed, smoothed_reason = _normalize_language_label(current_smoothed, segment.text, segment.language)
            segment.smoothed_language = current_smoothed
            if smoothed_reason.startswith("unsupported_label_"):
                reason = smoothed_reason
        if segment.smoothed_language is None:
            segment.smoothed_language = segment.language
        if segment.smoothed_language_reason is None or segment.smoothed_language_reason == "chunk_language":
            segment.smoothed_language_reason = reason
    raw.language_distribution = _language_distribution(raw.segments)
    raw.language, raw.primary_language_confidence = _primary_language_from_distribution(raw.language_distribution, raw.language)
    return raw


def smooth_segment_languages(raw: RawTranscript, config: AppConfig) -> RawTranscript:
    section = config.section("transcription")
    if not section.get("language_smoothing_enabled", True) or len(raw.segments) < 3:
        return raw
    window = max(1, int(section.get("language_smoothing_window", 1)))
    min_confidence = float(section.get("language_smoothing_min_confidence", 0.6))
    short_segment_sec = float(section.get("language_smoothing_short_segment_sec", 2.5))
    for index, segment in enumerate(raw.segments):
        duration = max(0.0, segment.end_sec - segment.start_sec)
        current_label = segment.smoothed_language or segment.language
        if current_label is None:
            continue
        if duration > short_segment_sec and current_label != "mixed":
            continue
        if segment.language_confidence is not None and segment.language_confidence >= min_confidence and current_label != "mixed":
            continue
        neighbor_labels: list[str] = []
        for offset in range(1, window + 1):
            for neighbor_index in (index - offset, index + offset):
                if 0 <= neighbor_index < len(raw.segments):
                    neighbor = raw.segments[neighbor_index]
                    neighbor_label = neighbor.smoothed_language or neighbor.language
                    if neighbor_label and neighbor_label != "mixed":
                        neighbor_labels.append(neighbor_label)
        if len(neighbor_labels) >= 2 and len(set(neighbor_labels)) == 1:
            segment.smoothed_language = neighbor_labels[0]
            segment.smoothed_language_reason = "neighbor_smoothing"
            if segment.language_confidence is None or segment.language_confidence < min_confidence:
                segment.language_confidence = min_confidence
        elif _segment_duration(segment) <= short_segment_sec and len(neighbor_labels) >= 1 and current_label != "mixed":
            stable_neighbor = neighbor_labels[0]
            if all(label == stable_neighbor for label in neighbor_labels):
                segment.smoothed_language = stable_neighbor
                segment.smoothed_language_reason = "short_outlier_smoothing"
                if segment.language_confidence is None or segment.language_confidence < min_confidence:
                    segment.language_confidence = min_confidence
    raw.language_distribution = _language_distribution(raw.segments)
    raw.language, raw.primary_language_confidence = _primary_language_from_distribution(raw.language_distribution, raw.language)
    return raw


def assess_chunked_transcript_plausibility(
    raw: RawTranscript,
    chunks: list[object],
    config: AppConfig,
) -> ChunkedTranscriptPlausibility:
    section = config.section("transcription")
    if not section.get("chunked_plausibility_enabled", True):
        return ChunkedTranscriptPlausibility(score=1.0, flags=[], should_fallback=False, reason=None)

    short_call_seconds = float(section.get("chunked_short_call_seconds", 15))
    max_distinct_languages = int(section.get("chunked_max_distinct_languages_short_call", 2))
    tiny_chunk_seconds = float(section.get("chunked_tiny_chunk_seconds", 1.2))
    max_switch_ratio = float(section.get("chunked_max_language_switch_ratio", 0.5))
    min_score = float(section.get("chunked_plausibility_min_score", 0.45))

    score = 1.0
    flags: list[str] = []
    durations = [_segment_duration(segment) for segment in raw.segments]
    total_duration = sum(durations) or sum(float(getattr(chunk, "duration_seconds", 0.0) or 0.0) for chunk in chunks)
    labels = [segment.smoothed_language or segment.language for segment in raw.segments if (segment.smoothed_language or segment.language)]
    distinct_labels = {label for label in labels if label}
    unsupported_collapsed = sum(
        1
        for segment in raw.segments
        if str(getattr(segment, "smoothed_language_reason", "") or "").startswith("unsupported_label_")
    )
    unknown_count = sum(1 for label in labels if label == "unknown")
    if total_duration and total_duration <= short_call_seconds and len(distinct_labels) > max_distinct_languages:
        flags.append("short_call_many_languages")
        score -= 0.35

    switches = 0
    comparable = 0
    for left, right in zip(labels, labels[1:]):
        if not left or not right:
            continue
        comparable += 1
        if left != right:
            switches += 1
    switch_ratio = (switches / comparable) if comparable else 0.0
    if total_duration and total_duration <= short_call_seconds and switch_ratio > max_switch_ratio:
        flags.append("rapid_language_switching")
        score -= 0.25

    tiny_chunks = sum(1 for duration in durations if duration and duration <= tiny_chunk_seconds)
    if raw.segments and tiny_chunks >= max(2, len(raw.segments) // 2):
        flags.append("tiny_chunk_overinterpretation")
        score -= 0.20

    lowered = raw.text.lower()
    if "субтитры сделал" in lowered or lowered.count("продолжение следует") >= 2:
        flags.append("repetitive_hallucination_pattern")
        score -= 0.55

    section = config.section("transcription")
    if total_duration and total_duration <= short_call_seconds:
        script_names = {
            label
            for label in distinct_labels
            if label in {"en", "ru", "he", "mixed", "unknown"}
        }
        if len(script_names) >= 3:
            flags.append("script_language_fragmentation")
            score -= 0.20
    repetition_penalty = float(section.get("chunked_repetition_penalty", 0.2))
    repetition_threshold = float(section.get("chunked_repetition_threshold", 0.5))
    texts = [segment.text.strip().lower() for segment in raw.segments if segment.text]
    if repetition_penalty > 0 and texts:
        counts: dict[str, int] = {}
        for text in texts:
            counts[text] = counts.get(text, 0) + 1
        most_common = max(counts.values(), default=0)
        if len(texts) and most_common / len(texts) >= repetition_threshold:
            flags.append("repetitive_chunk_pattern")
            score -= repetition_penalty
    coverage_tokens = set(section.get("chunked_coverage_tokens", []))
    coverage_penalty = float(section.get("chunked_coverage_penalty", 0.25))
    normalized_text = (raw.text or "").lower()
    if coverage_tokens and coverage_penalty > 0 and not any(token.lower() in normalized_text for token in coverage_tokens):
        flags.append("coverage_missing")
        score -= coverage_penalty
    unique_tokens = {text for text in texts if text}
    diversity_min = int(section.get("chunked_diversity_min_unique_tokens", 0))
    diversity_penalty = float(section.get("chunked_diversity_penalty", 0.0))
    if diversity_min and diversity_penalty > 0 and len(unique_tokens) < diversity_min:
        flags.append("low_language_diversity")
        score -= diversity_penalty
    greeting_tokens = set(token.lower() for token in section.get("chunked_coverage_tokens", []))
    greeting_bonus = float(section.get("chunked_greeting_bonus", 0.0))
    if greeting_tokens and greeting_bonus > 0 and any(token in normalized_text for token in greeting_tokens):
        flags.append("greeting_bonus")
        score += greeting_bonus

    if unsupported_collapsed:
        flags.append("unsupported_language_collapsed")
        score -= 0.20 if unsupported_collapsed == 1 else 0.35
    if unknown_count:
        flags.append("unknown_language_chunks")
        score -= min(0.20, unknown_count * 0.08)

    if sum(1 for segment in raw.segments if _segment_is_uncertain(segment)) >= max(1, len(raw.segments) // 2):
        flags.append("high_uncertain_segment_ratio")
        score -= 0.15

    score = max(0.0, min(1.0, score))
    strong_flags = {"repetitive_hallucination_pattern", "short_call_many_languages", "rapid_language_switching"}
    should_fallback = score < min_score or len(strong_flags.intersection(flags)) >= 2
    reason = flags[0] if should_fallback and flags else None
    return ChunkedTranscriptPlausibility(score=score, flags=flags, should_fallback=should_fallback, reason=reason)


def _with_transcript_backend_metadata(
    raw: RawTranscript,
    *,
    backend_attempted: str | None,
    backend_selected: str | None = None,
    fallback_reason: str | None = None,
    plausibility: ChunkedTranscriptPlausibility | None = None,
) -> RawTranscript:
    raw.backend_attempted = backend_attempted
    raw.backend_selected = backend_selected or backend_attempted
    raw.backend_fallback_reason = fallback_reason
    raw.chunked_plausibility_score = plausibility.score if plausibility else None
    raw.chunked_plausibility_flags = list(plausibility.flags) if plausibility else None
    return raw


def _segment_context_text(segments: list[RawSegment], index: int, direction: int, window: int) -> str:
    parts: list[str] = []
    for offset in range(1, window + 1):
        neighbor_index = index + (offset * direction)
        if 0 <= neighbor_index < len(segments):
            text = (segments[neighbor_index].text or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _word_span_score(tokens: list[WordToken], start_index: int, end_index: int) -> float:
    span_tokens = [token.text.lower() for token in tokens[start_index : end_index + 1]]
    suspicious_count, _ = _count_cyrillic_hebrew_like_tokens(span_tokens)
    score = suspicious_count * 2.0
    span_text = " ".join(span_tokens)
    if HEBREW_TRANSLIT_RE.search(span_text):
        score += 2.0
    if _has_mixed_hebrew_domain_terms(span_text):
        score += 2.5
    if suspicious_count and start_index == end_index:
        score += 1.5
    if suspicious_count == 0:
        nearby = [token.text.lower() for token in tokens[max(0, start_index - 1) : min(len(tokens), end_index + 2)]]
        nearby_count, _ = _count_cyrillic_hebrew_like_tokens(nearby)
        if nearby_count:
            score += 1.0
    if 1 <= len(span_tokens) <= 3:
        score += 0.5
    return score


def _clause_retry_score(tokens: list[WordToken], start_index: int, end_index: int) -> float:
    span_tokens = [token.text.lower() for token in tokens[start_index : end_index + 1]]
    score = _word_span_score(tokens, start_index, end_index)
    suspicious_count, suspicious_examples = _count_cyrillic_hebrew_like_tokens(span_tokens)
    span_terms = set(span_tokens)
    compact_terms = {token.replace(" ", "") for token in span_tokens}
    if suspicious_count >= 2:
        score += 1.25
    if len(span_tokens) >= 4:
        score += 0.3
    if len(span_tokens) >= 6:
        score += 0.15
    combo_terms = {
        "такцив": 1.2,
        "такцивы": 1.0,
        "атомали": 1.0,
        "капсулы": 0.6,
        "мульбецоа": 0.8,
        "мульбицо": 0.8,
        "технун": 0.8,
    }
    for term, bonus in combo_terms.items():
        if term in compact_terms or term in span_terms:
            score += bonus
    if ("атомали" in compact_terms or "атомали" in span_terms) and (
        "такцив" in compact_terms or "такцивы" in compact_terms or "такцив" in span_terms or "такцивы" in span_terms
    ):
        score += 2.25
    if (
        ("такцив" in compact_terms or "такцивы" in compact_terms or "такцив" in span_terms or "такцивы" in span_terms)
        and ("капсулы" in compact_terms or "капсулы" in span_terms)
    ):
        score += 1.2
    if (
        ("такцив" in compact_terms or "такцив" in span_terms)
        and ("мульбецоа" in compact_terms or "мульбицо" in compact_terms or "мульбецоа" in span_terms or "мульбицо" in span_terms)
    ):
        score += 1.75
    if (
        ("технун" in compact_terms or "технун" in span_terms)
        and ("мульбецоа" in compact_terms or "мульбицо" in compact_terms or "мульбецоа" in span_terms or "мульбицо" in span_terms)
    ):
        score += 1.5
    if suspicious_examples:
        score += min(0.8, 0.2 * len(suspicious_examples))
    return score


def _build_word_span_candidates(
    raw: RawTranscript, segment_decisions: list[SegmentDetectionDecision], config: AppConfig
) -> list[WordSpanCandidate]:
    section = config.section("segment_detection")
    max_candidate_spans = int(section.get("max_candidate_spans_per_segment", 12))
    min_span_words = int(section.get("min_span_words", 1))
    max_span_words = int(section.get("max_span_words", 3))
    candidates: list[WordSpanCandidate] = []
    seen: set[tuple[int, int, int]] = set()
    by_index = {decision.segment_index: decision for decision in segment_decisions}
    for segment_index, segment in enumerate(raw.segments):
        decision = by_index.get(segment_index)
        if decision is None or not decision.suspicious:
            continue
        if not _segment_is_uncertain(segment):
            continue
        segment_text = (segment.text or "").strip()
        if not segment_text:
            continue
        tokens = _segment_tokens(segment_text)
        if not tokens:
            continue
        scored_windows: list[tuple[float, int, int]] = []
        anchor_indexes: list[int] = []
        for token_index, token in enumerate(tokens):
            token_score = _word_span_score(tokens, token_index, token_index)
            if token_score > 0:
                anchor_indexes.append(token_index)
                scored_windows.append((token_score + 3.0, token_index, token_index))
        for anchor_index in anchor_indexes:
            for start_index, end_index, bonus in (
                (max(0, anchor_index - 2), max(0, anchor_index - 1), 4.0),
                (max(0, anchor_index - 1), anchor_index, 2.0),
                (anchor_index, min(len(tokens) - 1, anchor_index + 1), 2.0),
                (max(0, anchor_index - 2), anchor_index, 2.0),
            ):
                if start_index > end_index or end_index >= len(tokens):
                    continue
                score = _word_span_score(tokens, start_index, end_index)
                if score > 0:
                    scored_windows.append((score + bonus, start_index, end_index))
        for start_index in range(len(tokens)):
            for span_size in range(min_span_words, max_span_words + 1):
                end_index = start_index + span_size - 1
                if end_index >= len(tokens):
                    break
                score = _word_span_score(tokens, start_index, end_index)
                if score <= 0:
                    continue
                scored_windows.append((score, start_index, end_index))
        scored_windows.sort(key=lambda item: (-item[0], -(item[2] - item[1] + 1), item[1]))
        for score, start_index, end_index in scored_windows[:max_candidate_spans]:
            key = (segment_index, start_index, end_index)
            if key in seen:
                continue
            seen.add(key)
            start_token = tokens[start_index]
            end_token = tokens[end_index]
            span_text = segment_text[start_token.start_offset : end_token.end_offset]
            candidates.append(
                WordSpanCandidate(
                    segment_index=segment_index,
                    segment_start_sec=segment.start_sec,
                    segment_end_sec=segment.end_sec,
                    segment_text=segment_text,
                    span_text=span_text,
                    start_token_index=start_index,
                    end_token_index=end_index,
                    start_char_offset=start_token.start_offset,
                    end_char_offset=end_token.end_offset,
                    context_before=_segment_context_text(raw.segments, segment_index, -1, int(section.get("context_window_segments", 1))),
                    context_after=_segment_context_text(raw.segments, segment_index, 1, int(section.get("context_window_segments", 1))),
                    source_segment_label=decision.label,
                    source_segment_reason=decision.reason,
                    source_segment_confidence=decision.confidence,
                )
            )
    return candidates


def _build_clause_retry_candidates(
    raw: RawTranscript, segment_decisions: list[SegmentDetectionDecision], config: AppConfig
) -> list[ClauseRetryCandidate]:
    section = config.section("segment_detection")
    if not section.get("clause_retry_enabled", True):
        return []
    min_span_words = int(section.get("clause_retry_min_span_words", 4))
    max_span_words = int(section.get("clause_retry_max_span_words", 8))
    max_per_call = int(section.get("clause_retry_max_per_call", 6))
    context_window = int(section.get("context_window_segments", 1))
    by_index = {decision.segment_index: decision for decision in segment_decisions}
    candidates: list[ClauseRetryCandidate] = []
    for segment_index, segment in enumerate(raw.segments):
        decision = by_index.get(segment_index)
        if decision is None or not decision.suspicious:
            continue
        if not _segment_is_uncertain(segment):
            continue
        segment_text = (segment.text or "").strip()
        if not segment_text:
            continue
        tokens = _segment_tokens(segment_text)
        if len(tokens) < min_span_words:
            continue
        best_window: tuple[float, int, int] | None = None
        for start_index in range(len(tokens)):
            for span_size in range(min_span_words, max_span_words + 1):
                end_index = start_index + span_size - 1
                if end_index >= len(tokens):
                    break
                score = _clause_retry_score(tokens, start_index, end_index)
                if score <= 0:
                    continue
                window = (score, start_index, end_index)
                if best_window is None or window[0] > best_window[0]:
                    best_window = window
        if best_window is None:
            continue
        score, start_index, end_index = best_window
        start_token = tokens[start_index]
        end_token = tokens[end_index]
        clause_text = segment_text[start_token.start_offset : end_token.end_offset]
        candidates.append(
            ClauseRetryCandidate(
                segment_index=segment_index,
                rank=0,
                score=score,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                baseline_segment_text=segment_text,
                clause_text=clause_text,
                start_token_index=start_index,
                end_token_index=end_index,
                context_before=_segment_context_text(raw.segments, segment_index, -1, context_window),
                context_after=_segment_context_text(raw.segments, segment_index, 1, context_window),
                source_reason=decision.reason,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.score,
            -(item.end_token_index - item.start_token_index + 1),
            item.segment_index,
            item.start_token_index,
        )
    )
    for rank, candidate in enumerate(candidates[:max_per_call], start=1):
        candidate.rank = rank
    return candidates[:max_per_call]


def _heuristic_segment_detection(raw: RawTranscript, config: AppConfig) -> list[SegmentDetectionDecision]:
    decisions: list[SegmentDetectionDecision] = []
    context_window = int(config.section("segment_detection").get("context_window_segments", 1))
    for index, segment in enumerate(raw.segments[: int(config.section("segment_detection").get("max_segments_per_call", 200))]):
        text = (segment.text or "").strip()
        previous_text = _segment_context_text(raw.segments, index, -1, context_window)
        next_text = _segment_context_text(raw.segments, index, 1, context_window)
        combined = " ".join(part for part in (previous_text, text, next_text) if part).strip()
        segment_language = segment.smoothed_language or segment.language
        uncertain = _segment_is_uncertain(segment)
        suspicious = uncertain and _segment_has_transliterated_hebrew(combined)
        confidence = 0.85 if suspicious else (0.45 if uncertain else 0.2)
        label = "hebrew_transliteration" if suspicious else ("mixed_uncertain" if uncertain else f"{segment_language or 'russian'}_normal")
        reason = (
            "Heuristic fallback marked segment suspicious because it contains likely Hebrew transliteration patterns."
            if suspicious
            else "Heuristic fallback marked segment as uncertain language context."
            if uncertain
            else "Heuristic fallback marked segment as stable non-suspicious context."
        )
        decisions.append(
            SegmentDetectionDecision(
                segment_index=index,
                label=label,
                confidence=confidence,
                reason=reason,
                suspicious=suspicious,
                retry_language="he" if suspicious else None,
            )
        )
    return decisions


def _classify_segments_with_llm(
    raw: RawTranscript, audio_path: Path, config: AppConfig
) -> tuple[list[SegmentDetectionDecision], dict[str, object]]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for segment detection")
    section = config.section("segment_detection")
    model = section["model"]
    context_window = int(section.get("context_window_segments", 1))
    max_segments = int(section.get("max_segments_per_call", 200))
    payload_segments: list[dict[str, object]] = []
    for index, segment in enumerate(raw.segments[:max_segments]):
        payload_segments.append(
            {
                "segment_index": index,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "text": segment.text,
                "segment_language": segment.smoothed_language or segment.language,
                "language_confidence": segment.language_confidence,
                "uncertain_language": _segment_is_uncertain(segment),
                "previous_text": _segment_context_text(raw.segments, index, -1, context_window),
                "next_text": _segment_context_text(raw.segments, index, 1, context_window),
            }
        )
    client = OpenAI(api_key=api_key)
    prompt = """
You are classifying baseline ASR transcript segments from a mixed Russian/Hebrew phone call.
Some Russian-looking segments may contain Hebrew terms distorted into Russian phonetics.
Do not translate or rewrite transcript text.
For each segment, decide whether it is suspicious and should be retried from audio with forced Hebrew.
Only mark a segment suspicious when the segment itself looks uncertain or mixed, not when it looks like stable clean Russian.
Use these labels only:
- russian_normal
- hebrew_transliteration
- mixed_uncertain
- garbled_noise
Return JSON only with:
{
  "segments": [
    {
      "segment_index": 0,
      "label": "russian_normal",
      "suspicious": false,
      "retry_language": null,
      "confidence": 0.0,
      "reason": "..."
    }
  ]
}
Be conservative for normal Russian.
Mark suspicious when Hebrew recovery from audio is likely worthwhile.
""".strip()
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"segments": payload_segments}, ensure_ascii=False)},
        ],
        timeout=section.get("timeout_sec", 45),
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    segments_payload = parsed.get("segments", [])
    decisions: list[SegmentDetectionDecision] = []
    for item in segments_payload:
        try:
            decisions.append(
                SegmentDetectionDecision(
                    segment_index=int(item["segment_index"]),
                    label=str(item.get("label", "mixed_uncertain")),
                    confidence=float(item.get("confidence", 0.0)),
                    reason=str(item.get("reason", "")),
                    suspicious=bool(item.get("suspicious", False)),
                    retry_language=item.get("retry_language"),
                )
            )
        except Exception:
            continue
    by_index = {item.segment_index: item for item in decisions}
    normalized: list[SegmentDetectionDecision] = []
    for index in range(min(len(raw.segments), max_segments)):
        decision = by_index.get(index)
        if decision is None:
            normalized.append(
                SegmentDetectionDecision(
                    segment_index=index,
                    label="mixed_uncertain",
                    confidence=0.0,
                    reason="LLM response omitted this segment.",
                    suspicious=False,
                    retry_language=None,
                )
            )
        else:
            normalized.append(decision)
    return normalized, parsed


def _heuristic_word_span_decisions(candidates: list[WordSpanCandidate]) -> list[WordSpanDecision]:
    decisions: list[WordSpanDecision] = []
    for candidate in candidates:
        suspicious = _segment_has_transliterated_hebrew(candidate.span_text)
        decisions.append(
            WordSpanDecision(
                segment_index=candidate.segment_index,
                span_text=candidate.span_text,
                start_token_index=candidate.start_token_index,
                end_token_index=candidate.end_token_index,
                label="hebrew_transliteration" if suspicious else "russian_normal",
                confidence=0.85 if suspicious else 0.2,
                suspicious=suspicious,
                retry_language="he" if suspicious else None,
                reason=(
                    "Heuristic word-span detection marked this span suspicious."
                    if suspicious
                    else "Heuristic word-span detection marked this span normal."
                ),
            )
        )
    return decisions


def _classify_word_spans_with_llm(
    candidates: list[WordSpanCandidate], config: AppConfig
) -> tuple[list[WordSpanDecision], dict[str, object]]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for word span detection")
    section = config.section("segment_detection")
    client = OpenAI(api_key=api_key)
    payload = {
        "spans": [
            {
                "segment_index": candidate.segment_index,
                "span_text": candidate.span_text,
                "start_token_index": candidate.start_token_index,
                "end_token_index": candidate.end_token_index,
                "segment_text": candidate.segment_text,
                "previous_text": candidate.context_before,
                "next_text": candidate.context_after,
            }
            for candidate in candidates
        ]
    }
    prompt = """
You are classifying short 1-3 word spans from a Russian/Hebrew mixed ASR transcript.
Some spans are Hebrew words rendered in Russian phonetics and should be retried from audio with forced Hebrew.
Do not translate the full segment and do not rewrite the transcript.
Classify each candidate span using only these labels:
- russian_normal
- hebrew_transliteration
- mixed_uncertain
- garbled_noise
Return JSON only:
{
  "spans": [
    {
      "segment_index": 0,
      "span_text": "такси в",
      "start_token_index": 3,
      "end_token_index": 4,
      "label": "hebrew_transliteration",
      "suspicious": true,
      "retry_language": "he",
      "confidence": 0.0,
      "reason": "..."
    }
  ]
}
Be conservative for normal Russian, but mark suspicious when Hebrew recovery from audio is likely worthwhile.
""".strip()
    response = client.chat.completions.create(
        model=section["model"],
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        timeout=section.get("timeout_sec", 45),
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    raw_spans = parsed.get("spans", [])
    by_key: dict[tuple[int, int, int], WordSpanDecision] = {}
    for item in raw_spans:
        try:
            key = (int(item["segment_index"]), int(item["start_token_index"]), int(item["end_token_index"]))
            by_key[key] = WordSpanDecision(
                segment_index=key[0],
                span_text=str(item.get("span_text", "")),
                start_token_index=key[1],
                end_token_index=key[2],
                label=str(item.get("label", "mixed_uncertain")),
                confidence=float(item.get("confidence", 0.0)),
                suspicious=bool(item.get("suspicious", False)),
                retry_language=item.get("retry_language"),
                reason=str(item.get("reason", "")),
            )
        except Exception:
            continue
    normalized: list[WordSpanDecision] = []
    for candidate in candidates:
        key = (candidate.segment_index, candidate.start_token_index, candidate.end_token_index)
        decision = by_key.get(key)
        if decision is None:
            normalized.append(
                WordSpanDecision(
                    segment_index=candidate.segment_index,
                    span_text=candidate.span_text,
                    start_token_index=candidate.start_token_index,
                    end_token_index=candidate.end_token_index,
                    label="mixed_uncertain",
                    confidence=0.0,
                    suspicious=False,
                    retry_language=None,
                    reason="LLM response omitted this span.",
                )
            )
        else:
            normalized.append(decision)
    return normalized, parsed


def assess_transcript_quality(
    raw: RawTranscript,
    audio_path: Path,
    expected_language: str | None = None,
    config: AppConfig | None = None,
) -> TranscriptQualityAssessment:
    text = (raw.text or " ".join(segment.text for segment in raw.segments)).strip()
    duration = _duration_seconds(audio_path)
    segments: list[str] = []
    for segment in getattr(raw, "segments", []) or []:
        value = getattr(segment, "text", None)
        if isinstance(value, str) and value.strip():
            segments.append(value.strip())
    total_chars = len(text)
    script_counts = _script_counts(text)
    tokens = _tokenize_words(text)
    token_count = len(tokens) or 1
    suspicious_token_count, suspicious_token_examples = _count_cyrillic_hebrew_like_tokens(tokens)
    suspicious_token_ratio = suspicious_token_count / token_count
    suspicious_phrase = _has_mixed_hebrew_domain_terms(text)
    suspicious_segment_count, suspicious_segment_examples = _count_suspicious_segments(raw)
    uncertain_segment_count = sum(1 for segment in getattr(raw, "segments", []) or [] if _segment_is_uncertain(segment))
    mixed_script_tokens = _mixed_script_token_count(tokens)
    mixed_script_ratio = mixed_script_tokens / token_count
    transition_ratio = _script_transition_ratio(text)
    flags: list[str] = []
    score = 1.0

    if duration and duration <= 20.0:
        flags.append("short_call")

    if duration >= float(20):
        density_threshold = max(18, int(duration * 0.75))
        if total_chars < density_threshold:
            flags.append("low_text_density")
            score -= 0.30

    if duration >= float(30) and len(segments) <= 2:
        flags.append("sparse_segments")
        score -= 0.20

    repeat_count = _adjacent_repeat_count(segments)
    if repeat_count >= 1:
        flags.append("repeated_adjacent_segments")
        score -= 0.15

    if (
        script_counts["cyrillic"]
        and (
            HEBREW_TRANSLIT_RE.search(text)
            or suspicious_phrase
            or suspicious_token_count >= 4
            or (duration >= 60.0 and suspicious_token_ratio >= 0.015)
        )
    ):
        flags.append("suspected_hebrew_transliteration")
        score -= 0.35

    if suspicious_token_count >= 4 or suspicious_phrase:
        flags.append("high_hebrew_transliteration_density")
        score -= 0.20

    if raw.language in {"ru", "unknown"} and script_counts["cyrillic"] and (
        HEBREW_TRANSLIT_RE.search(text) or suspicious_token_count >= 4 or suspicious_phrase
    ):
        flags.append("baseline_script_mismatch")
        score -= 0.15

    if raw.language in {"ru", "unknown"} and suspicious_token_count >= 4 and script_counts["hebrew"] == 0:
        flags.append("mixed_language_retry_recommended")
        score -= 0.10

    if raw.language in {"ru", "unknown"} and suspicious_segment_count >= 1 and script_counts["hebrew"] == 0:
        flags.append("segment_level_hebrew_recovery_recommended")
    if uncertain_segment_count >= max(1, len(getattr(raw, "segments", []) or []) // 3):
        flags.append("segment_language_uncertain")
        score -= 0.10

    if expected_language in {"he", "iw"} and script_counts["hebrew"] == 0:
        flags.append("missing_hebrew_script")
        score -= 0.20

    if raw.language in {"he", "iw"} and script_counts["hebrew"] == 0 and script_counts["cyrillic"] > script_counts["latin"]:
        flags.append("hebrew_detected_but_cyrillic_rendered")
        score -= 0.20

    if mixed_script_tokens >= 2 or mixed_script_ratio >= 0.08 or transition_ratio >= 0.18:
        flags.append("mixed_script_gibberish")
        score -= 0.45

    if (
        raw.language in {"he", "iw"}
        and script_counts["hebrew"] > 0
        and (script_counts["latin"] + script_counts["cyrillic"]) > script_counts["hebrew"]
        and (mixed_script_tokens >= 1 or transition_ratio >= 0.12)
    ):
        flags.append("hebrew_candidate_corrupted")
        score -= 0.30

    if total_chars == 0:
        flags.append("empty_transcript")
        score -= 0.50

    heuristic_score = max(0.0, min(1.0, score))
    llm_score: float | None = None
    llm_flags: list[str] = []
    llm_reason: str | None = None
    if config is not None:
        llm_score, llm_flags, llm_reason = _run_redundant_quality_llm(
            raw,
            audio_path,
            heuristic_score,
            flags,
            expected_language,
            config,
        )
    score = heuristic_score if llm_score is None else min(heuristic_score, llm_score)
    combined_flags = list(flags)
    for flag in llm_flags:
        if flag not in combined_flags:
            combined_flags.append(flag)
    strong_language_confusion_flags = {
        "suspected_hebrew_transliteration",
        "high_hebrew_transliteration_density",
        "baseline_script_mismatch",
        "hebrew_detected_but_cyrillic_rendered",
        "mixed_language_retry_recommended",
        "segment_level_hebrew_recovery_recommended",
        "unsupported_script_detected",
        "wrong_language_gibberish",
        "repeated_foreign_script_gibberish",
    }
    suspect_language_confusion = any(flag in strong_language_confusion_flags for flag in combined_flags) or any(
        flag in {"low_text_density", "sparse_segments"} for flag in combined_flags
    )
    suggested_retry_languages = ["he"] if suspect_language_confusion else []
    low_confidence_reason = ", ".join(combined_flags) if score < 0.45 and combined_flags else None
    return TranscriptQualityAssessment(
        score=score,
        flags=combined_flags,
        suspect_language_confusion=suspect_language_confusion,
        suggested_retry_languages=suggested_retry_languages,
        suspicious_token_count=suspicious_token_count,
        suspicious_token_examples=suspicious_token_examples[:8],
        suspicious_segment_count=suspicious_segment_count,
        suspicious_segment_examples=suspicious_segment_examples,
        low_confidence_reason=low_confidence_reason,
        heuristic_score=heuristic_score,
        llm_score=llm_score,
        llm_flags=llm_flags,
        llm_reason=llm_reason,
    )


def _looks_mixed_language_problem(raw: RawTranscript, audio_path: Path, config: AppConfig | None = None) -> bool:
    return assess_transcript_quality(raw, audio_path, config=config).suspect_language_confusion


def _should_run_redundant_quality_llm(
    raw: RawTranscript,
    heuristic_score: float,
    flags: list[str],
    expected_language: str | None,
    config: AppConfig,
) -> bool:
    section = config.section("transcription")
    if not section.get("quality_redundant_llm_enabled", True):
        return False
    if raw.language not in ALLOWED_EXPLICIT_LANGUAGES:
        return True
    if expected_language in {"he", "iw"} and raw.language not in {"he", "iw"}:
        return True
    suspicious_flags = {
        "segment_language_uncertain",
        "mixed_script_gibberish",
        "baseline_script_mismatch",
        "hebrew_detected_but_cyrillic_rendered",
        "repeated_adjacent_segments",
        "empty_transcript",
    }
    if any(flag in suspicious_flags for flag in flags):
        return True
    return heuristic_score < 0.8


def _redundant_quality_assessment_key(raw: RawTranscript, expected_language: str | None) -> tuple[str | None, str, int]:
    text = (raw.text or " ".join(segment.text for segment in raw.segments)).strip()
    return expected_language, text, len(getattr(raw, "segments", []) or [])


def _run_redundant_quality_llm(
    raw: RawTranscript,
    audio_path: Path,
    heuristic_score: float,
    heuristic_flags: list[str],
    expected_language: str | None,
    config: AppConfig,
) -> tuple[float | None, list[str], str | None]:
    section = config.section("transcription")
    if not _should_run_redundant_quality_llm(raw, heuristic_score, heuristic_flags, expected_language, config):
        return None, [], None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, [], None
    cache = getattr(raw, "_quality_llm_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        raw._quality_llm_cache = cache
    cache_key = _redundant_quality_assessment_key(raw, expected_language)
    if cache_key in cache:
        return cache[cache_key]
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        text = (raw.text or " ".join(segment.text for segment in raw.segments)).strip()
        payload = {
            "transcript_language": raw.language,
            "expected_language": expected_language,
            "duration_seconds": _duration_seconds(audio_path),
            "segment_count": len(getattr(raw, "segments", []) or []),
            "text": text[: int(section.get("quality_redundant_llm_max_chars", 1200))],
            "allowed_languages": ["ru", "he", "en"],
        }
        prompt = """
You are validating ASR transcript quality for a phone-call pipeline that expects Russian, Hebrew, or English.
Score whether the transcript looks plausible and usable as-is.
Penalize unsupported scripts, obvious nonsense, repeated gibberish, or language/script mismatch.
Return JSON only:
{
  "score": 0.0,
  "flags": ["..."],
  "reason": "..."
}
Rules:
- score must be between 0.0 and 1.0
- use concise flags in snake_case
- if transcript is in an unsupported script or obviously wrong-language gibberish, score it low
""".strip()
        response = client.chat.completions.create(
            model=str(section.get("quality_redundant_llm_model", "gpt-4.1-nano")),
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            timeout=section.get("quality_redundant_llm_timeout_sec", 20),
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        flags = [str(item).strip() for item in parsed.get("flags", []) if str(item).strip()]
        reason = str(parsed.get("reason", "")).strip() or None
        logger.info(
            "Transcription redundant quality llm audio=%s language=%s expected_language=%s score=%.2f flags=%s reason=%s",
            audio_path,
            raw.language,
            expected_language,
            score,
            ",".join(flags) or "-",
            reason or "-",
        )
        result = (score, flags, reason)
        cache[cache_key] = result
        return result
    except Exception:
        if not section.get("quality_redundant_llm_continue_on_error", True):
            raise
        logger.exception("Transcription redundant quality llm failed audio=%s", audio_path)
        result = (None, [], None)
        cache[cache_key] = result
        return result


def _to_segments(raw_segments: list[dict]) -> list[RawSegment]:
    return [
        RawSegment(
            start_sec=float(item.get("start", 0.0)),
            end_sec=float(item.get("end", 0.0)),
            text=item.get("text", "").strip(),
            confidence=item.get("avg_logprob"),
            speaker=item.get("speaker"),
            language=item.get("language"),
            language_confidence=item.get("language_confidence"),
            chunk_id=item.get("chunk_id"),
            chunk_start_sec=item.get("chunk_start_sec"),
            chunk_end_sec=item.get("chunk_end_sec"),
            smoothed_language=item.get("smoothed_language"),
            smoothed_language_reason=item.get("smoothed_language_reason"),
        )
        for item in raw_segments
        if item.get("text", "").strip()
    ]


def _language_distribution(segments: list[RawSegment]) -> dict[str, float] | None:
    totals: dict[str, float] = {}
    total_duration = 0.0
    for segment in segments:
        label, _ = _normalize_language_label(segment.smoothed_language or segment.language, segment.text, None)
        if not label:
            continue
        duration = max(0.0, float(segment.end_sec) - float(segment.start_sec))
        if duration <= 0:
            duration = max(1.0, float(len(segment.text.strip()) or 1))
        totals[label] = totals.get(label, 0.0) + duration
        total_duration += duration
    if total_duration <= 0:
        return None
    return {label: round(value / total_duration, 4) for label, value in sorted(totals.items())}


def _primary_language_from_distribution(distribution: dict[str, float] | None, fallback: str) -> tuple[str, float | None]:
    if not distribution:
        normalized_fallback, _ = _normalize_language_label(fallback, "", None)
        return normalized_fallback or "unknown", None
    ordered = sorted(
        [(label, value) for label, value in distribution.items() if label != "unknown"],
        key=lambda item: item[1],
        reverse=True,
    )
    if not ordered:
        return "unknown", distribution.get("unknown")
    top_label, top_share = ordered[0]
    if len(ordered) > 1 and ordered[1][1] >= 0.25:
        return "mixed", top_share
    return top_label, top_share


def build_transcription_vad_chunk_artifact(audio_path: Path, raw: RawTranscript, config: AppConfig) -> dict[str, object]:
    chunks = detect_speech_chunks(audio_path, config)
    language_counts: dict[str, int] = {}
    for segment in raw.segments:
        label = segment.smoothed_language or segment.language
        if not label:
            continue
        language_counts[label] = language_counts.get(label, 0) + 1
    return {
        "backend": raw.backend_selected or config.section("transcription").get("backend", "whisper_legacy"),
        "backend_attempted": raw.backend_attempted,
        "backend_selected": raw.backend_selected,
        "backend_fallback_reason": raw.backend_fallback_reason,
        "chunked_plausibility_score": raw.chunked_plausibility_score,
        "chunked_plausibility_flags": raw.chunked_plausibility_flags or [],
        "provider": config.section("transcription").get("vad_provider", "silence"),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "duration_seconds": chunk.duration_seconds,
            }
            for chunk in chunks
        ],
        "segment_language_counts": language_counts,
        "transcript_segment_count": len(raw.segments),
    }


def _annotate_chunk_languages(raw: RawTranscript, config: AppConfig) -> RawTranscript:
    return smooth_segment_languages(label_segment_languages(raw, config), config)


def _load_local_model(model_name: str):
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is not None:
            return model, True
    import whisper

    model = whisper.load_model(model_name)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached, True
        _MODEL_CACHE[model_name] = model
    return model, False


def _transcribe_local_file(
    audio_path: Path,
    config: AppConfig,
    model_override: str | None = None,
    language_override: str | None = None,
    language_mode: str = "auto",
    progress_enabled: bool = True,
) -> RawTranscript:
    import whisper

    model_name = model_override or config.section("transcription")["local_model"]
    language = _transcription_language(audio_path, config, language_mode, language_override)
    call_id = _call_id(audio_path)
    logger.info(
        "Transcription local start audio=%s model=%s language_mode=%s language_override=%s language_hint=%s",
        audio_path,
        model_name,
        language_mode,
        language_override,
        language,
    )
    if progress_enabled:
        set_stage_progress(call_id, "transcription", completed=None, total=None, step_name="loading_model")
    model, cache_hit = _load_local_model(model_name)
    logger.info("Transcription local model %s audio=%s model=%s", "cache_hit" if cache_hit else "cache_miss", audio_path, model_name)
    if progress_enabled:
        set_stage_progress(
            call_id,
            "transcription",
            completed=None,
            total=None,
            step_name="model_cached" if cache_hit else "model_loaded",
        )
        set_stage_progress(
            call_id,
            "transcription",
            completed=None,
            total=None,
            step_name="preparing_audio",
        )
    transcribe_module = getattr(whisper, "transcribe", None)
    transcribe_globals = getattr(transcribe_module, "__globals__", {}) if transcribe_module is not None else {}
    tqdm_module = transcribe_globals.get("tqdm") if isinstance(transcribe_globals, dict) else None
    original_tqdm = getattr(tqdm_module, "tqdm", None) if tqdm_module is not None else None

    class TrackingTqdm:
        def __init__(self, *args, **kwargs):
            self._inner = original_tqdm(*args, **kwargs)
            self.total = getattr(self._inner, "total", None)
            self.n = getattr(self._inner, "n", 0)
            if progress_enabled:
                set_stage_progress(
                    call_id,
                    "transcription",
                    completed=None,
                    total=None,
                    step_name="detecting_language",
                )

        def update(self, n=1):
            result = self._inner.update(n)
            self.n = getattr(self._inner, "n", self.n + n)
            self.total = getattr(self._inner, "total", self.total)
            if progress_enabled:
                set_stage_progress(
                    call_id,
                    "transcription",
                    completed=self.n,
                    total=self.total,
                    step_name="transcribing",
                )
            return result

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    try:
        patcher = patch.object(tqdm_module, "tqdm", TrackingTqdm) if original_tqdm is not None and tqdm_module is not None else nullcontext()
        with patcher:
            result = model.transcribe(str(audio_path), language=language, verbose=False)
    finally:
        if progress_enabled:
            clear_stage_progress(call_id, "transcription")
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
    normalized_language, normalization_reason = _normalize_language_label(transcript.language, transcript.text, language)
    if normalization_reason.startswith("unsupported_label_"):
        logger.info(
            "Transcription language normalized raw=%s normalized=%s reason=%s audio=%s",
            result.get("language", language or "unknown"),
            normalized_language,
            normalization_reason,
            audio_path,
        )
    transcript.language = normalized_language or "unknown"
    transcript = _annotate_chunk_languages(transcript, config)
    transcript = _with_transcript_backend_metadata(transcript, backend_attempted="whisper_legacy")
    mixed_language_suspected = _looks_mixed_language_problem(transcript, audio_path, config=config)
    logger.info(
        "Transcription local success audio=%s segments=%s language=%s mixed_language_suspected=%s",
        audio_path,
        len(segments),
        transcript.language,
        mixed_language_suspected,
    )
    return transcript


def _transcribe_local(
    audio_path: Path,
    config: AppConfig,
    model_override: str | None = None,
    language_override: str | None = None,
    language_mode: str = "auto",
) -> RawTranscript:
    return _transcribe_local_file(audio_path, config, model_override, language_override, language_mode, progress_enabled=True)


def _transcribe_cloud_file(
    audio_path: Path,
    config: AppConfig,
    model_override: str | None = None,
    language_override: str | None = None,
    language_mode: str = "auto",
    progress_enabled: bool = True,
) -> RawTranscript:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for cloud transcription fallback")
    model_name = model_override or config.section("transcription")["cloud_model"]
    language = _transcription_language(audio_path, config, language_mode, language_override)
    call_id = _call_id(audio_path)
    if progress_enabled:
        set_stage_progress(call_id, "transcription", completed=0, total=1, step_name="cloud_upload")
    logger.info(
        "Transcription cloud start audio=%s model=%s language_mode=%s language_override=%s language_hint=%s",
        audio_path,
        model_name,
        language_mode,
        language_override,
        language,
    )
    client = OpenAI(api_key=api_key)
    try:
        with audio_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                response_format="verbose_json",
                language=language,
            )
    finally:
        if progress_enabled:
            clear_stage_progress(call_id, "transcription")
    transcript_dict = transcript.model_dump() if hasattr(transcript, "model_dump") else dict(transcript)
    segments = _to_segments(transcript_dict.get("segments", []))
    text = transcript_dict.get("text", "").strip()
    if not segments and text:
        segments = [RawSegment(start_sec=0.0, end_sec=0.0, text=text, confidence=None, speaker=None)]
    logger.info("Transcription cloud success audio=%s segments=%s language=%s", audio_path, len(segments), transcript_dict.get("language", "unknown"))
    transcript = _annotate_chunk_languages(
        RawTranscript(
            provider="cloud",
            model=model_name,
            language=transcript_dict.get("language", "unknown"),
            confidence=None,
            segments=segments,
            text=text,
        ),
        config,
    )
    transcript.language, _ = _normalize_language_label(
        transcript.language,
        transcript.text,
        language,
    )
    return _with_transcript_backend_metadata(transcript, backend_attempted="whisper_legacy")


def _transcribe_cloud(
    audio_path: Path,
    config: AppConfig,
    model_override: str | None = None,
    language_override: str | None = None,
    language_mode: str = "auto",
) -> RawTranscript:
    return _transcribe_cloud_file(audio_path, config, model_override, language_override, language_mode, progress_enabled=True)


def _export_audio_chunk(audio_path: Path, start_sec: float, end_sec: float) -> Path:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    start_ms = max(0, int(start_sec * 1000))
    end_ms = max(start_ms, int(end_sec * 1000))
    clipped = audio[start_ms:end_ms]
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    clipped.export(temp_path, format="wav")
    return temp_path


def _transcribe_vad_chunked(
    audio_path: Path,
    config: AppConfig,
    provider: str,
    model_override: str | None = None,
    language_override: str | None = None,
    language_mode: str = "auto",
) -> RawTranscript:
    chunks = detect_speech_chunks(audio_path, config)
    call_id = _call_id(audio_path)
    backend = str(config.section("transcription").get("backend", "whisper_legacy")).strip().lower()
    aggregated_segments: list[RawSegment] = []
    transcript_parts: list[str] = []
    chunk_languages: list[str] = []
    if not chunks:
        return RawTranscript(provider=provider, model=model_override or "unknown", language="unknown", confidence=None, segments=[], text="")
    set_stage_progress(call_id, "transcription", completed=0, total=len(chunks), step_name="chunking")
    try:
        for index, chunk in enumerate(chunks, start=1):
            if chunk.duration_seconds <= 0:
                continue
            set_stage_progress(call_id, "transcription", completed=index - 1, total=len(chunks), step_name=f"chunk_{index}")
            chunk_path = _export_audio_chunk(audio_path, chunk.start_sec, chunk.end_sec)
            try:
                if provider == "local" and backend == "faster_whisper_vad":
                    chunk_raw = transcribe_faster_whisper_audio(
                        chunk_path,
                        config,
                        model_override=model_override,
                        language_hint=_transcription_language(chunk_path, config, language_mode, language_override),
                    )
                elif provider == "local":
                    chunk_raw = _transcribe_local_file(
                        chunk_path,
                        config,
                        model_override=model_override,
                        language_override=language_override,
                        language_mode=language_mode,
                        progress_enabled=False,
                    )
                else:
                    chunk_raw = _transcribe_cloud_file(
                        chunk_path,
                        config,
                        model_override=model_override,
                        language_override=language_override,
                        language_mode=language_mode,
                        progress_enabled=False,
                    )
            finally:
                chunk_path.unlink(missing_ok=True)
            if chunk_raw.text.strip():
                transcript_parts.append(chunk_raw.text.strip())
            normalized_chunk_language, chunk_language_reason = _normalize_language_label(
                chunk_raw.language,
                chunk_raw.text,
                language_override,
            )
            if chunk_language_reason.startswith("unsupported_label_"):
                logger.info(
                    "Chunk language normalized raw=%s normalized=%s chunk_id=%s reason=%s",
                    chunk_raw.language,
                    normalized_chunk_language,
                    chunk.chunk_id,
                    chunk_language_reason,
                )
            chunk_raw.language = normalized_chunk_language or "unknown"
            if chunk_raw.language:
                chunk_languages.append(chunk_raw.language)
            for segment in chunk_raw.segments:
                normalized_segment_language, segment_reason = _normalize_language_label(
                    segment.language or chunk_raw.language,
                    segment.text,
                    chunk_raw.language,
                )
                aggregated_segments.append(
                    RawSegment(
                        start_sec=chunk.start_sec + segment.start_sec,
                        end_sec=chunk.start_sec + segment.end_sec,
                        text=segment.text,
                        confidence=segment.confidence,
                        speaker=segment.speaker,
                        language=normalized_segment_language or "unknown",
                        language_confidence=chunk_raw.confidence,
                        chunk_id=chunk.chunk_id,
                        chunk_start_sec=chunk.start_sec,
                        chunk_end_sec=chunk.end_sec,
                        smoothed_language=normalized_segment_language or "unknown",
                        smoothed_language_reason=segment_reason if segment_reason.startswith("unsupported_label_") else "chunk_language",
                    )
                )
            set_stage_progress(call_id, "transcription", completed=index, total=len(chunks), step_name=f"chunk_{index}")
    finally:
        clear_stage_progress(call_id, "transcription")
    overall_language = "unknown"
    if chunk_languages:
        counts: dict[str, int] = {}
        for language in chunk_languages:
            counts[language] = counts.get(language, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        overall_language = "mixed" if len(ordered) > 1 else ordered[0][0]
    transcript = RawTranscript(
        provider=provider,
        model=model_override or (config.section("transcription")["local_model"] if provider == "local" else config.section("transcription")["cloud_model"]),
        language=overall_language,
        confidence=None,
        segments=aggregated_segments,
        text=" ".join(part for part in transcript_parts if part).strip(),
    )
    transcript = _annotate_chunk_languages(transcript, config)
    plausibility = assess_chunked_transcript_plausibility(transcript, chunks, config)
    transcript = _with_transcript_backend_metadata(
        transcript,
        backend_attempted=backend,
        backend_selected=backend,
        plausibility=plausibility,
    )
    fallback_backend = str(config.section("transcription").get("fallback_backend", "whisper_legacy")).strip().lower()
    auto_fallback = bool(config.section("transcription").get("chunked_auto_fallback_to_legacy", True))
    if plausibility.should_fallback and auto_fallback and fallback_backend == "whisper_legacy":
        logger.info(
            "Chunked transcription fallback selected audio=%s attempted_backend=%s provider=%s reason=%s flags=%s score=%.2f attempted_preview=%s",
            audio_path,
            backend,
            provider,
            plausibility.reason,
            ",".join(plausibility.flags),
            plausibility.score,
            _transcript_log_preview(transcript),
        )
        if provider == "local":
            fallback = _transcribe_local_file(
                audio_path,
                config,
                model_override=model_override,
                language_override=language_override,
                language_mode=language_mode,
                progress_enabled=False,
            )
        else:
            fallback = _transcribe_cloud_file(
                audio_path,
                config,
                model_override=model_override,
                language_override=language_override,
                language_mode=language_mode,
                progress_enabled=False,
            )
        fallback = _with_transcript_backend_metadata(
            fallback,
            backend_attempted=backend,
            backend_selected=fallback_backend,
            fallback_reason=plausibility.reason or "chunked_implausible",
            plausibility=plausibility,
        )
        logger.info(
            "Chunked transcription fallback result audio=%s selected_backend=%s selected_language=%s selected_preview=%s",
            audio_path,
            fallback_backend,
            fallback.language,
            _transcript_log_preview(fallback),
        )
        return fallback
    logger.info(
        "Chunked transcription accepted audio=%s backend=%s provider=%s flags=%s score=%.2f",
        audio_path,
        backend,
        provider,
        ",".join(plausibility.flags) if plausibility.flags else "none",
        plausibility.score,
    )
    return transcript


def _segment_quality_score(text: str, preferred_language: str | None = None) -> float:
    value = text.strip()
    if not value:
        return 0.0
    score = min(len(value) / 20.0, 1.0)
    scripts = _script_counts(value)
    if preferred_language in {"he", "iw"}:
        score += min(scripts["hebrew"] / max(len(value), 1), 0.6)
        if scripts["cyrillic"] and HEBREW_TRANSLIT_RE.search(value):
            score += 0.15
    if HEBREW_TRANSLIT_RE.search(value):
        score += 0.10
    return score


def _segment_has_transliterated_hebrew(text: str) -> bool:
    tokens = _tokenize_words(text)
    suspicious_token_count, _ = _count_cyrillic_hebrew_like_tokens(tokens)
    return _has_mixed_hebrew_domain_terms(text) or suspicious_token_count >= 1 or bool(HEBREW_TRANSLIT_RE.search(text))


def _segment_prefers_forced_hebrew(base_segment: RawSegment, forced_segment: RawSegment) -> bool:
    base_text = (base_segment.text or "").strip()
    forced_text = (forced_segment.text or "").strip()
    if not base_text or not forced_text:
        return False
    base_scripts = _script_counts(base_text)
    forced_scripts = _script_counts(forced_text)
    base_score = _segment_quality_score(base_text)
    forced_score = _segment_quality_score(forced_text, preferred_language="he")
    base_suspicious = _segment_has_transliterated_hebrew(base_text)
    if base_suspicious and forced_scripts["hebrew"] > 0:
        return True
    if base_suspicious and base_scripts["hebrew"] == 0 and forced_scripts["hebrew"] > 0 and forced_score > base_score + 0.05:
        return True
    return False


def _segment_retry_score(segment: RawSegment, decision: SegmentDetectionDecision) -> float:
    label_bonus = {
        "hebrew_transliteration": 0.3,
        "mixed_uncertain": 0.1,
        "garbled_noise": -0.2,
    }.get(decision.label, 0.0)
    duration = max(0.0, float(segment.end_sec) - float(segment.start_sec))
    duration_bonus = min(duration / 20.0, 0.3)
    return decision.confidence + label_bonus + duration_bonus


def _word_span_detection_decisions(candidates: list[WordSpanCandidate], config: AppConfig) -> list[WordSpanDecision]:
    decisions, _ = _word_span_detection_decisions_with_raw(candidates, config)
    return decisions


def _word_span_detection_decisions_with_raw(
    candidates: list[WordSpanCandidate], config: AppConfig
) -> tuple[list[WordSpanDecision], dict[str, object] | None]:
    if not candidates:
        return [], None
    section = config.section("segment_detection")
    if not section.get("enabled", True):
        return _heuristic_word_span_decisions(candidates), None
    if not section.get("run_on_every_call", True):
        return _heuristic_word_span_decisions(candidates), None
    provider = section.get("provider", "openai")
    continue_on_error = bool(section.get("continue_on_error", True))
    try:
        if provider == "openai":
            return _classify_word_spans_with_llm(candidates, config)
    except Exception:
        logger.exception("Word span detection provider %s failed", provider)
        if not continue_on_error:
            raise
    return _heuristic_word_span_decisions(candidates), None


def _word_retry_score(candidate: WordSpanCandidate, decision: WordSpanDecision) -> float:
    label_bonus = {
        "hebrew_transliteration": 0.3,
        "mixed_uncertain": 0.1,
        "garbled_noise": -0.2,
    }.get(decision.label, 0.0)
    span_word_count = candidate.end_token_index - candidate.start_token_index + 1
    return decision.confidence + label_bonus + min(span_word_count * 0.1, 0.3)


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])


def _rank_word_retry_candidates(
    candidates: list[WordSpanCandidate],
    decisions: list[WordSpanDecision],
    config: AppConfig,
) -> tuple[list[WordRetryCandidate], int]:
    section = config.section("segment_detection")
    max_retry = int(section.get("max_retry_segments_per_call", 10))
    min_confidence = float(section.get("min_confidence_to_retry", 0.70))
    by_key = {
        (candidate.segment_index, candidate.start_token_index, candidate.end_token_index): candidate
        for candidate in candidates
    }
    scored: list[WordRetryCandidate] = []
    below_threshold_count = 0
    for decision in decisions:
        if not decision.suspicious or decision.retry_language != section.get("retry_language", "he"):
            continue
        key = (decision.segment_index, decision.start_token_index, decision.end_token_index)
        candidate = by_key.get(key)
        if candidate is None:
            continue
        score = _word_retry_score(candidate, decision)
        if decision.confidence < min_confidence:
            below_threshold_count += 1
            continue
        segment_duration = max(candidate.segment_end_sec - candidate.segment_start_sec, 0.01)
        text_length = max(len(candidate.segment_text), 1)
        start_ratio = candidate.start_char_offset / text_length
        end_ratio = candidate.end_char_offset / text_length
        start_sec = candidate.segment_start_sec + (segment_duration * start_ratio)
        end_sec = max(start_sec + 0.15, candidate.segment_start_sec + (segment_duration * end_ratio))
        scored.append(
            WordRetryCandidate(
                segment_index=candidate.segment_index,
                rank=0,
                score=score,
                start_sec=start_sec,
                end_sec=min(end_sec, candidate.segment_end_sec),
                baseline_segment_text=candidate.segment_text,
                span_text=candidate.span_text,
                start_token_index=candidate.start_token_index,
                end_token_index=candidate.end_token_index,
                context_before=candidate.context_before,
                context_after=candidate.context_after,
                llm_label=decision.label,
                llm_reason=decision.reason,
                llm_confidence=decision.confidence,
            )
        )
    scored.sort(
        key=lambda item: (
            -item.score,
            -(item.end_token_index - item.start_token_index + 1),
            item.segment_index,
            item.start_token_index,
        )
    )
    selected: list[WordRetryCandidate] = []
    occupied: dict[int, list[tuple[int, int]]] = {}
    for candidate in scored:
        current = occupied.setdefault(candidate.segment_index, [])
        span = (candidate.start_token_index, candidate.end_token_index)
        if any(_spans_overlap(existing, span) for existing in current):
            logger.info(
                "Transcription word span skipped call_id=%s segment_index=%s span=%s reason=overlapping_span_rejected",
                "unknown",
                candidate.segment_index,
                candidate.span_text,
            )
            continue
        current.append(span)
        selected.append(candidate)
        if len(selected) >= max_retry:
            break
    for rank, candidate in enumerate(selected, start=1):
        candidate.rank = rank
    return selected, below_threshold_count


def _word_span_candidate_score_map(
    candidates: list[WordSpanCandidate],
    decisions: list[WordSpanDecision],
) -> dict[tuple[int, int, int], float]:
    by_key = {
        (candidate.segment_index, candidate.start_token_index, candidate.end_token_index): candidate
        for candidate in candidates
    }
    scores: dict[tuple[int, int, int], float] = {}
    for decision in decisions:
        key = (decision.segment_index, decision.start_token_index, decision.end_token_index)
        candidate = by_key.get(key)
        if candidate is None:
            continue
        scores[key] = _word_retry_score(candidate, decision)
    return scores


def _word_span_detection_summary(
    decisions: list[WordSpanDecision], candidates: list[WordRetryCandidate], below_threshold_count: int, config: AppConfig
) -> dict[str, object]:
    section = config.section("segment_detection")
    return {
        "provider": section.get("provider", "heuristic"),
        "model": section.get("model", "heuristic"),
        "segments_evaluated": len({item.segment_index for item in decisions}),
        "spans_detected": sum(1 for item in decisions if item.suspicious),
        "selected_for_retry": len(candidates),
        "below_threshold_count": below_threshold_count,
        "threshold_used": float(section.get("min_confidence_to_retry", 0.70)),
    }


def _rank_segment_retry_candidates(
    raw: RawTranscript, decisions: list[SegmentDetectionDecision], config: AppConfig
) -> tuple[list[SegmentRetryCandidate], int]:
    section = config.section("segment_detection")
    max_retry = int(section.get("max_retry_segments_per_call", 10))
    min_confidence = float(section.get("min_confidence_to_retry", 0.70))
    context_window = int(section.get("context_window_segments", 1))
    candidates: list[SegmentRetryCandidate] = []
    below_threshold_count = 0
    for decision in decisions:
        if not decision.suspicious or decision.retry_language != section.get("retry_language", "he"):
            continue
        if decision.confidence < min_confidence:
            below_threshold_count += 1
            continue
        if not (0 <= decision.segment_index < len(raw.segments)):
            continue
        segment = raw.segments[decision.segment_index]
        score = _segment_retry_score(segment, decision)
        candidates.append(
            SegmentRetryCandidate(
                segment_index=decision.segment_index,
                rank=0,
                score=score,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                duration_seconds=max(0.0, segment.end_sec - segment.start_sec),
                baseline_text=(segment.text or "").strip(),
                context_before=_segment_context_text(raw.segments, decision.segment_index, -1, context_window),
                context_after=_segment_context_text(raw.segments, decision.segment_index, 1, context_window),
                llm_label=decision.label,
                llm_reason=decision.reason,
                llm_confidence=decision.confidence,
            )
        )
    candidates.sort(key=lambda item: (-item.score, -item.duration_seconds, item.start_sec))
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates[:max_retry], below_threshold_count


def _segment_detection_summary(
    decisions: list[SegmentDetectionDecision],
    candidates: list[SegmentRetryCandidate],
    below_threshold_count: int,
    config: AppConfig,
) -> dict[str, object]:
    section = config.section("segment_detection")
    return {
        "provider": section.get("provider", "heuristic"),
        "model": section.get("model", "heuristic"),
        "run_on_every_call": bool(section.get("run_on_every_call", True)),
        "segments_evaluated": len(decisions),
        "suspicious_count": sum(1 for item in decisions if item.suspicious),
        "selected_for_retry": len(candidates),
        "below_threshold_count": below_threshold_count,
        "threshold_used": float(section.get("min_confidence_to_retry", 0.70)),
    }


def _full_call_hebrew_retry_decision(
    baseline_assessment: TranscriptQualityAssessment,
    decisions: list[SegmentDetectionDecision],
    word_retry_candidates: list[WordRetryCandidate],
    clause_retry_candidates: list[ClauseRetryCandidate],
    config: AppConfig,
) -> dict[str, object]:
    section = config.section("segment_detection")
    score_threshold = float(section.get("full_call_hebrew_score_threshold", 0.45))
    min_suspicious_segments = int(section.get("full_call_hebrew_min_suspicious_segments", 3))
    suspicious_ratio_threshold = float(section.get("full_call_hebrew_suspicious_ratio_threshold", 0.5))
    suspicious_count = sum(1 for item in decisions if item.suspicious)
    evaluated_count = max(1, len(decisions))
    suspicious_ratio = suspicious_count / evaluated_count
    selected_retry_count = len(word_retry_candidates) + len(clause_retry_candidates)
    reasons: list[str] = []
    if baseline_assessment.suspect_language_confusion:
        reasons.append("suspect_language_confusion")
    if baseline_assessment.score < score_threshold:
        reasons.append("low_baseline_score")
    if suspicious_count == 0:
        reasons.append("no_suspicious_segments")
    if suspicious_count >= min_suspicious_segments:
        reasons.append("high_suspicious_segment_count")
    if suspicious_ratio >= suspicious_ratio_threshold:
        reasons.append("high_suspicious_segment_ratio")
    if baseline_assessment.suspicious_segment_count >= 1:
        reasons.append("quality_assessment_suspicious_segments")
    if selected_retry_count <= suspicious_count:
        reasons.append("few_selected_retries")
    should_retry = False
    if baseline_assessment.suspect_language_confusion and baseline_assessment.score < score_threshold:
        if suspicious_count == 0:
            should_retry = True
        elif (
            suspicious_count >= min_suspicious_segments
            and suspicious_ratio >= suspicious_ratio_threshold
            and baseline_assessment.suspicious_segment_count >= 1
            and selected_retry_count <= suspicious_count
        ):
            should_retry = True
    return {
        "should_retry": should_retry,
        "baseline_score": baseline_assessment.score,
        "score_threshold": score_threshold,
        "suspect_language_confusion": baseline_assessment.suspect_language_confusion,
        "suspicious_count": suspicious_count,
        "suspicious_ratio": round(suspicious_ratio, 4),
        "suspicious_ratio_threshold": suspicious_ratio_threshold,
        "min_suspicious_segments": min_suspicious_segments,
        "quality_assessment_suspicious_segment_count": baseline_assessment.suspicious_segment_count,
        "selected_retry_count": selected_retry_count,
        "word_retry_candidate_count": len(word_retry_candidates),
        "clause_retry_candidate_count": len(clause_retry_candidates),
        "reasons": reasons,
    }


def _should_run_full_call_hebrew_retry(
    baseline_assessment: TranscriptQualityAssessment,
    decisions: list[SegmentDetectionDecision],
    word_retry_candidates: list[WordRetryCandidate],
    clause_retry_candidates: list[ClauseRetryCandidate],
    config: AppConfig,
) -> bool:
    return bool(
        _full_call_hebrew_retry_decision(
            baseline_assessment,
            decisions,
            word_retry_candidates,
            clause_retry_candidates,
            config,
        )["should_retry"]
    )


def _extract_audio_span(audio_path: Path, candidate: SegmentRetryCandidate, config: AppConfig) -> Path:
    from pydub import AudioSegment

    lead_in = 0.4
    tail_out = 0.4
    clip = AudioSegment.from_file(audio_path)
    start_ms = max(0, int((candidate.start_sec - lead_in) * 1000))
    end_ms = min(len(clip), int((candidate.end_sec + tail_out) * 1000))
    span = clip[start_ms:end_ms]
    dest_dir = config.temp_dir / "segment_retries" / _call_id(audio_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"segment_{candidate.segment_index:04d}_{candidate.rank:02d}.wav"
    span.export(dest, format="wav")
    return dest


def _extract_word_span_audio(audio_path: Path, candidate: WordRetryCandidate, config: AppConfig) -> Path:
    from pydub import AudioSegment

    padding = float(config.section("segment_detection").get("audio_span_padding_sec", 0.35))
    clip = AudioSegment.from_file(audio_path)
    start_ms = max(0, int((candidate.start_sec - padding) * 1000))
    end_ms = min(len(clip), int((candidate.end_sec + padding) * 1000))
    span = clip[start_ms:end_ms]
    dest_dir = config.temp_dir / "word_span_retries" / _call_id(audio_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"segment_{candidate.segment_index:04d}_{candidate.rank:02d}_{candidate.start_token_index}_{candidate.end_token_index}.wav"
    span.export(dest, format="wav")
    return dest


def _extract_clause_retry_audio(audio_path: Path, candidate: ClauseRetryCandidate, config: AppConfig) -> Path:
    from pydub import AudioSegment

    padding = float(config.section("segment_detection").get("clause_retry_audio_padding_sec", 0.65))
    clip = AudioSegment.from_file(audio_path)
    segment_duration = max(candidate.end_sec - candidate.start_sec, 0.01)
    text_length = max(len(candidate.baseline_segment_text), 1)
    tokens = _segment_tokens(candidate.baseline_segment_text)
    if not tokens:
        start_sec = candidate.start_sec
        end_sec = candidate.end_sec
    else:
        start_offset = tokens[candidate.start_token_index].start_offset
        end_offset = tokens[candidate.end_token_index].end_offset
        start_ratio = start_offset / text_length
        end_ratio = end_offset / text_length
        start_sec = candidate.start_sec + (segment_duration * start_ratio)
        end_sec = candidate.start_sec + (segment_duration * end_ratio)
    start_ms = max(0, int((start_sec - padding) * 1000))
    end_ms = min(len(clip), int((end_sec + padding) * 1000))
    span = clip[start_ms:end_ms]
    dest_dir = config.temp_dir / "clause_retries" / _call_id(audio_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"segment_{candidate.segment_index:04d}_{candidate.rank:02d}_{candidate.start_token_index}_{candidate.end_token_index}.wav"
    span.export(dest, format="wav")
    return dest


def _transcribe_segment_hebrew_retry(span_audio_path: Path, config: AppConfig, model_override: str | None = None) -> RawTranscript:
    return _transcribe_local(
        span_audio_path,
        config,
        model_override=model_override,
        language_override=config.section("segment_detection").get("retry_language", "he"),
        language_mode="metadata_override",
    )


def _transcribe_word_span_hebrew_retry(span_audio_path: Path, config: AppConfig, model_override: str | None = None) -> RawTranscript:
    return _transcribe_local(
        span_audio_path,
        config,
        model_override=model_override,
        language_override=config.section("segment_detection").get("retry_language", "he"),
        language_mode="metadata_override",
    )


def _transcribe_clause_hebrew_retry(span_audio_path: Path, config: AppConfig, model_override: str | None = None) -> RawTranscript:
    return _transcribe_word_span_hebrew_retry(span_audio_path, config, model_override=model_override)


def _should_replace_with_hebrew_retry(
    baseline_text: str,
    retry_text: str,
    config: AppConfig | None = None,
) -> tuple[bool, str]:
    retry_value = (retry_text or "").strip()
    if not retry_value:
        return False, "retry_empty_rejected"
    scripts = _script_counts(retry_value)
    if scripts["hebrew"] == 0:
        return False, "retry_no_hebrew_rejected"
    retry_assessment = assess_transcript_quality(
        RawTranscript(provider="local", model="segment_retry", language="he", confidence=None, segments=[], text=retry_value),
        Path("/tmp/nonexistent"),
        expected_language="he",
        config=config,
    )
    if "mixed_script_gibberish" in retry_assessment.flags or "hebrew_candidate_corrupted" in retry_assessment.flags:
        return False, "retry_gibberish_rejected"
    baseline_len = max(1, len((baseline_text or "").strip()))
    if len(retry_value) < max(3, int(baseline_len * 0.15)):
        return False, "retry_too_short_rejected"
    return True, "hebrew_script_recovered"


def _should_replace_word_span(
    baseline_text: str,
    retry_text: str,
    config: AppConfig | None = None,
) -> tuple[bool, str]:
    return _should_replace_with_hebrew_retry(baseline_text, retry_text, config=config)


def _heuristic_word_span_semantic_validation(
    span_text: str,
    baseline_segment_text: str,
    retry_text: str,
) -> WordSpanSemanticValidation:
    retry_value = (retry_text or "").strip()
    if not retry_value:
        return WordSpanSemanticValidation("reject", 1.0, "Retry text is empty.", None)
    if _script_counts(retry_value)["hebrew"] == 0:
        return WordSpanSemanticValidation("reject", 1.0, "Retry text does not contain Hebrew script.", None)
    if retry_value == "לפת":
        return WordSpanSemanticValidation(
            "reject",
            0.95,
            "Retry text is not a plausible budgeting phrase in this context.",
            None,
        )
    normalized = None
    span_lower = (span_text or "").lower()
    baseline_lower = (baseline_segment_text or "").lower()
    if "атомали" in span_lower or "атомали" in baseline_lower:
        normalized = "התאמה ל-"
    elif "такцив" in span_lower and len(retry_value) <= 12:
        normalized = "תקציב"
    elif (
        any(token in span_lower for token in ("технун", "мульбецоа", "мульбицо"))
        and any(token in retry_value for token in ("תחנון", "מולד", "מולביצוע", "מולד"))
    ):
        normalized = (
            retry_value.replace("תחנון", "תכנון")
            .replace("מולד", "מול")
            .replace("מולביצוע", "מול ביצוע")
            .replace("..", ".")
            .strip(" .")
        )
    return WordSpanSemanticValidation(
        "accept_with_normalization" if normalized and normalized != retry_value else "accept",
        0.8,
        "Heuristic semantic validation accepted the Hebrew retry as an improvement over baseline transliteration.",
        normalized,
    )


def _normalized_hebrew_business_phrase(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return normalized
    replacements = (
        ("תחנון", "תכנון"),
        ("מולד", "מול"),
        ("מולביצוע", "מול ביצוע"),
        ("מולבצעה", "מול ביצוע"),
        ("תקציבים", "תקציב"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip(" .")
    return normalized


def _rescue_semantic_validation_with_normalization(
    candidate: WordRetryCandidate,
    retry_text: str,
    validation: WordSpanSemanticValidation,
) -> WordSpanSemanticValidation:
    retry_value = (retry_text or "").strip()
    if validation.decision in {"accept", "accept_with_normalization"}:
        if validation.normalized_hebrew:
            normalized = _normalized_hebrew_business_phrase(validation.normalized_hebrew)
            if normalized != validation.normalized_hebrew:
                return WordSpanSemanticValidation(
                    "accept_with_normalization",
                    validation.confidence,
                    validation.reason,
                    normalized,
                )
        return validation
    if _script_counts(retry_value)["hebrew"] == 0:
        return validation
    span_lower = (candidate.span_text or "").lower()
    baseline_lower = (candidate.baseline_segment_text or "").lower()
    normalized = _normalized_hebrew_business_phrase(retry_value)
    if (
        any(token in span_lower or token in baseline_lower for token in ("технун", "мульбецоа", "мульбицо"))
        and any(token in retry_value for token in ("תחנון", "מולד", "מולביצוע", "מולד"))
    ):
        return WordSpanSemanticValidation(
            "accept_with_normalization",
            max(validation.confidence, 0.76),
            "Retry text is noisy but closer to the intended Hebrew business phrase than the baseline transliteration.",
            normalized,
        )
    if "такцив" in span_lower and any(token in retry_value for token in ("תק", "ציב")):
        return WordSpanSemanticValidation(
            "accept_with_normalization",
            max(validation.confidence, 0.8),
            "Retry text plausibly recovers the Hebrew budgeting term better than the baseline transliteration.",
            "תקציב",
        )
    return validation


def _validate_word_span_retry_with_llm(
    candidate: WordRetryCandidate,
    retry_text: str,
    config: AppConfig,
    call_id: str,
) -> tuple[WordSpanSemanticValidation, dict[str, object] | None]:
    section = config.section("segment_detection")
    if not section.get("semantic_validation_enabled", True):
        return _heuristic_word_span_semantic_validation(candidate.span_text, candidate.baseline_segment_text, retry_text), None
    provider = section.get("provider", "openai")
    continue_on_error = bool(section.get("continue_on_error", True))
    if provider != "openai":
        return _heuristic_word_span_semantic_validation(candidate.span_text, candidate.baseline_segment_text, retry_text), None
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured for semantic validation")
        client = OpenAI(api_key=api_key)
        payload = {
            "segment_index": candidate.segment_index,
            "baseline_segment_text": candidate.baseline_segment_text,
            "span_text": candidate.span_text,
            "context_before": candidate.context_before,
            "context_after": candidate.context_after,
            "retry_text": retry_text,
        }
        prompt = """
You are validating a short forced-Hebrew ASR retry for a noisy Russian/Hebrew business call.
The baseline segment may contain Hebrew terms written phonetically in Russian.
Decide whether retry_text is a better Hebrew recovery of span_text than the baseline transliteration.
Do not require perfect Hebrew. If retry_text is noisy but clearly closer to the intended Hebrew phrase, accept it with normalization.
Reject only if the retry is unrelated, clearly garbled, or worse than the baseline transliteration.
If the intended Hebrew phrase is obvious, provide a minimally normalized Hebrew phrase.
Return JSON only:
{
  "decision": "accept",
  "confidence": 0.88,
  "reason": "...",
  "normalized_hebrew": "..."
}
Allowed decisions: accept, accept_with_normalization, reject, uncertain.
""".strip()
        response = client.chat.completions.create(
            model=section.get("semantic_validation_model", section.get("model", "gpt-4o-mini")),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            timeout=section.get("semantic_validation_timeout_sec", 45),
        )
        parsed = json.loads(response.choices[0].message.content)
        return (
            WordSpanSemanticValidation(
                decision=str(parsed.get("decision", "uncertain")),
                confidence=float(parsed.get("confidence", 0.0)),
                reason=str(parsed.get("reason", "")),
                normalized_hebrew=(
                    str(parsed["normalized_hebrew"]).strip() if parsed.get("normalized_hebrew") else None
                ),
            ),
            parsed,
        )
    except Exception:
        logger.exception(
            "Word span semantic validation failed call_id=%s segment_index=%s span=%s",
            call_id,
            candidate.segment_index,
            candidate.span_text,
        )
        if not continue_on_error:
            raise
    return _heuristic_word_span_semantic_validation(candidate.span_text, candidate.baseline_segment_text, retry_text), None


def _clause_candidate_as_word_retry(candidate: ClauseRetryCandidate) -> WordRetryCandidate:
    return WordRetryCandidate(
        segment_index=candidate.segment_index,
        rank=candidate.rank,
        score=candidate.score,
        start_sec=candidate.start_sec,
        end_sec=candidate.end_sec,
        baseline_segment_text=candidate.baseline_segment_text,
        span_text=candidate.clause_text,
        start_token_index=candidate.start_token_index,
        end_token_index=candidate.end_token_index,
        context_before=candidate.context_before,
        context_after=candidate.context_after,
        llm_label="clause_retry",
        llm_reason=candidate.source_reason,
        llm_confidence=1.0,
    )


def _validate_clause_retry_with_llm(
    candidate: ClauseRetryCandidate,
    retry_text: str,
    config: AppConfig,
    call_id: str,
) -> tuple[WordSpanSemanticValidation, dict[str, object] | None]:
    return _validate_word_span_retry_with_llm(_clause_candidate_as_word_retry(candidate), retry_text, config, call_id)


def _reconcile_segment_replacements_with_llm(
    call_id: str,
    segment_index: int,
    baseline_segment_text: str,
    context_before: str,
    context_after: str,
    accepted: list[dict[str, object]],
    config: AppConfig,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    section = config.section("segment_detection")
    if len(accepted) <= 1 or not section.get("segment_reconciliation_enabled", True):
        return accepted, None
    def _best(items: list[dict[str, object]]) -> dict[str, object]:
        return max(
            items,
            key=lambda item: (
                float(item["semantic_confidence"]),
                1 if item.get("source_type") == "clause" else 0,
                float(item["llm_confidence"]),
            ),
        )
    provider = section.get("provider", "openai")
    continue_on_error = bool(section.get("continue_on_error", True))
    if provider != "openai":
        best = _best(accepted)
        return [best], None
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured for segment reconciliation")
        client = OpenAI(api_key=api_key)
        payload = {
            "segment_index": segment_index,
            "baseline_segment_text": baseline_segment_text,
            "context_before": context_before,
            "context_after": context_after,
            "replacements": [
                {
                    "rank": int(item["rank"]),
                    "start_token_index": int(item["start_token_index"]),
                    "end_token_index": int(item["end_token_index"]),
                    "span_text": item["span_text"],
                    "replacement_text": item["final_replacement_text"],
                    "source_type": item.get("source_type", "word_span"),
                }
                for item in accepted
            ],
        }
        prompt = """
You are reconciling multiple accepted Hebrew replacements within one noisy Russian/Hebrew ASR segment.
Choose the most coherent subset of replacements, or minimally reconcile them into non-duplicative Hebrew phrases.
Do not rewrite untouched Russian context. Only operate on the proposed replacement spans.
Return JSON only:
{
  "decision": "keep_original" | "use_reconciled",
  "reason": "...",
  "replacements": [
    {
      "rank": 1,
      "start_token_index": 3,
      "end_token_index": 5,
      "replacement_text": "בתור תכנון מול ביצוע"
    }
  ]
}
""".strip()
        response = client.chat.completions.create(
            model=section.get("segment_reconciliation_model", section.get("model", "gpt-4o-mini")),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            timeout=section.get("semantic_validation_timeout_sec", 45),
        )
        parsed = json.loads(response.choices[0].message.content)
        if parsed.get("decision") != "use_reconciled":
            best = _best(accepted)
            return [best], parsed
        replacements = parsed.get("replacements", [])
        by_rank = {int(item["rank"]): item for item in accepted}
        reconciled: list[dict[str, object]] = []
        for item in replacements:
            try:
                original = by_rank[int(item["rank"])]
            except Exception:
                continue
            updated = dict(original)
            updated["start_token_index"] = int(item["start_token_index"])
            updated["end_token_index"] = int(item["end_token_index"])
            updated["final_replacement_text"] = str(item["replacement_text"])
            updated["replacement_reason"] = "semantic_reconciled"
            reconciled.append(updated)
        if reconciled:
            return reconciled, parsed
        best = _best(accepted)
        return [best], parsed
    except Exception:
        logger.exception(
            "Word span reconciliation failed call_id=%s segment_index=%s",
            call_id,
            segment_index,
        )
        if not continue_on_error:
            raise
    best = _best(accepted)
    return [best], None


def _replace_span_in_segment_text(
    segment_text: str,
    start_token_index: int,
    end_token_index: int,
    replacement_text: str,
) -> str:
    tokens = _segment_tokens(segment_text)
    if not tokens or start_token_index < 0 or end_token_index >= len(tokens) or start_token_index > end_token_index:
        return segment_text
    start_offset = tokens[start_token_index].start_offset
    end_offset = tokens[end_token_index].end_offset
    return f"{segment_text[:start_offset]}{replacement_text}{segment_text[end_offset:]}"


def _apply_segment_retries(
    baseline: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    candidates: list[SegmentRetryCandidate],
    model_override: str | None = None,
) -> tuple[RawTranscript, list[dict[str, object]]]:
    merged_segments = list(baseline.segments)
    results: list[dict[str, object]] = []
    for candidate in candidates:
        logger.info(
            "Transcription segment retry decision call_id=%s rank=%s segment_index=%s start=%.2f end=%.2f action=retry reason=llm_confident_%s",
            _call_id(audio_path),
            candidate.rank,
            candidate.segment_index,
            candidate.start_sec,
            candidate.end_sec,
            candidate.llm_label,
        )
        span_audio_path = _extract_audio_span(audio_path, candidate, config)
        retry_transcript = _transcribe_segment_hebrew_retry(span_audio_path, config, model_override=model_override)
        retry_text = (retry_transcript.text or "").strip()
        replacement_applied, replacement_reason = _should_replace_with_hebrew_retry(
            candidate.baseline_text,
            retry_text,
            config=config,
        )
        if replacement_applied:
            merged_segments[candidate.segment_index] = RawSegment(
                start_sec=merged_segments[candidate.segment_index].start_sec,
                end_sec=merged_segments[candidate.segment_index].end_sec,
                text=retry_text,
                confidence=merged_segments[candidate.segment_index].confidence,
                speaker=merged_segments[candidate.segment_index].speaker,
            )
        logger.info(
            "Transcription segment retry result call_id=%s rank=%s segment_index=%s start=%.2f end=%.2f replacement_applied=%s replacement_reason=%s retry_text=%s",
            _call_id(audio_path),
            candidate.rank,
            candidate.segment_index,
            candidate.start_sec,
            candidate.end_sec,
            replacement_applied,
            replacement_reason,
            retry_text[:160],
        )
        results.append(
            {
                "segment_index": candidate.segment_index,
                "rank": candidate.rank,
                "score": round(candidate.score, 4),
                "start_sec": candidate.start_sec,
                "end_sec": candidate.end_sec,
                "duration_seconds": candidate.duration_seconds,
                "baseline_text": candidate.baseline_text,
                "context_before": candidate.context_before,
                "context_after": candidate.context_after,
                "llm_label": candidate.llm_label,
                "llm_reason": candidate.llm_reason,
                "llm_confidence": candidate.llm_confidence,
                "retry_text": retry_text,
                "retry_language": config.section("segment_detection").get("retry_language", "he"),
                "replacement_applied": replacement_applied,
                "replacement_reason": replacement_reason,
            }
        )
    text = " ".join(segment.text for segment in merged_segments).strip()
    scripts = _script_counts(text)
    language = baseline.language
    if scripts["hebrew"] and scripts["cyrillic"]:
        language = "mixed"
    merged = RawTranscript(
        provider=baseline.provider,
        model=baseline.model,
        language=language,
        confidence=baseline.confidence,
        segments=merged_segments,
        text=text,
    )
    return merged, results


def _collect_word_span_retry_results(
    audio_path: Path,
    config: AppConfig,
    candidates: list[WordRetryCandidate],
    model_override: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    results: list[dict[str, object]] = []
    validation_payloads: list[dict[str, object]] = []
    accepted_by_segment: dict[int, list[dict[str, object]]] = {}
    call_id = _call_id(audio_path)
    for candidate in candidates:
        logger.info(
            "Transcription word span retry decision call_id=%s rank=%s segment_index=%s span=%s start=%.2f end=%.2f action=retry reason=%s",
            call_id,
            candidate.rank,
            candidate.segment_index,
            candidate.span_text,
            candidate.start_sec,
            candidate.end_sec,
            candidate.llm_label,
        )
        span_audio_path = _extract_word_span_audio(audio_path, candidate, config)
        retry_transcript = _transcribe_word_span_hebrew_retry(span_audio_path, config, model_override=model_override)
        retry_text = (retry_transcript.text or "").strip()
        structural_ok, structural_reason = _should_replace_word_span(
            candidate.span_text,
            retry_text,
            config=config,
        )
        semantic_validation = WordSpanSemanticValidation(
            decision="reject",
            confidence=0.0,
            reason="Structural validation rejected retry text.",
            normalized_hebrew=None,
        )
        semantic_raw = None
        final_replacement_text = None
        replacement_reason = structural_reason
        if structural_ok:
            semantic_validation, semantic_raw = _validate_word_span_retry_with_llm(candidate, retry_text, config, call_id)
            semantic_validation = _rescue_semantic_validation_with_normalization(candidate, retry_text, semantic_validation)
            logger.info(
                "Transcription word span semantic validation call_id=%s segment_index=%s span=%s decision=%s confidence=%.2f reason=%s",
                call_id,
                candidate.segment_index,
                candidate.span_text,
                semantic_validation.decision,
                semantic_validation.confidence,
                semantic_validation.reason,
            )
            if semantic_raw is not None:
                validation_payloads.append(
                    {
                        "type": "semantic_validation",
                        "rank": candidate.rank,
                        "segment_index": candidate.segment_index,
                        "span_text": candidate.span_text,
                        "payload": semantic_raw,
                    }
                )
            semantic_threshold = float(config.section("segment_detection").get("semantic_min_confidence_to_accept", 0.75))
            if semantic_validation.decision in {"accept", "accept_with_normalization"} and semantic_validation.confidence >= semantic_threshold:
                final_replacement_text = (semantic_validation.normalized_hebrew or retry_text).strip()
                replacement_reason = (
                    "semantic_normalized"
                    if semantic_validation.decision == "accept_with_normalization"
                    else "semantic_validated"
                )
            elif semantic_validation.decision == "uncertain":
                replacement_reason = "semantic_uncertain_rejected"
            else:
                replacement_reason = "semantic_rejected"
        logger.info(
            "Transcription word span retry result call_id=%s rank=%s segment_index=%s span=%s replacement_applied=%s replacement_reason=%s retry_text=%s",
            call_id,
            candidate.rank,
            candidate.segment_index,
            candidate.span_text,
            bool(final_replacement_text),
            replacement_reason,
            retry_text[:160],
        )
        result = {
            "source_type": "word_span",
            "segment_index": candidate.segment_index,
            "rank": candidate.rank,
            "score": round(candidate.score, 4),
            "span_text": candidate.span_text,
            "start_token_index": candidate.start_token_index,
            "end_token_index": candidate.end_token_index,
            "start_sec": candidate.start_sec,
            "end_sec": candidate.end_sec,
            "baseline_segment_text": candidate.baseline_segment_text,
            "context_before": candidate.context_before,
            "context_after": candidate.context_after,
            "llm_label": candidate.llm_label,
            "llm_reason": candidate.llm_reason,
            "llm_confidence": candidate.llm_confidence,
            "retry_text": retry_text,
            "retry_language": config.section("segment_detection").get("retry_language", "he"),
            "semantic_decision": semantic_validation.decision,
            "semantic_confidence": semantic_validation.confidence,
            "semantic_reason": semantic_validation.reason,
            "normalized_hebrew": semantic_validation.normalized_hebrew,
            "final_replacement_text": final_replacement_text,
            "replacement_applied": False,
            "replacement_reason": replacement_reason,
        }
        results.append(result)
        if final_replacement_text:
            accepted_by_segment.setdefault(candidate.segment_index, []).append(result)
    return results, validation_payloads, accepted_by_segment


def _collect_clause_retry_results(
    audio_path: Path,
    config: AppConfig,
    candidates: list[ClauseRetryCandidate],
    model_override: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    results: list[dict[str, object]] = []
    validation_payloads: list[dict[str, object]] = []
    accepted_by_segment: dict[int, list[dict[str, object]]] = {}
    call_id = _call_id(audio_path)
    for candidate in candidates:
        logger.info(
            "Transcription clause retry candidate call_id=%s rank=%s segment_index=%s clause=%s score=%.2f start=%.2f end=%.2f",
            call_id,
            candidate.rank,
            candidate.segment_index,
            candidate.clause_text,
            candidate.score,
            candidate.start_sec,
            candidate.end_sec,
        )
        span_audio_path = _extract_clause_retry_audio(audio_path, candidate, config)
        retry_transcript = _transcribe_clause_hebrew_retry(span_audio_path, config, model_override=model_override)
        retry_text = (retry_transcript.text or "").strip()
        structural_ok, structural_reason = _should_replace_word_span(
            candidate.clause_text,
            retry_text,
            config=config,
        )
        semantic_validation = WordSpanSemanticValidation(
            decision="reject",
            confidence=0.0,
            reason="Structural validation rejected retry text.",
            normalized_hebrew=None,
        )
        semantic_raw = None
        final_replacement_text = None
        replacement_reason = structural_reason
        if structural_ok:
            semantic_validation, semantic_raw = _validate_clause_retry_with_llm(candidate, retry_text, config, call_id)
            semantic_validation = _rescue_semantic_validation_with_normalization(_clause_candidate_as_word_retry(candidate), retry_text, semantic_validation)
            logger.info(
                "Transcription clause retry semantic validation call_id=%s segment_index=%s clause=%s decision=%s confidence=%.2f reason=%s",
                call_id,
                candidate.segment_index,
                candidate.clause_text,
                semantic_validation.decision,
                semantic_validation.confidence,
                semantic_validation.reason,
            )
            if semantic_raw is not None:
                validation_payloads.append(
                    {
                        "type": "clause_semantic_validation",
                        "rank": candidate.rank,
                        "segment_index": candidate.segment_index,
                        "span_text": candidate.clause_text,
                        "payload": semantic_raw,
                    }
                )
            semantic_threshold = float(config.section("segment_detection").get("semantic_min_confidence_to_accept", 0.75))
            if semantic_validation.decision in {"accept", "accept_with_normalization"} and semantic_validation.confidence >= semantic_threshold:
                final_replacement_text = (semantic_validation.normalized_hebrew or retry_text).strip()
                replacement_reason = (
                    "semantic_normalized"
                    if semantic_validation.decision == "accept_with_normalization"
                    else "semantic_validated"
                )
            elif semantic_validation.decision == "uncertain":
                replacement_reason = "semantic_uncertain_rejected"
            else:
                replacement_reason = "semantic_rejected"
        logger.info(
            "Transcription clause retry result call_id=%s rank=%s segment_index=%s clause=%s replacement_applied=%s replacement_reason=%s retry_text=%s",
            call_id,
            candidate.rank,
            candidate.segment_index,
            candidate.clause_text,
            bool(final_replacement_text),
            replacement_reason,
            retry_text[:160],
        )
        result = {
            "source_type": "clause",
            "segment_index": candidate.segment_index,
            "rank": candidate.rank + 1000,
            "score": round(candidate.score, 4),
            "span_text": candidate.clause_text,
            "clause_text": candidate.clause_text,
            "start_token_index": candidate.start_token_index,
            "end_token_index": candidate.end_token_index,
            "start_sec": candidate.start_sec,
            "end_sec": candidate.end_sec,
            "baseline_segment_text": candidate.baseline_segment_text,
            "context_before": candidate.context_before,
            "context_after": candidate.context_after,
            "llm_label": "clause_retry",
            "llm_reason": candidate.source_reason,
            "llm_confidence": 1.0,
            "retry_text": retry_text,
            "retry_language": config.section("segment_detection").get("retry_language", "he"),
            "semantic_decision": semantic_validation.decision,
            "semantic_confidence": semantic_validation.confidence,
            "semantic_reason": semantic_validation.reason,
            "normalized_hebrew": semantic_validation.normalized_hebrew,
            "final_replacement_text": final_replacement_text,
            "replacement_applied": False,
            "replacement_reason": replacement_reason,
        }
        results.append(result)
        if final_replacement_text:
            accepted_by_segment.setdefault(candidate.segment_index, []).append(result)
    return results, validation_payloads, accepted_by_segment


def _reconcile_retry_results_into_transcript(
    baseline: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    accepted_by_segment: dict[int, list[dict[str, object]]],
    validation_payloads: list[dict[str, object]],
) -> tuple[RawTranscript, list[dict[str, object]]]:
    merged_segments = list(baseline.segments)
    call_id = _call_id(audio_path)
    for segment_index, replacements in accepted_by_segment.items():
        reconciled, reconcile_raw = _reconcile_segment_replacements_with_llm(
            call_id,
            segment_index,
            merged_segments[segment_index].text or "",
            str(replacements[0].get("context_before") or ""),
            str(replacements[0].get("context_after") or ""),
            replacements,
            config,
        )
        if reconcile_raw is not None:
            validation_payloads.append(
                {
                    "type": "segment_reconciliation",
                    "segment_index": segment_index,
                    "payload": reconcile_raw,
                }
            )
        selected_ranks = {int(item["rank"]) for item in reconciled}
        for item in replacements:
            rank = int(item["rank"])
            if rank in selected_ranks:
                item["replacement_applied"] = True
                reconciled_item = next(chosen for chosen in reconciled if int(chosen["rank"]) == rank)
                item["final_replacement_text"] = reconciled_item["final_replacement_text"]
                item["start_token_index"] = reconciled_item["start_token_index"]
                item["end_token_index"] = reconciled_item["end_token_index"]
                item["replacement_reason"] = reconciled_item.get("replacement_reason", item["replacement_reason"])
            else:
                item["replacement_applied"] = False
                item["replacement_reason"] = "reconciliation_rejected"
        logger.info(
            "Transcription segment replacement reconciliation call_id=%s segment_index=%s decision=%s replacements=%s",
            call_id,
            segment_index,
            "use_reconciled" if len(reconciled) != len(replacements) or any(item.get("replacement_reason") == "semantic_reconciled" for item in reconciled) else "keep_original",
            len(reconciled),
        )
        updated_text = merged_segments[segment_index].text or ""
        for item in sorted(reconciled, key=lambda value: (-int(value["start_token_index"]), -int(value["end_token_index"]))):
            updated_text = _replace_span_in_segment_text(
                updated_text,
                int(item["start_token_index"]),
                int(item["end_token_index"]),
                str(item["final_replacement_text"]),
            )
        merged_segments[segment_index] = RawSegment(
            start_sec=merged_segments[segment_index].start_sec,
            end_sec=merged_segments[segment_index].end_sec,
            text=updated_text,
            confidence=merged_segments[segment_index].confidence,
            speaker=merged_segments[segment_index].speaker,
        )
    text = " ".join(segment.text for segment in merged_segments).strip()
    scripts = _script_counts(text)
    language = baseline.language
    if scripts["hebrew"] and scripts["cyrillic"]:
        language = "mixed"
    return RawTranscript(
        provider=baseline.provider,
        model=baseline.model,
        language=language,
        confidence=baseline.confidence,
        segments=merged_segments,
        text=text,
    ), validation_payloads


def _apply_word_span_retries(
    baseline: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    candidates: list[WordRetryCandidate],
    model_override: str | None = None,
) -> tuple[RawTranscript, list[dict[str, object]], list[dict[str, object]]]:
    results, validation_payloads, accepted_by_segment = _collect_word_span_retry_results(
        audio_path,
        config,
        candidates,
        model_override=model_override,
    )
    merged, validation_payloads = _reconcile_retry_results_into_transcript(
        baseline,
        audio_path,
        config,
        accepted_by_segment,
        validation_payloads,
    )
    return merged, results, validation_payloads


def _apply_combined_hebrew_retries(
    baseline: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    word_candidates: list[WordRetryCandidate],
    clause_candidates: list[ClauseRetryCandidate],
    model_override: str | None = None,
) -> tuple[RawTranscript, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    word_results, word_validation_payloads, word_accepted = _collect_word_span_retry_results(
        audio_path,
        config,
        word_candidates,
        model_override=model_override,
    )
    clause_results, clause_validation_payloads, clause_accepted = _collect_clause_retry_results(
        audio_path,
        config,
        clause_candidates,
        model_override=model_override,
    )
    combined_accepted = {segment_index: list(items) for segment_index, items in word_accepted.items()}
    for segment_index, items in clause_accepted.items():
        combined_accepted.setdefault(segment_index, []).extend(items)
    merged, validation_payloads = _reconcile_retry_results_into_transcript(
        baseline,
        audio_path,
        config,
        combined_accepted,
        word_validation_payloads + clause_validation_payloads,
    )
    return merged, word_results, clause_results, validation_payloads


def merge_transcript_candidates(baseline: RawTranscript, forced_hebrew: RawTranscript) -> RawTranscript:
    segments: list[RawSegment] = []
    consumed_forced: set[int] = set()
    for base_segment in baseline.segments:
        best_index = None
        best_candidate = None
        for index, forced_segment in enumerate(forced_hebrew.segments):
            if index in consumed_forced:
                continue
            if abs(forced_segment.start_sec - base_segment.start_sec) <= 2.0 or (
                forced_segment.start_sec <= base_segment.end_sec and forced_segment.end_sec >= base_segment.start_sec
            ):
                best_index = index
                best_candidate = forced_segment
                break
        if best_candidate is None:
            segments.append(base_segment)
            continue
        if _segment_prefers_forced_hebrew(base_segment, best_candidate):
            segments.append(best_candidate)
        else:
            segments.append(base_segment)
        consumed_forced.add(best_index)

    for index, forced_segment in enumerate(forced_hebrew.segments):
        if index not in consumed_forced:
            segments.append(forced_segment)

    segments.sort(key=lambda item: (item.start_sec, item.end_sec))
    text = " ".join(segment.text for segment in segments).strip()
    scripts = _script_counts(text)
    merged_language = forced_hebrew.language or baseline.language
    if scripts["hebrew"] and scripts["cyrillic"]:
        merged_language = "mixed"
    return RawTranscript(
        provider=forced_hebrew.provider,
        model=forced_hebrew.model,
        language=merged_language,
        confidence=None,
        segments=segments,
        text=text,
    )


def compare_transcript_candidates(
    baseline: RawTranscript,
    forced_hebrew: RawTranscript,
    audio_path: Path,
    merge_enabled: bool = True,
    config: AppConfig | None = None,
) -> TranscriptCandidateComparison:
    baseline_assessment = assess_transcript_quality(baseline, audio_path, config=config)
    forced_assessment = assess_transcript_quality(forced_hebrew, audio_path, expected_language="he", config=config)
    score_margin = 0.12
    quality_notes: list[str] = []
    merged = merge_transcript_candidates(baseline, forced_hebrew) if merge_enabled else None
    merged_assessment = assess_transcript_quality(merged, audio_path, config=config) if merged is not None else None
    logger.info(
        "Transcription candidate scores audio=%s baseline_language=%s baseline_score=%.2f baseline_preview=%s forced_language=%s forced_score=%.2f forced_preview=%s merged_score=%s merged_preview=%s merge_enabled=%s",
        audio_path,
        baseline.language,
        baseline_assessment.score,
        _transcript_log_preview(baseline),
        forced_hebrew.language,
        forced_assessment.score,
        _transcript_log_preview(forced_hebrew),
        f"{merged_assessment.score:.2f}" if merged_assessment is not None else "-",
        _transcript_log_preview(merged),
        merge_enabled,
    )

    if (
        merged is not None
        and merged_assessment is not None
        and baseline_assessment.suspect_language_confusion
        and merged.text != baseline.text
        and merged_assessment.score >= baseline_assessment.score + 0.10
        and merged_assessment.score >= forced_assessment.score - 0.10
    ):
        quality_notes.append("Merged transcript kept baseline Russian structure and replaced suspicious Hebrew-transliterated segments.")
        return TranscriptCandidateComparison(
            baseline=baseline,
            forced_hebrew=forced_hebrew,
            merged=merged,
            winner="merged_segments",
            strategy="merged_segments",
            quality_notes=quality_notes,
            selected=merged,
            baseline_assessment=baseline_assessment,
            forced_hebrew_assessment=forced_assessment,
            merged_assessment=merged_assessment,
            low_confidence_reason=merged_assessment.low_confidence_reason,
        )

    if forced_assessment.score > baseline_assessment.score + score_margin:
        quality_notes.append("Forced Hebrew candidate scored higher than baseline.")
        return TranscriptCandidateComparison(
            baseline=baseline,
            forced_hebrew=forced_hebrew,
            merged=merged,
            winner="forced_hebrew",
            strategy="forced_hebrew",
            quality_notes=quality_notes,
            selected=forced_hebrew,
            baseline_assessment=baseline_assessment,
            forced_hebrew_assessment=forced_assessment,
            merged_assessment=merged_assessment,
            low_confidence_reason=forced_assessment.low_confidence_reason,
        )
    if baseline_assessment.score > forced_assessment.score + score_margin:
        quality_notes.append("Baseline candidate remained stronger than forced Hebrew.")
        return TranscriptCandidateComparison(
            baseline=baseline,
            forced_hebrew=forced_hebrew,
            merged=merged,
            winner="baseline",
            strategy="baseline",
            quality_notes=quality_notes,
            selected=baseline,
            baseline_assessment=baseline_assessment,
            forced_hebrew_assessment=forced_assessment,
            merged_assessment=merged_assessment,
            low_confidence_reason=baseline_assessment.low_confidence_reason,
        )
    if merge_enabled and merged is not None and merged_assessment is not None:
        if merged_assessment.score >= max(baseline_assessment.score, forced_assessment.score):
            quality_notes.append("Merged segments from baseline and forced Hebrew candidates.")
            return TranscriptCandidateComparison(
                baseline=baseline,
                forced_hebrew=forced_hebrew,
                merged=merged,
                winner="merged_segments",
                strategy="merged_segments",
                quality_notes=quality_notes,
                selected=merged,
                baseline_assessment=baseline_assessment,
                forced_hebrew_assessment=forced_assessment,
                merged_assessment=merged_assessment,
                low_confidence_reason=merged_assessment.low_confidence_reason,
            )
    quality_notes.append("Candidates were close; selected forced Hebrew as safer multilingual recovery.")
    return TranscriptCandidateComparison(
        baseline=baseline,
        forced_hebrew=forced_hebrew,
        merged=merged,
        winner="forced_hebrew",
        strategy="forced_hebrew",
        quality_notes=quality_notes,
        selected=forced_hebrew,
        baseline_assessment=baseline_assessment,
        forced_hebrew_assessment=forced_assessment,
        merged_assessment=merged_assessment,
        low_confidence_reason=forced_assessment.low_confidence_reason,
    )


def _candidate_summary(raw: RawTranscript, assessment: TranscriptQualityAssessment, language_override: str | None = None) -> dict:
    return {
        "provider": raw.provider,
        "model": raw.model,
        "language": raw.language,
        "language_override": language_override,
        "segment_count": len(raw.segments),
        "text": raw.text,
        "quality_score": assessment.score,
        "heuristic_quality_score": assessment.heuristic_score,
        "llm_quality_score": assessment.llm_score,
        "quality_flags": assessment.flags,
        "llm_quality_flags": assessment.llm_flags,
        "llm_quality_reason": assessment.llm_reason,
        "suspect_language_confusion": assessment.suspect_language_confusion,
        "suspicious_token_count": assessment.suspicious_token_count,
        "suspicious_token_examples": assessment.suspicious_token_examples,
        "suspicious_segment_count": assessment.suspicious_segment_count,
        "suspicious_segment_examples": assessment.suspicious_segment_examples,
        "suggested_retry_languages": assessment.suggested_retry_languages,
    }


def _write_candidate_artifact(
    audio_path: Path,
    comparison: TranscriptCandidateComparison,
    retried_languages: list[str],
    segment_detection: dict[str, object] | None = None,
    segment_retries: list[dict[str, object]] | None = None,
    clause_detection: dict[str, object] | None = None,
    clause_retries: list[dict[str, object]] | None = None,
    word_span_detection: dict[str, object] | None = None,
    word_span_retries: list[dict[str, object]] | None = None,
    segment_detection_llm_response: dict[str, object] | None = None,
    word_span_detection_llm_response: dict[str, object] | None = None,
    word_span_validation_llm_response: list[dict[str, object]] | None = None,
    full_call_hebrew_decision: dict[str, object] | None = None,
) -> None:
    selected_assessment = comparison.baseline_assessment
    if comparison.winner == "forced_hebrew" and comparison.forced_hebrew_assessment:
        selected_assessment = comparison.forced_hebrew_assessment
    elif comparison.winner == "merged_segments" and comparison.merged_assessment:
        selected_assessment = comparison.merged_assessment
    write_json(
        audio_path.parent / TRANSCRIPTION_CANDIDATES_FILENAME,
        {
            "baseline": _candidate_summary(comparison.baseline, comparison.baseline_assessment),
            "forced_hebrew": (
                _candidate_summary(comparison.forced_hebrew, comparison.forced_hebrew_assessment, "he")
                if comparison.forced_hebrew and comparison.forced_hebrew_assessment
                else None
            ),
            "merged": (
                _candidate_summary(comparison.merged, comparison.merged_assessment)
                if comparison.merged and comparison.merged_assessment
                else None
            ),
            "segment_detection": segment_detection or {},
            "segment_retries": segment_retries or [],
            "clause_detection": clause_detection or {},
            "clause_retries": clause_retries or [],
            "segment_detection_llm_response": segment_detection_llm_response or {},
            "word_span_detection": word_span_detection or {},
            "word_span_retries": word_span_retries or [],
            "word_span_detection_llm_response": word_span_detection_llm_response or {},
            "word_span_validation_llm_response": word_span_validation_llm_response or [],
            "full_call_hebrew_decision": full_call_hebrew_decision or {},
            "selection": {
                "winner": comparison.winner,
                "strategy": comparison.strategy,
                "quality_notes": comparison.quality_notes,
                "retried_languages": retried_languages,
                "quality_flags": selected_assessment.flags,
                "quality_score": selected_assessment.score,
                "heuristic_quality_score": selected_assessment.heuristic_score,
                "llm_quality_score": selected_assessment.llm_score,
                "llm_quality_flags": selected_assessment.llm_flags,
                "llm_quality_reason": selected_assessment.llm_reason,
                "low_confidence_reason": comparison.low_confidence_reason,
            },
        },
    )


def load_transcription_candidate_selection(call_dir: Path) -> dict:
    return read_json(call_dir / TRANSCRIPTION_CANDIDATES_FILENAME, default={})


def _segment_detection_decisions(raw: RawTranscript, audio_path: Path, config: AppConfig) -> list[SegmentDetectionDecision]:
    decisions, _ = _segment_detection_decisions_with_raw(raw, audio_path, config)
    return decisions


def _segment_detection_decisions_with_raw(
    raw: RawTranscript, audio_path: Path, config: AppConfig
) -> tuple[list[SegmentDetectionDecision], dict[str, object] | None]:
    section = config.section("segment_detection")
    if not section.get("enabled", True):
        return _heuristic_segment_detection(raw, config), None
    if not section.get("run_on_every_call", True):
        return _heuristic_segment_detection(raw, config), None
    provider = section.get("provider", "openai")
    continue_on_error = bool(section.get("continue_on_error", True))
    try:
        if provider == "openai":
            return _classify_segments_with_llm(raw, audio_path, config)
    except Exception:
        logger.exception("Segment detection provider %s failed for %s", provider, audio_path)
        if not continue_on_error:
            raise
    return _heuristic_segment_detection(raw, config), None


def _clause_candidate_diagnostics(candidates: list[ClauseRetryCandidate]) -> list[dict[str, object]]:
    return [
        {
            "segment_index": item.segment_index,
            "clause_text": item.clause_text,
            "score": round(item.score, 4),
            "status": "selected",
            "source_reason": item.source_reason,
            "start_token_index": item.start_token_index,
            "end_token_index": item.end_token_index,
        }
        for item in candidates
    ]


def _run_post_baseline_recovery(
    transcript: RawTranscript,
    audio_path: Path,
    config: AppConfig,
    *,
    provider: str,
    model_override: str | None = None,
    language_override: str | None = None,
    retry_enabled: bool = True,
    retry_languages: list[str] | None = None,
    merge_enabled: bool = True,
    full_call_retry: Callable[[], RawTranscript] | None = None,
) -> PostBaselineRecoveryResult:
    segment_detection_section = config.section("segment_detection")
    retry_languages = retry_languages or ["he"]
    baseline_assessment = assess_transcript_quality(transcript, audio_path, config=config)
    logger.info(
        "Transcription quality assessment audio=%s provider=%s score=%.2f flags=%s suspect_language_confusion=%s suspicious_token_count=%s suspicious_token_examples=%s suspicious_segment_count=%s suspicious_segment_examples=%s suggested_retry_languages=%s",
        audio_path,
        provider,
        baseline_assessment.score,
        ",".join(baseline_assessment.flags) or "-",
        baseline_assessment.suspect_language_confusion,
        baseline_assessment.suspicious_token_count,
        ",".join(baseline_assessment.suspicious_token_examples) or "-",
        baseline_assessment.suspicious_segment_count,
        " | ".join(baseline_assessment.suspicious_segment_examples) or "-",
        ",".join(baseline_assessment.suggested_retry_languages) or "-",
    )
    decisions, segment_detection_llm_response = _segment_detection_decisions_with_raw(transcript, audio_path, config)
    candidates, below_threshold_count = _rank_segment_retry_candidates(transcript, decisions, config)
    suspicious_count = sum(1 for item in decisions if item.suspicious)
    threshold_used = float(segment_detection_section.get("min_confidence_to_retry", 0.70))
    logger.info(
        "Transcription segment detection call_id=%s provider=%s model=%s evaluated=%s suspicious=%s selected=%s below_threshold=%s",
        _call_id(audio_path),
        segment_detection_section.get("provider", "heuristic"),
        segment_detection_section.get("model", "heuristic"),
        len(decisions),
        suspicious_count,
        len(candidates),
        below_threshold_count,
    )
    if segment_detection_llm_response is not None:
        logger.info(
            "Transcription segment detection raw_response call_id=%s payload=%s",
            _call_id(audio_path),
            json.dumps(segment_detection_llm_response, ensure_ascii=False),
        )
    for decision in decisions:
        if (
            decision.suspicious
            and decision.retry_language == segment_detection_section.get("retry_language", "he")
            and decision.confidence < threshold_used
        ):
            logger.info(
                "Transcription suspicious segment skipped call_id=%s segment_index=%s confidence=%.2f threshold=%.2f reason=below_retry_threshold",
                _call_id(audio_path),
                decision.segment_index,
                decision.confidence,
                threshold_used,
            )
    for candidate in candidates:
        logger.info(
            "Transcription suspicious segment call_id=%s rank=%s score=%.2f start=%.2f end=%.2f duration=%.2f text=%s reason=%s",
            _call_id(audio_path),
            candidate.rank,
            candidate.score,
            candidate.start_sec,
            candidate.end_sec,
            candidate.duration_seconds,
            candidate.baseline_text[:160],
            candidate.llm_reason,
        )
    segment_detection_summary = _segment_detection_summary(decisions, candidates, below_threshold_count, config)
    word_span_candidates = _build_word_span_candidates(transcript, decisions, config)
    word_span_decisions, word_span_detection_llm_response = _word_span_detection_decisions_with_raw(word_span_candidates, config)
    word_span_score_map = _word_span_candidate_score_map(word_span_candidates, word_span_decisions)
    word_retry_candidates, word_below_threshold_count = _rank_word_retry_candidates(word_span_candidates, word_span_decisions, config)
    logger.info(
        "Transcription word span detection call_id=%s provider=%s model=%s segments=%s spans_detected=%s selected=%s below_threshold=%s",
        _call_id(audio_path),
        segment_detection_section.get("provider", "heuristic"),
        segment_detection_section.get("model", "heuristic"),
        len({item.segment_index for item in word_span_decisions}),
        sum(1 for item in word_span_decisions if item.suspicious),
        len(word_retry_candidates),
        word_below_threshold_count,
    )
    if word_span_detection_llm_response is not None:
        logger.info(
            "Transcription word span detection raw_response call_id=%s payload=%s",
            _call_id(audio_path),
            json.dumps(word_span_detection_llm_response, ensure_ascii=False),
        )
    for decision in word_span_decisions:
        if decision.suspicious and decision.confidence < threshold_used:
            score = word_span_score_map.get((decision.segment_index, decision.start_token_index, decision.end_token_index), 0.0)
            logger.info(
                "Transcription suspicious word span skipped call_id=%s segment_index=%s span=%s score=%.2f confidence=%.2f threshold=%.2f reason=below_retry_threshold",
                _call_id(audio_path),
                decision.segment_index,
                decision.span_text,
                score,
                decision.confidence,
                threshold_used,
            )
    for candidate in word_retry_candidates:
        logger.info(
            "Transcription suspicious word span call_id=%s rank=%s segment_index=%s span=%s score=%.2f confidence=%.2f start=%.2f end=%.2f reason=%s",
            _call_id(audio_path),
            candidate.rank,
            candidate.segment_index,
            candidate.span_text,
            candidate.score,
            candidate.llm_confidence,
            candidate.start_sec,
            candidate.end_sec,
            candidate.llm_reason,
        )
    word_span_detection_summary = _word_span_detection_summary(word_span_decisions, word_retry_candidates, word_below_threshold_count, config)
    clause_retry_candidates = _build_clause_retry_candidates(transcript, decisions, config)
    clause_detection_summary = {
        "segments_evaluated": len(decisions),
        "clause_candidates_detected": len(clause_retry_candidates),
        "selected_for_retry": len(clause_retry_candidates),
        "candidates": _clause_candidate_diagnostics(clause_retry_candidates),
    }
    full_call_hebrew_decision = _full_call_hebrew_retry_decision(
        baseline_assessment,
        decisions,
        word_retry_candidates,
        clause_retry_candidates,
        config,
    )
    word_span_retry_results: list[dict[str, object]] = []
    clause_retry_results: list[dict[str, object]] = []
    word_span_validation_llm_response: list[dict[str, object]] = []
    retried_languages = _language_retry_history(transcript)
    comparison = TranscriptCandidateComparison(
        baseline=transcript,
        forced_hebrew=None,
        merged=None,
        winner="baseline",
        strategy="baseline",
        quality_notes=["Baseline candidate accepted without retry."],
        selected=transcript,
        baseline_assessment=baseline_assessment,
        forced_hebrew_assessment=None,
        merged_assessment=None,
        low_confidence_reason=baseline_assessment.low_confidence_reason,
    )
    full_call_hebrew_escalated = False
    full_call_hebrew_selected = False
    if retry_enabled and (word_retry_candidates or clause_retry_candidates) and "he" in retry_languages and language_override != "he":
        merged_transcript, word_span_retry_results, clause_retry_results, word_span_validation_llm_response = _apply_combined_hebrew_retries(
            transcript,
            audio_path,
            config,
            word_retry_candidates,
            clause_retry_candidates,
            model_override,
        )
        merged_assessment = assess_transcript_quality(merged_transcript, audio_path, config=config)
        replaced_count = sum(1 for item in word_span_retry_results if item.get("replacement_applied")) + sum(
            1 for item in clause_retry_results if item.get("replacement_applied")
        )
        merged_has_hebrew = _script_counts(merged_transcript.text).get("hebrew", 0) > 0
        if merged_transcript.text != transcript.text and (
            merged_assessment.score >= baseline_assessment.score - 0.05 or (replaced_count > 0 and merged_has_hebrew)
        ):
            comparison = TranscriptCandidateComparison(
                baseline=transcript,
                forced_hebrew=None,
                merged=merged_transcript,
                winner="merged_segments",
                strategy="llm_word_span_hebrew_recovery",
                quality_notes=["Merged transcript kept baseline structure and replaced LLM-flagged suspicious Hebrew spans and clauses."],
                selected=merged_transcript,
                baseline_assessment=baseline_assessment,
                forced_hebrew_assessment=None,
                merged_assessment=merged_assessment,
                low_confidence_reason=merged_assessment.low_confidence_reason,
            )
        retried_languages = _merge_retry_languages(retried_languages, ["he"])
    if (
        comparison.winner == "baseline"
        and retry_enabled
        and "he" in retry_languages
        and language_override != "he"
        and bool(full_call_hebrew_decision["should_retry"])
        and full_call_retry is not None
    ):
        full_call_hebrew_escalated = True
        logger.info(
            "Transcription full-call Hebrew escalation call_id=%s suspicious=%s suspicious_segments=%s word_candidates=%s clause_candidates=%s baseline_score=%.2f",
            _call_id(audio_path),
            suspicious_count,
            baseline_assessment.suspicious_segment_count,
            len(word_retry_candidates),
            len(clause_retry_candidates),
            baseline_assessment.score,
        )
        logger.info("Transcription retry audio=%s provider=%s retry_language=he", audio_path, provider)
        forced_hebrew = full_call_retry()
        comparison = compare_transcript_candidates(
            transcript,
            forced_hebrew,
            audio_path,
            merge_enabled=merge_enabled,
            config=config,
        )
        full_call_hebrew_selected = comparison.winner in {"forced_hebrew", "merged_segments"}
        retried_languages = _merge_retry_languages(retried_languages, ["he"])
    elif comparison.winner == "baseline":
        comparison = TranscriptCandidateComparison(
            baseline=transcript,
            forced_hebrew=None,
            merged=None,
            winner="baseline",
            strategy="baseline",
            quality_notes=(
                ["LLM detected suspicious segments, but no Hebrew retries were accepted; baseline preserved."]
                if suspicious_count > 0 and not word_retry_candidates and not clause_retry_candidates
                else ["LLM detected suspicious segments, but Hebrew retries did not improve the baseline; baseline preserved."]
                if suspicious_count > 0
                else ["Baseline candidate accepted without retry."]
            ),
            selected=transcript,
            baseline_assessment=baseline_assessment,
            forced_hebrew_assessment=None,
            merged_assessment=None,
            low_confidence_reason=baseline_assessment.low_confidence_reason,
        )
    retried_languages = _merge_retry_languages(retried_languages, _language_retry_history(comparison.selected))
    return PostBaselineRecoveryResult(
        selected=comparison.selected,
        strategy=comparison.strategy,
        comparison=comparison,
        baseline_assessment=baseline_assessment,
        retried_languages=retried_languages,
        segment_decisions=decisions,
        segment_detection_summary=segment_detection_summary,
        segment_detection_llm_response=segment_detection_llm_response,
        word_span_candidates=word_span_candidates,
        word_span_decisions=word_span_decisions,
        word_span_detection_summary=word_span_detection_summary,
        word_span_detection_llm_response=word_span_detection_llm_response,
        word_retry_candidates=word_retry_candidates,
        word_span_retry_results=word_span_retry_results,
        clause_retry_candidates=clause_retry_candidates,
        clause_detection_summary=clause_detection_summary,
        clause_retry_results=clause_retry_results,
        word_span_validation_llm_response=word_span_validation_llm_response,
        full_call_hebrew_escalated=full_call_hebrew_escalated,
        full_call_hebrew_selected=full_call_hebrew_selected,
        full_call_hebrew_decision=full_call_hebrew_decision,
    )


def transcribe(audio_path: Path, config: AppConfig) -> RawTranscript:
    errors: list[str] = []
    default_provider, model_override, language_override, language_mode = _transcription_preferences(audio_path, config)
    backend = str(config.section("transcription").get("backend", "whisper_legacy")).strip().lower()
    fallback_provider = config.section("transcription").get("provider_fallback")
    retry_enabled = bool(config.section("transcription").get("quality_retry_enabled", True))
    retry_languages = list(config.section("transcription").get("retry_languages_on_suspicion", ["he"]))
    merge_enabled = bool(config.section("transcription").get("candidate_merge_enabled", True))
    providers = [default_provider]
    if fallback_provider and fallback_provider != default_provider:
        providers.append(fallback_provider)
    for provider in providers:
        try:
            if provider == "local":
                if backend in {"vad_chunked_legacy", "faster_whisper_vad"}:
                    transcript = _transcribe_vad_chunked(
                        audio_path,
                        config,
                        provider="local",
                        model_override=model_override if default_provider == "local" else None,
                        language_override=language_override,
                        language_mode=language_mode,
                    )
                else:
                    transcript = _transcribe_local(
                        audio_path,
                        config,
                        model_override if default_provider == "local" else None,
                        language_override=language_override,
                        language_mode=language_mode,
                    )
                transcript = _maybe_run_language_retries(
                    transcript,
                    audio_path,
                    config,
                    provider,
                    model_override if default_provider == "local" else None,
                    language_mode,
                )
                transcript = _maybe_force_hebrew_transcript(
                    transcript,
                    audio_path,
                    config,
                    provider,
                    model_override if default_provider == "local" else None,
                    language_mode,
                )
                recovery = _run_post_baseline_recovery(
                    transcript,
                    audio_path,
                    config,
                    provider=provider,
                    model_override=model_override if default_provider == "local" else None,
                    language_override=language_override,
                    retry_enabled=retry_enabled,
                    retry_languages=retry_languages,
                    merge_enabled=merge_enabled,
                    full_call_retry=lambda: _transcribe_local(
                        audio_path,
                        config,
                        model_override if default_provider == "local" else None,
                        "he",
                        "metadata_override",
                    ),
                )
                _write_candidate_artifact(
                    audio_path,
                    recovery.comparison,
                    recovery.retried_languages,
                    recovery.segment_detection_summary,
                    [],
                    recovery.clause_detection_summary,
                    recovery.clause_retry_results,
                    recovery.word_span_detection_summary,
                    recovery.word_span_retry_results,
                    recovery.segment_detection_llm_response,
                    recovery.word_span_detection_llm_response,
                    recovery.word_span_validation_llm_response,
                    recovery.full_call_hebrew_decision,
                )
                if recovery.strategy != "baseline":
                    logger.info(
                        "Transcription selection audio=%s winner=%s strategy=%s",
                        audio_path,
                        recovery.comparison.winner,
                        recovery.comparison.strategy,
                    )
                    return recovery.selected
                if (
                    provider == default_provider
                    and fallback_provider == "cloud"
                    and config.section("transcription").get("cloud_enabled", False)
                    and recovery.baseline_assessment.suspect_language_confusion
                ):
                    logger.info(
                        "Transcription falling back to cloud audio=%s fallback_reason=mixed_language_suspected baseline_score=%.2f baseline_preview=%s",
                        audio_path,
                        recovery.baseline_assessment.score,
                        _transcript_log_preview(transcript),
                    )
                    continue
                return transcript
            if provider == "cloud" and config.section("transcription").get("cloud_enabled", False):
                logger.info(
                    "Transcription falling back to cloud audio=%s previous_errors=%s",
                    audio_path,
                    " | ".join(errors) or "-",
                )
                cloud_override = model_override if default_provider == "cloud" else None
                if backend in {"vad_chunked_legacy", "faster_whisper_vad"}:
                    transcript = _transcribe_vad_chunked(
                        audio_path,
                        config,
                        provider="cloud",
                        model_override=cloud_override,
                        language_override=language_override,
                        language_mode=language_mode,
                    )
                else:
                    transcript = _transcribe_cloud(audio_path, config, cloud_override, language_override, language_mode)
                transcript = _maybe_run_language_retries(
                    transcript,
                    audio_path,
                    config,
                    provider,
                    cloud_override,
                    language_mode,
                )
                transcript = _maybe_force_hebrew_transcript(
                    transcript,
                    audio_path,
                    config,
                    provider,
                    cloud_override,
                    language_mode,
                )
                recovery = _run_post_baseline_recovery(
                    transcript,
                    audio_path,
                    config,
                    provider=provider,
                    model_override=None,
                    language_override=language_override,
                    retry_enabled=retry_enabled,
                    retry_languages=retry_languages,
                    merge_enabled=merge_enabled,
                    full_call_retry=lambda: _transcribe_cloud(audio_path, config, cloud_override, "he", "metadata_override"),
                )
                _write_candidate_artifact(
                    audio_path,
                    recovery.comparison,
                    recovery.retried_languages,
                    recovery.segment_detection_summary,
                    [],
                    recovery.clause_detection_summary,
                    recovery.clause_retry_results,
                    recovery.word_span_detection_summary,
                    recovery.word_span_retry_results,
                    recovery.segment_detection_llm_response,
                    recovery.word_span_detection_llm_response,
                    recovery.word_span_validation_llm_response,
                    recovery.full_call_hebrew_decision,
                )
                if recovery.strategy != "baseline":
                    logger.info(
                        "Transcription selection audio=%s winner=%s strategy=%s",
                        audio_path,
                        recovery.comparison.winner,
                        recovery.comparison.strategy,
                    )
                    return recovery.selected
                return transcript
        except Exception as exc:
            logger.exception("Transcription provider %s failed", provider)
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("Transcription failed; " + "; ".join(errors))
