from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawTranscript
from call_assistant.transcription.service import (
    _run_post_baseline_recovery,
    _transcribe_cloud,
    _transcribe_local,
    _transcribe_vad_chunked,
    _word_span_candidate_score_map,
)


DEFAULT_EXPECTED_TRANSCRIPT = """
[00:00] speaker_1 Unassigned
Алло.
[00:01] speaker_2 Unassigned
Жорик, ищи хорошо, потому что я сейчас прошелся по мейлам,
[00:07] speaker_2 Unassigned
я нашел файл, который лежал  בתור תכנון מול ביצוע
[00:13] speaker_2 Unassigned
Это, оказывается, файл Лероновский, который она послала в начале проекта.
[00:18] speaker_2 Unassigned
И там это не תקציב מול ביצוע капсулы 250, это больше תקציב 500,
[00:26] speaker_2 Unassigned
и потом его התאמה לתקציב капсулы 250.
[00:30] speaker_2 Unassigned
То есть это не совсем то, что...
[00:33] speaker_1 Unassigned
Не, должно быть что-то более нормальное.
[00:36] speaker_2 Unassigned
Вот, у меня это...
[00:38] speaker_2 Unassigned
Это, видимо, в сетке не лежало тогда.
[00:40] speaker_1 Unassigned
Угу.
[00:41] speaker_2 Unassigned
Так что, да, прищи, пожалуйста, будет очень интересно глянуть.
[00:44] speaker_1 Unassigned
Окей, давай.
""".strip()

DEFAULT_CALL_DIR = Path("/home/garik/CallAssistantData/calls/2026/04/04/call_20260404_164325_c124e9")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "reports"
DEFAULT_MANIFEST = ROOT / "data" / "eval" / "post_russian_recovery" / "manifest.real.json"

SPEAKER_LINE_RE = re.compile(r"^\[\d{2}:\d{2}\]\s+speaker_[^\n]+$", re.IGNORECASE)
TIMESTAMP_PREFIX_RE = re.compile(r"^\[\d{2}:\d{2}\]\s+speaker_[^\n]+\s*")
PUNCT_RE = re.compile(r"[^\w\u0400-\u04FF\u0590-\u05FF]+", re.UNICODE)
CRITICAL_EXPECTED_PHRASES = [
    "בתור תכנון מול ביצוע",
    "תקציב מול ביצוע",
    "תקציב 500",
    "התאמה לתקציב",
    "интересно глянуть",
]
TURN_LINE_RE = re.compile(r"^(speaker_[^:]+):\s*(.+)$", re.IGNORECASE)


@dataclass
class VariantSpec:
    name: str
    overrides: dict[str, Any]


@dataclass
class MatchScores:
    char_ratio: float
    token_precision: float
    token_recall: float
    token_f1: float
    critical_phrase_precision: float
    critical_phrase_recall: float
    critical_phrase_f1: float
    composite: float


@dataclass
class VariantResult:
    name: str
    overrides: dict[str, Any]
    elapsed_seconds: float
    selected_strategy: str
    selected_text: str
    scores: MatchScores
    suspicious_segments: int
    suspicious_word_spans: int
    selected_word_spans: int
    suspicious_clause_candidates: int
    selected_clause_candidates: int
    applied_replacements: int
    applied_clause_replacements: int
    selection_notes: list[str]
    clause_diagnostics: list[dict[str, Any]]
    candidate_diagnostics: list[dict[str, Any]]
    word_span_retries: list[dict[str, Any]]
    clause_retry_results: list[dict[str, Any]]
    full_call_hebrew_escalated: bool
    full_call_hebrew_selected: bool
    full_call_hebrew_decision: dict[str, Any]
    transcription_backend_attempted: str | None
    transcription_backend_selected: str | None
    transcription_backend_fallback_reason: str | None
    chunked_plausibility_score: float | None
    chunked_plausibility_flags: list[str]


@dataclass
class EvaluationManifestEntry:
    call_id: str
    expected_transcript_path: Path | None = None
    expected_case_path: Path | None = None
    call_dir: Path | None = None
    description: str = ""
    tags: list[str] | None = None
    priority_weight: float = 1.0


@dataclass
class CallEvaluationResult:
    entry: EvaluationManifestEntry
    baseline: RawTranscript | None
    expected_text: str | None
    variants: list[VariantResult]
    error: str | None = None


@dataclass
class AggregateVariantResult:
    name: str
    success_count: int
    failure_count: int
    weighted_composite: float
    weighted_char_ratio: float
    weighted_token_f1: float
    weighted_critical_phrase_f1: float
    weighted_delta_composite_vs_baseline: float
    weighted_delta_phrase_f1_vs_baseline: float
    regression_flagged_calls: int
    fallback_count: int


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_expected_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if SPEAKER_LINE_RE.match(line):
            continue
        line = TIMESTAMP_PREFIX_RE.sub("", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def normalize_compare_text(text: str) -> str:
    squashed = " ".join(text.split()).lower()
    squashed = PUNCT_RE.sub(" ", squashed)
    return " ".join(squashed.split())


def compute_match_scores(actual_text: str, expected_text: str) -> MatchScores:
    actual_norm = normalize_compare_text(actual_text)
    expected_norm = normalize_compare_text(expected_text)
    char_ratio = difflib.SequenceMatcher(a=actual_norm, b=expected_norm).ratio()
    actual_tokens = actual_norm.split()
    expected_tokens = expected_norm.split()
    if not actual_tokens and not expected_tokens:
        precision = recall = f1 = 1.0
    elif not actual_tokens or not expected_tokens:
        precision = recall = f1 = 0.0
    else:
        actual_counts: dict[str, int] = {}
        for token in actual_tokens:
            actual_counts[token] = actual_counts.get(token, 0) + 1
        expected_counts: dict[str, int] = {}
        for token in expected_tokens:
            expected_counts[token] = expected_counts.get(token, 0) + 1
        overlap = 0
        for token, count in actual_counts.items():
            overlap += min(count, expected_counts.get(token, 0))
        precision = overlap / len(actual_tokens)
        recall = overlap / len(expected_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    expected_phrases = [phrase for phrase in CRITICAL_EXPECTED_PHRASES if phrase in expected_text]
    matched_phrases = [phrase for phrase in expected_phrases if phrase in actual_text]
    if not expected_phrases:
        critical_phrase_precision = critical_phrase_recall = critical_phrase_f1 = 1.0
    else:
        critical_phrase_precision = len(matched_phrases) / max(len([phrase for phrase in CRITICAL_EXPECTED_PHRASES if phrase in actual_text]), 1)
        critical_phrase_recall = len(matched_phrases) / len(expected_phrases)
        critical_phrase_f1 = (
            2 * critical_phrase_precision * critical_phrase_recall / (critical_phrase_precision + critical_phrase_recall)
            if (critical_phrase_precision + critical_phrase_recall)
            else 0.0
        )
    composite = (char_ratio * 0.50) + (f1 * 0.25) + (critical_phrase_f1 * 0.25)
    return MatchScores(
        char_ratio=char_ratio,
        token_precision=precision,
        token_recall=recall,
        token_f1=f1,
        critical_phrase_precision=critical_phrase_precision,
        critical_phrase_recall=critical_phrase_recall,
        critical_phrase_f1=critical_phrase_f1,
        composite=composite,
    )


def build_default_variants() -> list[VariantSpec]:
    return [
        VariantSpec("current", {}),
        VariantSpec(
            "wide_spans",
            {"segment_detection": {"max_candidate_spans_per_segment": 18, "max_span_words": 4}},
        ),
        VariantSpec(
            "phrase_windows",
            {"segment_detection": {"min_span_words": 2, "max_span_words": 5, "max_candidate_spans_per_segment": 20}},
        ),
        VariantSpec(
            "more_context",
            {"segment_detection": {"context_window_segments": 2}},
        ),
        VariantSpec(
            "phrase_context",
            {"segment_detection": {"context_window_segments": 2, "min_span_words": 2, "max_span_words": 5, "max_candidate_spans_per_segment": 20}},
        ),
        VariantSpec(
            "semantic_loose",
            {"segment_detection": {"semantic_min_confidence_to_accept": 0.60}},
        ),
        VariantSpec(
            "semantic_strict",
            {"segment_detection": {"semantic_min_confidence_to_accept": 0.90}},
        ),
        VariantSpec(
            "more_padding",
            {"segment_detection": {"audio_span_padding_sec": 0.50}},
        ),
        VariantSpec(
            "phrase_padding",
            {"segment_detection": {"audio_span_padding_sec": 0.50, "min_span_words": 2, "max_span_words": 5, "max_candidate_spans_per_segment": 20}},
        ),
        VariantSpec(
            "clause_padding",
            {"segment_detection": {"clause_retry_audio_padding_sec": 0.90}},
        ),
        VariantSpec(
            "clause_context",
            {
                "segment_detection": {
                    "context_window_segments": 2,
                    "clause_retry_audio_padding_sec": 0.90,
                    "min_span_words": 2,
                    "max_span_words": 5,
                    "max_candidate_spans_per_segment": 20,
                }
            },
        ),
        VariantSpec(
            "full_call_hebrew_bias",
            {
                "segment_detection": {
                    "full_call_hebrew_score_threshold": 0.55,
                    "full_call_hebrew_min_suspicious_segments": 2,
                    "full_call_hebrew_suspicious_ratio_threshold": 0.40,
                }
            },
        ),
        VariantSpec(
            "no_reconcile",
            {"segment_detection": {"segment_reconciliation_enabled": False}},
        ),
        VariantSpec(
            "wide_loose_combo",
            {
                "segment_detection": {
                    "context_window_segments": 2,
                    "max_candidate_spans_per_segment": 18,
                    "max_span_words": 4,
                    "semantic_min_confidence_to_accept": 0.60,
                    "audio_span_padding_sec": 0.50,
                }
            },
        ),
    ]


def variant_config(base_config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    return AppConfig(data=_deep_merge(base_config.data, overrides), root_dir=base_config.root_dir)


def _repo_path(path_value: str | Path) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _find_call_dir(config: AppConfig, call_id: str) -> Path:
    matches = sorted(config.archive_root.rglob(f"call_{call_id}"))
    if not matches:
        raise FileNotFoundError(f"Could not locate call directory for {call_id}")
    return matches[0]


def load_manifest(manifest_path: Path, config: AppConfig) -> list[EvaluationManifestEntry]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload["calls"] if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Manifest must be a list or an object with a 'calls' list")
    entries: list[EvaluationManifestEntry] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Manifest entries must be objects")
        call_id = str(item.get("call_id") or "").strip()
        if not call_id:
            raise ValueError("Manifest entry missing call_id")
        expected_case_value = item.get("expected_case_path")
        expected_case_path = _repo_path(expected_case_value) if expected_case_value else None
        if expected_case_path is not None and not expected_case_path.exists():
            raise FileNotFoundError(f"Expected case path does not exist for {call_id}: {expected_case_path}")
        expected_path = None
        if expected_case_path is None:
            expected_value = item.get("expected_transcript_path")
            if not expected_value:
                raise ValueError(f"Manifest entry for {call_id} missing expected_case_path or expected_transcript_path")
            expected_path = _repo_path(expected_value)
            if not expected_path.exists():
                raise FileNotFoundError(f"Expected transcript path does not exist for {call_id}: {expected_path}")
        call_dir_value = item.get("call_dir")
        call_dir = _repo_path(call_dir_value) if call_dir_value else _find_call_dir(config, call_id)
        entries.append(
            EvaluationManifestEntry(
                call_id=call_id,
                call_dir=call_dir,
                expected_transcript_path=expected_path,
                expected_case_path=expected_case_path,
                description=str(item.get("description") or ""),
                tags=[str(tag) for tag in item.get("tags", [])],
                priority_weight=float(item.get("priority_weight", 1.0) or 1.0),
            )
        )
    return entries


def load_expected_case(entry: EvaluationManifestEntry) -> tuple[str, list[dict[str, Any]] | None]:
    if entry.expected_case_path is not None:
        if entry.expected_case_path.suffix.lower() == ".json":
            payload = json.loads(entry.expected_case_path.read_text(encoding="utf-8"))
            expected_text = normalize_expected_text(str(payload.get("expected_transcript") or ""))
            if not expected_text:
                raise ValueError(f"Expected case for {entry.call_id} is missing expected_transcript")
            turns = payload.get("expected_turns")
            return expected_text, turns if isinstance(turns, list) else None
        turns: list[dict[str, Any]] = []
        for raw_line in entry.expected_case_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = TURN_LINE_RE.match(line)
            if not match:
                continue
            speaker, text = match.groups()
            text = text.strip()
            if not text:
                continue
            turns.append({"speaker": speaker.strip(), "text": text})
        if not turns:
            raise ValueError(f"Expected case for {entry.call_id} does not contain any speaker turns")
        expected_text = normalize_expected_text(" ".join(item["text"] for item in turns))
        return expected_text, turns
    if entry.expected_transcript_path is None:
        raise ValueError(f"Expected transcript path is missing for {entry.call_id}")
    expected_text = normalize_expected_text(entry.expected_transcript_path.read_text(encoding="utf-8"))
    return expected_text, None


def select_variants(specs: list[VariantSpec], selected_names: list[str], baseline_only: bool) -> list[VariantSpec]:
    if baseline_only:
        return [item for item in specs if item.name == "current"]
    if not selected_names:
        return specs
    wanted = set(selected_names)
    filtered = [item for item in specs if item.name in wanted]
    missing = sorted(wanted - {item.name for item in filtered})
    if missing:
        raise ValueError(f"Unknown variant(s): {', '.join(missing)}")
    return filtered


def resolve_audio_path(call_dir: Path) -> Path:
    normalized = call_dir / "audio_normalized.wav"
    if normalized.exists():
        return normalized
    original = call_dir / "audio_original.m4a"
    if original.exists():
        return original
    raise FileNotFoundError(f"No audio file found in {call_dir}")


def expected_text_from_args(expected_file: Path | None) -> str:
    if expected_file is None:
        return normalize_expected_text(DEFAULT_EXPECTED_TRANSCRIPT)
    return normalize_expected_text(expected_file.read_text(encoding="utf-8"))


def run_variant(
    baseline: RawTranscript,
    audio_path: Path,
    base_config: AppConfig,
    spec: VariantSpec,
    expected_text: str,
    transcription_backend: str,
    transcription_provider: str,
) -> VariantResult:
    started = time.perf_counter()
    config = variant_config(base_config, spec.overrides)
    recovery = _run_post_baseline_recovery(
        baseline,
        audio_path,
        config,
        provider=transcription_provider,
        model_override=None,
        language_override="ru",
        retry_enabled=True,
        retry_languages=list(config.section("transcription").get("retry_languages_on_suspicion", ["he"])),
        merge_enabled=bool(config.section("transcription").get("candidate_merge_enabled", True)),
        full_call_retry=lambda: generate_baseline(
            audio_path=audio_path,
            config=config,
            baseline_language="he",
            transcription_backend=transcription_backend,
            transcription_provider=transcription_provider,
        ),
    )
    score_map = _word_span_candidate_score_map(recovery.word_span_candidates, recovery.word_span_decisions)
    scores = compute_match_scores(recovery.selected.text, expected_text)
    return VariantResult(
        name=spec.name,
        overrides=spec.overrides,
        elapsed_seconds=time.perf_counter() - started,
        selected_strategy=recovery.strategy,
        selected_text=recovery.selected.text,
        scores=scores,
        suspicious_segments=sum(1 for item in recovery.segment_decisions if item.suspicious),
        suspicious_word_spans=sum(1 for item in recovery.word_span_decisions if item.suspicious),
        selected_word_spans=len(recovery.word_retry_candidates),
        suspicious_clause_candidates=len(recovery.clause_retry_candidates),
        selected_clause_candidates=len(recovery.clause_retry_candidates),
        applied_replacements=sum(1 for item in recovery.word_span_retry_results if item.get("replacement_applied"))
        + sum(1 for item in recovery.clause_retry_results if item.get("replacement_applied")),
        applied_clause_replacements=sum(1 for item in recovery.clause_retry_results if item.get("replacement_applied")),
        selection_notes=list(recovery.comparison.quality_notes),
        clause_diagnostics=list(recovery.clause_detection_summary.get("candidates", [])),
        candidate_diagnostics=[
            {
                "segment_index": item.segment_index,
                "span_text": item.span_text,
                "start_token_index": item.start_token_index,
                "end_token_index": item.end_token_index,
                "score": round(
                    score_map.get((item.segment_index, item.start_token_index, item.end_token_index), 0.0),
                    4,
                ),
                "confidence": round(item.confidence, 4),
                "label": item.label,
                "retry_language": item.retry_language,
                "status": (
                    "selected"
                    if any(
                        candidate.segment_index == item.segment_index
                        and candidate.start_token_index == item.start_token_index
                        and candidate.end_token_index == item.end_token_index
                        for candidate in recovery.word_retry_candidates
                    )
                    else "below_threshold"
                    if item.confidence < float(config.section("segment_detection").get("min_confidence_to_retry", 0.70))
                    else "filtered_after_ranking"
                ),
                "reason": item.reason,
            }
            for item in recovery.word_span_decisions
            if item.suspicious
        ],
        word_span_retries=recovery.word_span_retry_results,
        clause_retry_results=recovery.clause_retry_results,
        full_call_hebrew_escalated=recovery.full_call_hebrew_escalated,
        full_call_hebrew_selected=recovery.full_call_hebrew_selected,
        full_call_hebrew_decision=recovery.full_call_hebrew_decision,
        transcription_backend_attempted=baseline.backend_attempted,
        transcription_backend_selected=baseline.backend_selected,
        transcription_backend_fallback_reason=baseline.backend_fallback_reason,
        chunked_plausibility_score=baseline.chunked_plausibility_score,
        chunked_plausibility_flags=list(baseline.chunked_plausibility_flags or []),
    )


def write_report(
    output_path: Path,
    call_dir: Path,
    baseline: RawTranscript,
    expected_text: str,
    results: list[VariantResult],
) -> None:
    lines: list[str] = []
    lines.append("Post-Russian Hebrew Recovery Tuning")
    lines.append("")
    lines.append(f"Call dir: {call_dir}")
    lines.append(f"Variants tested: {len(results)}")
    lines.append("")
    lines.append("Expected text:")
    lines.append(expected_text)
    lines.append("")
    lines.append("Baseline Russian transcript:")
    lines.append(baseline.text)
    lines.append("")
    lines.append("Ranked results:")
    lines.append("")
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.name}")
        lines.append(
            f"   composite={result.scores.composite:.4f} "
            f"char_ratio={result.scores.char_ratio:.4f} "
            f"token_f1={result.scores.token_f1:.4f} "
            f"critical_phrase_f1={result.scores.critical_phrase_f1:.4f}"
        )
        lines.append(f"   strategy={result.selected_strategy} elapsed_seconds={result.elapsed_seconds:.2f}")
        lines.append(
            f"   suspicious_segments={result.suspicious_segments} suspicious_word_spans={result.suspicious_word_spans} selected_word_spans={result.selected_word_spans} suspicious_clause_candidates={result.suspicious_clause_candidates} selected_clause_candidates={result.selected_clause_candidates} applied_replacements={result.applied_replacements} applied_clause_replacements={result.applied_clause_replacements}"
        )
        lines.append(f"   overrides={json.dumps(result.overrides, ensure_ascii=False, sort_keys=True)}")
        lines.append(f"   text={result.selected_text}")
        lines.append(f"   selection_notes={json.dumps(result.selection_notes, ensure_ascii=False)}")
        lines.append(
            "   full_call_hebrew="
            + json.dumps(
                {
                    "escalated": result.full_call_hebrew_escalated,
                    "selected": result.full_call_hebrew_selected,
                    "decision": result.full_call_hebrew_decision,
                },
                ensure_ascii=False,
            )
        )
        lines.append(
            "   critical_phrases_hit="
            + json.dumps([phrase for phrase in CRITICAL_EXPECTED_PHRASES if phrase in result.selected_text], ensure_ascii=False)
        )
        if result.clause_diagnostics:
            lines.append("   clause_candidates:")
            for candidate in result.clause_diagnostics:
                lines.append("   - " + json.dumps(candidate, ensure_ascii=False))
        if result.candidate_diagnostics:
            lines.append("   candidate_ranks:")
            for candidate in result.candidate_diagnostics:
                lines.append(
                    "   - "
                    + json.dumps(
                        {
                            "segment_index": candidate["segment_index"],
                            "span_text": candidate["span_text"],
                            "score": candidate["score"],
                            "confidence": candidate["confidence"],
                            "status": candidate["status"],
                            "label": candidate["label"],
                        },
                        ensure_ascii=False,
                    )
                )
        if result.word_span_retries:
            lines.append("   retries:")
            for retry in result.word_span_retries:
                lines.append(
                    "   - "
                    + json.dumps(
                        {
                            "span_text": retry.get("span_text"),
                            "retry_text": retry.get("retry_text"),
                            "normalized_hebrew": retry.get("normalized_hebrew"),
                            "final_replacement_text": retry.get("final_replacement_text"),
                            "replacement_applied": retry.get("replacement_applied"),
                            "replacement_reason": retry.get("replacement_reason"),
                        },
                        ensure_ascii=False,
                    )
                )
        if result.clause_retry_results:
            lines.append("   clause_retries:")
            for retry in result.clause_retry_results:
                lines.append(
                    "   - "
                    + json.dumps(
                        {
                            "clause_text": retry.get("clause_text") or retry.get("span_text"),
                            "retry_text": retry.get("retry_text"),
                            "normalized_hebrew": retry.get("normalized_hebrew"),
                            "final_replacement_text": retry.get("final_replacement_text"),
                            "replacement_applied": retry.get("replacement_applied"),
                            "replacement_reason": retry.get("replacement_reason"),
                        },
                        ensure_ascii=False,
                    )
                )
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_json_report(output_path: Path, call_dir: Path, baseline: RawTranscript, expected_text: str, results: list[VariantResult]) -> None:
    payload = {
        "call_dir": str(call_dir),
        "expected_text": expected_text,
        "baseline_text": baseline.text,
        "results": [
            {
                "name": result.name,
                "overrides": result.overrides,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "selected_strategy": result.selected_strategy,
                "selected_text": result.selected_text,
                "scores": asdict(result.scores),
                "suspicious_segments": result.suspicious_segments,
                "suspicious_word_spans": result.suspicious_word_spans,
                "selected_word_spans": result.selected_word_spans,
                "suspicious_clause_candidates": result.suspicious_clause_candidates,
                "selected_clause_candidates": result.selected_clause_candidates,
                "applied_replacements": result.applied_replacements,
                "applied_clause_replacements": result.applied_clause_replacements,
                "selection_notes": result.selection_notes,
                "clause_diagnostics": result.clause_diagnostics,
                "candidate_diagnostics": result.candidate_diagnostics,
                "word_span_retries": result.word_span_retries,
                "clause_retry_results": result.clause_retry_results,
                "full_call_hebrew_escalated": result.full_call_hebrew_escalated,
                "full_call_hebrew_selected": result.full_call_hebrew_selected,
                "full_call_hebrew_decision": result.full_call_hebrew_decision,
            }
            for result in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def persist_reports(
    text_output: Path,
    json_output: Path,
    call_dir: Path,
    baseline: RawTranscript,
    expected_text: str,
    results: list[VariantResult],
) -> list[VariantResult]:
    ranked = sorted(results, key=lambda item: item.scores.composite, reverse=True)
    write_report(text_output, call_dir, baseline, expected_text, ranked)
    write_json_report(json_output, call_dir, baseline, expected_text, ranked)
    return ranked


def evaluate_call(
    entry: EvaluationManifestEntry,
    config: AppConfig,
    specs: list[VariantSpec],
    baseline_language: str,
    transcription_backend: str,
    transcription_provider: str,
) -> CallEvaluationResult:
    try:
        call_dir = entry.call_dir or _find_call_dir(config, entry.call_id)
        audio_path = resolve_audio_path(call_dir)
        expected_text, _ = load_expected_case(entry)
        baseline = generate_baseline(
            audio_path=audio_path,
            config=config,
            baseline_language=baseline_language,
            transcription_backend=transcription_backend,
            transcription_provider=transcription_provider,
        )
        variants = [
            run_variant(
                baseline,
                audio_path,
                config,
                spec,
                expected_text,
                transcription_backend,
                transcription_provider,
            )
            for spec in specs
        ]
        return CallEvaluationResult(entry=entry, baseline=baseline, expected_text=expected_text, variants=variants)
    except Exception as exc:
        return CallEvaluationResult(entry=entry, baseline=None, expected_text=None, variants=[], error=str(exc))


def baseline_config(config: AppConfig, transcription_backend: str, transcription_provider: str) -> AppConfig:
    cloned = AppConfig(data=copy.deepcopy(config.data), root_dir=config.root_dir)
    cloned.data.setdefault("transcription", {})
    cloned.data["transcription"]["backend"] = transcription_backend
    cloned.data["transcription"]["provider_default"] = transcription_provider
    return cloned


def generate_baseline(
    audio_path: Path,
    config: AppConfig,
    baseline_language: str,
    transcription_backend: str,
    transcription_provider: str,
) -> RawTranscript:
    effective = baseline_config(config, transcription_backend, transcription_provider)
    if transcription_provider == "local":
        if transcription_backend == "whisper_legacy":
            return _transcribe_local(
                audio_path,
                effective,
                model_override=effective.section("transcription").get("local_model"),
                language_override=baseline_language,
                language_mode="metadata_override",
            )
        if transcription_backend in {"vad_chunked_legacy", "faster_whisper_vad"}:
            return _transcribe_vad_chunked(
                audio_path,
                effective,
                provider="local",
                model_override=effective.section("transcription").get("local_model"),
                language_override=baseline_language,
                language_mode="metadata_override",
            )
        raise ValueError(f"Unsupported transcription backend: {transcription_backend}")
    if transcription_provider == "cloud":
        if transcription_backend == "whisper_legacy":
            return _transcribe_cloud(
                audio_path,
                effective,
                model_override=effective.section("transcription").get("cloud_model"),
                language_override=baseline_language,
                language_mode="metadata_override",
            )
        if transcription_backend in {"vad_chunked_legacy", "faster_whisper_vad"}:
            return _transcribe_vad_chunked(
                audio_path,
                effective,
                provider="cloud",
                model_override=effective.section("transcription").get("cloud_model"),
                language_override=baseline_language,
                language_mode="metadata_override",
            )
        raise ValueError(f"Unsupported transcription backend: {transcription_backend}")
    raise ValueError(f"Unsupported transcription provider: {transcription_provider}")


def _regression_flag(baseline: VariantResult, candidate: VariantResult) -> bool:
    composite_delta = candidate.scores.composite - baseline.scores.composite
    lost_phrase_recall = candidate.scores.critical_phrase_recall < baseline.scores.critical_phrase_recall
    return composite_delta < -0.03 or lost_phrase_recall


def aggregate_variant_results(call_results: list[CallEvaluationResult]) -> list[AggregateVariantResult]:
    per_variant: dict[str, dict[str, float]] = {}
    failed_calls = sum(1 for item in call_results if item.error)
    for call_result in call_results:
        if call_result.error or not call_result.variants:
            continue
        baseline = next((item for item in call_result.variants if item.name == "current"), call_result.variants[0])
        weight = max(0.0, float(call_result.entry.priority_weight or 1.0))
        for variant in call_result.variants:
            bucket = per_variant.setdefault(
                variant.name,
                {
                    "weight_total": 0.0,
                    "weighted_composite": 0.0,
                    "weighted_char_ratio": 0.0,
                    "weighted_token_f1": 0.0,
                    "weighted_critical_phrase_f1": 0.0,
                    "weighted_delta_composite_vs_baseline": 0.0,
                    "weighted_delta_phrase_f1_vs_baseline": 0.0,
                    "success_count": 0.0,
                    "regression_flagged_calls": 0.0,
                    "fallback_count": 0.0,
                },
            )
            bucket["weight_total"] += weight
            bucket["weighted_composite"] += variant.scores.composite * weight
            bucket["weighted_char_ratio"] += variant.scores.char_ratio * weight
            bucket["weighted_token_f1"] += variant.scores.token_f1 * weight
            bucket["weighted_critical_phrase_f1"] += variant.scores.critical_phrase_f1 * weight
            bucket["weighted_delta_composite_vs_baseline"] += (variant.scores.composite - baseline.scores.composite) * weight
            bucket["weighted_delta_phrase_f1_vs_baseline"] += (
                variant.scores.critical_phrase_f1 - baseline.scores.critical_phrase_f1
            ) * weight
            bucket["success_count"] += 1
            if variant.transcription_backend_attempted and variant.transcription_backend_selected:
                if variant.transcription_backend_attempted != variant.transcription_backend_selected:
                    bucket["fallback_count"] += 1
            if variant.name != baseline.name and _regression_flag(baseline, variant):
                bucket["regression_flagged_calls"] += 1
    aggregate: list[AggregateVariantResult] = []
    for name, bucket in per_variant.items():
        weight_total = bucket["weight_total"] or 1.0
        aggregate.append(
            AggregateVariantResult(
                name=name,
                success_count=int(bucket["success_count"]),
                failure_count=failed_calls,
                weighted_composite=bucket["weighted_composite"] / weight_total,
                weighted_char_ratio=bucket["weighted_char_ratio"] / weight_total,
                weighted_token_f1=bucket["weighted_token_f1"] / weight_total,
                weighted_critical_phrase_f1=bucket["weighted_critical_phrase_f1"] / weight_total,
                weighted_delta_composite_vs_baseline=bucket["weighted_delta_composite_vs_baseline"] / weight_total,
                weighted_delta_phrase_f1_vs_baseline=bucket["weighted_delta_phrase_f1_vs_baseline"] / weight_total,
                regression_flagged_calls=int(bucket["regression_flagged_calls"]),
                fallback_count=int(bucket["fallback_count"]),
            )
        )
    return sorted(
        aggregate,
        key=lambda item: (
            item.weighted_composite,
            item.weighted_delta_composite_vs_baseline,
            item.weighted_critical_phrase_f1,
        ),
        reverse=True,
    )


def write_aggregate_text_report(
    output_path: Path,
    manifest_path: Path,
    call_results: list[CallEvaluationResult],
    aggregate: list[AggregateVariantResult],
) -> None:
    lines: list[str] = []
    lines.append("Post-Russian Recovery Aggregate Benchmark")
    lines.append("")
    lines.append(f"Manifest: {manifest_path}")
    lines.append(f"Calls evaluated: {len(call_results)}")
    lines.append(f"Successful calls: {sum(1 for item in call_results if not item.error)}")
    lines.append(f"Failed calls: {sum(1 for item in call_results if item.error)}")
    lines.append("")
    lines.append("Aggregate variant ranking:")
    lines.append("")
    for index, item in enumerate(aggregate, start=1):
        lines.append(f"{index}. {item.name}")
        lines.append(
            f"   weighted_composite={item.weighted_composite:.4f} "
            f"weighted_char_ratio={item.weighted_char_ratio:.4f} "
            f"weighted_token_f1={item.weighted_token_f1:.4f} "
            f"weighted_critical_phrase_f1={item.weighted_critical_phrase_f1:.4f}"
        )
        lines.append(
            f"   delta_composite_vs_baseline={item.weighted_delta_composite_vs_baseline:.4f} "
            f"delta_phrase_f1_vs_baseline={item.weighted_delta_phrase_f1_vs_baseline:.4f} "
            f"regression_flagged_calls={item.regression_flagged_calls} "
            f"fallback_count={item.fallback_count}"
        )
        lines.append(f"   success_count={item.success_count} failure_count={item.failure_count}")
        lines.append("")
    lines.append("Per-call summary:")
    lines.append("")
    for call_result in call_results:
        entry = call_result.entry
        lines.append(f"- {entry.call_id}")
        if call_result.error:
            lines.append(f"  error={call_result.error}")
            continue
        ranked = sorted(call_result.variants, key=lambda item: item.scores.composite, reverse=True)
        best = ranked[0]
        baseline = next((item for item in call_result.variants if item.name == "current"), ranked[0])
        lines.append(
            f"  best={best.name} composite={best.scores.composite:.4f} "
            f"baseline={baseline.scores.composite:.4f} "
            f"delta={best.scores.composite - baseline.scores.composite:.4f}"
        )
        if best.transcription_backend_attempted or best.transcription_backend_selected:
            lines.append(
                f"  backend_attempted={best.transcription_backend_attempted} "
                f"backend_selected={best.transcription_backend_selected} "
                f"fallback_reason={best.transcription_backend_fallback_reason}"
            )
        if entry.description:
            lines.append(f"  description={entry.description}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_aggregate_json_report(
    output_path: Path,
    manifest_path: Path,
    call_results: list[CallEvaluationResult],
    aggregate: list[AggregateVariantResult],
) -> None:
    payload = {
        "run_id": output_path.parent.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_path": str(manifest_path),
        "calls": [
            {
                "call_id": item.entry.call_id,
                "call_dir": str(item.entry.call_dir) if item.entry.call_dir else None,
                "expected_case_path": str(item.entry.expected_case_path) if item.entry.expected_case_path else None,
                "expected_transcript_path": str(item.entry.expected_transcript_path) if item.entry.expected_transcript_path else None,
                "description": item.entry.description,
                "tags": item.entry.tags or [],
                "priority_weight": item.entry.priority_weight,
                "error": item.error,
                "baseline_text": item.baseline.text if item.baseline else None,
                "expected_text": item.expected_text,
                "variants": [
                    {
                        "name": result.name,
                        "overrides": result.overrides,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                        "selected_strategy": result.selected_strategy,
                        "selected_text": result.selected_text,
                        "scores": asdict(result.scores),
                        "suspicious_segments": result.suspicious_segments,
                        "suspicious_word_spans": result.suspicious_word_spans,
                        "selected_word_spans": result.selected_word_spans,
                        "suspicious_clause_candidates": result.suspicious_clause_candidates,
                        "selected_clause_candidates": result.selected_clause_candidates,
                        "applied_replacements": result.applied_replacements,
                        "applied_clause_replacements": result.applied_clause_replacements,
                        "selection_notes": result.selection_notes,
                        "transcription_backend_attempted": result.transcription_backend_attempted,
                        "transcription_backend_selected": result.transcription_backend_selected,
                        "transcription_backend_fallback_reason": result.transcription_backend_fallback_reason,
                        "chunked_plausibility_score": result.chunked_plausibility_score,
                        "chunked_plausibility_flags": result.chunked_plausibility_flags,
                        "clause_diagnostics": result.clause_diagnostics,
                        "candidate_diagnostics": result.candidate_diagnostics,
                        "word_span_retries": result.word_span_retries,
                        "clause_retry_results": result.clause_retry_results,
                        "full_call_hebrew_escalated": result.full_call_hebrew_escalated,
                        "full_call_hebrew_selected": result.full_call_hebrew_selected,
                        "full_call_hebrew_decision": result.full_call_hebrew_decision,
                    }
                    for result in item.variants
                ],
            }
            for item in call_results
        ],
        "aggregate": {
            "ranked_variants": [asdict(item) for item in aggregate],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Hebrew recovery after the Russian baseline transcription stage.")
    parser.add_argument("--call-dir", type=Path, default=DEFAULT_CALL_DIR, help="Call directory that contains audio_normalized.wav or audio_original.m4a")
    parser.add_argument("--expected-file", type=Path, help="Optional file with expected transcript text; speaker/timestamp lines are stripped")
    parser.add_argument("--output", type=Path, help="Optional text report path")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report path")
    parser.add_argument("--manifest", type=Path, help=f"Optional evaluation manifest for multi-call offline tuning (default benchmark set: {DEFAULT_MANIFEST})")
    parser.add_argument("--output-dir", type=Path, help="Directory for aggregate and per-call reports in manifest mode")
    parser.add_argument("--call-id", action="append", default=[], help="Optional call_id filter for manifest mode; repeatable")
    parser.add_argument("--variant", action="append", default=[], help="Optional variant name filter; repeatable")
    parser.add_argument("--max-calls", type=int, help="Optional limit on number of manifest calls to evaluate")
    parser.add_argument("--write-per-call-reports", action="store_true", help="Write text/json reports for each call in manifest mode")
    parser.add_argument("--baseline-only", action="store_true", help="Only evaluate the current baseline variant")
    parser.add_argument("--compare-to", type=Path, help="Optional previous aggregate JSON report to compare manually later")
    parser.add_argument("--baseline-language", default="ru", help="Language forced for the one-time baseline transcription")
    parser.add_argument(
        "--transcription-backend",
        choices=["whisper_legacy", "vad_chunked_legacy", "faster_whisper_vad"],
        default="whisper_legacy",
        help="Baseline transcription backend used before running recovery variants",
    )
    parser.add_argument(
        "--transcription-provider",
        choices=["local", "cloud"],
        default="local",
        help="Baseline transcription provider used before running recovery variants",
    )
    parser.add_argument(
        "--benchmark-calls",
        action="store_true",
        help="Run manifest mode against the default benchmark call set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.load(ROOT / "config.yaml")
    specs = select_variants(build_default_variants(), args.variant, args.baseline_only)
    manifest_arg = args.manifest
    if args.benchmark_calls and manifest_arg is None:
        manifest_arg = DEFAULT_MANIFEST

    if manifest_arg:
        manifest_path = manifest_arg.resolve()
        entries = load_manifest(manifest_path, config)
        if args.call_id:
            allowed = set(args.call_id)
            entries = [item for item in entries if item.call_id in allowed]
        if args.max_calls is not None:
            entries = entries[: max(0, args.max_calls)]
        run_id = f"post_russian_recovery_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        output_dir = (args.output_dir.resolve() if args.output_dir else (DEFAULT_OUTPUT_DIR / run_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregate_text_output = output_dir / "aggregate_report.txt"
        aggregate_json_output = output_dir / "aggregate_report.json"
        results: list[CallEvaluationResult] = []
        try:
            for index, entry in enumerate(entries, start=1):
                print(f"[{index}/{len(entries)}] Evaluating call: {entry.call_id}", flush=True)
                result = evaluate_call(
                    entry,
                    config,
                    specs,
                    args.baseline_language,
                    args.transcription_backend,
                    args.transcription_provider,
                )
                results.append(result)
                if result.error:
                    print(f"[{index}/{len(entries)}] Failed {entry.call_id}: {result.error}", flush=True)
                else:
                    ranked = sorted(result.variants, key=lambda item: item.scores.composite, reverse=True)
                    best = ranked[0]
                    print(
                        f"[{index}/{len(entries)}] Best for {entry.call_id}: "
                        f"{best.name} composite={best.scores.composite:.4f} "
                        f"token_f1={best.scores.token_f1:.4f} "
                        f"critical_phrase_f1={best.scores.critical_phrase_f1:.4f}",
                        flush=True,
                    )
                    if args.write_per_call_reports and result.baseline and result.expected_text:
                        call_slug = f"{entry.call_id}"
                        persist_reports(
                            output_dir / f"{call_slug}.txt",
                            output_dir / f"{call_slug}.json",
                            entry.call_dir or _find_call_dir(config, entry.call_id),
                            result.baseline,
                            result.expected_text,
                            result.variants,
                        )
                aggregate = aggregate_variant_results(results)
                write_aggregate_text_report(aggregate_text_output, manifest_path, results, aggregate)
                write_aggregate_json_report(aggregate_json_output, manifest_path, results, aggregate)
        except KeyboardInterrupt:
            aggregate = aggregate_variant_results(results)
            write_aggregate_text_report(aggregate_text_output, manifest_path, results, aggregate)
            write_aggregate_json_report(aggregate_json_output, manifest_path, results, aggregate)
            print("Interrupted. Partial aggregate reports were written.", flush=True)
            print(f"Wrote {aggregate_text_output}", flush=True)
            print(f"Wrote {aggregate_json_output}", flush=True)
            return 130
        aggregate = aggregate_variant_results(results)
        write_aggregate_text_report(aggregate_text_output, manifest_path, results, aggregate)
        write_aggregate_json_report(aggregate_json_output, manifest_path, results, aggregate)
        print(f"Wrote {aggregate_text_output}", flush=True)
        print(f"Wrote {aggregate_json_output}", flush=True)
        if aggregate:
            best = aggregate[0]
            print(
                f"Best aggregate variant: {best.name} composite={best.weighted_composite:.4f} "
                f"delta_vs_baseline={best.weighted_delta_composite_vs_baseline:.4f} "
                f"regression_flagged_calls={best.regression_flagged_calls}",
                flush=True,
            )
        if args.compare_to:
            print(f"Previous report for manual comparison: {args.compare_to.resolve()}", flush=True)
        return 0

    call_dir = args.call_dir.resolve()
    audio_path = resolve_audio_path(call_dir)
    expected_text = expected_text_from_args(args.expected_file)
    baseline = generate_baseline(
        audio_path=audio_path,
        config=config,
        baseline_language=args.baseline_language,
        transcription_backend=args.transcription_backend,
        transcription_provider=args.transcription_provider,
    )
    stem = f"post_russian_recovery_tuning_{call_dir.name}"
    text_output = args.output or (DEFAULT_OUTPUT_DIR / f"{stem}.txt")
    json_output = args.json_output or (DEFAULT_OUTPUT_DIR / f"{stem}.json")
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    results: list[VariantResult] = []
    print(
        f"Baseline ready: provider={baseline.provider} model={baseline.model} language={baseline.language} "
        f"backend={args.transcription_backend}",
        flush=True,
    )
    print(f"Testing {len(specs)} variants", flush=True)
    try:
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] Running variant: {spec.name}", flush=True)
            result = run_variant(
                baseline,
                audio_path,
                config,
                spec,
                expected_text,
                args.transcription_backend,
                args.transcription_provider,
            )
            results.append(result)
            ranked = persist_reports(text_output, json_output, call_dir, baseline, expected_text, results)
            print(
                f"[{index}/{len(specs)}] Finished {spec.name}: "
                f"composite={result.scores.composite:.4f} "
                f"char_ratio={result.scores.char_ratio:.4f} "
                f"token_f1={result.scores.token_f1:.4f} "
                f"critical_phrase_f1={result.scores.critical_phrase_f1:.4f} "
                f"strategy={result.selected_strategy} "
                f"replacements={result.applied_replacements} "
                f"clause_replacements={result.applied_clause_replacements}",
                flush=True,
            )
            print(f"Partial reports updated: {text_output} | {json_output}", flush=True)
            if ranked:
                best = ranked[0]
                print(
                    f"Current best: {best.name} composite={best.scores.composite:.4f} "
                    f"char_ratio={best.scores.char_ratio:.4f} "
                    f"token_f1={best.scores.token_f1:.4f} "
                    f"critical_phrase_f1={best.scores.critical_phrase_f1:.4f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        ranked = persist_reports(text_output, json_output, call_dir, baseline, expected_text, results)
        print("Interrupted. Partial reports were written.", flush=True)
        if ranked:
            best = ranked[0]
            print(
                f"Best partial variant: {best.name} composite={best.scores.composite:.4f} "
                f"char_ratio={best.scores.char_ratio:.4f} "
                f"token_f1={best.scores.token_f1:.4f} "
                f"critical_phrase_f1={best.scores.critical_phrase_f1:.4f}",
                flush=True,
            )
        print(f"Wrote {text_output}", flush=True)
        print(f"Wrote {json_output}", flush=True)
        return 130
    ranked = persist_reports(text_output, json_output, call_dir, baseline, expected_text, results)
    print(f"Wrote {text_output}", flush=True)
    print(f"Wrote {json_output}", flush=True)
    if ranked:
        best = ranked[0]
        print(
            f"Best variant: {best.name} composite={best.scores.composite:.4f} "
            f"char_ratio={best.scores.char_ratio:.4f} "
            f"token_f1={best.scores.token_f1:.4f} "
            f"critical_phrase_f1={best.scores.critical_phrase_f1:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
