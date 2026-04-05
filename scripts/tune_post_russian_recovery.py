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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawTranscript
from call_assistant.transcription.service import (
    _apply_word_span_retries,
    _build_word_span_candidates,
    _rank_word_retry_candidates,
    _word_span_candidate_score_map,
    _segment_detection_decisions_with_raw,
    _transcribe_local,
    _word_span_detection_decisions_with_raw,
    assess_transcript_quality,
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
    applied_replacements: int
    candidate_diagnostics: list[dict[str, Any]]
    word_span_retries: list[dict[str, Any]]


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
) -> VariantResult:
    started = time.perf_counter()
    config = variant_config(base_config, spec.overrides)
    decisions, _ = _segment_detection_decisions_with_raw(baseline, audio_path, config)
    word_span_candidates = _build_word_span_candidates(baseline, decisions, config)
    word_span_decisions, _ = _word_span_detection_decisions_with_raw(word_span_candidates, config)
    score_map = _word_span_candidate_score_map(word_span_candidates, word_span_decisions)
    word_retry_candidates, _ = _rank_word_retry_candidates(word_span_candidates, word_span_decisions, config)
    threshold_used = float(config.section("segment_detection").get("min_confidence_to_retry", 0.70))
    selected_keys = {
        (item.segment_index, item.start_token_index, item.end_token_index)
        for item in word_retry_candidates
    }
    candidate_diagnostics: list[dict[str, Any]] = []
    for decision in word_span_decisions:
        if not decision.suspicious:
            continue
        key = (decision.segment_index, decision.start_token_index, decision.end_token_index)
        if key in selected_keys:
            status = "selected"
        elif decision.confidence < threshold_used:
            status = "below_threshold"
        else:
            status = "filtered_after_ranking"
        candidate_diagnostics.append(
            {
                "segment_index": decision.segment_index,
                "span_text": decision.span_text,
                "start_token_index": decision.start_token_index,
                "end_token_index": decision.end_token_index,
                "score": round(score_map.get(key, 0.0), 4),
                "confidence": round(decision.confidence, 4),
                "label": decision.label,
                "retry_language": decision.retry_language,
                "status": status,
                "reason": decision.reason,
            }
        )
    candidate_diagnostics.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["confidence"]),
            int(item["segment_index"]),
            int(item["start_token_index"]),
        )
    )
    selected: RawTranscript = baseline
    strategy = "baseline"
    word_span_retry_results: list[dict[str, Any]] = []
    if word_retry_candidates:
        merged, word_span_retry_results, _ = _apply_word_span_retries(
            baseline,
            audio_path,
            config,
            word_retry_candidates,
            model_override=base_config.section("transcription").get("local_model"),
        )
        baseline_assessment = assess_transcript_quality(baseline, audio_path)
        merged_assessment = assess_transcript_quality(merged, audio_path)
        replaced_count = sum(1 for item in word_span_retry_results if item.get("replacement_applied"))
        merged_has_hebrew = bool(re.search(r"[\u0590-\u05FF]", merged.text))
        if merged.text != baseline.text and (
            merged_assessment.score >= baseline_assessment.score - 0.05
            or (replaced_count > 0 and merged_has_hebrew)
        ):
            selected = merged
            strategy = "llm_word_span_hebrew_recovery"
    scores = compute_match_scores(selected.text, expected_text)
    return VariantResult(
        name=spec.name,
        overrides=spec.overrides,
        elapsed_seconds=time.perf_counter() - started,
        selected_strategy=strategy,
        selected_text=selected.text,
        scores=scores,
        suspicious_segments=sum(1 for item in decisions if item.suspicious),
        suspicious_word_spans=sum(1 for item in word_span_decisions if item.suspicious),
        selected_word_spans=len(word_retry_candidates),
        applied_replacements=sum(1 for item in word_span_retry_results if item.get("replacement_applied")),
        candidate_diagnostics=candidate_diagnostics,
        word_span_retries=word_span_retry_results,
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
            f"   suspicious_segments={result.suspicious_segments} suspicious_word_spans={result.suspicious_word_spans} selected_word_spans={result.selected_word_spans} applied_replacements={result.applied_replacements}"
        )
        lines.append(f"   overrides={json.dumps(result.overrides, ensure_ascii=False, sort_keys=True)}")
        lines.append(f"   text={result.selected_text}")
        lines.append(
            "   critical_phrases_hit="
            + json.dumps([phrase for phrase in CRITICAL_EXPECTED_PHRASES if phrase in result.selected_text], ensure_ascii=False)
        )
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
                "applied_replacements": result.applied_replacements,
                "candidate_diagnostics": result.candidate_diagnostics,
                "word_span_retries": result.word_span_retries,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Hebrew recovery after the Russian baseline transcription stage.")
    parser.add_argument("--call-dir", type=Path, default=DEFAULT_CALL_DIR, help="Call directory that contains audio_normalized.wav or audio_original.m4a")
    parser.add_argument("--expected-file", type=Path, help="Optional file with expected transcript text; speaker/timestamp lines are stripped")
    parser.add_argument("--output", type=Path, help="Optional text report path")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report path")
    parser.add_argument("--baseline-language", default="ru", help="Language forced for the one-time baseline transcription")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    call_dir = args.call_dir.resolve()
    audio_path = resolve_audio_path(call_dir)
    config = AppConfig.load(ROOT / "config.yaml")
    expected_text = expected_text_from_args(args.expected_file)
    baseline = _transcribe_local(
        audio_path,
        config,
        model_override=config.section("transcription").get("local_model"),
        language_override=args.baseline_language,
        language_mode="metadata_override",
    )
    stem = f"post_russian_recovery_tuning_{call_dir.name}"
    text_output = args.output or (DEFAULT_OUTPUT_DIR / f"{stem}.txt")
    json_output = args.json_output or (DEFAULT_OUTPUT_DIR / f"{stem}.json")
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    specs = build_default_variants()
    results: list[VariantResult] = []
    print(f"Baseline ready: provider={baseline.provider} model={baseline.model} language={baseline.language}", flush=True)
    print(f"Testing {len(specs)} variants", flush=True)
    try:
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] Running variant: {spec.name}", flush=True)
            result = run_variant(baseline, audio_path, config, spec, expected_text)
            results.append(result)
            ranked = persist_reports(text_output, json_output, call_dir, baseline, expected_text, results)
            print(
                f"[{index}/{len(specs)}] Finished {spec.name}: "
                f"composite={result.scores.composite:.4f} "
                f"char_ratio={result.scores.char_ratio:.4f} "
                f"token_f1={result.scores.token_f1:.4f} "
                f"critical_phrase_f1={result.scores.critical_phrase_f1:.4f} "
                f"strategy={result.selected_strategy} "
                f"replacements={result.applied_replacements}",
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
