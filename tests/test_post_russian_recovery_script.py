from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tune_post_russian_recovery.py"
SPEC = importlib.util.spec_from_file_location("tune_post_russian_recovery", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawSegment, RawTranscript
from call_assistant.transcription.service import (
    PostBaselineRecoveryResult,
    TranscriptCandidateComparison,
    assess_transcript_quality,
)


class PostRussianRecoveryScriptTests(unittest.TestCase):
    def test_load_manifest_resolves_relative_expected_transcript_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            archive_root = temp_root / "calls"
            archive_call_dir = archive_root / "2026" / "04" / "04" / "call_test_call"
            archive_call_dir.mkdir(parents=True)
            expected_dir = ROOT / "data" / "eval" / "post_russian_recovery" / "examples" / "call_example"
            manifest_path = temp_root / "manifest.json"
            manifest_path.write_text(
                json_dumps(
                    {
                        "calls": [
                            {
                                "call_id": "test_call",
                                "call_dir": str(archive_call_dir),
                                "expected_transcript_path": "data/eval/post_russian_recovery/examples/call_example/expected_transcript.txt",
                                "description": "fixture",
                                "tags": ["mixed"],
                                "priority_weight": 2.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = AppConfig.load(ROOT / "config.yaml")

            entries = MODULE.load_manifest(manifest_path, config)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].call_id, "test_call")
        self.assertEqual(entries[0].expected_transcript_path, expected_dir / "expected_transcript.txt")
        self.assertEqual(entries[0].priority_weight, 2.0)

    def test_normalize_expected_text_strips_speaker_lines(self) -> None:
        text = """
        [00:00] speaker_1 Unassigned
        Алло.
        [00:01] speaker_2 Unassigned
        Жорик, ищи хорошо.
        """.strip()

        normalized = MODULE.normalize_expected_text(text)

        self.assertEqual(normalized, "Алло. Жорик, ищи хорошо.")

    def test_compute_match_scores_prefers_closer_text(self) -> None:
        expected = "я нашел файл который лежал בתור תכנון מול ביצוע и потом его התאמה לתקציב"
        closer = MODULE.compute_match_scores(
            "я нашел файл который лежал בתור תכנון מול ביצוע и потом его התאמה לתקציב",
            expected,
        )
        farther = MODULE.compute_match_scores(
            "я нашел файл который лежал в бутуре Технун Мульбецоа",
            expected,
        )

        self.assertGreater(closer.composite, farther.composite)
        self.assertGreater(closer.char_ratio, farther.char_ratio)
        self.assertGreater(closer.critical_phrase_f1, farther.critical_phrase_f1)

    def test_build_default_variants_includes_phrase_oriented_variants(self) -> None:
        names = [item.name for item in MODULE.build_default_variants()]

        self.assertIn("phrase_windows", names)
        self.assertIn("phrase_context", names)
        self.assertIn("phrase_padding", names)
        self.assertIn("clause_padding", names)
        self.assertIn("full_call_hebrew_bias", names)

    def test_generate_baseline_uses_chunked_backend_when_requested(self) -> None:
        config = AppConfig.load(ROOT / "config.yaml")
        audio_path = ROOT / "tests" / "dummy_audio.m4a"
        expected = RawTranscript(provider="local", model="large-v3-turbo", language="ru", confidence=None, segments=[], text="baseline")

        with patch.object(MODULE, "_transcribe_vad_chunked", return_value=expected) as chunked_mock:
            baseline = MODULE.generate_baseline(
                audio_path=audio_path,
                config=config,
                baseline_language="ru",
                transcription_backend="vad_chunked_legacy",
                transcription_provider="local",
            )

        self.assertIs(baseline, expected)
        chunked_mock.assert_called_once()

    def test_parse_args_accepts_benchmark_calls_flag(self) -> None:
        argv = sys.argv[:]
        try:
            sys.argv = ["tune_post_russian_recovery.py", "--benchmark-calls"]
            args = MODULE.parse_args()
        finally:
            sys.argv = argv

        self.assertTrue(args.benchmark_calls)

    def test_run_variant_preserves_baseline_without_word_retries(self) -> None:
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=1.0, text="Алло.", confidence=None, speaker=None)],
            text="Алло.",
        )
        config = AppConfig.load(ROOT / "config.yaml")
        spec = MODULE.VariantSpec(name="current", overrides={})
        comparison = TranscriptCandidateComparison(
            baseline=baseline,
            forced_hebrew=None,
            merged=None,
            winner="baseline",
            strategy="baseline",
            quality_notes=["Baseline candidate accepted without retry."],
            selected=baseline,
            baseline_assessment=assess_transcript_quality(baseline, ROOT / "tests" / "dummy_audio.m4a"),
            forced_hebrew_assessment=None,
            merged_assessment=None,
            low_confidence_reason=None,
        )
        recovery = PostBaselineRecoveryResult(
            selected=baseline,
            strategy="baseline",
            comparison=comparison,
            baseline_assessment=comparison.baseline_assessment,
            segment_decisions=[],
            segment_detection_summary={},
            segment_detection_llm_response=None,
            word_span_candidates=[],
            word_span_decisions=[],
            word_span_detection_summary={},
            word_span_detection_llm_response=None,
            word_retry_candidates=[],
            word_span_retry_results=[],
            clause_retry_candidates=[],
            clause_detection_summary={"candidates": []},
            clause_retry_results=[],
            word_span_validation_llm_response=[],
            full_call_hebrew_escalated=False,
            full_call_hebrew_selected=False,
            full_call_hebrew_decision={"should_retry": False},
        )

        with patch.object(MODULE, "_run_post_baseline_recovery", return_value=recovery):
            result = MODULE.run_variant(
                baseline,
                ROOT / "tests" / "dummy_audio.m4a",
                config,
                spec,
                "Алло.",
                "whisper_legacy",
                "local",
            )

        self.assertEqual(result.selected_strategy, "baseline")
        self.assertEqual(result.selected_text, "Алло.")
        self.assertEqual(result.selected_word_spans, 0)
        self.assertEqual(result.selected_clause_candidates, 0)
        self.assertEqual(result.applied_replacements, 0)
        self.assertEqual(result.candidate_diagnostics, [])
        self.assertEqual(result.clause_diagnostics, [])

    def test_write_report_includes_variant_metrics(self) -> None:
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[],
            text="baseline text",
        )
        result = MODULE.VariantResult(
            name="current",
            overrides={"segment_detection": {"max_span_words": 3}},
            elapsed_seconds=1.23,
            selected_strategy="baseline",
            selected_text="baseline text",
            scores=MODULE.MatchScores(
                char_ratio=0.9,
                token_precision=0.8,
                token_recall=0.7,
                token_f1=0.75,
                critical_phrase_precision=1.0,
                critical_phrase_recall=0.8,
                critical_phrase_f1=0.8889,
                composite=0.8597,
            ),
            suspicious_segments=2,
            suspicious_word_spans=4,
            selected_word_spans=2,
            suspicious_clause_candidates=1,
            selected_clause_candidates=1,
            applied_replacements=1,
            applied_clause_replacements=1,
            selection_notes=["Recovered budgeting clause."],
            clause_diagnostics=[{"segment_index": 2, "clause_text": "его атомали, такцив", "score": 2.5, "status": "selected"}],
            candidate_diagnostics=[
                {
                    "segment_index": 2,
                    "span_text": "такцив Мульбецоа",
                    "score": 0.5,
                    "confidence": 0.0,
                    "status": "selected",
                    "label": "hebrew_transliteration",
                }
            ],
            word_span_retries=[],
            clause_retry_results=[],
            full_call_hebrew_escalated=False,
            full_call_hebrew_selected=False,
            full_call_hebrew_decision={"should_retry": False},
            transcription_backend_attempted="whisper_legacy",
            transcription_backend_selected="whisper_legacy",
            transcription_backend_fallback_reason=None,
            chunked_plausibility_score=None,
            chunked_plausibility_flags=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.txt"
            MODULE.write_report(output_path, ROOT, baseline, "expected text", [result])
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("current", text)
        self.assertIn("critical_phrase_f1=0.8889", text)
        self.assertIn("expected text", text)
        self.assertIn("candidate_ranks:", text)
        self.assertIn("clause_candidates:", text)
        self.assertIn("\"span_text\": \"такцив Мульбецоа\"", text)

    def test_persist_reports_sorts_results_by_composite(self) -> None:
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[],
            text="baseline text",
        )
        slower = MODULE.VariantResult(
            name="slower",
            overrides={},
            elapsed_seconds=2.0,
            selected_strategy="baseline",
            selected_text="slower text",
            scores=MODULE.MatchScores(0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.6),
            suspicious_segments=0,
            suspicious_word_spans=0,
            selected_word_spans=0,
            suspicious_clause_candidates=0,
            selected_clause_candidates=0,
            applied_replacements=0,
            applied_clause_replacements=0,
            selection_notes=[],
            clause_diagnostics=[],
            candidate_diagnostics=[],
            word_span_retries=[],
            clause_retry_results=[],
            full_call_hebrew_escalated=False,
            full_call_hebrew_selected=False,
            full_call_hebrew_decision={"should_retry": False},
            transcription_backend_attempted="whisper_legacy",
            transcription_backend_selected="whisper_legacy",
            transcription_backend_fallback_reason=None,
            chunked_plausibility_score=None,
            chunked_plausibility_flags=[],
        )
        better = MODULE.VariantResult(
            name="better",
            overrides={},
            elapsed_seconds=1.0,
            selected_strategy="baseline",
            selected_text="better text",
            scores=MODULE.MatchScores(0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 0.9),
            suspicious_segments=0,
            suspicious_word_spans=0,
            selected_word_spans=0,
            suspicious_clause_candidates=0,
            selected_clause_candidates=0,
            applied_replacements=0,
            applied_clause_replacements=0,
            selection_notes=[],
            clause_diagnostics=[],
            candidate_diagnostics=[],
            word_span_retries=[],
            clause_retry_results=[],
            full_call_hebrew_escalated=False,
            full_call_hebrew_selected=False,
            full_call_hebrew_decision={"should_retry": False},
            transcription_backend_attempted="whisper_legacy",
            transcription_backend_selected="whisper_legacy",
            transcription_backend_fallback_reason=None,
            chunked_plausibility_score=None,
            chunked_plausibility_flags=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            text_output = Path(temp_dir) / "report.txt"
            json_output = Path(temp_dir) / "report.json"
            ranked = MODULE.persist_reports(
                text_output,
                json_output,
                ROOT,
                baseline,
                "expected text",
                [slower, better],
            )
            text = text_output.read_text(encoding="utf-8")

        self.assertEqual(ranked[0].name, "better")
        self.assertIn("1. better", text)

    def test_aggregate_variant_results_ranks_by_weighted_composite(self) -> None:
        baseline = MODULE.VariantResult(
            name="current",
            overrides={},
            elapsed_seconds=1.0,
            selected_strategy="baseline",
            selected_text="baseline",
            scores=MODULE.MatchScores(0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.6),
            suspicious_segments=0,
            suspicious_word_spans=0,
            selected_word_spans=0,
            suspicious_clause_candidates=0,
            selected_clause_candidates=0,
            applied_replacements=0,
            applied_clause_replacements=0,
            selection_notes=[],
            clause_diagnostics=[],
            candidate_diagnostics=[],
            word_span_retries=[],
            clause_retry_results=[],
            full_call_hebrew_escalated=False,
            full_call_hebrew_selected=False,
            full_call_hebrew_decision={"should_retry": False},
            transcription_backend_attempted="whisper_legacy",
            transcription_backend_selected="whisper_legacy",
            transcription_backend_fallback_reason=None,
            chunked_plausibility_score=None,
            chunked_plausibility_flags=[],
        )
        better = MODULE.VariantResult(
            name="phrase_windows",
            overrides={},
            elapsed_seconds=1.0,
            selected_strategy="baseline",
            selected_text="better",
            scores=MODULE.MatchScores(0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.8),
            suspicious_segments=0,
            suspicious_word_spans=0,
            selected_word_spans=0,
            suspicious_clause_candidates=0,
            selected_clause_candidates=0,
            applied_replacements=0,
            applied_clause_replacements=0,
            selection_notes=[],
            clause_diagnostics=[],
            candidate_diagnostics=[],
            word_span_retries=[],
            clause_retry_results=[],
            full_call_hebrew_escalated=False,
            full_call_hebrew_selected=False,
            full_call_hebrew_decision={"should_retry": False},
            transcription_backend_attempted="whisper_legacy",
            transcription_backend_selected="whisper_legacy",
            transcription_backend_fallback_reason=None,
            chunked_plausibility_score=None,
            chunked_plausibility_flags=[],
        )
        call_result = MODULE.CallEvaluationResult(
            entry=MODULE.EvaluationManifestEntry(
                call_id="call_a",
                call_dir=Path("/tmp/call_a"),
                expected_transcript_path=Path("/tmp/expected.txt"),
                priority_weight=1.5,
            ),
            baseline=None,
            expected_text="expected",
            variants=[baseline, better],
        )

        aggregate = MODULE.aggregate_variant_results([call_result])

        self.assertEqual(aggregate[0].name, "phrase_windows")
        self.assertGreater(aggregate[0].weighted_composite, aggregate[1].weighted_composite)

    def test_write_aggregate_json_report_includes_calls_and_ranked_variants(self) -> None:
        call_result = MODULE.CallEvaluationResult(
            entry=MODULE.EvaluationManifestEntry(
                call_id="call_a",
                call_dir=Path("/tmp/call_a"),
                expected_transcript_path=Path("/tmp/expected.txt"),
                description="fixture call",
                tags=["mixed"],
                priority_weight=1.0,
            ),
            baseline=RawTranscript(provider="local", model="m", language="ru", confidence=None, segments=[], text="baseline"),
            expected_text="expected",
            variants=[],
        )
        aggregate = [
            MODULE.AggregateVariantResult(
                name="current",
                success_count=1,
                failure_count=0,
                weighted_composite=0.7,
                weighted_char_ratio=0.7,
                weighted_token_f1=0.7,
                weighted_critical_phrase_f1=0.7,
                weighted_delta_composite_vs_baseline=0.0,
                weighted_delta_phrase_f1_vs_baseline=0.0,
                regression_flagged_calls=0,
                fallback_count=0,
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "aggregate.json"
            MODULE.write_aggregate_json_report(output_path, ROOT / "manifest.json", [call_result], aggregate)
            payload = json_loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["calls"][0]["call_id"], "call_a")
        self.assertEqual(payload["aggregate"]["ranked_variants"][0]["name"], "current")
        self.assertEqual(payload["aggregate"]["ranked_variants"][0]["fallback_count"], 0)


def json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def json_loads(text: str):
    import json

    return json.loads(text)


if __name__ == "__main__":
    unittest.main()
