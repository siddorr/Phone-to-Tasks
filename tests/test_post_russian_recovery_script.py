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


class PostRussianRecoveryScriptTests(unittest.TestCase):
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

        with (
            patch.object(MODULE, "_segment_detection_decisions_with_raw", return_value=([], None)),
            patch.object(MODULE, "_build_word_span_candidates", return_value=[]),
            patch.object(MODULE, "_word_span_detection_decisions_with_raw", return_value=([], None)),
            patch.object(MODULE, "_rank_word_retry_candidates", return_value=([], 0)),
        ):
            result = MODULE.run_variant(
                baseline,
                ROOT / "tests" / "dummy_audio.m4a",
                config,
                spec,
                "Алло.",
            )

        self.assertEqual(result.selected_strategy, "baseline")
        self.assertEqual(result.selected_text, "Алло.")
        self.assertEqual(result.selected_word_spans, 0)
        self.assertEqual(result.applied_replacements, 0)
        self.assertEqual(result.candidate_diagnostics, [])

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
            applied_replacements=1,
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
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.txt"
            MODULE.write_report(output_path, ROOT, baseline, "expected text", [result])
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("current", text)
        self.assertIn("critical_phrase_f1=0.8889", text)
        self.assertIn("expected text", text)
        self.assertIn("candidate_ranks:", text)
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
            applied_replacements=0,
            candidate_diagnostics=[],
            word_span_retries=[],
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
            applied_replacements=0,
            candidate_diagnostics=[],
            word_span_retries=[],
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


if __name__ == "__main__":
    unittest.main()
