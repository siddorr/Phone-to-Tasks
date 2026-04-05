from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawSegment, RawTranscript, TranscriptSegment
from call_assistant.analysis.service import _fallback_analysis
from call_assistant.diarization.service import SpeakerTurn, apply_speaker_mapping, diarize
from call_assistant.transcript_cleaner.service import clean_transcript


class DiarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config_path = root / "config.yaml"
        self.config_path.write_text(
            """
paths:
  incoming_folder: "./incoming"
  archive_root: "./calls"
  sqlite_path: "./index/test.db"
  logs_dir: "./logs"
  temp_dir: "./temp"
diarization:
  provider: "pyannote"
  enabled: true
  fallback_single_speaker: true
  confidence_default: "medium"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        self.config = AppConfig.load(self.config_path)
        self.config.ensure_directories()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_diarize_falls_back_to_single_speaker(self) -> None:
        raw = RawTranscript(
            provider="local",
            model="test",
            language="ru",
            confidence=None,
            text="Алло",
            segments=[RawSegment(start_sec=0.0, end_sec=1.0, text="Алло")],
        )
        result = diarize(raw, self.config, audio_path=None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker_cluster_id, "speaker_1")
        self.assertEqual(result[0].speaker_label, "speaker_1")
        self.assertEqual(result[0].diarization_confidence, "low")

    def test_diarize_prefers_largest_overlap_not_midpoint_only(self) -> None:
        raw = RawTranscript(
            provider="local",
            model="test",
            language="ru",
            confidence=None,
            text="Привет да",
            segments=[RawSegment(start_sec=0.0, end_sec=10.0, text="Привет да")],
        )
        audio_path = Path(self.temp_dir.name) / "dummy.wav"
        audio_path.write_bytes(b"dummy")
        turns = [
            SpeakerTurn(start_sec=0.0, end_sec=6.1, speaker_cluster_id="speaker_1", confidence="medium"),
            SpeakerTurn(start_sec=6.1, end_sec=10.0, speaker_cluster_id="speaker_2", confidence="medium"),
        ]
        with mock.patch("call_assistant.diarization.service._run_pyannote", return_value=turns):
            result = diarize(raw, self.config, audio_path=audio_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker_cluster_id, "speaker_1")

    def test_diarize_splits_segment_when_multiple_turns_overlap_and_text_has_sentences(self) -> None:
        raw = RawTranscript(
            provider="local",
            model="test",
            language="he",
            confidence=None,
            text="הלו. שלום זה שליר. שלום.",
            segments=[RawSegment(start_sec=0.0, end_sec=5.0, text="הלו. שלום זה שליר. שלום.")],
        )
        audio_path = Path(self.temp_dir.name) / "dummy_split.wav"
        audio_path.write_bytes(b"dummy")
        turns = [
            SpeakerTurn(start_sec=0.0, end_sec=1.5, speaker_cluster_id="speaker_1", confidence="medium"),
            SpeakerTurn(start_sec=1.5, end_sec=5.0, speaker_cluster_id="speaker_2", confidence="medium"),
        ]
        with mock.patch("call_assistant.diarization.service._run_pyannote", return_value=turns):
            result = diarize(raw, self.config, audio_path=audio_path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].speaker_cluster_id, "speaker_1")
        self.assertEqual(result[0].text, "הלו.")
        self.assertEqual(result[1].speaker_cluster_id, "speaker_2")
        self.assertEqual(result[1].text, "שלום זה שליר. שלום.")

    def test_apply_speaker_mapping_preserves_cluster_id(self) -> None:
        mapped = apply_speaker_mapping(
            [
                {
                    "segment_id": "seg_0001",
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_cluster_id": "speaker_1",
                    "speaker_label": "speaker_1",
                    "speaker_channel_label": None,
                    "text": "Алло",
                    "confidence": None,
                    "diarization_confidence": "medium",
                }
            ],
            {"speaker_1": "me"},
        )
        self.assertEqual(mapped[0]["speaker_cluster_id"], "speaker_1")
        self.assertEqual(mapped[0]["speaker_label"], "me")

    def test_apply_speaker_mapping_accepts_transcript_segment_objects(self) -> None:
        mapped = apply_speaker_mapping(
            [
                TranscriptSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=1.0,
                    speaker_cluster_id="speaker_1",
                    speaker_label="speaker_1",
                    speaker_channel_label=None,
                    text="Алло",
                    confidence=None,
                    diarization_confidence="medium",
                )
            ],
            {"speaker_1": "me"},
        )
        self.assertEqual(mapped[0]["speaker_cluster_id"], "speaker_1")
        self.assertEqual(mapped[0]["speaker_label"], "me")

    def test_clean_transcript_renders_cluster_labels(self) -> None:
        clean = clean_transcript(
            [
                TranscriptSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=1.0,
                    speaker_cluster_id="speaker_1",
                    speaker_label="speaker_1",
                    speaker_channel_label=None,
                    text="Алло",
                    confidence=None,
                    diarization_confidence="medium",
                )
            ]
        )
        self.assertIn("SPEAKER 1", clean.text)

    def test_fallback_analysis_short_summary_is_not_first_line(self) -> None:
        segments = [
            TranscriptSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=1.0,
                speaker_cluster_id="speaker_1",
                speaker_label="speaker_1",
                speaker_channel_label=None,
                text="Алло",
                confidence=None,
                diarization_confidence="medium",
            )
        ]
        clean = clean_transcript(segments)
        analysis = _fallback_analysis(clean, segments)
        self.assertNotEqual(analysis.short_summary, "Алло")
        self.assertIn("automatic summary confidence is low", analysis.short_summary.lower())


if __name__ == "__main__":
    unittest.main()
