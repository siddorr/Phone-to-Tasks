from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawSegment, RawTranscript, TranscriptSegment
from call_assistant.diarization.service import apply_speaker_mapping, diarize
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


if __name__ == "__main__":
    unittest.main()
