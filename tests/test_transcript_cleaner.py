from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.models import TranscriptSegment
from call_assistant.transcript_cleaner.service import clean_transcript


class TranscriptCleanerTests(unittest.TestCase):
    def test_clean_transcript_preserves_order(self) -> None:
        segments = [
            TranscriptSegment("seg_1", 0.0, 1.0, "unknown", None, "Hello   there!!!", None),
            TranscriptSegment("seg_2", 1.0, 2.0, "unknown", None, "Need   to   send docs", None),
        ]
        clean = clean_transcript(segments)
        self.assertIn("[0000.00] UNKNOWN: Hello there!", clean.text)
        self.assertIn("[0001.00] UNKNOWN: Need to send docs", clean.text)


if __name__ == "__main__":
    unittest.main()
