from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.ingest.recorded_time import fallback_recorded_at_from_mtime
from call_assistant.ingest.recorded_time import parse_recorded_at_from_filename
from call_assistant.ingest.recorded_time import resolve_recorded_at


class RecordedTimeTests(unittest.TestCase):
    def test_parse_recorded_at_from_filename(self) -> None:
        recorded_at, source = parse_recorded_at_from_filename(Path("Call recording test_250216_100928.m4a"))
        self.assertEqual(source, "filename")
        self.assertIsNotNone(recorded_at)
        self.assertIn("2025-02-16T10:09:28", recorded_at)

    def test_resolve_recorded_at_prefers_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Call recording test_250216_100928.wav"
            path.write_bytes(b"abc")
            recorded_at, source, confidence = resolve_recorded_at(path)
            self.assertEqual(source, "filename")
            self.assertEqual(confidence, "high")
            self.assertIn("2025-02-16T10:09:28", recorded_at)

    def test_fallback_recorded_at_from_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            path.write_bytes(b"abc")
            timestamp = datetime(2024, 1, 2, 3, 4, 5).timestamp()
            path.touch()
            import os
            os.utime(path, (timestamp, timestamp))
            recorded_at = fallback_recorded_at_from_mtime(path)
            expected = datetime.fromtimestamp(timestamp, datetime.now().astimezone().tzinfo).isoformat()
            self.assertEqual(expected, recorded_at)


if __name__ == "__main__":
    unittest.main()
