from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import openai


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_transcription_comparison.py"
SPEC = importlib.util.spec_from_file_location("generate_transcription_comparison", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ComparisonScriptTests(unittest.TestCase):
    def test_parse_model_names_rejects_unknown_model(self) -> None:
        with self.assertRaises(SystemExit):
            MODULE.parse_model_names("small,not-a-model")

    def test_classify_openai_exception_transport(self) -> None:
        exc = openai.APIConnectionError(request=None)
        self.assertEqual(MODULE.classify_openai_exception(exc), "transport")

    def test_write_report_includes_failure_kind(self) -> None:
        audio_path = ROOT / "tests" / "dummy_audio.m4a"
        audio_path.write_bytes(b"dummy")
        self.addCleanup(audio_path.unlink)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.txt"
            original_output = MODULE.OUTPUT_PATH
            MODULE.OUTPUT_PATH = output_path
            try:
                MODULE.write_report(
                    [audio_path],
                    {
                        audio_path.name: [
                            MODULE.TranscriptRun(
                                model_name="whisper-1",
                                provider="openai",
                                ok=False,
                                language=None,
                                elapsed_seconds=1.2,
                                text="",
                                error="APIConnectionError: Connection error.",
                                failure_kind="transport",
                                input_path="/tmp/openai_compare_call1_x.wav",
                            )
                        ]
                    },
                    include_local=False,
                    include_openai=True,
                    openai_mode="shared-client-sequential",
                )
                text = output_path.read_text(encoding="utf-8")
            finally:
                MODULE.OUTPUT_PATH = original_output

        self.assertIn("failure_kind: transport", text)
        self.assertIn("OpenAI inputs are normalized", text)
        self.assertIn("OpenAI request mode: shared-client-sequential", text)

    def test_build_openai_client_without_key(self) -> None:
        import os

        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            self.assertIsNone(MODULE.build_openai_client())
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_write_report_handles_partial_results(self) -> None:
        audio_one = ROOT / "tests" / "dummy_audio_one.m4a"
        audio_two = ROOT / "tests" / "dummy_audio_two.m4a"
        audio_one.write_bytes(b"one")
        audio_two.write_bytes(b"two")
        self.addCleanup(audio_one.unlink)
        self.addCleanup(audio_two.unlink)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.txt"
            original_output = MODULE.OUTPUT_PATH
            MODULE.OUTPUT_PATH = output_path
            try:
                MODULE.write_report(
                    [audio_one, audio_two],
                    {
                        audio_one.name: [
                            MODULE.TranscriptRun(
                                model_name="small",
                                provider="local",
                                ok=True,
                                language="ru",
                                elapsed_seconds=1.0,
                                text="Алло",
                                input_path=str(audio_one),
                            )
                        ]
                    },
                    include_local=True,
                    include_openai=False,
                    openai_mode="shared-client-sequential",
                )
                text = output_path.read_text(encoding="utf-8")
            finally:
                MODULE.OUTPUT_PATH = original_output

        self.assertIn(audio_one.name, text)
        self.assertIn(audio_two.name, text)


if __name__ == "__main__":
    unittest.main()
