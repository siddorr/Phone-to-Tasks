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
from call_assistant.common.db import connect
from call_assistant.common.io import write_json
from call_assistant.ingest.watcher import file_sha256
from call_assistant.ingest.watcher import import_file
from call_assistant.indexing.service import index_call
from call_assistant.orchestrator.queue import claim_next_job, complete_job, enqueue
from call_assistant.transcription.service import _transcription_preferences
from call_assistant.ui.app import _reset_call_for_retranscription


class QueueTests(unittest.TestCase):
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
ingest:
  supported_extensions: [".wav"]
  stability_check_seconds: 1
  stability_poll_interval_seconds: 1
  file_hash_algorithm: "sha256"
  scan_interval_seconds: 1
""".strip()
            + "\n",
            encoding="utf-8",
        )
        self.config = AppConfig.load(self.config_path)
        self.config.ensure_directories()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_enqueue_claim_complete(self) -> None:
        enqueue(self.config, "call_1", "audio_prepare")
        job = claim_next_job(self.config)
        self.assertIsNotNone(job)
        self.assertEqual(job.call_id, "call_1")
        complete_job(self.config, job.job_id)

    def test_claim_next_job_reclaims_stale_running_job(self) -> None:
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts,
                available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "stale-job",
                "call_stale",
                "analysis",
                "running",
                0,
                1,
                2,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts,
                available_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "queued-job",
                "call_queued",
                "audio_prepare",
                "queued",
                0,
                0,
                2,
                "2026-01-01T00:00:01+00:00",
            ),
        )
        db.commit()

        job = claim_next_job(self.config)
        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, "stale-job")

        reclaimed = db.execute("SELECT status, error_message FROM queue_jobs WHERE job_id = ?", ("stale-job",)).fetchone()
        self.assertEqual(reclaimed["status"], "running")
        self.assertIn("stale", reclaimed["error_message"])

    def test_file_sha256(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")
        self.assertEqual(file_sha256(sample), file_sha256(sample))

    def test_import_registers_call_before_indexing_and_dedups(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        first_call_id = import_file(sample, self.config)
        self.assertIsNotNone(first_call_id)

        db = connect(self.config.sqlite_path)
        row = db.execute("SELECT call_id, sha256, current_state FROM calls").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["call_id"], first_call_id)
        self.assertEqual(row["current_state"], "imported")

        second_call_id = import_file(sample, self.config)
        self.assertIsNone(second_call_id)
        total = db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        self.assertEqual(total, 1)

    def test_index_call_updates_existing_imported_row_state(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )

        metadata = {
            **connect(self.config.sqlite_path)
            .execute("SELECT * FROM calls WHERE call_id = ?", (call_id,))
            .fetchone(),
            "call_id": call_id,
            "schema_version": "1.0",
            "app_version": "test",
            "source_filename": sample.name,
            "source_path": str(sample),
            "file_size_bytes": sample.stat().st_size,
            "language_hints": ["en"],
            "current_state": "transcribed",
            "audio_format": "wav",
            "duration_seconds": 1.5,
            "errors": [],
        }
        write_json(archive_path / "metadata.json", dict(metadata))
        write_json(archive_path / "tasks_reviewed.json", [])

        index_call(archive_path, self.config)

        row = db.execute(
            "SELECT current_state, audio_format, duration_seconds FROM calls WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        self.assertEqual(row["current_state"], "transcribed")
        self.assertEqual(row["audio_format"], "wav")
        self.assertEqual(row["duration_seconds"], 1.5)

    def test_index_call_handles_duplicate_task_ids_within_call(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )

        metadata = {
            "schema_version": "1.0",
            "app_version": "test",
            "call_id": call_id,
            "source_filename": sample.name,
            "source_path": str(sample),
            "imported_at": db.execute(
                "SELECT imported_at FROM calls WHERE call_id = ?",
                (call_id,),
            ).fetchone()["imported_at"],
            "recorded_at": None,
            "file_size_bytes": sample.stat().st_size,
            "sha256": db.execute("SELECT sha256 FROM calls WHERE call_id = ?", (call_id,)).fetchone()["sha256"],
            "language_hints": ["en"],
            "current_state": "indexed",
            "review_state": "pending",
            "low_confidence": False,
            "errors": [],
        }
        write_json(archive_path / "metadata.json", metadata)
        write_json(
            archive_path / "tasks.json",
            [
                {"task_id": "dup", "text": "First task"},
                {"task_id": "dup", "text": "Second task"},
            ],
        )
        write_json(archive_path / "tasks_reviewed.json", [])

        index_call(archive_path, self.config)

        total = db.execute("SELECT COUNT(*) FROM tasks WHERE call_id = ?", (call_id,)).fetchone()[0]
        self.assertEqual(total, 2)

    def test_index_call_handles_same_task_id_across_calls(self) -> None:
        first = self.config.incoming_folder / "first.wav"
        second = self.config.incoming_folder / "second.wav"
        first.write_bytes(b"abc")
        second.write_bytes(b"def")

        first_call_id = import_file(first, self.config)
        second_call_id = import_file(second, self.config)
        self.assertIsNotNone(first_call_id)
        self.assertIsNotNone(second_call_id)

        db = connect(self.config.sqlite_path)
        for call_id, sample in ((first_call_id, first), (second_call_id, second)):
            archive_path = Path(
                db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
            )
            metadata = {
                "schema_version": "1.0",
                "app_version": "test",
                "call_id": call_id,
                "source_filename": sample.name,
                "source_path": str(sample),
                "imported_at": db.execute(
                    "SELECT imported_at FROM calls WHERE call_id = ?",
                    (call_id,),
                ).fetchone()["imported_at"],
                "recorded_at": None,
                "file_size_bytes": sample.stat().st_size,
                "sha256": db.execute("SELECT sha256 FROM calls WHERE call_id = ?", (call_id,)).fetchone()["sha256"],
                "language_hints": ["en"],
                "current_state": "indexed",
                "review_state": "pending",
                "low_confidence": False,
                "errors": [],
            }
            write_json(archive_path / "metadata.json", metadata)
            write_json(archive_path / "tasks.json", [{"task_id": "same", "text": f"Task for {call_id}"}])
            write_json(archive_path / "tasks_reviewed.json", [])
            index_call(archive_path, self.config)

        total = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        self.assertEqual(total, 2)

    def test_transcription_preferences_read_call_override(self) -> None:
        call_dir = self.config.archive_root / "2026" / "04" / "04" / "call_test"
        call_dir.mkdir(parents=True)
        write_json(
            call_dir / "metadata.json",
            {"transcription_preference": {"provider": "cloud", "model": "whisper-1"}},
        )
        provider, model = _transcription_preferences(call_dir / "audio_normalized.wav", self.config)
        self.assertEqual(provider, "cloud")
        self.assertEqual(model, "whisper-1")

    def test_reset_call_for_retranscription_clears_downstream_and_enqueues(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")
        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )

        write_json(
            archive_path / "metadata.json",
            {
                "call_id": call_id,
                "current_state": "indexed",
                "review_state": "reviewed",
                "errors": ["old error"],
            },
        )
        write_json(archive_path / "tasks_reviewed.json", [{"task_id": "x"}])
        write_json(archive_path / "transcript_raw.json", {"text": "old"})
        write_json(archive_path / "transcript_segments.json", [{"text": "old"}])
        (archive_path / "transcript_clean.txt").write_text("old", encoding="utf-8")
        write_json(archive_path / "summary.json", {"short_summary": "old"})
        write_json(archive_path / "tasks.json", [{"task_id": "old"}])

        _reset_call_for_retranscription(self.config, call_id, archive_path, "local:large-v3-turbo")

        metadata = db.execute(
            "SELECT current_state, review_state FROM calls WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        self.assertEqual(metadata["current_state"], "audio_prepared")
        self.assertEqual(metadata["review_state"], "pending")

        job = db.execute(
            "SELECT stage, status FROM queue_jobs WHERE call_id = ? ORDER BY available_at DESC LIMIT 1",
            (call_id,),
        ).fetchone()
        self.assertEqual(job["stage"], "transcription")
        self.assertEqual(job["status"], "queued")

        metadata_json = (archive_path / "metadata.json").read_text(encoding="utf-8")
        self.assertIn("large-v3-turbo", metadata_json)
        self.assertFalse((archive_path / "transcript_raw.json").exists())
        self.assertFalse((archive_path / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
