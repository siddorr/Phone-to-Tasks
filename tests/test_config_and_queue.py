from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
import types
import sys
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import write_json
from call_assistant.ingest.watcher import file_sha256
from call_assistant.ingest.watcher import import_file
from call_assistant.ingest.watcher import already_imported_source_path
from call_assistant.indexing.service import index_call
from call_assistant.orchestrator.queue import _stale_after_seconds_for_job, claim_next_job, complete_job, enqueue, reset_running_jobs_on_startup
from call_assistant.orchestrator.worker import WorkerThread, processing_mode
from call_assistant.reprocess import reset_all_calls_for_retranscription
from call_assistant.reprocess import reset_call_for_retranscription
from call_assistant.transcription.service import _looks_mixed_language_problem, _transcribe_local, _transcription_preferences
from call_assistant.ui.app import _compare_call_rows, _filter_calls_by_recent, _format_duration, _normalize_call_sort, _queue_manual_process_notice


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
processing:
  startup_mode: "manual_step"
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

    def test_transcription_stale_timeout_scales_with_duration(self) -> None:
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
            "current_state": "audio_prepared",
            "review_state": "pending",
            "low_confidence": False,
            "duration_seconds": 1200,
            "errors": [],
        }
        write_json(archive_path / "metadata.json", metadata)
        db.execute("UPDATE calls SET duration_seconds = ? WHERE call_id = ?", (1200, call_id))
        db.commit()

        row = {
            "call_id": call_id,
            "stage": "transcription",
        }
        self.assertEqual(_stale_after_seconds_for_job(self.config, row), 4200)

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
        self.assertTrue(already_imported_source_path(self.config, sample))

    def test_reset_running_jobs_on_startup_requeues_jobs(self) -> None:
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts,
                available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "running-job",
                "call_1",
                "transcription",
                "running",
                0,
                1,
                2,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        db.commit()

        recovered = reset_running_jobs_on_startup(self.config)
        self.assertEqual(recovered, 1)

        row = db.execute("SELECT status, error_message FROM queue_jobs WHERE job_id = ?", ("running-job",)).fetchone()
        self.assertEqual(row["status"], "queued")
        self.assertIn("startup", row["error_message"].lower())

    def test_processing_mode_defaults_to_manual_step_from_config(self) -> None:
        self.assertEqual(processing_mode(self.config), "manual_step")

    def test_worker_start_skips_background_thread_in_manual_mode(self) -> None:
        worker = WorkerThread(self.config)
        with patch("call_assistant.orchestrator.worker.reset_running_jobs_on_startup") as reset_mock:
            worker.start()
        self.assertFalse(worker._started)
        reset_mock.assert_not_called()

    def test_worker_start_keeps_automatic_mode_behavior(self) -> None:
        self.config.data["processing"]["startup_mode"] = "automatic"
        worker = WorkerThread(self.config)
        with patch("call_assistant.orchestrator.worker.reset_running_jobs_on_startup", return_value=0) as reset_mock:
            with patch.object(worker._thread, "start") as thread_start:
                worker.start()
        self.assertTrue(worker._started)
        reset_mock.assert_called_once()
        thread_start.assert_called_once()

    def test_queue_manual_process_notice_runs_manual_step(self) -> None:
        with patch("call_assistant.ui.app.run_manual_step", return_value=(2, True)) as run_step:
            notice = _queue_manual_process_notice(self.config)
        self.assertEqual(notice, "Imported 2 new calls; processed 1 job.")
        run_step.assert_called_once_with(self.config)

    def test_queue_manual_process_notice_rejects_automatic_mode(self) -> None:
        self.config.data["processing"]["startup_mode"] = "automatic"
        with patch("call_assistant.ui.app.run_manual_step") as run_step:
            notice = _queue_manual_process_notice(self.config)
        self.assertEqual(notice, "Manual processing is disabled in automatic mode.")
        run_step.assert_not_called()

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
            {
                "transcription_preference": {"provider": "cloud", "model": "whisper-1"},
                "transcription_language_override": "he",
                "transcription_language_mode": "metadata_override",
            },
        )
        provider, model, language_override, language_mode = _transcription_preferences(call_dir / "audio_normalized.wav", self.config)
        self.assertEqual(provider, "cloud")
        self.assertEqual(model, "whisper-1")
        self.assertEqual(language_override, "he")
        self.assertEqual(language_mode, "metadata_override")

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

        reset_call_for_retranscription(self.config, call_id, archive_path, "local:large-v3-turbo", "he")

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
        self.assertIn('"transcription_language_override": "he"', metadata_json)
        self.assertFalse((archive_path / "transcript_raw.json").exists())
        self.assertFalse((archive_path / "summary.json").exists())

    def test_reset_all_calls_for_retranscription_queues_each_call(self) -> None:
        first = self.config.incoming_folder / "first.wav"
        second = self.config.incoming_folder / "second.wav"
        first.write_bytes(b"abc")
        second.write_bytes(b"def")

        first_call_id = import_file(first, self.config)
        second_call_id = import_file(second, self.config)
        self.assertIsNotNone(first_call_id)
        self.assertIsNotNone(second_call_id)

        db = connect(self.config.sqlite_path)
        reset = reset_all_calls_for_retranscription(self.config, "local:large-v3-turbo")
        self.assertEqual(reset, [first_call_id, second_call_id])

        rows = db.execute(
            "SELECT call_id, stage FROM queue_jobs WHERE stage = 'transcription' ORDER BY call_id"
        ).fetchall()
        self.assertEqual(
            [(row["call_id"], row["stage"]) for row in rows],
            [(first_call_id, "transcription"), (second_call_id, "transcription")],
        )


class CallsPageTests(unittest.TestCase):
    def _call(
        self,
        *,
        call_id: str,
        respondent: str,
        recorded_at: datetime,
        summary: str,
        task_count: int = 0,
        state: str = "indexed",
    ) -> dict:
        return {
            "call_id": call_id,
            "recorded_at": recorded_at.isoformat(),
            "imported_at": recorded_at.isoformat(),
            "display_recorded_at": recorded_at.isoformat(),
            "display_respondent": respondent,
            "current_state": state,
            "display_task_count": task_count,
            "display_short_description": summary,
        }

    def test_calls_page_sorts_by_respondent_ascending(self) -> None:
        now = datetime.now(timezone.utc)
        aaron = self._call(call_id="call_a", respondent="Aaron", recorded_at=now - timedelta(hours=1), summary="First")
        bella = self._call(call_id="call_b", respondent="Bella", recorded_at=now - timedelta(hours=2), summary="Second")
        self.assertLess(_compare_call_rows(aaron, bella, "respondent", "asc"), 0)

    def test_calls_page_recent_filter_last_day_excludes_older_calls(self) -> None:
        now = datetime.now(timezone.utc)
        recent_call = self._call(
            call_id="recent_call",
            respondent="Recent",
            recorded_at=now - timedelta(hours=6),
            summary="Fresh",
        )
        old_call = self._call(call_id="old_call", respondent="Older", recorded_at=now - timedelta(days=8), summary="Stale")

        filtered = _filter_calls_by_recent([recent_call, old_call], "1d")

        self.assertEqual([item["call_id"] for item in filtered], ["recent_call"])

    def test_calls_page_exact_date_overrides_recent_filter(self) -> None:
        self.assertEqual(_normalize_call_sort("invalid", "invalid"), ("recorded_at", "desc"))
        self.assertEqual(_normalize_call_sort("tasks", "invalid"), ("tasks", "desc"))

    def test_format_duration(self) -> None:
        self.assertEqual(_format_duration(None), "-")
        self.assertEqual(_format_duration(65), "01:05")
        self.assertEqual(_format_duration(3665), "1:01:05")

    def test_transcribe_local_auto_mode_does_not_force_language(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "shalom привет", "segments": [], "language": "he"}
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        config_path = root / "config.yaml"
        config_path.write_text(
            """
paths:
  incoming_folder: "./incoming"
  archive_root: "./calls"
  sqlite_path: "./index/test.db"
  logs_dir: "./logs"
  temp_dir: "./temp"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {})

        fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=mock_model))
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            _transcribe_local(audio_path, config, language_mode="auto")

        self.assertIsNone(mock_model.transcribe.call_args.kwargs["language"])
        temp_dir.cleanup()

    def test_mixed_language_problem_detects_hebrew_in_non_hebrew_result(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        config_path = root / "config.yaml"
        config_path.write_text(
            """
paths:
  incoming_folder: "./incoming"
  archive_root: "./calls"
  sqlite_path: "./index/test.db"
  logs_dir: "./logs"
  temp_dir: "./temp"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 45})

        raw = type(
            "RawLike",
            (),
            {"text": "Привет שלום", "segments": [1], "language": "ru"},
        )()

        self.assertTrue(_looks_mixed_language_problem(raw, audio_path))
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
