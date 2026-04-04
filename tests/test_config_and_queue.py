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
from call_assistant.common.io import read_json, write_json
from call_assistant.common.models import TranscriptSegment
from call_assistant.ingest.watcher import file_sha256
from call_assistant.ingest.watcher import import_file
from call_assistant.ingest.watcher import already_imported_source_path
from call_assistant.indexing.service import index_call
from call_assistant.orchestrator.queue import _stale_after_seconds_for_job, claim_next_job, claim_next_job_for_call, complete_job, enqueue, reset_running_jobs_on_startup
from call_assistant.orchestrator.worker import WorkerThread, _process_diarization, _process_transcript_clean, _run_claimed_job, processing_mode
from call_assistant.reprocess import reset_all_calls_for_retranscription
from call_assistant.reprocess import reset_call_for_retranscription
from call_assistant.transcription.service import _looks_mixed_language_problem, _transcribe_local, _transcription_preferences
from call_assistant.ui.app import _app_status_payload, _compare_call_rows, _diarization_context, _estimate_stage_runtime_seconds, _filter_calls_by_recent, _format_duration, _manual_processing_notice, _normalize_call_sort


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
        with patch("call_assistant.ui.app.run_manual_step", return_value=(2, "call_123", 4)) as run_step:
            notice = _manual_processing_notice(self.config)
        self.assertEqual(notice, "Imported 2 new calls; completed 4 stage(s) for call call_123.")
        run_step.assert_called_once_with(self.config)

    def test_queue_manual_process_notice_rejects_automatic_mode(self) -> None:
        self.config.data["processing"]["startup_mode"] = "automatic"
        with patch("call_assistant.ui.app.run_manual_step") as run_step:
            notice = _manual_processing_notice(self.config)
        self.assertEqual(notice, "Manual processing is disabled in automatic mode.")
        run_step.assert_not_called()

    def test_claim_next_job_for_call_claims_only_target_call(self) -> None:
        enqueue(self.config, "call_a", "audio_prepare")
        enqueue(self.config, "call_b", "audio_prepare")

        job = claim_next_job_for_call(self.config, "call_b")

        self.assertIsNotNone(job)
        self.assertEqual(job.call_id, "call_b")

    def test_app_status_payload_idle_when_no_jobs_exist(self) -> None:
        db = connect(self.config.sqlite_path)
        payload = _app_status_payload(self.config, db)
        self.assertEqual(payload["app_status"], "Idle")
        self.assertEqual(payload["counts"], {"running": 0, "queued": 0, "failed": 0})
        self.assertIn("server_now", payload)
        self.assertIsNone(payload["current_task"])

    def test_app_status_payload_paused_in_manual_mode_with_queued_jobs(self) -> None:
        enqueue(self.config, "call_queued", "audio_prepare")
        db = connect(self.config.sqlite_path)
        payload = _app_status_payload(self.config, db)
        self.assertEqual(payload["app_status"], "Paused")
        self.assertEqual(payload["counts"]["queued"], 1)

    def test_app_status_payload_uses_oldest_running_job(self) -> None:
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO calls (
                call_id, archive_path, source_filename, source_path, sha256, imported_at,
                current_state, review_state, low_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "call_old",
                str(self.config.archive_root / "call_old"),
                "old.wav",
                "old.wav",
                "sha_old",
                "2026-01-01T00:00:00+00:00",
                "diarizing",
                "pending",
                0,
            ),
        )
        db.execute(
            """
            INSERT INTO calls (
                call_id, archive_path, source_filename, source_path, sha256, imported_at,
                current_state, review_state, low_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "call_new",
                str(self.config.archive_root / "call_new"),
                "new.wav",
                "new.wav",
                "sha_new",
                "2026-01-01T00:00:01+00:00",
                "diarizing",
                "pending",
                0,
            ),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("job_new", "call_new", "diarization", "running", 0, 1, 2, "2026-01-01T00:00:02+00:00", "2026-01-01T00:00:20+00:00"),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("job_old", "call_old", "analysis", "running", 0, 1, 2, "2026-01-01T00:00:03+00:00", "2026-01-01T00:00:10+00:00"),
        )
        db.commit()

        payload = _app_status_payload(self.config, db)
        self.assertEqual(payload["app_status"], "Running")
        self.assertEqual(payload["current_task"]["call_id"], "call_old")
        self.assertEqual(payload["current_task"]["stage"], "analysis")
        self.assertIn("started_at", payload["current_task"])
        self.assertIn("estimated_finish_at", payload["current_task"])
        self.assertIn("elapsed_seconds", payload["current_task"])
        self.assertIn("remaining_seconds", payload["current_task"])

    def test_estimate_stage_runtime_seconds_uses_stage_default_without_history(self) -> None:
        db = connect(self.config.sqlite_path)
        running_job = {"stage": "analysis", "duration_seconds": None}
        self.assertEqual(_estimate_stage_runtime_seconds(db, running_job), 45.0)

    def test_estimate_stage_runtime_seconds_uses_audio_ratio_for_diarization(self) -> None:
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO calls (
                call_id, archive_path, source_filename, source_path, sha256, imported_at,
                duration_seconds, current_state, review_state, low_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "call_hist",
                str(self.config.archive_root / "call_hist"),
                "hist.wav",
                "hist.wav",
                "sha_hist",
                "2026-01-01T00:00:00+00:00",
                100.0,
                "diarized",
                "pending",
                0,
            ),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job_hist",
                "call_hist",
                "diarization",
                "done",
                0,
                1,
                2,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:02:30+00:00",
            ),
        )
        db.commit()

        estimate = _estimate_stage_runtime_seconds(db, {"stage": "diarization", "duration_seconds": 200.0})
        self.assertEqual(estimate, 300.0)

    def test_app_status_payload_recalculates_eta_after_estimate_is_exceeded(self) -> None:
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO calls (
                call_id, archive_path, source_filename, source_path, sha256, imported_at,
                current_state, review_state, low_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "call_recalc",
                str(self.config.archive_root / "call_recalc"),
                "recalc.wav",
                "recalc.wav",
                "sha_recalc",
                "2026-01-01T00:00:00+00:00",
                "analyzing",
                "pending",
                0,
            ),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job_recalc",
                "call_recalc",
                "analysis",
                "running",
                0,
                1,
                2,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        db.commit()

        payload = _app_status_payload(self.config, db)
        self.assertEqual(payload["current_task"]["eta_status"], "recalculating")
        self.assertIsNotNone(payload["current_task"]["eta_extension_seconds"])
        self.assertIn("recalculating", payload["current_task"]["estimated_finish_display"])

    def test_diarization_context_detects_single_speaker_fallback(self) -> None:
        context = _diarization_context(
            [{"event": "fallback_diarization_completed", "mode": "single_speaker"}],
            [{"speaker_cluster_id": "speaker_1"}],
        )
        self.assertEqual(context["diarization_mode"], "single_speaker_fallback")
        self.assertIn("single-speaker fallback", context["diarization_notice"])

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

    def test_process_transcript_clean_writes_clean_transcript_and_updates_state(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )
        call_row = db.execute(
            "SELECT source_filename, source_path, sha256, imported_at FROM calls WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        write_json(
            archive_path / "metadata.json",
            {
                "call_id": call_id,
                "source_filename": call_row["source_filename"],
                "source_path": call_row["source_path"],
                "sha256": call_row["sha256"],
                "imported_at": call_row["imported_at"],
                "current_state": "speaker_identity",
                "review_state": "pending",
                "errors": [],
            },
        )
        write_json(
            archive_path / "transcript_segments.json",
            [
                {
                    "segment_id": "seg_1",
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_cluster_id": "speaker_1",
                    "speaker_label": "speaker_1",
                    "speaker_channel_label": None,
                    "text": "Hello   there!!!",
                    "confidence": None,
                    "speaker_display_name": None,
                    "speaker_identity_id": None,
                    "identity_confidence": None,
                    "diarization_confidence": None,
                }
            ],
        )

        _process_transcript_clean(self.config, archive_path)

        self.assertTrue((archive_path / "transcript_clean.txt").exists())
        row = db.execute("SELECT current_state FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        self.assertEqual(row["current_state"], "transcript_clean")

    def test_process_diarization_accepts_transcript_segment_objects(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )
        call_row = db.execute(
            "SELECT source_filename, source_path, sha256, imported_at FROM calls WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        write_json(
            archive_path / "metadata.json",
            {
                "call_id": call_id,
                "source_filename": call_row["source_filename"],
                "source_path": call_row["source_path"],
                "sha256": call_row["sha256"],
                "imported_at": call_row["imported_at"],
                "current_state": "transcribed",
                "review_state": "pending",
                "errors": [],
            },
        )
        write_json(
            archive_path / "transcript_raw.json",
            {
                "provider": "test",
                "model": "test",
                "language": "en",
                "confidence": 1.0,
                "text": "Hello",
                "segments": [{"start_sec": 0.0, "end_sec": 1.0, "text": "Hello"}],
            },
        )

        segments = [
            TranscriptSegment(
                segment_id="seg_1",
                start_sec=0.0,
                end_sec=1.0,
                speaker_cluster_id="speaker_1",
                speaker_label="speaker_1",
                speaker_channel_label=None,
                text="Hello",
                confidence=None,
                diarization_confidence="medium",
            )
        ]
        with patch("call_assistant.orchestrator.worker.transcribe_data_to_segments", return_value=segments):
            _process_diarization(self.config, archive_path)

        metadata = read_json(archive_path / "metadata.json", default={})
        self.assertEqual(metadata.get("current_state"), "diarized")
        self.assertEqual(metadata.get("diarization_mode"), "single_speaker_fallback")
        self.assertEqual(metadata.get("stage_outcomes", {}).get("diarization", {}).get("status"), "degraded")

    def test_run_claimed_job_clears_stale_errors_after_success(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )
        metadata = read_json(archive_path / "metadata.json", default={})
        metadata["current_state"] = "failed"
        metadata["errors"] = ["old failure"]
        write_json(archive_path / "metadata.json", metadata)
        db.execute("UPDATE calls SET current_state = 'failed', last_error = 'old failure' WHERE call_id = ?", (call_id,))
        db.commit()

        job = types.SimpleNamespace(job_id="job_1", call_id=call_id, stage="transcript_clean", attempt_count=1)

        with patch("call_assistant.orchestrator.worker.process_job") as process_mock:
            with patch("call_assistant.orchestrator.worker.complete_job") as complete_mock:
                result = _run_claimed_job(self.config, job)

        self.assertTrue(result)
        process_mock.assert_called_once_with(self.config, job)
        complete_mock.assert_called_once_with(self.config, "job_1")

        metadata = read_json(archive_path / "metadata.json", default={})
        self.assertEqual(metadata.get("errors"), [])
        row = db.execute("SELECT last_error FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        self.assertIsNone(row["last_error"])

    def test_run_claimed_job_retryable_failure_preserves_last_stable_state(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )
        metadata = read_json(archive_path / "metadata.json", default={})
        metadata["current_state"] = "transcribing"
        write_json(archive_path / "metadata.json", metadata)
        db.execute("UPDATE calls SET current_state = 'transcribing' WHERE call_id = ?", (call_id,))
        db.commit()

        job = types.SimpleNamespace(job_id="job_retry", call_id=call_id, stage="transcription", attempt_count=1, max_attempts=3)

        with patch("call_assistant.orchestrator.worker.process_job", side_effect=RuntimeError("boom")):
            result = _run_claimed_job(self.config, job)

        self.assertTrue(result)
        metadata = read_json(archive_path / "metadata.json", default={})
        self.assertEqual(metadata.get("current_state"), "audio_prepared")
        self.assertEqual(metadata.get("last_blocking_stage"), "transcription")
        self.assertEqual(metadata.get("last_blocking_error"), "boom")
        self.assertEqual(metadata.get("stage_outcomes", {}).get("transcription", {}).get("status"), "failed_retryable")

    def test_run_claimed_job_terminal_failure_marks_call_failed(self) -> None:
        sample = self.config.incoming_folder / "sample.wav"
        sample.write_bytes(b"abc")

        call_id = import_file(sample, self.config)
        self.assertIsNotNone(call_id)

        db = connect(self.config.sqlite_path)
        archive_path = Path(
            db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (call_id,)).fetchone()["archive_path"]
        )
        metadata = read_json(archive_path / "metadata.json", default={})
        metadata["current_state"] = "analyzing"
        write_json(archive_path / "metadata.json", metadata)
        db.execute("UPDATE calls SET current_state = 'analyzing' WHERE call_id = ?", (call_id,))
        db.commit()

        job = types.SimpleNamespace(job_id="job_fail", call_id=call_id, stage="analysis", attempt_count=3, max_attempts=3)

        with patch("call_assistant.orchestrator.worker.process_job", side_effect=RuntimeError("kaput")):
            result = _run_claimed_job(self.config, job)

        self.assertTrue(result)
        metadata = read_json(archive_path / "metadata.json", default={})
        self.assertEqual(metadata.get("current_state"), "failed")
        self.assertEqual(metadata.get("last_blocking_stage"), "analysis")
        self.assertEqual(metadata.get("last_blocking_error"), "kaput")
        self.assertEqual(metadata.get("stage_outcomes", {}).get("analysis", {}).get("status"), "failed_terminal")

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
