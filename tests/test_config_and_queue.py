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
from call_assistant.orchestrator.queue import _stale_after_seconds_for_job, claim_next_job, claim_next_job_for_call, complete_job, enqueue, is_job_stale, reset_running_jobs_on_startup
from call_assistant.orchestrator.worker import WorkerThread, _process_diarization, _process_transcript_clean, _run_claimed_job, processing_mode, run_manual_step, run_once
from call_assistant.reprocess import reset_all_calls_for_retranscription
from call_assistant.reprocess import reset_call_for_retranscription
from call_assistant.common.models import RawSegment, RawTranscript
from call_assistant.transcription.service import (
    _MODEL_CACHE,
    _apply_combined_hebrew_retries,
    _apply_word_span_retries,
    _build_clause_retry_candidates,
    _build_word_span_candidates,
    _heuristic_word_span_semantic_validation,
    _load_local_model,
    _looks_mixed_language_problem,
    _replace_span_in_segment_text,
    _rescue_semantic_validation_with_normalization,
    _transcribe_local,
    _transcription_preferences,
    SegmentDetectionDecision,
    ClauseRetryCandidate,
    TranscriptQualityAssessment,
    WordRetryCandidate,
    WordSpanSemanticValidation,
    WordSpanDecision,
    assess_transcript_quality,
    compare_transcript_candidates,
    load_transcription_candidate_selection,
    merge_transcript_candidates,
    transcribe,
)
from call_assistant.ui.app import _app_status_payload, _clean_transcript_preview, _clear_queue_notice, _compare_call_rows, _compact_preview, _diarization_context, _estimate_stage_runtime_seconds, _filter_calls_by_recent, _format_duration, _manual_processing_notice, _normalize_call_sort, _processing_log_preview, _set_archive_root_notice, _set_processing_mode_notice, _summary_preview


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

    def test_run_manual_step_skips_scan_by_default(self) -> None:
        with patch("call_assistant.orchestrator.worker.detect_new_calls") as detect_mock:
            imported_count, call_id, stage_count = run_manual_step(self.config)
        detect_mock.assert_not_called()
        self.assertEqual(imported_count, 0)
        self.assertIsNone(call_id)
        self.assertEqual(stage_count, 0)

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

    def test_clear_queue_notice_deletes_queued_and_failed_and_switches_to_manual(self) -> None:
        self.config.data.setdefault("processing", {})["startup_mode"] = "automatic"
        self.config.save()
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("queued-job", "call_queued", "transcription", "queued", 0, 0, 2, "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("running-job", "call_running", "diarization", "running", 0, 1, 2, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:05+00:00"),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, finished_at, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("failed-job", "call_failed", "analysis", "failed", 0, 2, 2, "2026-01-01T00:00:00+00:00", "2026-01-01T00:01:00+00:00", "boom"),
        )
        db.commit()

        notice = _clear_queue_notice(self.config)

        self.assertIn("switched to manual_step", notice)
        self.assertIn("running job(s) may still finish", notice)
        self.assertEqual(processing_mode(self.config), "manual_step")
        rows = db.execute("SELECT job_id, status FROM queue_jobs ORDER BY job_id").fetchall()
        self.assertEqual([(row["job_id"], row["status"]) for row in rows], [("running-job", "running")])

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
        reset_mock.assert_called_once()

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
        self.assertEqual(notice, "Completed 4 stage(s) for call call_123.")
        run_step.assert_called_once_with(self.config, scan_first=False)

    def test_queue_manual_process_notice_rejects_automatic_mode(self) -> None:
        self.config.data["processing"]["startup_mode"] = "automatic"
        with patch("call_assistant.ui.app.run_manual_step") as run_step:
            notice = _manual_processing_notice(self.config)
        self.assertEqual(notice, "Manual processing is disabled in automatic mode.")
        run_step.assert_not_called()

    def test_run_once_drains_same_call_end_to_end(self) -> None:
        self.config.data["processing"]["startup_mode"] = "automatic"
        with (
            patch("call_assistant.orchestrator.worker.detect_new_calls", return_value=[]),
            patch("call_assistant.orchestrator.worker.claim_next_job") as claim_next,
            patch("call_assistant.orchestrator.worker.claim_next_job_for_call") as claim_for_call,
            patch("call_assistant.orchestrator.worker._run_claimed_job", return_value=True) as run_claimed,
        ):
            first_job = types.SimpleNamespace(call_id="call_1")
            second_job = types.SimpleNamespace(call_id="call_1")
            third_job = types.SimpleNamespace(call_id="call_1")
            claim_next.return_value = first_job
            claim_for_call.side_effect = [second_job, third_job, None]

            processed = run_once(self.config, scan_first=False)

        self.assertTrue(processed)
        claim_next.assert_called_once_with(self.config)
        self.assertEqual(run_claimed.call_args_list, [unittest.mock.call(self.config, first_job), unittest.mock.call(self.config, second_job), unittest.mock.call(self.config, third_job)])

    def test_set_processing_mode_notice_updates_config_and_persists(self) -> None:
        notice = _set_processing_mode_notice(self.config, "automatic")
        self.assertEqual(notice, "Switched processing mode to automatic.")
        self.assertEqual(processing_mode(self.config), "automatic")
        reloaded = AppConfig.load(self.config_path)
        self.assertEqual(processing_mode(reloaded), "automatic")

    def test_set_archive_root_notice_updates_config_and_persists(self) -> None:
        target = Path(self.temp_dir.name) / "other_calls"
        notice = _set_archive_root_notice(self.config, str(target))
        self.assertEqual(notice, f"Switched calls folder to {target.resolve()}.")
        self.assertEqual(self.config.archive_root, target.resolve())
        self.assertTrue(target.exists())
        reloaded = AppConfig.load(self.config_path)
        self.assertEqual(reloaded.archive_root, target.resolve())

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

    def test_summary_preview_prefers_short_summary(self) -> None:
        preview = _summary_preview({"short_summary": "This is a compact summary preview for the call detail page."})
        self.assertIn("compact summary", preview)

    def test_clean_transcript_preview_uses_first_non_empty_line(self) -> None:
        preview = _clean_transcript_preview("\n\nFirst useful line\nSecond line")
        self.assertEqual(preview, "First useful line")

    def test_processing_log_preview_uses_latest_event(self) -> None:
        preview = _processing_log_preview(
            [
                {"event": "audio_prepare_started"},
                {"event": "analysis_finished", "at": "2026-01-01T10:00:00+00:00"},
            ]
        )
        self.assertIn("analysis_finished", preview)
        self.assertIn("2026-01-01", preview)

    def test_compact_preview_truncates_long_text(self) -> None:
        preview = _compact_preview("word " * 40, limit=20)
        self.assertTrue(preview.endswith("…"))

    def test_app_status_payload_uses_oldest_running_job(self) -> None:
        db = connect(self.config.sqlite_path)
        now = datetime.now(timezone.utc)
        started_old = (now - timedelta(seconds=40)).isoformat()
        started_new = (now - timedelta(seconds=30)).isoformat()
        available_old = (now - timedelta(seconds=50)).isoformat()
        available_new = (now - timedelta(seconds=45)).isoformat()
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
            ("job_new", "call_new", "diarization", "running", 0, 1, 2, available_new, started_new),
        )
        db.execute(
            """
            INSERT INTO queue_jobs (
                job_id, call_id, stage, status, priority, attempt_count, max_attempts, available_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("job_old", "call_old", "analysis", "running", 0, 1, 2, available_old, started_old),
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

    def test_is_job_stale_detects_old_running_job(self) -> None:
        row = {
            "call_id": "call_old",
            "stage": "analysis",
            "started_at": "2026-01-01T00:00:00+00:00",
        }
        self.assertTrue(is_job_stale(self.config, row, now=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)))

    def test_app_status_payload_treats_stale_running_job_as_queued(self) -> None:
        db = connect(self.config.sqlite_path)
        db.execute(
            """
            INSERT INTO calls (
                call_id, archive_path, source_filename, source_path, sha256, imported_at,
                current_state, review_state, low_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "call_stale",
                str(self.config.archive_root / "call_stale"),
                "stale.wav",
                "stale.wav",
                "sha_stale",
                "2026-01-01T00:00:00+00:00",
                "transcribing",
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
                "job_stale",
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
        db.commit()

        payload = _app_status_payload(self.config, db)
        self.assertEqual(payload["app_status"], "Paused")
        self.assertEqual(payload["counts"]["running"], 0)
        self.assertEqual(payload["counts"]["queued"], 1)
        self.assertIsNone(payload["current_task"])

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
        now = datetime.now(timezone.utc)
        started_at = (now - timedelta(seconds=120)).isoformat()
        available_at = (now - timedelta(seconds=130)).isoformat()
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
                available_at,
                started_at,
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

    def test_load_local_model_reuses_cached_instance(self) -> None:
        _MODEL_CACHE.clear()
        fake_model = object()
        fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=fake_model))

        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            first_model, first_cache_hit = _load_local_model("large-v3-turbo")
            second_model, second_cache_hit = _load_local_model("large-v3-turbo")

        self.assertIs(first_model, fake_model)
        self.assertIs(second_model, fake_model)
        self.assertFalse(first_cache_hit)
        self.assertTrue(second_cache_hit)
        fake_whisper.load_model.assert_called_once_with("large-v3-turbo")
        _MODEL_CACHE.clear()

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

    def test_quality_assessment_suspects_hebrew_transliteration_without_hebrew_script(self) -> None:
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
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 12})

        raw = RawTranscript(
            provider="cloud",
            model="whisper-1",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=11.0, text="Баруха шэм, йом тов", confidence=None, speaker=None)],
            text="Баруха шэм, йом тов",
        )

        assessment = assess_transcript_quality(raw, audio_path)
        self.assertTrue(assessment.suspect_language_confusion)
        self.assertIn("he", assessment.suggested_retry_languages)
        self.assertIn("suspected_hebrew_transliteration", assessment.flags)
        self.assertGreaterEqual(assessment.suspicious_token_count, 3)
        self.assertTrue(assessment.suspicious_token_examples)
        temp_dir.cleanup()

    def test_quality_assessment_detects_dense_cyrillic_hebrew_transliteration(self) -> None:
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
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 392.0})

        text = (
            "алё жора была такая сиха и потом у них бэдик бахаян "
            "масляни микроним хивра мид хаббер и еще тахан"
        )
        raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=120.0, text=text, confidence=None, speaker=None)],
            text=text,
        )

        assessment = assess_transcript_quality(raw, audio_path)
        self.assertTrue(assessment.suspect_language_confusion)
        self.assertIn("suspected_hebrew_transliteration", assessment.flags)
        self.assertIn("high_hebrew_transliteration_density", assessment.flags)
        self.assertIn("mixed_language_retry_recommended", assessment.flags)
        self.assertGreaterEqual(assessment.suspicious_token_count, 4)
        self.assertIn("сиха", assessment.suspicious_token_examples)
        temp_dir.cleanup()

    def test_quality_assessment_does_not_flag_normal_russian_words_as_hebrew_transliteration(self) -> None:
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
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 47.0})

        text = (
            "Алло. Жорик, ищи хорошо, потому что я сейчас прошелся по мейлам, "
            "я нашел файл, который лежал в бутуре Технун Мульбецоа. "
            "Это, оказывается, файл Лероновский."
        )
        raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=47.0, text=text, confidence=None, speaker=None)],
            text=text,
        )

        assessment = assess_transcript_quality(raw, audio_path)
        self.assertNotIn("high_hebrew_transliteration_density", assessment.flags)
        self.assertLessEqual(assessment.suspicious_token_count, 2)
        self.assertNotIn("лежал", assessment.suspicious_token_examples)
        self.assertNotIn("лероновский", assessment.suspicious_token_examples)
        temp_dir.cleanup()

    def test_quality_assessment_recommends_hebrew_retry_for_readable_russian_with_hebrew_terms(self) -> None:
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
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 47.0})

        segments = [
            RawSegment(start_sec=0.0, end_sec=12.0, text="Алло. Жорик, ищи хорошо, потому что я сейчас прошелся по мейлам.", confidence=None, speaker=None),
            RawSegment(start_sec=12.0, end_sec=24.0, text="Я нашел файл, который лежал в бутуре Технун Мульбецоа.", confidence=None, speaker=None),
            RawSegment(start_sec=24.0, end_sec=47.0, text="И там это не такцив Мульбецоа, капсулы 250, это больше такцивы 500.", confidence=None, speaker=None),
        ]
        text = " ".join(segment.text for segment in segments)
        raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=segments,
            text=text,
        )

        assessment = assess_transcript_quality(raw, audio_path)
        self.assertIn("segment_level_hebrew_recovery_recommended", assessment.flags)
        self.assertIn("he", assessment.suggested_retry_languages)
        self.assertGreaterEqual(assessment.suspicious_segment_count, 1)
        temp_dir.cleanup()

    def test_quality_assessment_penalizes_mixed_script_gibberish_in_forced_hebrew_candidate(self) -> None:
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
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 47.0})

        text = "י noted בחטר póm א outset הורטדност With Kickstarter זה נ lekker ב kunst"
        raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="he",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=47.0, text=text, confidence=None, speaker=None)],
            text=text,
        )

        assessment = assess_transcript_quality(raw, audio_path, expected_language="he")
        self.assertIn("mixed_script_gibberish", assessment.flags)
        self.assertIn("hebrew_candidate_corrupted", assessment.flags)
        self.assertLess(assessment.score, 0.5)
        temp_dir.cleanup()

    def test_transcribe_retries_with_forced_hebrew_and_records_selection(self) -> None:
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
transcription:
  provider_default: "local"
  provider_fallback: "cloud"
  cloud_enabled: false
  quality_retry_enabled: true
  candidate_merge_enabled: true
  retry_languages_on_suspicion: ["he"]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 12})

        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=11.0, text="Баруха шэм, йом тов", confidence=None, speaker=None)],
            text="Баруха шэм, йом тов",
        )
        forced_hebrew = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="he",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=11.0, text="ברוך השם, יום טוב", confidence=None, speaker=None)],
            text="ברוך השם, יום טוב",
        )
        decisions = [
            SegmentDetectionDecision(
                segment_index=0,
                label="hebrew_transliteration",
                confidence=0.95,
                reason="Likely Hebrew greeting rendered in Cyrillic.",
                suspicious=True,
                retry_language="he",
            )
        ]
        word_span_decisions = [
            WordSpanDecision(
                segment_index=0,
                span_text="Баруха шэм, йом тов",
                start_token_index=0,
                end_token_index=3,
                label="hebrew_transliteration",
                confidence=0.95,
                suspicious=True,
                retry_language="he",
                reason="Likely Hebrew greeting rendered in Cyrillic.",
            )
        ]
        retry_results = [
            {
                "segment_index": 0,
                "rank": 1,
                "score": 1.25,
                "span_text": "Баруха шэм, йом тов",
                "start_token_index": 0,
                "end_token_index": 3,
                "start_sec": 0.0,
                "end_sec": 11.0,
                "baseline_segment_text": "Баруха шэм, йом тов",
                "context_before": "",
                "context_after": "",
                "llm_label": "hebrew_transliteration",
                "llm_reason": "Likely Hebrew greeting rendered in Cyrillic.",
                "llm_confidence": 0.95,
                "retry_text": "ברוך השם, יום טוב",
                "retry_language": "he",
                "replacement_applied": True,
                "replacement_reason": "hebrew_script_recovered",
            }
        ]
        word_span_candidates = [
            types.SimpleNamespace(
                segment_index=0,
                segment_start_sec=0.0,
                segment_end_sec=11.0,
                segment_text="Баруха шэм, йом тов",
                span_text="Баруха шэм, йом тов",
                start_token_index=0,
                end_token_index=3,
                start_char_offset=0,
                end_char_offset=len("Баруха шэм, йом тов"),
                context_before="",
                context_after="",
                source_segment_label="hebrew_transliteration",
                source_segment_reason="Likely Hebrew greeting rendered in Cyrillic.",
                source_segment_confidence=0.95,
            )
        ]
        with (
            patch("call_assistant.transcription.service._transcribe_local", return_value=baseline),
            patch("call_assistant.transcription.service._segment_detection_decisions_with_raw", return_value=(decisions, {"segments": []})),
            patch("call_assistant.transcription.service._build_word_span_candidates", return_value=word_span_candidates),
            patch("call_assistant.transcription.service._word_span_detection_decisions_with_raw", return_value=(word_span_decisions, {"spans": []})),
            patch("call_assistant.transcription.service._build_clause_retry_candidates", return_value=[]),
            patch("call_assistant.transcription.service._apply_combined_hebrew_retries", return_value=(forced_hebrew, retry_results, [], [])),
        ):
            selected = transcribe(audio_path, config)

        self.assertIn(selected.language, {"he", "mixed"})
        artifact = load_transcription_candidate_selection(audio_path.parent)
        self.assertEqual(artifact["selection"]["strategy"], "llm_word_span_hebrew_recovery")
        self.assertEqual(artifact["selection"]["retried_languages"], ["he"])
        self.assertEqual(artifact["word_span_detection"]["selected_for_retry"], 1)
        self.assertEqual(len(artifact["word_span_retries"]), 1)
        self.assertIn("suspected_hebrew_transliteration", artifact["baseline"]["quality_flags"])
        self.assertGreaterEqual(artifact["baseline"]["suspicious_token_count"], 3)
        self.assertTrue(artifact["baseline"]["suspicious_token_examples"])
        temp_dir.cleanup()

    def test_transcribe_records_segment_detection_and_merged_artifact(self) -> None:
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
transcription:
  provider_default: "local"
  provider_fallback: "cloud"
  cloud_enabled: false
  quality_retry_enabled: true
  candidate_merge_enabled: true
segment_detection:
  enabled: true
  run_on_every_call: true
  max_retry_segments_per_call: 10
  min_confidence_to_retry: 0.70
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 47})

        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="Алло.", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="И там это не такцив Мульбецоа.", confidence=None, speaker=None),
            ],
            text="Алло. И там это не такцив Мульбецоа.",
        )
        merged = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="mixed",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="Алло.", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="ושם זה לא תקציב מולבצעה.", confidence=None, speaker=None),
            ],
            text="Алло. ושם זה לא תקציב מולבצעה.",
        )
        decisions = [
            SegmentDetectionDecision(
                segment_index=0,
                label="russian_normal",
                confidence=0.1,
                reason="Normal Russian greeting.",
                suspicious=False,
                retry_language=None,
            ),
            SegmentDetectionDecision(
                segment_index=1,
                label="hebrew_transliteration",
                confidence=0.95,
                reason="Hebrew budgeting term rendered in Russian phonetics.",
                suspicious=True,
                retry_language="he",
            ),
        ]
        word_span_decisions = [
            WordSpanDecision(
                segment_index=1,
                span_text="такцив Мульбецоа",
                start_token_index=4,
                end_token_index=5,
                label="hebrew_transliteration",
                confidence=0.95,
                suspicious=True,
                retry_language="he",
                reason="Hebrew budgeting term rendered in Russian phonetics.",
            )
        ]
        retry_results = [
            {
                "segment_index": 1,
                "rank": 1,
                "score": 1.2,
                "span_text": "такцив Мульбецоа",
                "start_token_index": 4,
                "end_token_index": 5,
                "start_sec": 5.0,
                "end_sec": 10.0,
                "baseline_segment_text": "И там это не такцив Мульбецоа.",
                "context_before": "Алло.",
                "context_after": "",
                "llm_label": "hebrew_transliteration",
                "llm_reason": "Hebrew budgeting term rendered in Russian phonetics.",
                "llm_confidence": 0.95,
                "retry_text": "ושם זה לא תקציב מולבצעה.",
                "retry_language": "he",
                "replacement_applied": True,
                "replacement_reason": "hebrew_script_recovered",
            }
        ]

        with (
            patch("call_assistant.transcription.service._transcribe_local", return_value=baseline),
            patch("call_assistant.transcription.service._segment_detection_decisions_with_raw", return_value=(decisions, {"segments": []})),
            patch("call_assistant.transcription.service._word_span_detection_decisions_with_raw", return_value=(word_span_decisions, {"spans": []})),
            patch("call_assistant.transcription.service._build_clause_retry_candidates", return_value=[]),
            patch("call_assistant.transcription.service._apply_combined_hebrew_retries", return_value=(merged, retry_results, [], [])),
        ):
            selected = transcribe(audio_path, config)

        self.assertEqual(selected.language, "mixed")
        artifact = load_transcription_candidate_selection(audio_path.parent)
        self.assertEqual(artifact["selection"]["strategy"], "llm_word_span_hebrew_recovery")
        self.assertEqual(artifact["selection"]["winner"], "merged_segments")
        self.assertEqual(artifact["segment_detection"]["selected_for_retry"], 1)
        self.assertEqual(artifact["word_span_detection"]["selected_for_retry"], 1)
        self.assertEqual(len(artifact["word_span_retries"]), 1)
        self.assertEqual(artifact["merged"]["text"], merged.text)
        temp_dir.cleanup()

    def test_transcribe_records_segment_detection_on_baseline_only(self) -> None:
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
transcription:
  provider_default: "local"
  provider_fallback: "cloud"
  cloud_enabled: false
  quality_retry_enabled: true
segment_detection:
  enabled: true
  run_on_every_call: true
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 10})
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=2.0, text="Привет.", confidence=None, speaker=None)],
            text="Привет.",
        )
        decisions = [
            SegmentDetectionDecision(
                segment_index=0,
                label="russian_normal",
                confidence=0.05,
                reason="Normal Russian segment.",
                suspicious=False,
                retry_language=None,
            )
        ]
        with (
            patch("call_assistant.transcription.service._transcribe_local", return_value=baseline),
            patch("call_assistant.transcription.service._segment_detection_decisions_with_raw", return_value=(decisions, {"segments": []})),
        ):
            selected = transcribe(audio_path, config)

        self.assertEqual(selected.text, "Привет.")
        artifact = load_transcription_candidate_selection(audio_path.parent)
        self.assertEqual(artifact["selection"]["strategy"], "baseline")
        self.assertEqual(artifact["segment_detection"]["segments_evaluated"], 1)
        self.assertEqual(artifact["segment_detection"]["selected_for_retry"], 0)
        self.assertEqual(artifact["word_span_detection"]["selected_for_retry"], 0)
        temp_dir.cleanup()

    def test_transcribe_preserves_baseline_when_segments_are_suspicious_but_below_threshold(self) -> None:
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
transcription:
  provider_default: "local"
  provider_fallback: "cloud"
  cloud_enabled: false
  quality_retry_enabled: true
segment_detection:
  enabled: true
  run_on_every_call: true
  min_confidence_to_retry: 0.70
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 90})

        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="И там это не такцив Мульбецоа.", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="Потом его атомали.", confidence=None, speaker=None),
            ],
            text="И там это не такцив Мульбецоа. Потом его атомали.",
        )
        decisions = [
            SegmentDetectionDecision(
                segment_index=0,
                label="hebrew_transliteration",
                confidence=0.55,
                reason="Suspicious Hebrew budgeting term rendered in Russian phonetics.",
                suspicious=True,
                retry_language="he",
            ),
            SegmentDetectionDecision(
                segment_index=1,
                label="mixed_uncertain",
                confidence=0.61,
                reason="Context suggests corrupted Hebrew-adjacent wording.",
                suspicious=True,
                retry_language="he",
            ),
        ]

        with (
            patch("call_assistant.transcription.service._transcribe_local", return_value=baseline) as mock_local,
            patch("call_assistant.transcription.service._segment_detection_decisions_with_raw", return_value=(decisions, {"segments": []})),
            patch(
                "call_assistant.transcription.service._word_span_detection_decisions_with_raw",
                return_value=(
                    [
                        WordSpanDecision(
                            segment_index=0,
                            span_text="такцив Мульбецоа",
                            start_token_index=4,
                            end_token_index=5,
                            label="hebrew_transliteration",
                            confidence=0.55,
                            suspicious=True,
                            retry_language="he",
                            reason="Below threshold retry candidate.",
                        ),
                        WordSpanDecision(
                            segment_index=1,
                            span_text="атомали",
                            start_token_index=1,
                            end_token_index=1,
                            label="mixed_uncertain",
                            confidence=0.61,
                            suspicious=True,
                            retry_language="he",
                            reason="Below threshold retry candidate.",
                        ),
                    ],
                    {"spans": []},
                ),
            ),
            patch("call_assistant.transcription.service._build_clause_retry_candidates", return_value=[]),
            patch("call_assistant.transcription.service._transcribe_cloud") as mock_cloud,
        ):
            selected = transcribe(audio_path, config)

        self.assertEqual(selected.text, baseline.text)
        mock_local.assert_called_once()
        mock_cloud.assert_not_called()
        artifact = load_transcription_candidate_selection(audio_path.parent)
        self.assertEqual(artifact["selection"]["strategy"], "baseline")
        self.assertEqual(artifact["selection"]["winner"], "baseline")
        self.assertEqual(artifact["segment_detection"]["suspicious_count"], 2)
        self.assertEqual(artifact["segment_detection"]["selected_for_retry"], 0)
        self.assertEqual(artifact["segment_detection"]["below_threshold_count"], 2)
        self.assertEqual(artifact["segment_detection"]["threshold_used"], 0.7)
        self.assertEqual(artifact["word_span_detection"]["selected_for_retry"], 0)
        self.assertEqual(artifact["selection"]["retried_languages"], [])
        self.assertIn("baseline preserved", " ".join(artifact["selection"]["quality_notes"]).lower())
        temp_dir.cleanup()

    def test_transcribe_escalates_to_full_call_hebrew_for_hebrew_dominant_low_quality_baseline(self) -> None:
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
transcription:
  provider_default: "local"
  provider_fallback: "cloud"
  cloud_enabled: false
  quality_retry_enabled: true
  candidate_merge_enabled: true
  retry_languages_on_suspicion: ["he"]
segment_detection:
  enabled: true
  run_on_every_call: true
  min_confidence_to_retry: 0.0
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 20})

        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=2.0, text="Хай, салам.", confidence=None, speaker=None),
                RawSegment(start_sec=2.0, end_sec=4.0, text="Салам, бессадар.", confidence=None, speaker=None),
                RawSegment(start_sec=4.0, end_sec=6.0, text="Ей, баруха шэм, юм то.", confidence=None, speaker=None),
            ],
            text="Хай, салам. Салам, бессадар. Ей, баруха шэм, юм то.",
        )
        forced_hebrew = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="he",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=2.0, text="היי, שלום.", confidence=None, speaker=None),
                RawSegment(start_sec=2.0, end_sec=4.0, text="שלום, בסדר.", confidence=None, speaker=None),
                RawSegment(start_sec=4.0, end_sec=6.0, text="אוקיי ברוך השם יום טוב.", confidence=None, speaker=None),
            ],
            text="היי, שלום. שלום, בסדר. אוקיי ברוך השם יום טוב.",
        )
        decisions = [
            SegmentDetectionDecision(0, "hebrew_transliteration", 0.9, "Hebrew greeting transliteration.", True, "he"),
            SegmentDetectionDecision(1, "hebrew_transliteration", 0.9, "Hebrew greeting transliteration.", True, "he"),
            SegmentDetectionDecision(2, "hebrew_transliteration", 0.9, "Hebrew blessing transliteration.", True, "he"),
        ]
        baseline_assessment = TranscriptQualityAssessment(
            score=0.30,
            flags=["suspected_hebrew_transliteration", "baseline_script_mismatch"],
            suspect_language_confusion=True,
            suspicious_token_count=6,
            suspicious_token_examples=["хай", "салам", "бессадар", "баруха", "шэм", "юм"],
            suspicious_segment_count=3,
            suspicious_segment_examples=[segment.text for segment in baseline.segments],
            suggested_retry_languages=["he"],
            low_confidence_reason="suspected_hebrew_transliteration",
        )
        forced_assessment = TranscriptQualityAssessment(
            score=0.95,
            flags=[],
            suspect_language_confusion=False,
            suspicious_token_count=0,
            suspicious_token_examples=[],
            suspicious_segment_count=0,
            suspicious_segment_examples=[],
            suggested_retry_languages=[],
            low_confidence_reason=None,
        )

        with (
            patch("call_assistant.transcription.service._transcribe_local", side_effect=[baseline, forced_hebrew]) as mock_local,
            patch(
                "call_assistant.transcription.service.assess_transcript_quality",
                side_effect=[baseline_assessment, forced_assessment, forced_assessment, forced_assessment],
            ),
            patch("call_assistant.transcription.service._segment_detection_decisions_with_raw", return_value=(decisions, {"segments": []})),
            patch("call_assistant.transcription.service._word_span_detection_decisions_with_raw", return_value=([], {"spans": []})),
            patch("call_assistant.transcription.service._build_clause_retry_candidates", return_value=[]),
            patch("call_assistant.transcription.service._build_word_span_candidates", return_value=[]),
        ):
            selected = transcribe(audio_path, config)

        self.assertIn(selected.language, {"he", "mixed"})
        self.assertEqual(mock_local.call_count, 2)
        self.assertEqual(mock_local.call_args_list[1].args[3], "he")
        artifact = load_transcription_candidate_selection(audio_path.parent)
        self.assertIn(artifact["selection"]["strategy"], {"forced_hebrew", "merged_segments"})
        self.assertEqual(artifact["selection"]["retried_languages"], ["he"])
        temp_dir.cleanup()

    def test_build_word_span_candidates_includes_split_phrase(self) -> None:
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
segment_detection:
  enabled: true
  run_on_every_call: true
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(
                    start_sec=0.0,
                    end_sec=6.0,
                    text="И там это не такси в мульбицо капсулы 250",
                    confidence=None,
                    speaker=None,
                )
            ],
            text="И там это не такси в мульбицо капсулы 250",
        )
        decisions = [
            SegmentDetectionDecision(
                segment_index=0,
                label="hebrew_transliteration",
                confidence=0.9,
                reason="Suspicious Hebrew budgeting phrase.",
                suspicious=True,
                retry_language="he",
            )
        ]
        candidates = _build_word_span_candidates(raw, decisions, config)
        spans = {candidate.span_text.lower() for candidate in candidates}
        self.assertIn("такси в", spans)
        self.assertTrue(any("мульбицо" in span for span in spans))
        temp_dir.cleanup()

    def test_build_clause_retry_candidates_includes_budget_clause(self) -> None:
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
segment_detection:
  enabled: true
  run_on_every_call: true
  clause_retry_enabled: true
  clause_retry_min_span_words: 4
  clause_retry_max_span_words: 8
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(
                    start_sec=0.0,
                    end_sec=6.0,
                    text="и потом его атомали, такцив капсулы 250",
                    confidence=None,
                    speaker=None,
                )
            ],
            text="и потом его атомали, такцив капсулы 250",
        )
        decisions = [
            SegmentDetectionDecision(
                segment_index=0,
                label="hebrew_transliteration",
                confidence=0.9,
                reason="Budgeting clause.",
                suspicious=True,
                retry_language="he",
            )
        ]

        candidates = _build_clause_retry_candidates(raw, decisions, config)

        self.assertEqual(len(candidates), 1)
        self.assertIn("атомали", candidates[0].clause_text.lower())
        self.assertIn("такцив", candidates[0].clause_text.lower())
        temp_dir.cleanup()

    def test_replace_span_in_segment_text_preserves_surrounding_russian(self) -> None:
        updated = _replace_span_in_segment_text(
            "И там это не такси в мульбицо капсулы 250.",
            4,
            5,
            "תקציב",
        )
        self.assertEqual(updated, "И там это не תקציב мульбицо капсулы 250.")

    def test_heuristic_word_span_semantic_validation_normalizes_budget_phrase(self) -> None:
        validation = _heuristic_word_span_semantic_validation(
            "такцив",
            "И там это не такцив Мульбецоа.",
            "תקציב",
        )

        self.assertEqual(validation.decision, "accept")
        self.assertEqual(validation.normalized_hebrew, "תקציב")

    def test_rescue_semantic_validation_accepts_noisy_hebrew_business_phrase(self) -> None:
        candidate = WordRetryCandidate(
            segment_index=0,
            rank=1,
            score=0.4,
            start_sec=1.0,
            end_sec=2.0,
            baseline_segment_text="который лежал в бутуре Технун Мульбецоа",
            span_text="в бутуре Технун",
            start_token_index=2,
            end_token_index=4,
            context_before="",
            context_after="",
            llm_label="mixed_uncertain",
            llm_reason="Contains Hebrew transliterations.",
            llm_confidence=0.0,
        )
        rescued = _rescue_semantic_validation_with_normalization(
            candidate,
            "בתור תחנון מולד",
            WordSpanSemanticValidation("reject", 0.7, "Too noisy.", None),
        )

        self.assertEqual(rescued.decision, "accept_with_normalization")
        self.assertEqual(rescued.normalized_hebrew, "בתור תכנון מול")

    def test_apply_word_span_retries_rejects_semantically_bad_hebrew_output(self) -> None:
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
segment_detection:
  semantic_validation_enabled: true
  semantic_min_confidence_to_accept: 0.75
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"call_id": "call_x", "duration_seconds": 30})
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=5.0, text="И там это не такцив 500.", confidence=None, speaker=None)],
            text="И там это не такцив 500.",
        )
        candidate = WordRetryCandidate(
            segment_index=0,
            rank=1,
            score=0.9,
            start_sec=1.0,
            end_sec=2.0,
            baseline_segment_text="И там это не такцив 500.",
            span_text="такцив",
            start_token_index=4,
            end_token_index=4,
            context_before="",
            context_after="",
            llm_label="hebrew_transliteration",
            llm_reason="Budget term",
            llm_confidence=0.0,
        )
        retry_raw = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="he",
            confidence=None,
            segments=[],
            text="לפת",
        )

        with (
            patch("call_assistant.transcription.service._extract_word_span_audio", return_value=audio_path),
            patch("call_assistant.transcription.service._transcribe_word_span_hebrew_retry", return_value=retry_raw),
            patch(
                "call_assistant.transcription.service._validate_word_span_retry_with_llm",
                return_value=(WordSpanSemanticValidation("reject", 0.95, "Bad budgeting phrase.", None), {"decision": "reject"}),
            ),
        ):
            merged, results, validation_payloads = _apply_word_span_retries(baseline, audio_path, config, [candidate])

        self.assertEqual(merged.text, baseline.text)
        self.assertFalse(results[0]["replacement_applied"])
        self.assertEqual(results[0]["replacement_reason"], "semantic_rejected")
        self.assertEqual(results[0]["semantic_decision"], "reject")
        self.assertEqual(validation_payloads[0]["type"], "semantic_validation")
        temp_dir.cleanup()

    def test_apply_word_span_retries_reconciles_multiple_replacements_same_segment(self) -> None:
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
segment_detection:
  semantic_validation_enabled: true
  segment_reconciliation_enabled: true
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"call_id": "call_y", "duration_seconds": 30})
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=6.0, text="который лежал в бутуре Технун Мульбецоа", confidence=None, speaker=None)],
            text="который лежал в бутуре Технун Мульбецоа",
        )
        candidates = [
            WordRetryCandidate(
                segment_index=0,
                rank=1,
                score=0.9,
                start_sec=1.0,
                end_sec=2.0,
                baseline_segment_text=baseline.text,
                span_text="в бутуре Технун",
                start_token_index=2,
                end_token_index=4,
                context_before="",
                context_after="",
                llm_label="hebrew_transliteration",
                llm_reason="Planning phrase",
                llm_confidence=0.0,
            ),
            WordRetryCandidate(
                segment_index=0,
                rank=2,
                score=0.8,
                start_sec=2.1,
                end_sec=2.8,
                baseline_segment_text=baseline.text,
                span_text="Мульбецоа",
                start_token_index=5,
                end_token_index=5,
                context_before="",
                context_after="",
                llm_label="hebrew_transliteration",
                llm_reason="Execution phrase",
                llm_confidence=0.0,
            ),
        ]
        retry_values = [
            RawTranscript(provider="local", model="large-v3-turbo", language="he", confidence=None, segments=[], text="בתור תחנון מולד"),
            RawTranscript(provider="local", model="large-v3-turbo", language="he", confidence=None, segments=[], text="תחנון מולביצוע"),
        ]
        validations = [
            (WordSpanSemanticValidation("accept_with_normalization", 0.9, "Accept", "בתור תכנון"), {"decision": "accept_with_normalization"}),
            (WordSpanSemanticValidation("accept_with_normalization", 0.9, "Accept", "מול ביצוע"), {"decision": "accept_with_normalization"}),
        ]
        reconciled = [
            {
                "segment_index": 0,
                "rank": 1,
                "start_token_index": 2,
                "end_token_index": 5,
                "final_replacement_text": "בתור תכנון מול ביצוע",
                "replacement_reason": "semantic_reconciled",
            }
        ]

        with (
            patch("call_assistant.transcription.service._extract_word_span_audio", return_value=audio_path),
            patch("call_assistant.transcription.service._transcribe_word_span_hebrew_retry", side_effect=retry_values),
            patch("call_assistant.transcription.service._validate_word_span_retry_with_llm", side_effect=validations),
            patch("call_assistant.transcription.service._reconcile_segment_replacements_with_llm", return_value=(reconciled, {"decision": "use_reconciled"})),
        ):
            merged, results, validation_payloads = _apply_word_span_retries(baseline, audio_path, config, candidates)

        self.assertIn("בתור תכנון מול ביצוע", merged.text)
        self.assertTrue(results[0]["replacement_applied"])
        self.assertEqual(results[0]["replacement_reason"], "semantic_reconciled")
        self.assertFalse(results[1]["replacement_applied"])
        self.assertEqual(results[1]["replacement_reason"], "reconciliation_rejected")
        self.assertTrue(any(item["type"] == "segment_reconciliation" for item in validation_payloads))
        temp_dir.cleanup()

    def test_apply_combined_hebrew_retries_prefers_clause_in_reconciliation(self) -> None:
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
segment_detection:
  semantic_validation_enabled: true
  segment_reconciliation_enabled: true
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = AppConfig.load(config_path)
        config.ensure_directories()
        audio_path = config.temp_dir / "sample.wav"
        audio_path.write_bytes(b"abc")
        write_json(audio_path.parent / "metadata.json", {"call_id": "call_clause", "duration_seconds": 30})
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[RawSegment(start_sec=0.0, end_sec=6.0, text="и потом его атомали, такцив капсулы 250", confidence=None, speaker=None)],
            text="и потом его атомали, такцив капсулы 250",
        )
        word_candidate = WordRetryCandidate(
            segment_index=0,
            rank=1,
            score=0.9,
            start_sec=1.0,
            end_sec=2.0,
            baseline_segment_text=baseline.text,
            span_text="атомали, такцив",
            start_token_index=3,
            end_token_index=4,
            context_before="",
            context_after="",
            llm_label="hebrew_transliteration",
            llm_reason="Budget clause",
            llm_confidence=0.9,
        )
        clause_candidate = ClauseRetryCandidate(
            segment_index=0,
            rank=1,
            score=1.2,
            start_sec=0.0,
            end_sec=6.0,
            baseline_segment_text=baseline.text,
            clause_text="его атомали, такцив капсулы",
            start_token_index=2,
            end_token_index=5,
            context_before="",
            context_after="",
            source_reason="Budget clause",
        )
        word_retry = RawTranscript(provider="local", model="large-v3-turbo", language="he", confidence=None, segments=[], text="התאמה ל")
        clause_retry = RawTranscript(provider="local", model="large-v3-turbo", language="he", confidence=None, segments=[], text="התאמה לתקציב")
        validations = [
            (WordSpanSemanticValidation("accept_with_normalization", 0.9, "Accept", "התאמה ל-"), {"decision": "accept_with_normalization"}),
            (WordSpanSemanticValidation("accept", 0.95, "Accept", "התאמה לתקציב"), {"decision": "accept"}),
        ]
        reconciled = [
            {
                "segment_index": 0,
                "rank": 1001,
                "start_token_index": 2,
                "end_token_index": 5,
                "final_replacement_text": "התאמה לתקציב",
                "replacement_reason": "semantic_reconciled",
                "source_type": "clause",
            }
        ]

        with (
            patch("call_assistant.transcription.service._extract_word_span_audio", return_value=audio_path),
            patch("call_assistant.transcription.service._extract_clause_retry_audio", return_value=audio_path),
            patch("call_assistant.transcription.service._transcribe_word_span_hebrew_retry", return_value=word_retry),
            patch("call_assistant.transcription.service._transcribe_clause_hebrew_retry", return_value=clause_retry),
            patch("call_assistant.transcription.service._validate_word_span_retry_with_llm", side_effect=[validations[0]]),
            patch("call_assistant.transcription.service._validate_clause_retry_with_llm", side_effect=[validations[1]]),
            patch("call_assistant.transcription.service._reconcile_segment_replacements_with_llm", return_value=(reconciled, {"decision": "use_reconciled"})),
        ):
            merged, word_results, clause_results, _ = _apply_combined_hebrew_retries(
                baseline,
                audio_path,
                config,
                [word_candidate],
                [clause_candidate],
            )

        self.assertIn("התאמה לתקציב", merged.text)
        self.assertFalse(word_results[0]["replacement_applied"])
        self.assertTrue(clause_results[0]["replacement_applied"])
        self.assertEqual(clause_results[0]["replacement_reason"], "semantic_reconciled")
        temp_dir.cleanup()

    def test_merge_transcript_candidates_prefers_forced_hebrew_only_for_suspicious_segments(self) -> None:
        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="Алё, жора", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="У меня была такая сиха", confidence=None, speaker=None),
            ],
            text="Алё, жора У меня была такая сиха",
        )
        forced_hebrew = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="he",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="טקסט משובש לגמרי", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="הייתה לי שיחה כזאת", confidence=None, speaker=None),
            ],
            text="טקסט משובש לגמרי הייתה לי שיחה כזאת",
        )

        merged = merge_transcript_candidates(baseline, forced_hebrew)

        self.assertEqual(merged.segments[0].text, "Алё, жора")
        self.assertEqual(merged.segments[1].text, "הייתה לי שיחה כזאת")

    def test_compare_transcript_candidates_prefers_merged_for_mixed_language_recovery(self) -> None:
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
        write_json(audio_path.parent / "metadata.json", {"duration_seconds": 120})

        baseline = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="ru",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="Алё, жора", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="У меня была такая сиха", confidence=None, speaker=None),
            ],
            text="Алё, жора У меня была такая сиха",
        )
        forced_hebrew = RawTranscript(
            provider="local",
            model="large-v3-turbo",
            language="he",
            confidence=None,
            segments=[
                RawSegment(start_sec=0.0, end_sec=5.0, text="עיוות מוחלט של הרוסית", confidence=None, speaker=None),
                RawSegment(start_sec=5.0, end_sec=10.0, text="הייתה לי שיחה כזאת", confidence=None, speaker=None),
            ],
            text="עיוות מוחלט של הרוסית הייתה לי שיחה כזאת",
        )

        comparison = compare_transcript_candidates(baseline, forced_hebrew, audio_path, merge_enabled=True)

        self.assertEqual(comparison.strategy, "merged_segments")
        self.assertEqual(comparison.selected.segments[0].text, "Алё, жора")
        self.assertEqual(comparison.selected.segments[1].text, "הייתה לי שיחה כזאת")
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
