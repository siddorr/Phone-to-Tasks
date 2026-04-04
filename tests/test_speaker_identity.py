from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import read_json, write_json
from call_assistant.common.models import TranscriptSegment
from call_assistant.speaker_identity.service import (
    _best_profile_match,
    _cosine_similarity,
    apply_identity_assignments,
    archive_speaker_profile,
    assign_speaker_identity,
    call_speaker_assignments,
    accept_suggested_speaker_identity,
    create_speaker_profile,
    list_assignable_speaker_profiles,
    reject_suggested_speaker_identity,
    rename_speaker_profile,
    run_speaker_identity_stage,
)


class SpeakerIdentityTests(unittest.TestCase):
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
speaker_identity:
  enabled: true
  provider: "pyannote"
  auto_assign_threshold: 0.75
  suggest_threshold: 0.60
  min_cluster_duration_seconds: 6
  max_segments_per_cluster: 20
  continue_on_error: true
""".strip()
            + "\n",
            encoding="utf-8",
        )
        self.config = AppConfig.load(self.config_path)
        self.config.ensure_directories()
        self.db = connect(self.config.sqlite_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _call_dir(self, call_id: str = "call_1") -> Path:
        call_dir = self.config.archive_root / "2026" / "04" / "04" / f"call_{call_id}"
        call_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            call_dir / "metadata.json",
            {
                "call_id": call_id,
                "source_filename": "sample.wav",
                "source_path": "sample.wav",
                "sha256": f"sha-{call_id}",
                "imported_at": "2026-04-04T00:00:00+00:00",
                "review_state": "pending",
                "current_state": "diarized",
                "errors": [],
            },
        )
        (call_dir / "audio_normalized.wav").write_bytes(b"fake")
        return call_dir

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_best_profile_match_picks_highest_similarity(self) -> None:
        speaker_id = create_speaker_profile(self.db, "Natasha")
        self.db.execute(
            """
            INSERT INTO speaker_embeddings (
                embedding_id, speaker_identity_id, call_id, speaker_cluster_id, segment_count,
                duration_seconds, embedding_vector_json, model_name, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("emb-1", speaker_id, "call_a", "speaker_1", 3, 8.0, "[1.0, 0.0]", "test", 1.0, "now"),
        )
        self.db.commit()

        match = _best_profile_match(self.db, [0.9, 0.1])

        self.assertEqual(match.speaker_identity_id, speaker_id)
        self.assertGreater(match.match_score, 0.8)

    def test_apply_identity_assignments_prefers_profile_name(self) -> None:
        speaker_id = create_speaker_profile(self.db, "Natasha")
        self.db.execute(
            """
            INSERT INTO speaker_assignments (
                assignment_id, call_id, speaker_cluster_id, speaker_identity_id,
                assignment_source, match_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("asg-1", "call_1", "speaker_1", speaker_id, "user", 1.0, "now", "now"),
        )
        self.db.commit()

        segments = [
            TranscriptSegment(
                segment_id="seg_1",
                start_sec=0.0,
                end_sec=8.0,
                speaker_label="speaker_1",
                speaker_channel_label=None,
                text="Hello",
                confidence=None,
                speaker_cluster_id="speaker_1",
            )
        ]
        updated = apply_identity_assignments(segments, self.db, "call_1")

        self.assertEqual(updated[0].speaker_display_name, "Natasha")
        self.assertEqual(updated[0].speaker_identity_id, speaker_id)

    def test_run_speaker_identity_stage_skips_short_clusters(self) -> None:
        call_dir = self._call_dir()
        write_json(
            call_dir / "transcript_segments.json",
            [
                {
                    "segment_id": "seg_1",
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                    "speaker_label": "speaker_1",
                    "speaker_channel_label": None,
                    "text": "Hello",
                    "confidence": None,
                    "speaker_cluster_id": "speaker_1",
                    "diarization_confidence": "medium",
                }
            ],
        )

        result = run_speaker_identity_stage(self.config, call_dir, self.db)

        self.assertEqual(result.segments[0]["speaker_identity_id"], None)
        self.assertEqual(result.outcome_status, "skipped")
        count = self.db.execute("SELECT COUNT(*) FROM speaker_embeddings").fetchone()[0]
        self.assertEqual(count, 0)

    def test_user_assignment_overrides_auto_assignment(self) -> None:
        call_dir = self._call_dir()
        speaker_id = create_speaker_profile(self.db, "Natasha")
        assign_speaker_identity(self.config, call_dir, "speaker_1", speaker_id, assignment_source="user")
        write_json(
            call_dir / "transcript_segments.json",
            [
                {
                    "segment_id": "seg_1",
                    "start_sec": 0.0,
                    "end_sec": 8.0,
                    "speaker_label": "speaker_1",
                    "speaker_channel_label": None,
                    "text": "Hello",
                    "confidence": None,
                    "speaker_cluster_id": "speaker_1",
                    "diarization_confidence": "medium",
                }
            ],
        )

        with patch("call_assistant.speaker_identity.service._compute_embedding", return_value=[1.0, 0.0]):
            result = run_speaker_identity_stage(self.config, call_dir, self.db)

        self.assertEqual(result.segments[0]["speaker_identity_id"], speaker_id)
        self.assertEqual(result.outcome_status, "skipped")
        assignment = self.db.execute(
            "SELECT assignment_source FROM speaker_assignments WHERE call_id = ? AND speaker_cluster_id = ?",
            ("call_1", "speaker_1"),
        ).fetchone()
        self.assertEqual(assignment["assignment_source"], "user")

    def test_speaker_identity_stage_is_degraded_when_embedding_fails(self) -> None:
        call_dir = self._call_dir()
        write_json(
            call_dir / "transcript_segments.json",
            [
                {
                    "segment_id": "seg_1",
                    "start_sec": 0.0,
                    "end_sec": 8.0,
                    "speaker_label": "speaker_1",
                    "speaker_channel_label": None,
                    "text": "Hello",
                    "confidence": None,
                    "speaker_cluster_id": "speaker_1",
                    "diarization_confidence": "medium",
                }
            ],
        )

        with patch("call_assistant.speaker_identity.service._compute_embedding", side_effect=RuntimeError("boom")):
            result = run_speaker_identity_stage(self.config, call_dir, self.db)

        self.assertEqual(result.outcome_status, "degraded")
        self.assertIn("No embeddings produced", result.outcome_detail)

    def test_speaker_identity_stage_creates_suggested_assignment_with_profile(self) -> None:
        call_dir = self._call_dir()
        speaker_id = create_speaker_profile(self.db, "Natasha")
        self.db.execute(
            """
            INSERT INTO speaker_embeddings (
                embedding_id, speaker_identity_id, call_id, speaker_cluster_id, segment_count,
                duration_seconds, embedding_vector_json, model_name, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("emb-1", speaker_id, "call_a", "speaker_1", 3, 8.0, "[1.0, 0.0]", "test", 1.0, "now"),
        )
        self.db.commit()
        write_json(
            call_dir / "transcript_segments.json",
            [
                {
                    "segment_id": "seg_1",
                    "start_sec": 0.0,
                    "end_sec": 8.0,
                    "speaker_label": "speaker_1",
                    "speaker_channel_label": None,
                    "text": "Hello",
                    "confidence": None,
                    "speaker_cluster_id": "speaker_1",
                    "diarization_confidence": "medium",
                }
            ],
        )

        with patch("call_assistant.speaker_identity.service._compute_embedding", return_value=[0.65, 0.76]):
            result = run_speaker_identity_stage(self.config, call_dir, self.db)

        self.assertEqual(result.outcome_status, "success")
        self.assertEqual(result.suggested_cluster_count, 1)
        row = self.db.execute(
            """
            SELECT speaker_identity_id, assignment_source
            FROM speaker_assignments
            WHERE call_id = ? AND speaker_cluster_id = ?
            """,
            ("call_1", "speaker_1"),
        ).fetchone()
        self.assertEqual(row["speaker_identity_id"], speaker_id)
        self.assertEqual(row["assignment_source"], "suggested")

    def test_rename_speaker_profile_updates_display_name(self) -> None:
        speaker_id = create_speaker_profile(self.db, "Natasha")
        rename_speaker_profile(self.db, speaker_id, "Nata")
        row = self.db.execute(
            "SELECT display_name FROM speaker_profiles WHERE speaker_identity_id = ?",
            (speaker_id,),
        ).fetchone()
        self.assertEqual(row["display_name"], "Nata")

    def test_archive_profile_excludes_from_assignable_profiles(self) -> None:
        visible_id = create_speaker_profile(self.db, "Visible")
        hidden_id = create_speaker_profile(self.db, "Hidden")
        archive_speaker_profile(self.db, hidden_id)

        profiles = list_assignable_speaker_profiles(self.db)

        self.assertEqual([profile["speaker_identity_id"] for profile in profiles], [visible_id])

    def test_accept_suggested_speaker_identity_converts_to_user_assignment(self) -> None:
        call_dir = self._call_dir()
        speaker_id = create_speaker_profile(self.db, "Natasha")
        self.db.execute(
            """
            INSERT INTO speaker_assignments (
                assignment_id, call_id, speaker_cluster_id, speaker_identity_id,
                assignment_source, match_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("asg-suggested", "call_1", "speaker_1", speaker_id, "suggested", 0.67, "now", "now"),
        )
        self.db.commit()

        accepted_id = accept_suggested_speaker_identity(self.config, call_dir, "speaker_1")

        self.assertEqual(accepted_id, speaker_id)
        row = self.db.execute(
            "SELECT assignment_source, speaker_identity_id, match_score FROM speaker_assignments WHERE call_id = ? AND speaker_cluster_id = ?",
            ("call_1", "speaker_1"),
        ).fetchone()
        self.assertEqual(row["assignment_source"], "user")
        self.assertEqual(row["speaker_identity_id"], speaker_id)
        self.assertEqual(row["match_score"], 0.67)

    def test_reject_suggested_speaker_identity_removes_assignment(self) -> None:
        call_dir = self._call_dir()
        speaker_id = create_speaker_profile(self.db, "Natasha")
        self.db.execute(
            """
            INSERT INTO speaker_assignments (
                assignment_id, call_id, speaker_cluster_id, speaker_identity_id,
                assignment_source, match_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("asg-suggested", "call_1", "speaker_1", speaker_id, "suggested", 0.67, "now", "now"),
        )
        self.db.commit()

        reject_suggested_speaker_identity(self.config, call_dir, "speaker_1")

        row = self.db.execute(
            "SELECT COUNT(*) AS count FROM speaker_assignments WHERE call_id = ? AND speaker_cluster_id = ?",
            ("call_1", "speaker_1"),
        ).fetchone()
        self.assertEqual(row["count"], 0)

    def test_call_speaker_assignments_includes_display_name_and_source(self) -> None:
        speaker_id = create_speaker_profile(self.db, "Natasha")
        self.db.execute(
            """
            INSERT INTO speaker_assignments (
                assignment_id, call_id, speaker_cluster_id, speaker_identity_id,
                assignment_source, match_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("asg-1", "call_1", "speaker_1", speaker_id, "suggested", 0.62, "now", "now"),
        )
        self.db.commit()

        assignments = call_speaker_assignments(self.db, "call_1")

        self.assertEqual(assignments["speaker_1"]["display_name"], "Natasha")
        self.assertEqual(assignments["speaker_1"]["assignment_source"], "suggested")


if __name__ == "__main__":
    unittest.main()
