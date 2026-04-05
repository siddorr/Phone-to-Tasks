from __future__ import annotations

import logging
import re
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawTranscript, TranscriptSegment
from call_assistant.common.progress import clear_stage_progress, set_stage_progress

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    start_sec: float
    end_sec: float
    speaker_cluster_id: str
    confidence: str | None = None


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _split_text_chunks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", stripped)
    chunks = [part.strip() for part in parts if part.strip()]
    return chunks or [stripped]


def _segment_for_turn(
    segment_id: str,
    start_sec: float,
    end_sec: float,
    cluster_id: str,
    speaker_mapping: dict[str, str] | None,
    text: str,
    confidence: float | None,
    diarization_confidence: str | None,
    speaker_channel_label: str | None,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment_id,
        start_sec=start_sec,
        end_sec=end_sec,
        speaker_cluster_id=cluster_id,
        speaker_label=_resolve_label(cluster_id, speaker_mapping),
        speaker_channel_label=speaker_channel_label,
        text=text.strip(),
        confidence=confidence,
        diarization_confidence=diarization_confidence or "medium",
    )


def _resolve_label(cluster_id: str, speaker_mapping: dict[str, str] | None = None) -> str:
    if speaker_mapping:
        mapped = speaker_mapping.get(cluster_id)
        if mapped in {"me", "other", "unknown"}:
            return mapped
    return cluster_id


def _fallback_single_speaker(raw: RawTranscript, speaker_mapping: dict[str, str] | None = None) -> list[TranscriptSegment]:
    label = _resolve_label("speaker_1", speaker_mapping)
    return [
        TranscriptSegment(
            segment_id=f"seg_{index:04d}",
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            speaker_cluster_id="speaker_1",
            speaker_label=label,
            speaker_channel_label=item.speaker,
            text=item.text.strip(),
            confidence=item.confidence,
            diarization_confidence="low",
        )
        for index, item in enumerate(raw.segments, start=1)
        if item.text.strip()
    ]


def _run_pyannote(audio_path: Path) -> list[SpeakerTurn]:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import Hooks, ProgressHook
    from pydub import AudioSegment
    import torch

    call_id = audio_path.parent.name[len("call_") :] if audio_path.parent.name.startswith("call_") else audio_path.parent.name
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    audio = AudioSegment.from_file(audio_path)
    sample_rate = audio.frame_rate
    sample_width = audio.sample_width
    channels = audio.channels
    raw = audio.raw_data

    if sample_width == 1:
        dtype = torch.uint8
        waveform = torch.tensor(list(raw), dtype=dtype).float()
        waveform = (waveform - 128.0) / 128.0
    elif sample_width == 2:
        waveform = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif sample_width == 4:
        waveform = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width for diarization: {sample_width}")

    waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()
    class TrackingHook:
        def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
            set_stage_progress(call_id, "diarization", completed=completed, total=total, step_name=str(step_name))

    set_stage_progress(call_id, "diarization", completed=0, total=1, step_name="initializing")
    try:
        with Hooks(TrackingHook(), ProgressHook(hidden=True)) as hook:
            diarization_output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    finally:
        clear_stage_progress(call_id, "diarization")
    diarization = getattr(diarization_output, "speaker_diarization", diarization_output)
    speaker_map: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        cluster_id = speaker_map.setdefault(speaker, f"speaker_{len(speaker_map) + 1}")
        turns.append(
            SpeakerTurn(
                start_sec=float(turn.start),
                end_sec=float(turn.end),
                speaker_cluster_id=cluster_id,
                confidence="medium",
            )
        )
    return turns


def _align_turns_to_transcript_segments(
    raw: RawTranscript,
    turns: list[SpeakerTurn],
    speaker_mapping: dict[str, str] | None = None,
) -> list[TranscriptSegment]:
    if not turns:
        return _fallback_single_speaker(raw, speaker_mapping)

    segments: list[TranscriptSegment] = []
    for index, item in enumerate(raw.segments, start=1):
        if not item.text.strip():
            continue
        midpoint = (item.start_sec + item.end_sec) / 2 if item.end_sec >= item.start_sec else item.start_sec
        overlapping_turns = sorted(
            [
                (
                    _overlap_seconds(item.start_sec, item.end_sec, turn.start_sec, turn.end_sec),
                    turn,
                )
                for turn in turns
            ],
            key=lambda item_overlap: (item_overlap[1].start_sec, item_overlap[1].end_sec),
        )
        matching_turn = None
        positive_overlaps = [(overlap, turn) for overlap, turn in overlapping_turns if overlap > 0.0]
        text_chunks = _split_text_chunks(item.text)
        if len(positive_overlaps) > 1 and len(text_chunks) > 1:
            total_overlap = sum(overlap for overlap, _ in positive_overlaps)
            assigned_chunk_count = 0
            total_chunks = len(text_chunks)
            for overlap_index, (overlap, turn) in enumerate(positive_overlaps, start=1):
                if overlap_index == len(positive_overlaps):
                    chunk_count = total_chunks - assigned_chunk_count
                else:
                    chunk_count = max(1, round(total_chunks * (overlap / total_overlap)))
                    remaining_turns = len(positive_overlaps) - overlap_index
                    max_allowed = total_chunks - assigned_chunk_count - remaining_turns
                    chunk_count = min(chunk_count, max_allowed)
                chunk_end = assigned_chunk_count + chunk_count
                chunk_text = " ".join(text_chunks[assigned_chunk_count:chunk_end]).strip()
                assigned_chunk_count = chunk_end
                if not chunk_text:
                    continue
                seg_start = max(item.start_sec, turn.start_sec)
                seg_end = min(item.end_sec, turn.end_sec)
                if seg_end < seg_start:
                    seg_start = item.start_sec
                    seg_end = item.end_sec
                segments.append(
                    _segment_for_turn(
                        segment_id=f"seg_{index:04d}_{overlap_index}",
                        start_sec=seg_start,
                        end_sec=seg_end,
                        cluster_id=turn.speaker_cluster_id,
                        speaker_mapping=speaker_mapping,
                        text=chunk_text,
                        confidence=item.confidence,
                        diarization_confidence=turn.confidence,
                        speaker_channel_label=item.speaker,
                    )
                )
            continue
        if positive_overlaps:
            matching_turn = max(
                positive_overlaps,
                key=lambda item_overlap: (
                    item_overlap[0],
                    -min(
                        abs(item_overlap[1].start_sec - midpoint),
                        abs(item_overlap[1].end_sec - midpoint),
                    ),
                ),
            )[1]
        if matching_turn is None:
            matching_turn = min(
                turns,
                key=lambda turn: min(abs(turn.start_sec - midpoint), abs(turn.end_sec - midpoint)),
            )
        cluster_id = matching_turn.speaker_cluster_id
        segments.append(
            _segment_for_turn(
                segment_id=f"seg_{index:04d}",
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                cluster_id=cluster_id,
                speaker_mapping=speaker_mapping,
                text=item.text.strip(),
                confidence=item.confidence,
                diarization_confidence=matching_turn.confidence,
                speaker_channel_label=item.speaker,
            )
        )
    return segments


def apply_speaker_mapping(segments: list[dict | TranscriptSegment], mapping: dict[str, str]) -> list[dict]:
    normalized: list[dict] = []
    for item in segments:
        if is_dataclass(item):
            item = asdict(item)
        cluster_id = item.get("speaker_cluster_id") or item.get("speaker_label") or "speaker_1"
        item = {**item}
        item["speaker_cluster_id"] = cluster_id
        item["speaker_label"] = _resolve_label(cluster_id, mapping)
        normalized.append(item)
    return normalized


def diarize(
    raw: RawTranscript,
    config: AppConfig,
    audio_path: Path | None = None,
    speaker_mapping: dict[str, str] | None = None,
) -> list[TranscriptSegment]:
    if not raw.segments:
        return []
    if audio_path is None or not config.section("diarization").get("enabled", True):
        return _fallback_single_speaker(raw, speaker_mapping)

    provider = config.section("diarization").get("provider", "pyannote")
    try:
        if provider == "pyannote":
            turns = _run_pyannote(audio_path)
            if turns:
                return _align_turns_to_transcript_segments(raw, turns, speaker_mapping)
    except Exception:
        logger.exception("Diarization backend %s failed for %s", provider, audio_path)

    if config.section("diarization").get("fallback_single_speaker", True):
        return _fallback_single_speaker(raw, speaker_mapping)
    return []
