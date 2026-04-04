from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.models import RawTranscript, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    start_sec: float
    end_sec: float
    speaker_cluster_id: str
    confidence: str | None = None


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
    from pydub import AudioSegment
    import torch

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
    diarization_output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
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
        matching_turn = next(
            (
                turn
                for turn in turns
                if turn.start_sec <= midpoint <= turn.end_sec
            ),
            None,
        )
        if matching_turn is None:
            matching_turn = min(
                turns,
                key=lambda turn: min(abs(turn.start_sec - midpoint), abs(turn.end_sec - midpoint)),
            )
        cluster_id = matching_turn.speaker_cluster_id
        segments.append(
            TranscriptSegment(
                segment_id=f"seg_{index:04d}",
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                speaker_cluster_id=cluster_id,
                speaker_label=_resolve_label(cluster_id, speaker_mapping),
                speaker_channel_label=item.speaker,
                text=item.text.strip(),
                confidence=item.confidence,
                diarization_confidence=matching_turn.confidence or "medium",
            )
        )
    return segments


def apply_speaker_mapping(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    normalized: list[dict] = []
    for item in segments:
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
