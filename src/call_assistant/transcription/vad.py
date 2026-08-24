from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.io import read_json

logger = logging.getLogger(__name__)


@dataclass
class VadChunk:
    chunk_id: str
    start_sec: float
    end_sec: float
    duration_seconds: float


def _merge_chunks(chunks: list[VadChunk], gap_sec: float, min_chunk_sec: float) -> list[VadChunk]:
    if not chunks:
        return []
    merged: list[VadChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        previous = merged[-1]
        gap = chunk.start_sec - previous.end_sec
        if gap <= gap_sec or previous.duration_seconds < min_chunk_sec or chunk.duration_seconds < min_chunk_sec:
            merged[-1] = VadChunk(
                chunk_id=previous.chunk_id,
                start_sec=previous.start_sec,
                end_sec=chunk.end_sec,
                duration_seconds=max(0.0, chunk.end_sec - previous.start_sec),
            )
            continue
        merged.append(chunk)
    return [
        VadChunk(
            chunk_id=f"chunk_{index:04d}",
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            duration_seconds=item.duration_seconds,
        )
        for index, item in enumerate(merged, start=1)
    ]


def _split_long_chunks(chunks: list[VadChunk], max_chunk_sec: float) -> list[VadChunk]:
    if max_chunk_sec <= 0:
        return chunks
    split: list[VadChunk] = []
    for chunk in chunks:
        if chunk.duration_seconds <= max_chunk_sec:
            split.append(chunk)
            continue
        cursor = chunk.start_sec
        index = 1
        while cursor < chunk.end_sec:
            end_sec = min(chunk.end_sec, cursor + max_chunk_sec)
            split.append(
                VadChunk(
                    chunk_id=f"{chunk.chunk_id}_{index}",
                    start_sec=cursor,
                    end_sec=end_sec,
                    duration_seconds=max(0.0, end_sec - cursor),
                )
            )
            cursor = end_sec
            index += 1
    return [
        VadChunk(
            chunk_id=f"chunk_{index:04d}",
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            duration_seconds=item.duration_seconds,
        )
        for index, item in enumerate(split, start=1)
    ]


def _whole_file_chunk(audio_path: Path) -> list[VadChunk]:
    metadata = read_json(audio_path.parent / "metadata.json", default={})
    duration = float(metadata.get("duration_seconds") or 0.0)
    if duration <= 0:
        return [VadChunk(chunk_id="chunk_0001", start_sec=0.0, end_sec=0.0, duration_seconds=0.0)]
    return [VadChunk(chunk_id="chunk_0001", start_sec=0.0, end_sec=duration, duration_seconds=duration)]


def detect_speech_chunks(audio_path: Path, config: AppConfig) -> list[VadChunk]:
    section = config.section("transcription")
    if not section.get("vad_enabled", True):
        return _whole_file_chunk(audio_path)
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except Exception:
        logger.exception("VAD dependencies unavailable for %s", audio_path)
        return _whole_file_chunk(audio_path)

    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0
    if duration_sec <= 0:
        return [VadChunk(chunk_id="chunk_0001", start_sec=0.0, end_sec=0.0, duration_seconds=0.0)]

    min_silence_ms = int(section.get("vad_min_silence_ms", 300))
    min_chunk_ms = int(section.get("vad_min_chunk_ms", 500))
    merge_gap_sec = float(section.get("vad_merge_gap_ms", 250)) / 1000.0
    max_chunk_sec = float(section.get("vad_max_chunk_sec", 20))
    silence_threshold = audio.dBFS - 16 if audio.dBFS != float("-inf") else -48

    ranges = detect_nonsilent(audio, min_silence_len=min_silence_ms, silence_thresh=silence_threshold)
    if not ranges:
        return _whole_file_chunk(audio_path)

    chunks = [
        VadChunk(
            chunk_id=f"chunk_{index:04d}",
            start_sec=max(0.0, start_ms / 1000.0),
            end_sec=min(duration_sec, end_ms / 1000.0),
            duration_seconds=max(0.0, (end_ms - start_ms) / 1000.0),
        )
        for index, (start_ms, end_ms) in enumerate(ranges, start=1)
        if (end_ms - start_ms) >= min_chunk_ms
    ]
    if not chunks:
        return _whole_file_chunk(audio_path)
    return _split_long_chunks(_merge_chunks(chunks, merge_gap_sec, min_chunk_ms / 1000.0), max_chunk_sec)
