from __future__ import annotations

from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.common.models import AudioMetadata


def normalize_audio(source: Path, dest: Path, config: AppConfig) -> tuple[AudioMetadata, list[dict]]:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    def _silence_threshold(segment: AudioSegment, filters: dict) -> float:
        explicit = filters.get("trim_silence_threshold_db")
        if isinstance(explicit, (int, float)):
            return float(explicit)
        offset = float(filters.get("trim_silence_threshold_offset_db", 16))
        dbfs = segment.dBFS
        if dbfs == float("-inf"):
            return -48.0
        return dbfs - offset

    def _detect_ranges(segment: AudioSegment, filters: dict) -> list[tuple[int, int]]:
        min_silence_ms = int(filters.get("trim_min_silence_ms", 120))
        silence_threshold = _silence_threshold(segment, filters)
        return detect_nonsilent(segment, min_silence_len=min_silence_ms, silence_thresh=silence_threshold)

    def _trim_segment(segment: AudioSegment, filters: dict) -> AudioSegment:
        if not filters.get("trim_silence_enabled"):
            return segment
        ranges = _detect_ranges(segment, filters)
        if not ranges:
            return segment
        padding = int(filters.get("trim_padding_ms", 0))
        start = max(0, ranges[0][0] - padding)
        end = min(len(segment), ranges[-1][1] + padding)
        return segment[start:end]

    def _split_ranges(
        ranges: list[tuple[int, int]], segment_ms: int, filters: dict
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        if not ranges:
            return result
        max_ms = int(filters.get("chunk_max_seconds", 20) * 1000)
        min_ms = int(filters.get("chunk_min_seconds", 0.5) * 1000)
        overlap_ms = int(filters.get("chunk_overlap_ms", 0))
        if max_ms <= 0:
            return result
        step = max(max_ms - overlap_ms, 1)
        for start, end in ranges:
            chunk_start = max(0, start - overlap_ms)
            chunk_end = min(segment_ms, end + overlap_ms)
            while chunk_start < chunk_end:
                current_end = min(chunk_end, chunk_start + max_ms)
                duration = current_end - chunk_start
                if duration >= min_ms:
                    result.append((chunk_start, current_end))
                chunk_start += step
        return result

    def _export_chunks(
        segment: AudioSegment, ranges: list[tuple[int, int]], filters: dict, base_dir: Path
    ) -> list[dict]:
        if not ranges:
            return []
        chunk_dir = base_dir / filters.get("chunk_subdir", "./audio_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        exported: list[dict] = []
        for index, (start_ms, end_ms) in enumerate(ranges, start=1):
            chunk_segment = segment[start_ms:end_ms]
            chunk_path = chunk_dir / f"chunk_{index:04d}.wav"
            chunk_segment.export(chunk_path, format="wav")
            exported.append(
                {
                    "chunk_id": f"chunk_{index:04d}",
                    "path": str(chunk_path),
                    "start_sec": round(start_ms / 1000.0, 3),
                    "end_sec": round(end_ms / 1000.0, 3),
                    "duration_seconds": round((end_ms - start_ms) / 1000.0, 3),
                }
            )
        return exported

    filters = config.section("audio_processing")
    audio = AudioSegment.from_file(source)
    processed = audio.set_channels(1).set_frame_rate(filters.get("target_sample_rate", 16000))
    processed = _trim_segment(processed, filters)
    high_pass = filters.get("high_pass_hz")
    if high_pass:
        processed = processed.high_pass_filter(high_pass)
    low_pass = filters.get("low_pass_hz")
    if low_pass:
        processed = processed.low_pass_filter(low_pass)
    denoise_mode = filters.get("denoise")
    if denoise_mode == "mild":
        processed = processed.low_pass_filter(filters.get("low_pass_hz", 7000))
    target_db = filters.get("target_db")
    if target_db is not None:
        processed = processed.normalize(headroom=target_db)
    chunk_info: list[dict] = []
    if filters.get("chunking_enabled"):
        speech_ranges = _detect_ranges(processed, filters)
        chunk_ranges = _split_ranges(speech_ranges, len(processed), filters)
        chunk_info = _export_chunks(processed, chunk_ranges, filters, dest.parent)
    padding = int(filters.get("silence_pad_ms", 0))
    if padding:
        processed = AudioSegment.silent(duration=padding) + processed + AudioSegment.silent(duration=padding)
    dest.parent.mkdir(parents=True, exist_ok=True)
    processed.export(dest, format="wav")
    metadata = AudioMetadata(
        duration_seconds=round(len(processed) / 1000.0, 3),
        sample_rate=processed.frame_rate,
        channels=processed.channels,
        audio_format=source.suffix.lower().lstrip("."),
    )
    return metadata, chunk_info
