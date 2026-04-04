from __future__ import annotations

from pathlib import Path

from call_assistant.common.models import AudioMetadata


def normalize_audio(source: Path, dest: Path) -> AudioMetadata:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(source)
    normalized = audio.set_channels(1).set_frame_rate(16000)
    dest.parent.mkdir(parents=True, exist_ok=True)
    normalized.export(dest, format="wav")
    return AudioMetadata(
        duration_seconds=round(len(normalized) / 1000.0, 3),
        sample_rate=normalized.frame_rate,
        channels=normalized.channels,
        audio_format=source.suffix.lower().lstrip("."),
    )
