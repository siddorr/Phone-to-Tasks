from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from call_assistant.audio.normalize import normalize_audio
from call_assistant.common.config import AppConfig


def test_normalize_audio_trim_and_chunk(tmp_path: Path) -> None:
    config = AppConfig.load("config.yaml")
    filters = config.section("audio_processing").copy()
    filters.update(
        {
            "trim_silence_enabled": True,
            "trim_min_silence_ms": 50,
            "trim_padding_ms": 0,
            "chunking_enabled": True,
            "chunk_max_seconds": 0.8,
            "chunk_min_seconds": 0.1,
            "chunk_overlap_ms": 50,
            "silence_pad_ms": 0,
        }
    )
    config.data["audio_processing"] = filters
    source = tmp_path / "source.wav"
    silence = AudioSegment.silent(duration=400)
    tone = Sine(400).to_audio_segment(duration=900)
    (silence + tone + silence).export(source, format="wav")
    dest = tmp_path / "output" / "audio_normalized.wav"
    metadata, chunks = normalize_audio(source, dest, config)
    assert metadata.sample_rate == filters["target_sample_rate"]
    assert metadata.channels == 1
    assert metadata.duration_seconds <= 1.5
    assert len(chunks) > 0
    assert all(entry["duration_seconds"] <= filters["chunk_max_seconds"] + 0.05 for entry in chunks)
