from pathlib import Path

from call_assistant.audio.normalize import normalize_audio
from call_assistant.common.config import AppConfig


VARIANTS = {
    "prefix": {"target_db": -9.0},
    "warm": {"high_pass_hz": 100, "low_pass_hz": 8000},
    "sharp": {"high_pass_hz": 250, "low_pass_hz": 6000},
    "soft": {"target_sample_rate": 16000, "target_db": -3.0},
    "padded": {
        "silence_pad_ms": 300,
        "trim_silence_enabled": True,
        "chunking_enabled": False,
    },
    "recommended": {
        "trim_silence_enabled": True,
        "chunking_enabled": True,
        "chunk_max_seconds": 15,
        "chunk_overlap_ms": 250,
        "chunk_min_seconds": 0.5,
        "silence_pad_ms": 160,
    },
}


def main() -> None:
    config = AppConfig.load("config.yaml")
    call_dir = config.archive_root / "2026" / "04" / "04" / "call_20260404_164304_1b9540"
    source = next(path for path in call_dir.iterdir() if path.name.startswith("audio_original"))
    base_audio = config.section("audio_processing").copy()
    temp_root = call_dir / "audio_variants"
    for name, overrides in VARIANTS.items():
        filters = base_audio.copy()
        filters.update(overrides)
        config.data["audio_processing"] = filters
        dest_dir = temp_root / name
        dest_file = dest_dir / "audio_normalized.wav"
        _, _ = normalize_audio(source, dest_file, config)


if __name__ == "__main__":
    main()
