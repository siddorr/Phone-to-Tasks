from pathlib import Path
from call_assistant.common.config import AppConfig
from call_assistant.audio.normalize import normalize_audio

CALL_DIR = Path("/home/garik/CallAssistantData/calls/2026/04/04/call_20260404_164304_1b9540")
SOURCE = CALL_DIR / "audio_normalized.wav"
VARIANTS = {"vol_05": 5, "vol_10": 10, "vol_20": 20}

def main() -> None:
    config = AppConfig.load("config.yaml")
    for name, gain in VARIANTS.items():
        dest = CALL_DIR / "audio_variants" / name
        dest.mkdir(parents=True, exist_ok=True)
        filters = config.section("audio_processing").copy()
        filters["target_db"] = gain
        config.data["audio_processing"] = filters
        _, _ = normalize_audio(SOURCE, dest / "audio_normalized.wav", config)

if __name__ == "__main__":
    main()
