from pathlib import Path

from call_assistant.common.config import AppConfig
from call_assistant.orchestrator.worker import _process_audio_prepare
from call_assistant.transcription.service import transcribe


def main() -> None:
    config = AppConfig.load("config.yaml")
    config.data["audio_processing"]["temp_subdir"] = "audio_variants/padded"
    call_dir = config.archive_root / "2026" / "04" / "04" / "call_20260404_164304_1b9540"
    _process_audio_prepare(config, call_dir)
    audio_path = call_dir / "audio_normalized.wav"

    print("LOCAL TRANSCRIPTION")
    local = transcribe(audio_path, config)
    print(local.text)

    config.data["transcription"]["provider_default"] = "cloud"
    print("CLOUD TRANSCRIPTION")
    cloud = transcribe(audio_path, config)
    print(cloud.text)


if __name__ == "__main__":
    main()
