import subprocess
from pathlib import Path

CALL_DIR = Path("/home/garik/CallAssistantData/calls/2026/04/04/call_20260404_164304_1b9540")
SOURCE = CALL_DIR / "audio_normalized.wav"
DEST_DIR = CALL_DIR / "audio_variants" / "boost_weak"
DEST_DIR.mkdir(exist_ok=True, parents=True)
DEST = DEST_DIR / "audio_normalized.wav"

# compress quiet sections and boost them slightly with ffmpeg
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        str(SOURCE),
        "-af",
        "acompressor=threshold=-45dB:ratio=5:attack=10:release=250,volume=2dB",
        str(DEST),
    ],
    check=True,
)
