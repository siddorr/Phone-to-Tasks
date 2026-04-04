from __future__ import annotations

import re

from call_assistant.common.models import CleanTranscript, TranscriptSegment


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(r"([?.!,])\1+", r"\1", cleaned)
    return cleaned


def clean_transcript(segments: list[TranscriptSegment]) -> CleanTranscript:
    lines: list[str] = []
    for segment in segments:
        cleaned = _clean_text(segment.text)
        segment.text = cleaned
        label = segment.speaker_label.replace("speaker_", "speaker ").upper()
        start = f"{segment.start_sec:07.2f}"
        lines.append(f"[{start}] {label}: {cleaned}")
    return CleanTranscript(text="\n".join(lines), segments=segments)
