from __future__ import annotations

from call_assistant.common.models import RawTranscript, TranscriptSegment


def diarize(raw: RawTranscript) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for index, item in enumerate(raw.segments, start=1):
        speaker = item.speaker if item.speaker in {"me", "other", "unknown"} else "unknown"
        segments.append(
            TranscriptSegment(
                segment_id=f"seg_{index:04d}",
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                speaker_label=speaker,
                speaker_channel_label=item.speaker,
                text=item.text.strip(),
                confidence=item.confidence,
            )
        )
    return segments
