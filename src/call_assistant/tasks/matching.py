from __future__ import annotations

from call_assistant.common.models import TaskRecord, TranscriptSegment


def attach_task_source(task: TaskRecord, segments: list[TranscriptSegment]) -> TaskRecord:
    lowered = task.text.lower()
    for segment in segments:
        if lowered and lowered[:40] in segment.text.lower():
            task.source_timestamp = segment.start_sec
            task.source_quote = segment.text
            return task
    for segment in segments:
        if any(token in segment.text.lower() for token in lowered.split()[:3]):
            task.source_timestamp = segment.start_sec
            task.source_quote = segment.text
            return task
    if segments:
        task.source_timestamp = segments[0].start_sec
        task.source_quote = segments[0].text
    return task
