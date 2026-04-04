from __future__ import annotations

import json
import os
import re

from call_assistant.common.config import AppConfig
from call_assistant.common.models import AnalysisBundle, CleanTranscript, TaskRecord, TranscriptSegment
from call_assistant.tasks.matching import attach_task_source


def _fallback_analysis(clean: CleanTranscript, segments: list[TranscriptSegment]) -> AnalysisBundle:
    lines = [line.strip() for line in clean.text.splitlines() if line.strip()]
    key_points = lines[:5]
    task_candidates: list[TaskRecord] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text
        lowered = text.lower()
        if any(marker in lowered for marker in ("need to", "should", "will", "follow up", "send", "call", "review")):
            task = TaskRecord(
                task_id=f"task_{index:04d}",
                text=text,
                owner="unknown",
                type="task",
                source_timestamp=segment.start_sec,
                source_quote=segment.text,
                confidence=0.5,
            )
            task_candidates.append(task)
    decisions = [segment.text for segment in segments if re.search(r"\b(decided|agreed|confirm)\b", segment.text, re.I)]
    questions = [segment.text for segment in segments if "?" in segment.text]
    commitments = [segment.text for segment in segments if re.search(r"\b(i will|we will|promise)\b", segment.text, re.I)]
    short_summary = lines[0] if lines else "No transcript content available."
    detailed_summary = " ".join(lines[:8]) if lines else short_summary
    return AnalysisBundle(
        short_summary=short_summary,
        detailed_summary=detailed_summary,
        key_points=key_points,
        decisions=decisions[:10],
        open_questions=questions[:10],
        commitments=commitments[:10],
        tasks=task_candidates,
        analysis_confidence=0.45 if task_candidates else 0.3,
        analysis_language="mixed",
        low_confidence_reason="deterministic fallback analysis",
    )


def _openai_analysis(clean: CleanTranscript, segments: list[TranscriptSegment], config: AppConfig) -> AnalysisBundle:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=api_key)
    prompt = """
Analyze this call transcript and return JSON with:
{
  "short_summary": "string",
  "detailed_summary": "string",
  "key_points": ["string"],
  "decisions": ["string"],
  "open_questions": ["string"],
  "commitments": ["string"],
  "analysis_confidence": 0.0,
  "analysis_language": "string",
  "tasks": [
    {
      "text": "string",
      "owner": "me|other|unknown",
      "type": "task|waiting|follow_up|reminder",
      "status": "new",
      "deadline": "optional string",
      "confidence": 0.0
    }
  ]
}
Return JSON only.
"""
    response = client.chat.completions.create(
        model=config.section("analysis")["model"],
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": clean.text},
        ],
        timeout=config.section("analysis").get("timeout_sec", 90),
    )
    payload = json.loads(response.choices[0].message.content)
    tasks: list[TaskRecord] = []
    for index, task_data in enumerate(payload.get("tasks", []), start=1):
        task = TaskRecord(
            task_id=f"task_{index:04d}",
            text=task_data["text"],
            owner=task_data.get("owner"),
            type=task_data.get("type", "task"),
            status=task_data.get("status", "new"),
            deadline=task_data.get("deadline"),
            confidence=task_data.get("confidence"),
            source_timestamp=None,
            source_quote=None,
        )
        tasks.append(attach_task_source(task, segments))
    return AnalysisBundle(
        short_summary=payload.get("short_summary", ""),
        detailed_summary=payload.get("detailed_summary", ""),
        key_points=payload.get("key_points", []),
        decisions=payload.get("decisions", []),
        open_questions=payload.get("open_questions", []),
        commitments=payload.get("commitments", []),
        tasks=tasks,
        analysis_confidence=payload.get("analysis_confidence"),
        analysis_language=payload.get("analysis_language", "unknown"),
        low_confidence_reason=None,
    )


def analyze_call(clean: CleanTranscript, segments: list[TranscriptSegment], config: AppConfig) -> AnalysisBundle:
    if not config.section("analysis").get("enabled", True):
        return _fallback_analysis(clean, segments)
    provider = config.section("analysis").get("provider", "openai")
    if provider == "openai":
        try:
            return _openai_analysis(clean, segments, config)
        except Exception:
            return _fallback_analysis(clean, segments)
    return _fallback_analysis(clean, segments)
