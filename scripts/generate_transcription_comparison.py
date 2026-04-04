from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import openai
import whisper
from openai import OpenAI
from pydub import AudioSegment

from call_assistant.common.config import AppConfig


OUTPUT_PATH = ROOT / "transcription_model_comparison.txt"
OPENAI_MODEL = "whisper-1"
DEFAULT_OPENAI_RETRIES = 3
DEFAULT_OPENAI_BACKOFF_SECONDS = 1.0
DEFAULT_OPENAI_CALL_PAUSE_SECONDS = 2.0


@dataclass
class TranscriptRun:
    model_name: str
    provider: str
    ok: bool
    language: str | None
    elapsed_seconds: float
    text: str
    error: str | None = None
    failure_kind: str | None = None
    input_path: str | None = None


def select_calls(limit: int) -> list[Path]:
    candidates = [path for path in (ROOT / "data" / "incoming").iterdir() if path.is_file()]
    return sorted(candidates, key=lambda path: (path.stat().st_size, path.name))[:limit]


def excerpt(text: str, limit: int = 240) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return squashed[: limit - 3] + "..."


def parse_model_names(spec: str | None) -> list[str]:
    available = sorted(whisper.available_models())
    if not spec:
        return available
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise SystemExit(f"Unknown Whisper model(s): {', '.join(unknown)}")
    return requested


def classify_openai_exception(exc: Exception) -> str:
    if isinstance(exc, openai.APIConnectionError):
        return "transport"
    if isinstance(exc, openai.APITimeoutError | openai.Timeout):
        return "timeout"
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return "auth"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
        return "api_validation"
    if isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code in (401, 403):
            return "auth"
        if status_code == 429:
            return "rate_limit"
        if status_code and 400 <= status_code < 500:
            return "api_validation"
    return "unknown"


def normalize_for_openai(audio_path: Path, index: int, keep_temp: bool) -> Path:
    prefix = f"openai_compare_call{index}_"
    temp_dir = None if keep_temp else tempfile.gettempdir()
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", dir=temp_dir, delete=False)
    temp_file.close()
    normalized_path = Path(temp_file.name)
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(normalized_path, format="wav")
    return normalized_path


def build_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def transcribe_local(audio_path: Path, model_name: str, language_hint: str | None) -> TranscriptRun:
    started = time.perf_counter()
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(str(audio_path), language=language_hint, verbose=False)
        elapsed = time.perf_counter() - started
        text = (result.get("text") or "").strip()
        return TranscriptRun(
            model_name=model_name,
            provider="local",
            ok=True,
            language=result.get("language"),
            elapsed_seconds=elapsed,
            text=text,
            input_path=str(audio_path),
        )
    except Exception as exc:
        return TranscriptRun(
            model_name=model_name,
            provider="local",
            ok=False,
            language=None,
            elapsed_seconds=time.perf_counter() - started,
            text="",
            error=str(exc),
            failure_kind="unknown",
            input_path=str(audio_path),
        )


def _transcribe_openai_once(normalized_audio_path: Path, client: OpenAI) -> tuple[str | None, str]:
    with normalized_audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=OPENAI_MODEL,
            file=audio_file,
            response_format="verbose_json",
        )
    payload = transcript.model_dump() if hasattr(transcript, "model_dump") else dict(transcript)
    return payload.get("language"), (payload.get("text") or "").strip()


def transcribe_openai(
    audio_path: Path,
    index: int,
    keep_temp: bool,
    client: OpenAI | None,
) -> TranscriptRun:
    started = time.perf_counter()
    if client is None:
        return TranscriptRun(
            model_name=OPENAI_MODEL,
            provider="openai",
            ok=False,
            language=None,
            elapsed_seconds=0.0,
            text="",
            error="OPENAI_API_KEY is not set",
            failure_kind="auth",
        )

    normalized_path = normalize_for_openai(audio_path, index=index, keep_temp=keep_temp)
    try:
        delay = DEFAULT_OPENAI_BACKOFF_SECONDS
        last_exception: Exception | None = None
        for attempt in range(1, DEFAULT_OPENAI_RETRIES + 1):
            try:
                language, text = _transcribe_openai_once(normalized_path, client)
                return TranscriptRun(
                    model_name=OPENAI_MODEL,
                    provider="openai",
                    ok=True,
                    language=language,
                    elapsed_seconds=time.perf_counter() - started,
                    text=text,
                    input_path=str(normalized_path),
                )
            except Exception as exc:  # pragma: no cover - network/path dependent
                last_exception = exc
                if attempt == DEFAULT_OPENAI_RETRIES:
                    break
                time.sleep(delay)
                delay *= 2
        assert last_exception is not None
        return TranscriptRun(
            model_name=OPENAI_MODEL,
            provider="openai",
            ok=False,
            language=None,
            elapsed_seconds=time.perf_counter() - started,
            text="",
            error=f"{type(last_exception).__name__}: {last_exception}",
            failure_kind=classify_openai_exception(last_exception),
            input_path=str(normalized_path),
        )
    finally:
        if normalized_path.exists() and not keep_temp:
            normalized_path.unlink()


def run_local_models(
    audio_path: Path,
    model_names: Iterable[str],
    language_hint: str | None,
) -> Iterable[TranscriptRun]:
    for model_name in model_names:
        yield transcribe_local(audio_path, model_name, language_hint)


def write_report(
    selected_calls: list[Path],
    results: dict[str, list[TranscriptRun]],
    include_local: bool,
    include_openai: bool,
    openai_mode: str,
) -> None:
    lines: list[str] = []
    lines.append("Phone-to-Tasks Transcription Model Comparison")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Calls compared: {len(selected_calls)}")
    lines.append(
        "Selected files: " + ", ".join(audio_path.name for audio_path in selected_calls)
    )
    if include_local and include_openai:
        lines.append(
            "Models compared: selected local Whisper models plus OpenAI whisper-1"
        )
    elif include_local:
        lines.append("Models compared: selected local Whisper models only")
    else:
        lines.append("Models compared: OpenAI whisper-1 only")
    if include_openai:
        lines.append("OpenAI inputs are normalized to mono 16 kHz WAV with ASCII-safe temp filenames before upload")
        lines.append(f"OpenAI request mode: {openai_mode}")
    lines.append("")

    for audio_path in selected_calls:
        call_key = audio_path.name
        lines.append("=" * 80)
        lines.append(call_key)
        lines.append(f"File size bytes: {audio_path.stat().st_size}")
        lines.append("")
        for run in results.get(call_key, []):
            header = f"{run.provider}:{run.model_name}"
            lines.append(header)
            lines.append("-" * len(header))
            lines.append(f"status: {'ok' if run.ok else 'failed'}")
            lines.append(f"language: {run.language or 'unknown'}")
            lines.append(f"elapsed_seconds: {run.elapsed_seconds:.2f}")
            if run.input_path:
                lines.append(f"input_path: {run.input_path}")
            if run.ok:
                lines.append(f"chars: {len(run.text)}")
                lines.append(f"excerpt: {excerpt(run.text)}")
            else:
                lines.append(f"failure_kind: {run.failure_kind or 'unknown'}")
                lines.append(f"error: {run.error}")
            lines.append("")
    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a text comparison of transcription models on a few calls.")
    parser.add_argument("--calls", type=int, default=3, help="Number of smallest incoming calls to compare")
    parser.add_argument("--local-only", action="store_true", help="Run only local Whisper models")
    parser.add_argument("--openai-only", action="store_true", help="Run only OpenAI whisper-1")
    parser.add_argument("--models", help="Comma-separated local Whisper models to run")
    parser.add_argument(
        "--keep-temp-openai-files",
        action="store_true",
        help="Keep normalized OpenAI WAV files for debugging",
    )
    args = parser.parse_args()

    if args.local_only and args.openai_only:
        raise SystemExit("--local-only and --openai-only are mutually exclusive")

    include_local = not args.openai_only
    include_openai = not args.local_only
    openai_mode = "shared-client-sequential"

    config = AppConfig.load(ROOT / "config.yaml")
    language_hint = config.section("transcription").get("language_hints", [None])[0]
    local_models = parse_model_names(args.models)
    selected_calls = select_calls(args.calls)
    openai_client = build_openai_client() if include_openai else None

    if not selected_calls:
        raise SystemExit("No audio files found in data/incoming")

    results: dict[str, list[TranscriptRun]] = {}
    for index, audio_path in enumerate(selected_calls, start=1):
        call_results: list[TranscriptRun] = []
        results[audio_path.name] = call_results
        if include_local:
            for run in run_local_models(audio_path, local_models, language_hint):
                call_results.append(run)
                write_report(
                    selected_calls,
                    results,
                    include_local=include_local,
                    include_openai=include_openai,
                    openai_mode=openai_mode,
                )
        if include_openai:
            call_results.append(
                transcribe_openai(
                    audio_path,
                    index=index,
                    keep_temp=args.keep_temp_openai_files,
                    client=openai_client,
                )
            )
            write_report(
                selected_calls,
                results,
                include_local=include_local,
                include_openai=include_openai,
                openai_mode=openai_mode,
            )
            if index < len(selected_calls):
                time.sleep(DEFAULT_OPENAI_CALL_PAUSE_SECONDS)

    write_report(
        selected_calls,
        results,
        include_local=include_local,
        include_openai=include_openai,
        openai_mode=openai_mode,
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
