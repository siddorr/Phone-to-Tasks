from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.logging_utils import configure_logging
from call_assistant.orchestrator.worker import process_pending
from call_assistant.reprocess import reset_all_calls_for_retranscription, transcription_choices


def main() -> None:
    parser = argparse.ArgumentParser(description="Retranscribe all indexed calls with a selected model")
    parser.add_argument(
        "--model",
        default=None,
        help="Model choice in provider:model format, e.g. local:large-v3-turbo or cloud:whisper-1",
    )
    parser.add_argument("--no-process", action="store_true", help="Queue retranscription without draining the worker queue")
    parser.add_argument("--call-id", action="append", default=[], help="Optional specific call_id to retranscribe; repeatable")
    args = parser.parse_args()

    config = AppConfig.load(ROOT / "config.yaml")
    config.ensure_directories()
    configure_logging(config.logs_dir, config.section("logging")["level"])

    choices = transcription_choices(config)
    selected_model = args.model or choices[0]["value"]
    valid_choices = {item["value"] for item in choices}
    if selected_model not in valid_choices:
        available = ", ".join(sorted(valid_choices))
        raise SystemExit(f"Unsupported --model '{selected_model}'. Available: {available}")

    reset = reset_all_calls_for_retranscription(config, selected_model, args.call_id or None)
    print(f"Queued retranscription for {len(reset)} call(s) using {selected_model}")

    if not args.no_process:
        processed = process_pending(config, scan_first=False)
        print(f"Processed {processed} queued job(s)")


if __name__ == "__main__":
    main()
