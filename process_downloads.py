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
from call_assistant.ingest.watcher import detect_new_calls, scan_incoming
from call_assistant.orchestrator.worker import process_pending


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and process incoming call recordings")
    parser.add_argument("--list", action="store_true", help="List candidate files only")
    parser.add_argument("--no-process", action="store_true", help="Import new files without draining the queue")
    args = parser.parse_args()

    config = AppConfig.load(ROOT / "config.yaml")
    config.ensure_directories()
    configure_logging(config.logs_dir, config.section("logging")["level"])

    if args.list:
        for path in scan_incoming(config):
            print(path)
        return

    imported = detect_new_calls(config)
    print(f"Imported {len(imported)} new call(s)")
    if not args.no_process:
        processed = process_pending(config, scan_first=False)
        print(f"Processed {processed} queued job(s)")


if __name__ == "__main__":
    main()
