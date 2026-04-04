from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uvicorn

from call_assistant.common.config import AppConfig
from call_assistant.common.logging_utils import configure_logging
from call_assistant.orchestrator.worker import WorkerThread
from call_assistant.ui.app import create_app


def main() -> None:
    config = AppConfig.load(ROOT / "config.yaml")
    config.ensure_directories()
    configure_logging(config.logs_dir, config.section("logging")["level"])
    worker = WorkerThread(config)
    worker.start()
    try:
        uvicorn.run(
            create_app(config),
            host=config.section("ui")["host"],
            port=int(config.section("ui")["port"]),
        )
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
