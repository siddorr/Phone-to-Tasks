from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect


def main() -> None:
    parser = argparse.ArgumentParser(description="Show transcript for a call")
    parser.add_argument("call_id", nargs="?", help="Call ID to show")
    args = parser.parse_args()

    config = AppConfig.load(ROOT / "config.yaml")
    db = connect(config.sqlite_path)
    if args.call_id:
        row = db.execute("SELECT archive_path FROM calls WHERE call_id = ?", (args.call_id,)).fetchone()
    else:
        row = db.execute("SELECT archive_path FROM calls ORDER BY imported_at DESC LIMIT 1").fetchone()
    if not row:
        print("No indexed calls found")
        return
    transcript_path = Path(row["archive_path"]) / "transcript_clean.txt"
    if transcript_path.exists():
        print(transcript_path.read_text(encoding="utf-8"))
    else:
        print(f"Transcript not found at {transcript_path}")


if __name__ == "__main__":
    main()
