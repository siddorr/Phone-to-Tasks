from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from call_assistant.common.config import AppConfig
from call_assistant.common.db import connect
from call_assistant.common.io import read_json, write_json
from call_assistant.ingest.recorded_time import resolve_recorded_at


def _best_source_path(call_dir: Path, metadata: dict) -> Path | None:
    source_path = metadata.get("source_path")
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return candidate
    originals = sorted(call_dir.glob("audio_original.*"))
    return originals[0] if originals else None


def main() -> None:
    config = AppConfig.load(ROOT / "config.yaml")
    db = connect(config.sqlite_path)
    rows = db.execute("SELECT call_id, archive_path FROM calls ORDER BY imported_at ASC").fetchall()
    updated = 0
    skipped = 0
    for row in rows:
        call_dir = Path(row["archive_path"])
        metadata_path = call_dir / "metadata.json"
        metadata = read_json(metadata_path, default={})
        source_path = _best_source_path(call_dir, metadata)
        if source_path is None:
            skipped += 1
            continue
        recorded_at, source, confidence = resolve_recorded_at(source_path)
        metadata["recorded_at"] = recorded_at
        metadata["recorded_at_source"] = source
        metadata["recorded_at_confidence"] = confidence
        write_json(metadata_path, metadata)
        db.execute(
            """
            UPDATE calls
            SET recorded_at = ?, recorded_at_source = ?, recorded_at_confidence = ?
            WHERE call_id = ?
            """,
            (recorded_at, source, confidence, row["call_id"]),
        )
        updated += 1
    db.commit()
    print(f"Updated {updated} call(s); skipped {skipped} call(s)")


if __name__ == "__main__":
    main()
