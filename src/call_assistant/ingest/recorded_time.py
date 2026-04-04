from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path


FILENAME_TIMESTAMP_RE = re.compile(r"_(\d{6})_(\d{6})(?:_|$)")


def _local_tzinfo():
    return datetime.now().astimezone().tzinfo


def parse_recorded_at_from_filename(path: Path) -> tuple[str | None, str | None]:
    match = FILENAME_TIMESTAMP_RE.search(path.stem)
    if not match:
        return None, None
    date_part, time_part = match.groups()
    try:
        parsed = datetime.strptime(f"{date_part}{time_part}", "%y%m%d%H%M%S")
    except ValueError:
        return None, None
    tzinfo = _local_tzinfo()
    if tzinfo is not None:
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed.isoformat(), "filename"


def parse_recorded_at_from_media_metadata(path: Path) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=creation_time",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None

    if result.returncode != 0:
        return None, None

    value = result.stdout.strip()
    if not value:
        return None, None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    return parsed.isoformat(), "media_metadata"


def fallback_recorded_at_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, _local_tzinfo()).isoformat()


def resolve_recorded_at(path: Path) -> tuple[str | None, str, str]:
    for resolver, confidence in (
        (parse_recorded_at_from_filename, "high"),
        (parse_recorded_at_from_media_metadata, "medium"),
    ):
        recorded_at, source = resolver(path)
        if recorded_at:
            return recorded_at, source or "unknown", confidence
    return fallback_recorded_at_from_mtime(path), "filesystem_mtime", "low"
