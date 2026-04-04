from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "app": {"name": "call-assistant", "environment": "local"},
    "paths": {
        "incoming_folder": "./data/incoming",
        "archive_root": "./data/calls",
        "sqlite_path": "./data/index/call_assistant.db",
        "logs_dir": "./data/logs",
        "temp_dir": "./data/temp",
    },
    "ingest": {
        "supported_extensions": [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"],
        "stability_check_seconds": 20,
        "stability_poll_interval_seconds": 5,
        "file_hash_algorithm": "sha256",
        "scan_interval_seconds": 10,
    },
    "queue": {
        "worker_count": 1,
        "stale_job_seconds": 300,
        "transcription_stale_multiplier": 3,
        "transcription_stale_buffer_seconds": 600,
        "transcription_stale_min_seconds": 1800,
        "retry_limits": {
            "import": 2,
            "audio_prepare": 2,
            "transcription": 2,
            "diarization": 2,
            "transcript_clean": 2,
            "analysis": 2,
            "indexing": 2,
        },
    },
    "processing": {
        "startup_mode": "automatic",
    },
    "transcription": {
        "provider_default": "local",
        "provider_fallback": "cloud",
        "local_model": "large-v3-turbo",
        "local_device": "auto",
        "cloud_enabled": True,
        "cloud_model": "whisper-1",
        "language_hints": ["ru", "he", "en"],
        "language_mode": "auto",
        "force_language_for_single_language_calls": False,
        "mixed_language_triggers": ["he", "ru"],
    },
    "diarization": {
        "provider": "pyannote",
        "enabled": True,
        "fallback_single_speaker": True,
        "confidence_default": "medium",
    },
    "speaker_identity": {
        "enabled": True,
        "provider": "pyannote",
        "auto_assign_threshold": 0.75,
        "suggest_threshold": 0.60,
        "min_cluster_duration_seconds": 6,
        "max_segments_per_cluster": 20,
        "continue_on_error": True,
    },
    "analysis": {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "summary_language": "en",
        "timeout_sec": 90,
        "max_retries": 2,
    },
    "ui": {"host": "127.0.0.1", "port": 8080},
    "logging": {"level": "INFO"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class AppConfig:
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    root_dir: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "AppConfig":
        path = Path(config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        loaded: dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        return cls(data=_deep_merge(DEFAULT_CONFIG, loaded), root_dir=path.parent if path.exists() else Path.cwd())

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]

    def path(self, *parts: str) -> Path:
        value: Any = self.data
        for part in parts:
            value = value[part]
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (self.root_dir / candidate).resolve()

    @property
    def incoming_folder(self) -> Path:
        return self.path("paths", "incoming_folder")

    @property
    def archive_root(self) -> Path:
        return self.path("paths", "archive_root")

    @property
    def sqlite_path(self) -> Path:
        return self.path("paths", "sqlite_path")

    @property
    def logs_dir(self) -> Path:
        return self.path("paths", "logs_dir")

    @property
    def temp_dir(self) -> Path:
        return self.path("paths", "temp_dir")

    def retry_limit(self, stage: str) -> int:
        return int(self.data["queue"]["retry_limits"].get(stage, 1))

    def ensure_directories(self) -> None:
        for directory in (
            self.incoming_folder,
            self.archive_root,
            self.sqlite_path.parent,
            self.logs_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
