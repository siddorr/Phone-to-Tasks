from __future__ import annotations

from threading import Lock
from time import monotonic

_LOCK = Lock()
_PROGRESS: dict[tuple[str, str], dict[str, object]] = {}


def set_stage_progress(
    call_id: str,
    stage: str,
    *,
    completed: float | int | None,
    total: float | int | None,
    step_name: str | None = None,
) -> None:
    with _LOCK:
        _PROGRESS[(call_id, stage)] = {
            "completed": float(completed) if completed is not None else None,
            "total": float(total) if total is not None else None,
            "step_name": step_name,
            "updated_at_monotonic": monotonic(),
        }


def get_stage_progress(call_id: str, stage: str) -> dict[str, object] | None:
    with _LOCK:
        progress = _PROGRESS.get((call_id, stage))
        return dict(progress) if progress else None


def clear_stage_progress(call_id: str, stage: str) -> None:
    with _LOCK:
        _PROGRESS.pop((call_id, stage), None)
