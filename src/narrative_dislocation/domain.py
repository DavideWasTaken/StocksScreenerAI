"""Small shared contracts and JSON normalization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataProvenance:
    source: str
    as_of: str | None = None
    url: str | None = None
    field: str | None = None
    note: str | None = None


@dataclass(slots=True)
class RunManifest:
    run_id: str
    started_at: str
    completed_at: str | None = None
    status: str = "running"
    configuration: dict[str, Any] = field(default_factory=dict)
    universe_size: int = 0
    fetched_symbols: int = 0
    quant_candidates: int = 0
    ai_analyzed: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def json_safe(value: Any) -> Any:
    """Recursively convert common scientific/Python values to strict JSON values.

    NaN and infinities become ``None``. That preserves the distinction between
    missing data and real zeroes and prevents non-standard JSON output.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)
