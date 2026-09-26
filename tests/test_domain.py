from __future__ import annotations

import json
from datetime import UTC, datetime

from narrative_dislocation.domain import DataProvenance, json_safe


def test_json_safe_converts_nan_and_dataclasses():
    value = {
        "missing": float("nan"),
        "zero": 0.0,
        "when": datetime(2025, 1, 2, tzinfo=UTC),
        "source": DataProvenance(source="unit-test", as_of="2025-01-02"),
    }

    normalized = json_safe(value)

    assert normalized["missing"] is None
    assert normalized["zero"] == 0.0
    assert normalized["when"] == "2025-01-02T00:00:00+00:00"
    assert normalized["source"]["source"] == "unit-test"
    json.dumps(normalized, allow_nan=False)
