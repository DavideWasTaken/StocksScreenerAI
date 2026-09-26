"""Deterministic CSV, JSON, and Markdown output for screener results.

All functions accept a pandas DataFrame, a mapping, or an iterable of mappings.
Nested narrative results are preserved in JSON, flattened in CSV, and rendered with
an explicit measured-fact/AI-inference distinction in Markdown.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

REPORT_SCHEMA_VERSION = 1
DEFAULT_SCORE_PATHS = (
    "dislocation_score",
    "final_dislocation_score",
    "scores.dislocation_score",
    "measured_facts.dislocation_score",
)

PREFERRED_FACT_NAMES = (
    "drawdown_from_2y_high",
    "drawdown_2y",
    "latest_fundamental_period_end",
    "latest_fundamental_age_days",
    "revenue_growth",
    "fcf_growth",
    "fcf_per_share_growth",
    "fcf_share_growth",
    "fcf_margin",
    "gross_margin_stability",
    "share_dilution",
    "net_debt_to_fcf",
    "roic",
    "roic_proxy",
    "ev_to_fcf",
    "fcf_yield",
    "pe",
    "pe_ratio",
    "valuation_compression",
    "relative_performance",
    "price_dislocation_score",
    "fundamental_resilience_score",
    "valuation_compression_score",
    "balance_sheet_quality_score",
)


def ranked_dataframe(
    results: Any,
    *,
    top_n: int | None = 20,
    score_column: str | None = None,
) -> Any:
    """Return a flattened, stably ranked DataFrame.

    If no recognized score is present, the caller's existing order is retained. This
    avoids inventing a ranking when upstream scoring is unavailable.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - requirements include pandas.
        raise RuntimeError("pandas is required for CSV/DataFrame reporting") from exc

    ranked = _rank_records(results, top_n=top_n, score_column=score_column)
    frame = pd.json_normalize(ranked, sep=".") if ranked else pd.DataFrame()
    if frame.empty:
        if "rank" not in frame.columns:
            frame.insert(0, "rank", [])
        return frame

    # Lists/dicts should be valid JSON in CSV, not Python repr with ambiguous quotes.
    for column in frame.columns:
        frame[column] = frame[column].map(_csv_cell)

    ordered = _ordered_csv_columns(list(frame.columns), score_column)
    return frame.loc[:, ordered]


def write_ranked_csv(
    results: Any,
    path: str | Path,
    *,
    top_n: int | None = 20,
    score_column: str | None = None,
) -> Path:
    """Write the ranked Top N CSV and return its resolved output path."""

    destination = _prepare_destination(path)
    frame = ranked_dataframe(results, top_n=top_n, score_column=score_column)
    temporary = _temporary_path(destination)
    try:
        frame.to_csv(temporary, index=False, na_rep="")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def build_detailed_payload(
    results: Any,
    *,
    top_n: int | None = None,
    score_column: str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the detailed, nested JSON payload without writing it."""

    ranked = _rank_records(results, top_n=top_n, score_column=score_column)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "count": len(ranked),
        "run_metadata": _json_safe(dict(run_metadata or {})),
        "candidates": ranked,
    }


def write_detailed_json(
    results: Any,
    path: str | Path,
    *,
    top_n: int | None = None,
    score_column: str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write all ranked details as valid JSON (NaN/Infinity become null)."""

    destination = _prepare_destination(path)
    payload = build_detailed_payload(
        results,
        top_n=top_n,
        score_column=score_column,
        run_metadata=run_metadata,
    )
    _atomic_text_write(
        destination,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n",
    )
    return destination


def build_markdown_report(
    results: Any,
    *,
    top_n: int = 10,
    score_column: str | None = None,
    title: str = "Narrative Dislocation — Top Candidates",
) -> str:
    """Render concise Top N stock briefs with explicit epistemic labels."""

    if top_n < 0:
        raise ValueError("top_n must be >= 0")
    ranked = _rank_records(results, top_n=top_n, score_column=score_column)
    lines = [
        f"# {title}",
        "",
        f"Generated: {_utc_now()}",
        "",
        (
            "> **Reading guide:** “Measured fact” means a value supplied by the quant "
            "pipeline. “AI inference” means model interpretation or web-grounded "
            "research. Missing financial data is not estimated. This is research "
            "triage, not investment advice."
        ),
        "",
    ]

    if not ranked:
        lines.extend(["No candidates were available.", ""])
        return "\n".join(lines)

    for record in ranked:
        lines.extend(_candidate_markdown(record, score_column=score_column))

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    results: Any,
    path: str | Path,
    *,
    top_n: int = 10,
    score_column: str | None = None,
    title: str = "Narrative Dislocation — Top Candidates",
) -> Path:
    """Write concise Top N Markdown briefs and return the output path."""

    destination = _prepare_destination(path)
    content = build_markdown_report(
        results,
        top_n=top_n,
        score_column=score_column,
        title=title,
    )
    _atomic_text_write(destination, content)
    return destination


def write_reports(
    results: Any,
    output_dir: str | Path,
    *,
    ranked_top_n: int = 20,
    report_top_n: int = 10,
    score_column: str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the complete output bundle under one run directory.

    Filenames are stable for CLI integration; the returned paths make handoff and
    logging straightforward.
    """

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "ranked_csv": write_ranked_csv(
            results,
            directory / "top_20.csv",
            top_n=ranked_top_n,
            score_column=score_column,
        ),
        "detailed_json": write_detailed_json(
            results,
            directory / "detailed_results.json",
            score_column=score_column,
            run_metadata=run_metadata,
        ),
        "markdown_report": write_markdown_report(
            results,
            directory / "top_10_report.md",
            top_n=report_top_n,
            score_column=score_column,
        ),
    }


def _candidate_markdown(record: Mapping[str, Any], *, score_column: str | None) -> list[str]:
    rank = record.get("rank", "?")
    ticker = _first_value(record, ("ticker", "symbol", "Ticker", "Symbol")) or "UNKNOWN"
    company = _first_value(
        record,
        ("company_name", "name", "long_name", "short_name", "longName", "shortName"),
    )
    inference = _find_nested_mapping(record, "ai_inference") or _inference_at_top(record)
    facts = _find_nested_mapping(record, "measured_facts") or _top_level_facts(record)
    evidence = _find_nested_sequence(record, "evidence_sources")

    score = _score_value(record, score_column)[0]
    gap = _mapping_value(inference, "narrative_gap_score")
    value_trap = _mapping_value(inference, "value_trap_probability")
    status = _find_nested_mapping(record, "ai_metadata") or {}
    ai_error = status.get("error") if isinstance(status, Mapping) else None

    heading = f"## {rank}. {_one_line(ticker)}"
    if company and _one_line(company).casefold() != _one_line(ticker).casefold():
        heading += f" — {_one_line(company)}"

    metrics: list[str] = []
    if score is not None:
        metrics.append(f"Dislocation Score: **{_score_text(score)} / 100**")
    if gap is not None:
        metrics.append(f"Narrative Gap: **{_score_text(gap)} / 100**")
    if value_trap is not None:
        metrics.append(f"Value-trap probability: **{_score_text(value_trap)}%**")

    lines = [heading, ""]
    if metrics:
        lines.extend([" · ".join(metrics), ""])
    if ai_error:
        lines.extend(
            [
                f"> AI analysis unavailable for this run: {_one_line(ai_error)}",
                "",
            ]
        )

    lines.extend(
        [
            "### Why it fell",
            "",
            f"**AI inference:** {_inference_text(inference, 'why_it_fell')}",
            "",
            "### What the market believes",
            "",
            (
                "**AI inference — dominant bearish narrative:** "
                + _inference_text(inference, "dominant_bearish_narrative")
            ),
            "",
            "**AI inference — expected damage if bears are right:**",
            "",
        ]
    )
    lines.extend(
        _markdown_bullets(_mapping_value(inference, "expected_damage_if_bear_thesis_is_correct"))
    )

    lines.extend(
        [
            "",
            "### What fundamentals show",
            "",
            "**Measured facts supplied by the quant pipeline:**",
            "",
        ]
    )
    lines.extend(_measured_fact_bullets(facts))
    lines.extend(
        [
            "",
            "**AI inference — interpretation of observed damage:**",
            "",
        ]
    )
    lines.extend(_markdown_bullets(_mapping_value(inference, "observed_fundamental_damage")))

    lines.extend(
        [
            "",
            "### Why it may be mispriced",
            "",
            f"**AI inference:** {_inference_text(inference, 'concise_thesis')}",
            "",
            "**AI inference — possible catalysts:**",
            "",
        ]
    )
    lines.extend(_markdown_bullets(_mapping_value(inference, "catalysts")))

    lines.extend(
        [
            "",
            "### What invalidates the thesis",
            "",
            "**AI inference — invalidation conditions:**",
            "",
        ]
    )
    lines.extend(_markdown_bullets(_mapping_value(inference, "invalidation_conditions")))
    lines.extend(["", "**AI inference — structural risks:**", ""])
    lines.extend(_markdown_bullets(_mapping_value(inference, "structural_risks")))

    lines.extend(["", "### Evidence", ""])
    lines.extend(_evidence_bullets(evidence))
    lines.extend(["", "---", ""])
    return lines


def _rank_records(
    results: Any,
    *,
    top_n: int | None,
    score_column: str | None,
) -> list[dict[str, Any]]:
    if top_n is not None and top_n < 0:
        raise ValueError("top_n must be >= 0 or None")
    records = _records(results)

    scored: list[tuple[int, dict[str, Any], float | None]] = []
    any_score = False
    for index, record in enumerate(records):
        numeric, _ = _score_value(record, score_column)
        any_score = any_score or numeric is not None
        scored.append((index, record, numeric))

    if any_score:
        scored.sort(
            key=lambda item: (
                item[2] is None,
                -(item[2] if item[2] is not None else 0.0),
                item[0],
            )
        )

    if top_n is not None:
        scored = scored[:top_n]
    ranked: list[dict[str, Any]] = []
    for rank, (_, record, _) in enumerate(scored, start=1):
        row = _json_safe(dict(record))
        # Rank is output state, not a measured fundamental.
        row["rank"] = rank
        ranked.append(row)
    return ranked


def _records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, BaseModel):
        return [_json_safe(data.model_dump(mode="python"))]
    if isinstance(data, Mapping):
        candidates = data.get("candidates")
        if isinstance(candidates, list):
            return [_record(item) for item in candidates]
        return [_record(data)]
    if hasattr(data, "to_dict"):
        try:
            values = data.to_dict(orient="records")
        except TypeError:
            values = None
        if isinstance(values, list):
            return [_record(item) for item in values]
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return [_record(item) for item in data]
    raise TypeError("results must be a mapping, DataFrame, or iterable of mappings")


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    elif not isinstance(value, Mapping) and hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("each result must be mapping-like")
    return _json_safe(dict(value))


def _score_value(
    record: Mapping[str, Any], score_column: str | None
) -> tuple[float | None, str | None]:
    paths = (score_column,) if score_column else DEFAULT_SCORE_PATHS
    for path in paths:
        if not path:
            continue
        found, value, actual_path = _path_value(record, path)
        if found:
            numeric = _finite_float(value)
            if numeric is not None:
                return numeric, actual_path
    return None, None


def _path_value(record: Mapping[str, Any], requested_path: str) -> tuple[bool, Any, str | None]:
    current: Any = record
    actual: list[str] = []
    for part in requested_path.split("."):
        if not isinstance(current, Mapping):
            return False, None, None
        key = _matching_key(current, part)
        if key is None:
            return False, None, None
        actual.append(str(key))
        current = current[key]
    return True, current, ".".join(actual)


def _matching_key(mapping: Mapping[str, Any], requested: str) -> str | None:
    if requested in mapping:
        return requested
    normalized = _normalized_key(requested)
    for key in mapping:
        if _normalized_key(str(key)) == normalized:
            return str(key)
    return None


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _find_nested_mapping(
    record: Mapping[str, Any], target: str, depth: int = 0
) -> dict[str, Any] | None:
    if depth > 5:
        return None
    key = _matching_key(record, target)
    if key is not None and isinstance(record[key], Mapping):
        return dict(record[key])
    for value in record.values():
        if isinstance(value, Mapping):
            found = _find_nested_mapping(value, target, depth + 1)
            if found is not None:
                return found
    return None


def _find_nested_sequence(record: Mapping[str, Any], target: str, depth: int = 0) -> list[Any]:
    if depth > 5:
        return []
    key = _matching_key(record, target)
    if (
        key is not None
        and isinstance(record[key], Sequence)
        and not isinstance(record[key], (str, bytes))
    ):
        return list(record[key])
    for value in record.values():
        if isinstance(value, Mapping):
            found = _find_nested_sequence(value, target, depth + 1)
            if found:
                return found
    return []


def _inference_at_top(record: Mapping[str, Any]) -> dict[str, Any]:
    if _matching_key(record, "why_it_fell") is not None:
        return dict(record)
    return {}


def _top_level_facts(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "aiinference",
        "aimetadata",
        "evidencesources",
        "narrativeanalysis",
        "analysis",
        "narrative",
        "rank",
        "whyitfell",
        "dominantbearishnarrative",
        "expecteddamageifbearthesisiscorrect",
        "observedfundamentaldamage",
        "catalysts",
        "structuralrisks",
        "valuetrapprobability",
        "narrativegapscore",
        "concisethesis",
        "invalidationconditions",
        "aiinferences",
        "evidence",
    }
    return {
        key: value
        for key, value in record.items()
        if _normalized_key(str(key)) not in excluded and not isinstance(value, Mapping)
    }


def _measured_fact_bullets(facts: Mapping[str, Any]) -> list[str]:
    if not facts:
        return ["- No measured facts were supplied for display."]

    selected: list[tuple[str, Any]] = []
    used: set[str] = set()
    by_normalized = {_normalized_key(str(key)): str(key) for key in facts}
    for preferred in PREFERRED_FACT_NAMES:
        actual = by_normalized.get(_normalized_key(preferred))
        if actual is not None and actual not in used:
            selected.append((actual, facts[actual]))
            used.add(actual)
        if len(selected) >= 18:
            break

    if not selected:
        for key, value in facts.items():
            if _normalized_key(str(key)) in {
                "ticker",
                "symbol",
                "companyname",
                "name",
                "rank",
            }:
                continue
            if isinstance(value, (Mapping, list, tuple)):
                continue
            selected.append((str(key), value))
            if len(selected) >= 10:
                break
    if not selected:
        return ["- No scalar measured facts were supplied for display."]
    return [f"- `{_one_line(key)}`: {_fact_value(value)}" for key, value in selected]


def _evidence_bullets(evidence: Sequence[Any]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        if isinstance(item, BaseModel):
            item = item.model_dump(mode="python")
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        title = _one_line(item.get("title") or url).replace("[", "\\[").replace("]", "\\]")
        supports = _one_line(item.get("supports") or "")
        suffix = f" — {supports}" if supports else ""
        rows.append(f"- [{title}]({url}){suffix}")
    return rows or ["- No evidence URL was returned for this analysis."]


def _markdown_bullets(value: Any) -> list[str]:
    if value is None or value == "":
        return ["- AI analysis unavailable."]
    values = value if isinstance(value, (list, tuple)) else [value]
    rows = [f"- {_one_line(item)}" for item in values if _one_line(item)]
    return rows or ["- AI analysis unavailable."]


def _inference_text(inference: Mapping[str, Any], key: str) -> str:
    value = _mapping_value(inference, key)
    if isinstance(value, (list, tuple)):
        value = "; ".join(_one_line(item) for item in value if _one_line(item))
    return _one_line(value) if value not in (None, "") else "AI analysis unavailable."


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    actual = _matching_key(mapping, key)
    return mapping.get(actual) if actual is not None else None


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        actual = _matching_key(mapping, key)
        if actual is not None and mapping.get(actual) not in (None, ""):
            return mapping[actual]
    # Analyzer envelopes may put identifying data in measured_facts.
    facts = _find_nested_mapping(mapping, "measured_facts")
    if facts:
        for key in keys:
            actual = _matching_key(facts, key)
            if actual is not None and facts.get(actual) not in (None, ""):
                return facts[actual]
    return None


def _ordered_csv_columns(columns: list[str], score_column: str | None) -> list[str]:
    preferences = [
        "rank",
        "ticker",
        "symbol",
        "company_name",
        score_column or "dislocation_score",
        "price_dislocation_score",
        "fundamental_resilience_score",
        "valuation_compression_score",
        "balance_sheet_quality_score",
        "ai_inference.narrative_gap_score",
        "narrative_gap_score",
        "ai_inference.value_trap_probability",
        "value_trap_probability",
    ]
    ordered: list[str] = []
    for preferred in preferences:
        if not preferred:
            continue
        normalized = _normalized_key(preferred)
        for column in columns:
            if column not in ordered and _normalized_key(column) == normalized:
                ordered.append(column)
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
            if item is not value:
                return _json_safe(item)
        except (TypeError, ValueError):
            pass
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, str) else str(value)


def _fact_value(value: Any) -> str:
    if value is None:
        return "not supplied"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "not supplied"
        return f"{value:.6g}"
    return _one_line(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _score_text(value: Any) -> str:
    numeric = _finite_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.1f}".rstrip("0").rstrip(".")


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _prepare_destination(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{os.getpid()}.tmp")


def _atomic_text_write(destination: Path, content: str) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# Friendly aliases for callers that prefer "export" terminology.
export_ranked_csv = write_ranked_csv
export_detailed_json = write_detailed_json
export_markdown_report = write_markdown_report


__all__ = [
    "build_detailed_payload",
    "build_markdown_report",
    "export_detailed_json",
    "export_markdown_report",
    "export_ranked_csv",
    "ranked_dataframe",
    "write_detailed_json",
    "write_markdown_report",
    "write_ranked_csv",
    "write_reports",
]
