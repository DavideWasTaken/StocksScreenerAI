"""OpenAI-backed narrative research for quant-screened stock candidates.

The module deliberately treats every value received from the quant pipeline as a
*supplied measured fact* and every model-produced statement as an *AI inference*.
It never fills missing financial values.  One successful response is cached per
ticker and input fingerprint, so an interrupted batch can be resumed safely.

The implementation follows the current Responses API Python SDK patterns:
``client.responses.parse`` for Pydantic structured output, and the optional
built-in ``web_search`` tool with source metadata included in the response.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:  # Keep reporting/offline imports usable before optional dependencies install.
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only in a partial installation.
    OpenAI = None  # type: ignore[assignment,misc]


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "narrative-gap-v1"
CACHE_SCHEMA_VERSION = 3

# These fields describe when/how the local pipeline ran, not a change in the
# analytical evidence sent to the model.  In particular, a cache-first rerun
# receives a new batch-completion ``as_of`` and flips ``source_cached`` even
# though every financial and price fact is unchanged.  Including either value
# in the fingerprint would defeat per-candidate resume after an interruption.
_FINGERPRINT_IGNORED_TOP_LEVEL_FIELDS = frozenset({"as_of"})
_FINGERPRINT_IGNORED_PROVENANCE_FIELDS = frozenset({"source_cached"})


class NarrativeAnalysisError(RuntimeError):
    """Raised when a candidate cannot be analyzed after all retry attempts."""


class MissingOpenAIKeyError(NarrativeAnalysisError):
    """Raised when no injected client and no ``OPENAI_API_KEY`` are available."""


class ClaimEvidence(BaseModel):
    """A source the model says supports a specific inference."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    supports: str

    @field_validator("url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        if not _is_http_url(value):
            raise ValueError("evidence URL must be an absolute http(s) URL")
        return value


class NarrativeAssessment(BaseModel):
    """Strict model output: all fields below are inference, not raw financials."""

    model_config = ConfigDict(extra="forbid")

    why_it_fell: str
    dominant_bearish_narrative: str
    expected_damage_if_bear_thesis_is_correct: list[str]
    observed_fundamental_damage: list[str]
    catalysts: list[str]
    structural_risks: list[str]
    value_trap_probability: int = Field(ge=0, le=100)
    narrative_gap_score: int = Field(ge=0, le=100)
    concise_thesis: str
    invalidation_conditions: list[str]
    ai_inferences: list[str]
    evidence: list[ClaimEvidence]


class EvidenceSource(BaseModel):
    """Normalized evidence exposed to downstream reports."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    supports: str
    source_type: Literal[
        "structured_evidence", "url_citation", "web_search_source", "response_source"
    ]

    @field_validator("url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        if not _is_http_url(value):
            raise ValueError("evidence URL must be an absolute http(s) URL")
        return value


class AnalysisMetadata(BaseModel):
    """Operational metadata kept outside both facts and inference."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "error"] = "completed"
    model: str
    prompt_version: str = PROMPT_VERSION
    analyzed_at_utc: str
    web_search_enabled: bool
    used_web_search: bool
    cache_hit: bool = False
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None


class NarrativeResult(BaseModel):
    """Public result envelope with an explicit fact/inference boundary."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    company_name: str | None = None
    measured_facts: dict[str, Any]
    ai_inference: NarrativeAssessment | None
    evidence_sources: list[EvidenceSource]
    ai_metadata: AnalysisMetadata


SYSTEM_PROMPT = """You are a skeptical equity-research assistant analyzing a stock that
has already passed a quantitative narrative-dislocation screen.

FACT DISCIPLINE
- The user message contains a JSON object named measured_facts. Those values are the
  only measured financial facts available to you. Treat null as unavailable.
- Never estimate, interpolate, backfill, or invent a missing financial value.
- Keep supplied measurements distinct from interpretation. Numeric claims about the
  company's fundamentals must quote the supplied metric name/value or be omitted.
- Web material is evidence about events, management commentary, consensus narrative,
  and risks; it does not silently replace missing measured_facts.
- Every output field is AI inference. Be explicit about uncertainty and conflicting
  evidence. This is research triage, not investment advice.

RESEARCH
- When web search is available, use it where useful. Prefer recent company filings,
  investor-relations releases, earnings-call material, SEC filings, and reputable
  reporting. Distinguish event dates from article publication dates.
- Evidence URLs must be URLs actually consulted during this response. Do not fabricate
  a URL. State that evidence is limited when reliable sources are unavailable.

SCORING
- Narrative Gap = market-implied deterioration minus observed fundamental
  deterioration. Score 100 only when the selloff appears to imply far more durable
  damage than supplied fundamentals and evidence currently show; score 0 when observed
  damage meets/exceeds the narrative or the apparent gap is unsupported.
- Value-trap probability is 0-100, where 100 means a very high chance that apparently
  cheap valuation reflects lasting deterioration rather than temporary dislocation.
- Expected damage describes what should become observable if the bear thesis is right.
- Observed damage must be grounded in measured_facts and may say data is unavailable.

Return the requested structured fields concisely. Include thesis invalidators that are
observable and specific rather than generic price movements."""


def configured_model() -> str:
    """Return the model configured at call time, allowing dotenv/env overrides."""

    return os.getenv("NARRATIVE_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def build_candidate_prompt(candidate: Mapping[str, Any]) -> str:
    """Build the auditable user prompt for one already-selected candidate."""

    facts = _json_safe(dict(candidate))
    ticker = _extract_ticker(facts)
    payload = {
        "research_as_of": date.today().isoformat(),
        "ticker": ticker,
        "company_name": _extract_company_name(facts),
        "measured_facts": facts,
        "task": (
            "Research why this stock fell, the dominant bearish narrative, damage "
            "expected if bears are right versus damage observable in the supplied "
            "facts, catalysts, structural risks, value-trap probability, and the "
            "Narrative Gap Score."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)


class NarrativeAnalyzer:
    """Analyze only the candidates explicitly supplied by the calling quant pipeline.

    Parameters are injectable so the complete behavior can be tested offline.  The
    OpenAI client is constructed lazily; an injected fake client does not require an
    API key.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        cache_dir: str | Path = "data/cache/ai",
        cache_ttl_hours: float | None = None,
        use_web_search: bool = True,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        timeout: float = 120.0,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay must be >= 0")
        if cache_ttl_hours is not None and cache_ttl_hours < 0:
            raise ValueError("cache_ttl_hours must be >= 0 or None")

        self.model = model or configured_model()
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_hours = cache_ttl_hours
        self.use_web_search = use_web_search
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.timeout = timeout
        self._api_key = api_key
        self._client = client

    def analyze_candidate(
        self,
        candidate: Mapping[str, Any] | Any,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Analyze one candidate, returning a JSON-safe fact/inference envelope."""

        facts = _coerce_record(candidate)
        ticker = _extract_ticker(facts)
        company_name = _extract_company_name(facts)
        fingerprint = self._fingerprint(facts)
        cache_path = self._cache_path(ticker)

        if not force_refresh:
            cached = self._load_cache(cache_path, fingerprint, current_facts=facts)
            if cached is not None:
                return cached

        prompt = build_candidate_prompt(facts)
        response = self._request_with_retries(prompt, ticker)
        assessment = _parse_assessment(response)
        response_evidence = _extract_response_evidence(response)
        evidence = _merge_evidence(assessment.evidence, response_evidence)

        result = NarrativeResult(
            ticker=ticker,
            company_name=company_name,
            measured_facts=facts,
            ai_inference=assessment,
            evidence_sources=evidence,
            ai_metadata=AnalysisMetadata(
                model=self.model,
                analyzed_at_utc=_utc_now(),
                web_search_enabled=self.use_web_search,
                used_web_search=(
                    self.use_web_search and _response_used_web_search(response, response_evidence)
                ),
                cache_hit=False,
                response_id=_optional_string(getattr(response, "id", None)),
                usage=_extract_usage(response),
            ),
        )
        result_dict = result.model_dump(mode="json")
        self._write_cache(cache_path, fingerprint, result_dict)
        return result_dict

    def analyze_candidates(
        self,
        candidates: Any,
        *,
        force_refresh: bool = False,
        continue_on_error: bool = False,
        limit: int | None = None,
        as_dataframe: bool = False,
    ) -> list[dict[str, Any]] | Any:
        """Analyze supplied rows in order; this method never expands the universe.

        ``limit`` simply takes the first N rows, allowing the caller's quant ranking to
        decide which names are "best". Successful earlier rows are already cached if a
        later row fails, which makes the batch resumable.
        """

        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0 or None")
        records = _coerce_records(candidates)
        if limit is not None:
            records = records[:limit]

        results: list[dict[str, Any]] = []
        for record in records:
            try:
                results.append(self.analyze_candidate(record, force_refresh=force_refresh))
            except Exception as exc:
                if not continue_on_error:
                    raise
                ticker = _best_effort_ticker(record)
                LOGGER.error("Narrative analysis failed for %s: %s", ticker, exc)
                results.append(self._error_result(record, ticker, exc))

        if as_dataframe:
            return narrative_results_to_dataframe(results)
        return results

    def _request_with_retries(self, prompt: str, ticker: str) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "text_format": NarrativeAssessment,
        }
        if self.use_web_search:
            request.update(
                {
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto",
                    "include": ["web_search_call.action.sources"],
                }
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._send_request(request)
            except MissingOpenAIKeyError:
                raise
            except Exception as exc:  # SDK/API/validation failures are safe to retry.
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2**attempt)
                LOGGER.warning(
                    "OpenAI narrative request for %s failed (%s/%s): %s; retrying in %.1fs",
                    ticker,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                if delay:
                    time.sleep(delay)

        raise NarrativeAnalysisError(
            f"OpenAI narrative analysis failed for {ticker} after "
            f"{self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def _send_request(self, request: Mapping[str, Any]) -> Any:
        """Use the SDK Pydantic helper, with a REST-schema fallback for older SDKs."""

        responses_api = self._get_client().responses
        parse = getattr(responses_api, "parse", None)
        if callable(parse):
            return parse(**dict(request))

        # ``responses.parse`` is the current official Python interface. A few older
        # SDK versions exposed Responses ``create`` first, so keep installations that
        # satisfy the project's broad minimum usable without falling back to Chat
        # Completions. ``_parse_assessment`` validates ``output_text`` afterward.
        create = getattr(responses_api, "create", None)
        if not callable(create):
            raise NarrativeAnalysisError(
                "Installed OpenAI SDK exposes neither responses.parse nor responses.create"
            )
        fallback = dict(request)
        fallback.pop("text_format", None)
        fallback["text"] = {
            "format": {
                "type": "json_schema",
                "name": "narrative_assessment",
                "strict": True,
                "schema": NarrativeAssessment.model_json_schema(),
            }
        }
        return create(**fallback)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise NarrativeAnalysisError(
                "The 'openai' package is required for AI analysis. Install project dependencies."
            )
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise MissingOpenAIKeyError(
                "OPENAI_API_KEY is not set. Add it to the environment or run quant-only."
            )
        # SDK retries are disabled here because this class owns bounded, visible retries.
        self._client = OpenAI(
            api_key=api_key,
            timeout=self.timeout,
            max_retries=0,
        )
        return self._client

    def _fingerprint(self, facts: Mapping[str, Any]) -> str:
        payload = {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "use_web_search": self.use_web_search,
            "measured_facts": _stable_fingerprint_facts(facts),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, ticker: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker).strip("._-")
        if not safe:
            safe = "ticker"
        if safe != ticker:
            suffix = hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8]
            safe = f"{safe}-{suffix}"
        return self.cache_dir / f"{safe}.json"

    def _load_cache(
        self,
        path: Path,
        fingerprint: str,
        *,
        current_facts: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            if self.cache_ttl_hours is not None:
                age_hours = max(0.0, time.time() - path.stat().st_mtime) / 3600.0
                if age_hours >= self.cache_ttl_hours:
                    return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION
                or payload.get("fingerprint") != fingerprint
            ):
                return None
            result = NarrativeResult.model_validate(payload["result"])
            # Inference/evidence are reusable because the stable analytical
            # fingerprint matched. Return the caller's current fact envelope,
            # however, so operational timestamps and cache provenance are never
            # stale or misrepresented in this run's report.
            result.measured_facts = _json_safe(dict(current_facts))
            if result.ai_metadata.web_search_enabled and _evidence_implies_web_search(
                result.evidence_sources
            ):
                # Older cache entries could under-report a real call when the
                # SDK omitted/serialized the output-item discriminator but
                # returned `web_search_call.action.sources`. The source type is
                # definitive API metadata, not model-authored evidence.
                result.ai_metadata.used_web_search = True
            result.ai_metadata.cache_hit = True
            return result.model_dump(mode="json")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring unreadable narrative cache %s: %s", path, exc)
            return None

    def _write_cache(self, path: Path, fingerprint: str, result: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "result": _json_safe(dict(result)),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            # os.replace removes the temporary path on success. On a failed write it is
            # safe to unlink only this exact process-specific temporary file.
            if temporary.exists():
                temporary.unlink()

    def _error_result(
        self, record: Mapping[str, Any], ticker: str, exc: Exception
    ) -> dict[str, Any]:
        facts = _json_safe(dict(record))
        result = NarrativeResult(
            ticker=ticker,
            company_name=_extract_company_name(facts),
            measured_facts=facts,
            ai_inference=None,
            evidence_sources=[],
            ai_metadata=AnalysisMetadata(
                status="error",
                model=self.model,
                analyzed_at_utc=_utc_now(),
                web_search_enabled=self.use_web_search,
                used_web_search=False,
                error=str(exc),
            ),
        )
        return result.model_dump(mode="json")


def analyze_candidates(
    candidates: Any,
    *,
    analyzer: NarrativeAnalyzer | None = None,
    **batch_options: Any,
) -> list[dict[str, Any]] | Any:
    """Convenience wrapper around :class:`NarrativeAnalyzer`."""

    return (analyzer or NarrativeAnalyzer()).analyze_candidates(candidates, **batch_options)


def narrative_results_to_dataframe(results: Any) -> Any:
    """Flatten narrative results for score integration while retaining clear labels."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - project requirements include pandas.
        raise RuntimeError("pandas is required for DataFrame output") from exc

    rows: list[dict[str, Any]] = []
    for raw in _coerce_records(results):
        facts = raw.get("measured_facts") or {}
        inference = raw.get("ai_inference") or {}
        metadata = raw.get("ai_metadata") or {}
        row = dict(facts) if isinstance(facts, Mapping) else {}
        row["ticker"] = raw.get("ticker") or row.get("ticker")
        row["company_name"] = raw.get("company_name") or row.get("company_name")
        if isinstance(inference, Mapping):
            row.update(inference)
        row["evidence_sources"] = raw.get("evidence_sources", [])
        row["ai_status"] = metadata.get("status") if isinstance(metadata, Mapping) else None
        row["ai_cache_hit"] = metadata.get("cache_hit") if isinstance(metadata, Mapping) else None
        rows.append(_json_safe(row))
    return pd.DataFrame.from_records(rows)


def _parse_assessment(response: Any) -> NarrativeAssessment:
    parsed = getattr(response, "output_parsed", None)
    if parsed is None and isinstance(response, Mapping):
        parsed = response.get("output_parsed")
    if parsed is not None:
        return NarrativeAssessment.model_validate(parsed)

    output_text = getattr(response, "output_text", None)
    if output_text is None and isinstance(response, Mapping):
        output_text = response.get("output_text")
    if output_text:
        return NarrativeAssessment.model_validate_json(output_text)

    refusal = _find_refusal(response)
    suffix = f": {refusal}" if refusal else ""
    raise NarrativeAnalysisError(f"OpenAI response contained no parsed output{suffix}")


def _extract_response_evidence(response: Any) -> list[EvidenceSource]:
    # Restrict traversal to API output items. ``output_parsed`` can contain the
    # model's structured evidence too, but those entries are already handled as
    # ``structured_evidence`` and should not be mislabeled as API citations.
    output = getattr(response, "output", None)
    if output is None and isinstance(response, Mapping):
        output = response.get("output")
    plain = _plain_object(output if output is not None else response)
    found: list[EvidenceSource] = []

    def walk(value: Any, parent_key: str = "", depth: int = 0) -> None:
        if depth > 16:
            return
        if isinstance(value, Mapping):
            url = value.get("url")
            if isinstance(url, str) and _is_http_url(url):
                item_type = str(value.get("type") or "")
                if item_type == "url_citation":
                    source_type = "url_citation"
                elif parent_key == "sources":
                    source_type = "web_search_source"
                else:
                    source_type = "response_source"
                found.append(
                    EvidenceSource(
                        title=str(value.get("title") or value.get("name") or ""),
                        url=url,
                        supports="",
                        source_type=source_type,
                    )
                )
            for key, child in value.items():
                walk(child, str(key), depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child, parent_key, depth + 1)

    walk(plain)
    return found


def _evidence_implies_web_search(evidence: Iterable[EvidenceSource]) -> bool:
    return any(item.source_type == "web_search_source" for item in evidence)


def _response_used_web_search(
    response: Any,
    extracted_evidence: Iterable[EvidenceSource] = (),
) -> bool:
    """Return whether the API output contains an actual web-search tool call."""

    if _evidence_implies_web_search(extracted_evidence):
        return True

    output = getattr(response, "output", None)
    if output is None and isinstance(response, Mapping):
        output = response.get("output")
    plain = _plain_object(output)

    def walk(value: Any, depth: int = 0) -> bool:
        if depth > 16:
            return False
        if isinstance(value, Mapping):
            if str(value.get("type") or "") == "web_search_call":
                return True
            return any(walk(child, depth + 1) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(walk(child, depth + 1) for child in value)
        return False

    return walk(plain)


def _merge_evidence(
    structured: Iterable[ClaimEvidence], extracted: Iterable[EvidenceSource]
) -> list[EvidenceSource]:
    structured_items = list(structured)
    extracted_items = list(extracted)
    extracted_urls = {_canonical_url(item.url) for item in extracted_items}
    merged: dict[str, EvidenceSource] = {}
    for item in structured_items:
        # When the API returns authoritative citation/search metadata, do not expose a
        # model-written URL unless it appears there. This prevents a plausible-looking
        # but unconsulted URL from becoming report evidence.
        if extracted_urls and _canonical_url(item.url) not in extracted_urls:
            continue
        source = EvidenceSource(
            title=item.title,
            url=item.url,
            supports=item.supports,
            source_type="structured_evidence",
        )
        merged[_canonical_url(source.url)] = source

    source_priority = {
        "structured_evidence": 0,
        "response_source": 1,
        "web_search_source": 2,
        "url_citation": 3,
    }
    for source in extracted_items:
        key = _canonical_url(source.url)
        existing = merged.get(key)
        if existing is None:
            merged[key] = source
            continue
        updates: dict[str, Any] = {}
        if not existing.title and source.title:
            updates["title"] = source.title
        if not existing.supports and source.supports:
            updates["supports"] = source.supports
        if source_priority[source.source_type] > source_priority[existing.source_type]:
            updates["source_type"] = source.source_type
        if updates:
            merged[key] = existing.model_copy(update=updates)
    return list(merged.values())


def _stable_fingerprint_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Return analytical facts without volatile local-run bookkeeping."""

    stable = _json_safe(dict(facts))
    for field in _FINGERPRINT_IGNORED_TOP_LEVEL_FIELDS:
        stable.pop(field, None)

    provenance = stable.get("data_provenance")
    if isinstance(provenance, Mapping):
        stable_provenance = dict(provenance)
        for field in _FINGERPRINT_IGNORED_PROVENANCE_FIELDS:
            stable_provenance.pop(field, None)
        stable["data_provenance"] = stable_provenance
    return stable


def _extract_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return None
    plain = _plain_object(usage)
    return plain if isinstance(plain, dict) else None


def _find_refusal(response: Any) -> str | None:
    plain = _plain_object(response)

    def walk(value: Any) -> str | None:
        if isinstance(value, Mapping):
            if value.get("type") == "refusal" and value.get("refusal"):
                return str(value["refusal"])
            for child in value.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = walk(child)
                if found:
                    return found
        return None

    return walk(plain)


def _coerce_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, Mapping) or isinstance(data, BaseModel):
        return [_coerce_record(data)]
    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict(orient="records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [_coerce_record(item) for item in records]
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return [_coerce_record(item) for item in data]
    raise TypeError("candidates must be a mapping, DataFrame, or iterable of mappings")


def _coerce_record(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, BaseModel):
        candidate = candidate.model_dump(mode="python")
    elif not isinstance(candidate, Mapping) and hasattr(candidate, "to_dict"):
        candidate = candidate.to_dict()
    if not isinstance(candidate, Mapping):
        raise TypeError("each candidate must be mapping-like")
    return _json_safe(dict(candidate))


def _extract_ticker(record: Mapping[str, Any]) -> str:
    for key in ("ticker", "symbol", "Ticker", "Symbol"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().upper()
    raise ValueError("candidate is missing a non-empty ticker/symbol")


def _best_effort_ticker(record: Mapping[str, Any]) -> str:
    try:
        return _extract_ticker(record)
    except (TypeError, ValueError):
        return "UNKNOWN"


def _extract_company_name(record: Mapping[str, Any]) -> str | None:
    for key in (
        "company_name",
        "name",
        "long_name",
        "short_name",
        "longName",
        "shortName",
    ):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _json_safe(value: Any) -> Any:
    """Recursively normalize pandas/numpy/datetime values without inventing data."""

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

    # numpy scalars expose .item(); pandas NA/NaT may raise during bool conversion.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
            if item is not value:
                return _json_safe(item)
        except (TypeError, ValueError):
            pass
    type_name = type(value).__name__
    if type_name in {"NAType", "NaTType"}:
        return None
    try:
        if value != value:  # NaN-like scalar without importing numpy/pandas.
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value
    return str(value)


def _plain_object(value: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    if depth > 16:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)

    if isinstance(value, BaseModel):
        return _plain_object(value.model_dump(mode="python"), depth + 1, seen)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _plain_object(model_dump(), depth + 1, seen)
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(key): _plain_object(item, depth + 1, seen) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_object(item, depth + 1, seen) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _plain_object(item, depth + 1, seen)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            parsed.query,
            "",  # Fragments do not identify a distinct evidence document.
        )
    )


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "AnalysisMetadata",
    "ClaimEvidence",
    "DEFAULT_MODEL",
    "EvidenceSource",
    "MissingOpenAIKeyError",
    "NarrativeAnalyzer",
    "NarrativeAnalysisError",
    "NarrativeAssessment",
    "NarrativeResult",
    "analyze_candidates",
    "build_candidate_prompt",
    "configured_model",
    "narrative_results_to_dataframe",
]
