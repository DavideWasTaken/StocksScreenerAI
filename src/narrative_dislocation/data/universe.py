"""Current US large-cap universe construction.

S&P 500 and Nasdaq-100 constituents come from their public Wikipedia tables.
Russell 1000 constituents are intentionally opt-in via a user supplied CSV or
CSV URL: no stable, freely licensed official constituent feed is assumed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .cache import DiskCache, utc_now
from .models import (
    EquitySnapshot,
    MarketCapFilterResult,
    SourceMetadata,
    UniverseMember,
    UniverseSnapshot,
    ensure_utc,
)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia moved the live component table off the general index article in 2026.
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

SP500_MEMBERSHIP = "sp500"
NASDAQ100_MEMBERSHIP = "nasdaq100"
RUSSELL1000_MEMBERSHIP = "russell1000"
_MEMBERSHIP_ORDER = {
    SP500_MEMBERSHIP: 0,
    NASDAQ100_MEMBERSHIP: 1,
    RUSSELL1000_MEMBERSHIP: 2,
}


class UniverseLoadError(RuntimeError):
    """Raised when no requested constituent source can be loaded."""


def normalize_symbol(symbol: Any) -> str:
    """Normalize a constituent ticker to yfinance's symbol convention.

    In particular, share-class dots such as ``BRK.B`` become ``BRK-B``.  The
    unmodified source ticker remains available in ``UniverseMember.raw_symbols``.
    """

    if symbol is None:
        return ""
    value = str(symbol).strip().upper()
    if not value or value in {"NAN", "NONE", "NULL", "N/A", "NA", "-"}:
        return ""
    value = re.sub(r"\s+", "", value)
    return value.replace(".", "-")


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _canonical_column(value: Any) -> str:
    if isinstance(value, tuple):
        value = " ".join(str(part) for part in value if str(part) != "nan")
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Any | None:
    by_name = {_canonical_column(column): column for column in frame.columns}
    for alias in aliases:
        found = by_name.get(_canonical_column(alias))
        if found is not None:
            return found
    return None


class UniverseLoader:
    """Load, normalize and deduplicate current index constituents."""

    def __init__(
        self,
        cache: DiskCache | None = None,
        *,
        cache_dir: str | Path = ".cache/narrative_dislocation",
        cache_ttl: timedelta = timedelta(days=7),
        timeout_seconds: float = 30.0,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        session: requests.Session | Any | None = None,
        now_fn: Callable[[], datetime] = utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least one")
        self.cache = cache or DiskCache(
            cache_dir, default_ttl=cache_ttl, namespace="universe", now_fn=now_fn
        )
        self.cache_ttl = cache_ttl
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update(
                {
                    "User-Agent": (
                        "NarrativeDislocationScreener/0.1 "
                        "(public constituent data; contact: local-user)"
                    )
                }
            )
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn

    def _now(self) -> datetime:
        return ensure_utc(self._now_fn())

    def load(
        self,
        *,
        include_russell: bool = True,
        russell_source: str | Path | None = None,
        force_refresh: bool = False,
    ) -> UniverseSnapshot:
        """Return a deduplicated current constituent snapshot.

        ``russell_source`` should be a local CSV path or an HTTP(S) CSV URL.
        When omitted, ``RUSSELL_1000_SOURCE`` is consulted.  A missing optional
        Russell source produces an explicit warning rather than guessed members.
        """

        all_members: list[UniverseMember] = []
        sources: list[SourceMetadata] = []
        warnings: list[str] = []

        for display_name, loader in (
            ("S&P 500", self._load_sp500),
            ("Nasdaq-100", self._load_nasdaq100),
        ):
            try:
                members, metadata, warning = loader(force_refresh=force_refresh)
            except Exception as exc:
                warnings.append(f"{display_name} unavailable: {type(exc).__name__}: {exc}")
                continue
            all_members.extend(members)
            sources.append(metadata)
            if warning:
                warnings.append(warning)

        selected_russell_source = russell_source or os.getenv("RUSSELL_1000_SOURCE")
        if include_russell:
            if selected_russell_source:
                try:
                    members, metadata, warning = self._load_russell1000(
                        selected_russell_source, force_refresh=force_refresh
                    )
                except Exception as exc:
                    warnings.append(f"Russell 1000 unavailable: {type(exc).__name__}: {exc}")
                else:
                    all_members.extend(members)
                    sources.append(metadata)
                    if warning:
                        warnings.append(warning)
            else:
                warnings.append(
                    "Russell 1000 not loaded: provide russell_source or "
                    "RUSSELL_1000_SOURCE; no constituent list was inferred."
                )

        if not all_members:
            raise UniverseLoadError("no constituent source could be loaded")

        members = self._merge_members(all_members)
        as_of = max((source.as_of or source.retrieved_at) for source in sources)
        return UniverseSnapshot(
            members=members,
            sources=tuple(sources),
            as_of=as_of,
            warnings=tuple(warnings),
        )

    def _load_sp500(
        self, *, force_refresh: bool
    ) -> tuple[tuple[UniverseMember, ...], SourceMetadata, str | None]:
        return self._load_public_index(
            cache_key="universe:sp500:v2",
            source_name="Wikipedia S&P 500 constituents",
            url=SP500_URL,
            membership=SP500_MEMBERSHIP,
            symbol_aliases=("Symbol", "Ticker"),
            company_aliases=("Security", "Company", "Name"),
            sector_aliases=("GICS Sector", "Sector"),
            industry_aliases=("GICS Sub-Industry", "Industry", "Sub-Industry"),
            force_refresh=force_refresh,
        )

    def _load_nasdaq100(
        self, *, force_refresh: bool
    ) -> tuple[tuple[UniverseMember, ...], SourceMetadata, str | None]:
        return self._load_public_index(
            cache_key="universe:nasdaq100:v3",
            source_name="Wikipedia Nasdaq-100 constituents",
            url=NASDAQ100_URL,
            membership=NASDAQ100_MEMBERSHIP,
            symbol_aliases=("Ticker", "Symbol"),
            company_aliases=("Company", "Security", "Name"),
            sector_aliases=("GICS Sector", "ICB Industry", "Sector"),
            industry_aliases=(
                "GICS Sub-Industry",
                "ICB Subsector",
                "Industry",
                "Sub-Industry",
            ),
            force_refresh=force_refresh,
        )

    def _load_public_index(
        self,
        *,
        cache_key: str,
        source_name: str,
        url: str,
        membership: str,
        symbol_aliases: Sequence[str],
        company_aliases: Sequence[str],
        sector_aliases: Sequence[str],
        industry_aliases: Sequence[str],
        force_refresh: bool,
    ) -> tuple[tuple[UniverseMember, ...], SourceMetadata, str | None]:
        hit = self.cache.lookup(cache_key, ttl=self.cache_ttl, force_refresh=force_refresh)
        if hit.found:
            members, metadata = hit.value
            return members, replace(metadata, cached=True), None

        try:
            response = self._get(url)
            frame = self._select_constituent_table(response.text, symbol_aliases)
            members = self._members_from_frame(
                frame,
                membership=membership,
                source_url=url,
                symbol_aliases=symbol_aliases,
                company_aliases=company_aliases,
                sector_aliases=sector_aliases,
                industry_aliases=industry_aliases,
            )
            if not members:
                raise UniverseLoadError(f"{source_name} returned no valid symbols")
            retrieved_at = self._now()
            source_as_of = self._last_modified(response) or retrieved_at
            metadata = SourceMetadata(
                name=source_name,
                url=url,
                retrieved_at=retrieved_at,
                as_of=source_as_of,
                cached=False,
                details={"row_count": len(members), "membership": membership},
            )
            value = (members, metadata)
            self.cache.set(cache_key, value, ttl=self.cache_ttl)
            return members, metadata, None
        except Exception:
            stale = self.cache.lookup(cache_key, ttl=self.cache_ttl, allow_stale=True)
            if not stale.found:
                raise
            members, metadata = stale.value
            warning = f"{source_name}: live refresh failed; using stale cached constituents."
            return members, replace(metadata, cached=True), warning

    def _load_russell1000(
        self, source: str | Path, *, force_refresh: bool
    ) -> tuple[tuple[UniverseMember, ...], SourceMetadata, str | None]:
        source_text = str(source)
        is_remote = source_text.casefold().startswith(("http://", "https://"))
        source_url: str
        fingerprint = source_text
        if is_remote:
            source_url = source_text
        else:
            path = Path(source).expanduser().resolve()
            source_url = path.as_uri()
            if not path.is_file():
                raise FileNotFoundError(f"Russell CSV not found: {path}")
            stat = path.stat()
            fingerprint = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"

        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
        cache_key = f"universe:russell1000:v2:{digest}"
        hit = self.cache.lookup(cache_key, ttl=self.cache_ttl, force_refresh=force_refresh)
        if hit.found:
            members, metadata = hit.value
            return members, replace(metadata, cached=True), None

        try:
            if is_remote:
                response = self._get(source_text)
                frame = pd.read_csv(StringIO(response.text))
                source_as_of = self._last_modified(response) or self._now()
            else:
                path = Path(source).expanduser().resolve()
                frame = pd.read_csv(path)
                source_as_of = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

            members = self._members_from_frame(
                frame,
                membership=RUSSELL1000_MEMBERSHIP,
                source_url=source_url,
                symbol_aliases=("Symbol", "Ticker", "Ticker Symbol"),
                company_aliases=("Company", "Security", "Name", "Company Name"),
                sector_aliases=("GICS Sector", "Sector"),
                industry_aliases=("GICS Sub-Industry", "Industry", "Sub-Industry"),
            )
            if not members:
                raise UniverseLoadError("Russell CSV returned no valid ticker column/rows")
            retrieved_at = self._now()
            metadata = SourceMetadata(
                name="User-supplied Russell 1000 constituents",
                url=source_url,
                retrieved_at=retrieved_at,
                as_of=source_as_of,
                cached=False,
                details={
                    "row_count": len(members),
                    "membership": RUSSELL1000_MEMBERSHIP,
                    "user_supplied": True,
                },
            )
            self.cache.set(cache_key, (members, metadata), ttl=self.cache_ttl)
            return members, metadata, None
        except Exception:
            stale = self.cache.lookup(cache_key, ttl=self.cache_ttl, allow_stale=True)
            if not stale.found:
                raise
            members, metadata = stale.value
            warning = "Russell 1000 refresh failed; using stale user-supplied constituents."
            return members, replace(metadata, cached=True), warning

    def _get(self, url: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response
            except Exception as exc:  # requests and injected test sessions vary
                last_error = exc
                if attempt + 1 < self.retries:
                    self._sleep_fn(self.backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _last_modified(response: Any) -> datetime | None:
        value = getattr(response, "headers", {}).get("Last-Modified")
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return ensure_utc(parsed)

    @staticmethod
    def _select_constituent_table(html: str, symbol_aliases: Sequence[str]) -> pd.DataFrame:
        tables = pd.read_html(StringIO(html))
        for frame in tables:
            if _find_column(frame, symbol_aliases) is not None:
                return frame
        aliases = ", ".join(symbol_aliases)
        raise UniverseLoadError(f"no HTML table contained a symbol column ({aliases})")

    @staticmethod
    def _members_from_frame(
        frame: pd.DataFrame,
        *,
        membership: str,
        source_url: str,
        symbol_aliases: Sequence[str],
        company_aliases: Sequence[str],
        sector_aliases: Sequence[str],
        industry_aliases: Sequence[str],
    ) -> tuple[UniverseMember, ...]:
        symbol_column = _find_column(frame, symbol_aliases)
        if symbol_column is None:
            raise UniverseLoadError("constituent data has no recognized ticker column")
        company_column = _find_column(frame, company_aliases)
        sector_column = _find_column(frame, sector_aliases)
        industry_column = _find_column(frame, industry_aliases)

        members: list[UniverseMember] = []
        seen: set[str] = set()
        for _, row in frame.iterrows():
            raw_symbol = _clean_text(row[symbol_column])
            symbol = normalize_symbol(raw_symbol)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            members.append(
                UniverseMember(
                    symbol=symbol,
                    raw_symbols=(raw_symbol,) if raw_symbol else (),
                    company_name=(
                        _clean_text(row[company_column]) if company_column is not None else None
                    ),
                    sector=(_clean_text(row[sector_column]) if sector_column is not None else None),
                    industry=(
                        _clean_text(row[industry_column]) if industry_column is not None else None
                    ),
                    memberships=(membership,),
                    source_urls=(source_url,),
                )
            )
        return tuple(members)

    @staticmethod
    def _merge_members(members: Iterable[UniverseMember]) -> tuple[UniverseMember, ...]:
        merged: dict[str, UniverseMember] = {}
        for member in members:
            previous = merged.get(member.symbol)
            if previous is None:
                merged[member.symbol] = member
                continue
            memberships = tuple(
                sorted(
                    set(previous.memberships) | set(member.memberships),
                    key=lambda item: (_MEMBERSHIP_ORDER.get(item, 99), item),
                )
            )
            merged[member.symbol] = UniverseMember(
                symbol=member.symbol,
                raw_symbols=tuple(dict.fromkeys(previous.raw_symbols + member.raw_symbols)),
                company_name=previous.company_name or member.company_name,
                sector=previous.sector or member.sector,
                industry=previous.industry or member.industry,
                memberships=memberships,
                source_urls=tuple(dict.fromkeys(previous.source_urls + member.source_urls)),
            )
        return tuple(merged[symbol] for symbol in sorted(merged))


def filter_by_market_cap(
    universe: UniverseSnapshot,
    market_caps: Mapping[str, float | int | Mapping[str, Any] | EquitySnapshot | None],
    *,
    minimum: float = 5_000_000_000.0,
    keep_missing: bool = False,
) -> MarketCapFilterResult:
    """Apply the measured ``market cap > minimum`` rule without guessing gaps.

    Values may be plain numbers, provider metadata dictionaries, or
    ``EquitySnapshot`` objects.  Missing/non-finite observations are reported
    separately and excluded by default.
    """

    normalized_caps = {normalize_symbol(key): value for key, value in market_caps.items()}
    included: list[UniverseMember] = []
    included_caps: dict[str, float] = {}
    below: list[str] = []
    missing: list[str] = []

    for member in universe.members:
        raw_value = normalized_caps.get(member.symbol)
        if isinstance(raw_value, EquitySnapshot):
            raw_value = raw_value.market_cap
        elif isinstance(raw_value, Mapping):
            raw_value = raw_value.get("market_cap")

        cap: float | None
        if isinstance(raw_value, bool):
            cap = None
        else:
            try:
                candidate = float(raw_value)
                cap = candidate if math.isfinite(candidate) else None
            except (TypeError, ValueError):
                cap = None

        if cap is None:
            missing.append(member.symbol)
            if keep_missing:
                included.append(member)
            continue
        if cap > minimum:
            included.append(member)
            included_caps[member.symbol] = cap
        else:
            below.append(member.symbol)

    warnings = list(universe.warnings)
    if missing:
        action = "retained" if keep_missing else "excluded"
        warnings.append(
            f"Market-cap data missing for {len(missing)} constituents; they were {action}, "
            "not estimated."
        )
    filtered = UniverseSnapshot(
        members=tuple(included),
        sources=universe.sources,
        as_of=universe.as_of,
        warnings=tuple(warnings),
    )
    return MarketCapFilterResult(
        universe=filtered,
        included_market_caps=included_caps,
        excluded_below_minimum=tuple(below),
        excluded_missing_market_cap=tuple(missing),
    )
