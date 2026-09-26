"""Typed boundary objects shared by universe and market-data providers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd


def _serialise_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _serialise_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialise_scalar(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    if hasattr(value, "item"):
        try:
            return _serialise_scalar(value.item())
        except (TypeError, ValueError):
            pass
    return value


def frame_to_split_dict(frame: pd.DataFrame) -> dict[str, Any]:
    """Convert a frame to JSON-friendly split form, retaining missing as null."""

    return {
        "index": [_serialise_scalar(value) for value in frame.index.tolist()],
        "columns": [_serialise_scalar(value) for value in frame.columns.tolist()],
        "data": [
            [_serialise_scalar(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


@dataclass(frozen=True)
class SourceMetadata:
    name: str
    url: str | None
    retrieved_at: datetime
    as_of: datetime | None = None
    cached: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "cached": self.cached,
            "details": _serialise_scalar(self.details),
        }


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    raw_symbols: tuple[str, ...]
    company_name: str | None
    sector: str | None
    industry: str | None
    memberships: tuple[str, ...]
    source_urls: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "raw_symbols": list(self.raw_symbols),
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "memberships": list(self.memberships),
            "source_urls": list(self.source_urls),
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    members: tuple[UniverseMember, ...]
    sources: tuple[SourceMetadata, ...]
    as_of: datetime
    warnings: tuple[str, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members)

    def by_symbol(self) -> dict[str, UniverseMember]:
        return {member.symbol: member for member in self.members}

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for member in self.members:
            row = member.to_dict()
            row["raw_symbols"] = "|".join(member.raw_symbols)
            row["memberships"] = "|".join(member.memberships)
            row["source_urls"] = "|".join(member.source_urls)
            rows.append(row)
        return pd.DataFrame.from_records(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "members": [member.to_dict() for member in self.members],
            "sources": [source.to_dict() for source in self.sources],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MarketCapFilterResult:
    universe: UniverseSnapshot
    included_market_caps: Mapping[str, float]
    excluded_below_minimum: tuple[str, ...]
    excluded_missing_market_cap: tuple[str, ...]


@dataclass(frozen=True)
class FinancialStatements:
    annual_income: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_income: pd.DataFrame = field(default_factory=pd.DataFrame)
    annual_balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    annual_cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return all(
            frame.empty
            for frame in (
                self.annual_income,
                self.quarterly_income,
                self.annual_balance_sheet,
                self.quarterly_balance_sheet,
                self.annual_cash_flow,
                self.quarterly_cash_flow,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annual_income": frame_to_split_dict(self.annual_income),
            "quarterly_income": frame_to_split_dict(self.quarterly_income),
            "annual_balance_sheet": frame_to_split_dict(self.annual_balance_sheet),
            "quarterly_balance_sheet": frame_to_split_dict(self.quarterly_balance_sheet),
            "annual_cash_flow": frame_to_split_dict(self.annual_cash_flow),
            "quarterly_cash_flow": frame_to_split_dict(self.quarterly_cash_flow),
        }


@dataclass(frozen=True)
class EquitySnapshot:
    symbol: str
    prices: pd.DataFrame
    metadata: Mapping[str, Any]
    statements: FinancialStatements
    source: SourceMetadata
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def market_cap(self) -> float | None:
        value = self.metadata.get("market_cap")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "symbol": self.symbol,
            "metadata": _serialise_scalar(self.metadata),
            "source": self.source.to_dict(),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
        if include_frames:
            result["prices"] = frame_to_split_dict(self.prices)
            result["statements"] = self.statements.to_dict()
        return result


@dataclass(frozen=True)
class BatchFetchResult:
    snapshots: Mapping[str, EquitySnapshot]
    errors: Mapping[str, str]
    started_at: datetime
    completed_at: datetime

    @property
    def succeeded(self) -> tuple[str, ...]:
        return tuple(self.snapshots)


@dataclass(frozen=True)
class PriceBatchResult:
    prices: Mapping[str, pd.DataFrame]
    errors: Mapping[str, str]
    source: SourceMetadata


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
