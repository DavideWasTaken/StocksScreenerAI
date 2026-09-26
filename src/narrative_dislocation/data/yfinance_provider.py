"""Resilient, cache-first Yahoo Finance adapter implemented with yfinance.

Yahoo Finance is a convenient free source, not an exchange-grade point-in-time
fundamentals database.  Every snapshot therefore names the source and retrieval
time, retains unavailable fields as ``None``/empty frames, and records component
errors instead of silently manufacturing values.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from .cache import DiskCache, utc_now
from .models import (
    BatchFetchResult,
    EquitySnapshot,
    FinancialStatements,
    PriceBatchResult,
    SourceMetadata,
    ensure_utc,
)
from .universe import normalize_symbol

YAHOO_SOURCE_NAME = "Yahoo Finance via yfinance"


class DataProviderError(RuntimeError):
    """Base error for provider failures."""


class DataUnavailableError(DataProviderError):
    """Raised when a symbol has no usable returned component."""


_INFO_FIELDS: Mapping[str, tuple[str, ...]] = {
    "company_name": ("longName", "shortName", "displayName"),
    "sector": ("sector",),
    "industry": ("industry",),
    "market_cap": ("marketCap",),
    "enterprise_value": ("enterpriseValue",),
    "shares_outstanding": ("sharesOutstanding",),
    "float_shares": ("floatShares",),
    "currency": ("currency",),
    "financial_currency": ("financialCurrency",),
    "exchange": ("exchange", "fullExchangeName"),
    "quote_type": ("quoteType",),
    "country": ("country",),
    "website": ("website",),
    "full_time_employees": ("fullTimeEmployees",),
    "fiscal_year_end": ("lastFiscalYearEnd",),
    "most_recent_quarter": ("mostRecentQuarter",),
    "trailing_pe": ("trailingPE",),
    "forward_pe": ("forwardPE",),
    "price_to_book": ("priceToBook",),
    "beta": ("beta",),
}

_FAST_INFO_FIELDS: Mapping[str, tuple[str, ...]] = {
    "market_cap": ("market_cap", "marketCap"),
    "shares_outstanding": ("shares", "shares_outstanding", "sharesOutstanding"),
    "currency": ("currency",),
    "exchange": ("exchange",),
    "last_price": ("last_price", "lastPrice"),
    "previous_close": ("previous_close", "previousClose"),
    "year_high": ("year_high", "yearHigh"),
    "year_low": ("year_low", "yearLow"),
}

_PRICE_COLUMN_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjclose": "adj_close",
    "volume": "volume",
    "dividends": "dividends",
    "stocksplits": "stock_splits",
    "capitalgains": "capital_gains",
}


def _not_missing(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    try:
        result = pd.isna(value)
        if not hasattr(result, "__len__"):
            return not bool(result)
    except (TypeError, ValueError, OverflowError):
        pass
    return True


def _clean_scalar(value: Any) -> Any:
    if not _not_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _first_present(
    mapping: Mapping[str, Any], candidates: Sequence[str]
) -> tuple[Any, str] | tuple[None, None]:
    for candidate in candidates:
        try:
            value = mapping.get(candidate)
        except (AttributeError, KeyError, TypeError):
            try:
                value = mapping[candidate]
            except (KeyError, TypeError, AttributeError):
                continue
        if _not_missing(value):
            return _clean_scalar(value), candidate
    return None, None


class YFinanceProvider:
    """Fetch prices, metadata, and annual/quarterly statements.

    The full ``EquitySnapshot`` and each price series have independent cache
    keys.  This keeps bulk runs resumable and lets benchmark-only workflows
    avoid downloading financial statements.
    """

    def __init__(
        self,
        cache: DiskCache | None = None,
        *,
        cache_dir: str | Path = ".cache/narrative_dislocation",
        cache_ttl: timedelta = timedelta(hours=18),
        price_cache_ttl: timedelta = timedelta(hours=6),
        retries: int = 3,
        backoff_seconds: float = 1.0,
        yf_module: Any | None = None,
        now_fn: Callable[[], datetime] = utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least one")
        self.cache = cache or DiskCache(
            cache_dir, default_ttl=cache_ttl, namespace="yfinance", now_fn=now_fn
        )
        self.cache_ttl = cache_ttl
        self.price_cache_ttl = price_cache_ttl
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self._yf_module = yf_module
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn

    @property
    def yf(self) -> Any:
        if self._yf_module is None:
            try:
                import yfinance as yf  # type: ignore[import-not-found]
            except ImportError as exc:
                raise DataProviderError(
                    "yfinance is required for market data; install project dependencies"
                ) from exc
            self._yf_module = yf
        return self._yf_module

    def _now(self) -> datetime:
        return ensure_utc(self._now_fn())

    def _retry_call(self, operation: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    self._sleep_fn(self.backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _range_key(
        lookback_years: int, start: str | date | datetime | None, end: str | date | datetime | None
    ) -> str:
        def render(value: Any) -> str:
            return value.isoformat() if hasattr(value, "isoformat") else str(value or "")

        return f"years={lookback_years}:start={render(start)}:end={render(end)}"

    def _snapshot_key(
        self,
        symbol: str,
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> str:
        return f"yf:snapshot:v3:{symbol}:{self._range_key(lookback_years, start, end)}"

    def _price_key(
        self,
        symbol: str,
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> str:
        return f"yf:prices:v3:{symbol}:{self._range_key(lookback_years, start, end)}"

    @staticmethod
    def _validate_history_request(
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> None:
        if lookback_years < 2 and start is None and end is None:
            raise ValueError("lookback_years must be at least 2 for screener inputs")

    @staticmethod
    def _history_kwargs(
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> dict[str, Any]:
        if start is None and end is None:
            return {"period": f"{lookback_years}y"}
        kwargs: dict[str, Any] = {}
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
        return kwargs

    def fetch_price_history(
        self,
        symbol: str,
        *,
        lookback_years: int = 3,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        force_refresh: bool = False,
        _ticker: Any | None = None,
    ) -> pd.DataFrame:
        """Fetch a lossless daily OHLCV/actions frame for one symbol."""

        self._validate_history_request(lookback_years, start, end)
        normalized = normalize_symbol(symbol)
        if not normalized:
            raise ValueError("symbol cannot be empty")
        cache_key = self._price_key(normalized, lookback_years, start, end)
        hit = self.cache.lookup(cache_key, ttl=self.price_cache_ttl, force_refresh=force_refresh)
        if hit.found:
            return hit.value.copy()

        try:
            ticker = _ticker or self._retry_call(lambda: self.yf.Ticker(normalized))
            raw = self._retry_call(lambda: self._ticker_history(ticker, lookback_years, start, end))
            frame = self._normalise_price_frame(raw, normalized)
            if not self._has_price_observations(frame):
                raise DataUnavailableError(f"no price history returned for {normalized}")
            self.cache.set(cache_key, frame, ttl=self.price_cache_ttl)
            return frame.copy()
        except Exception:
            stale = self.cache.lookup(cache_key, ttl=self.price_cache_ttl, allow_stale=True)
            if stale.found:
                return stale.value.copy()
            raise

    @staticmethod
    def _ticker_history(
        ticker: Any,
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> Any:
        kwargs = YFinanceProvider._history_kwargs(lookback_years, start, end)
        common = {"auto_adjust": False, "actions": True}
        # ``repair=True`` is intentionally omitted. Recent yfinance releases
        # route it through optional scikit-learn code, and it also transforms
        # rather than faithfully preserving the vendor observations.
        return ticker.history(**kwargs, **common)

    @staticmethod
    def _normalise_price_frame(raw: Any, symbol: str) -> pd.DataFrame:
        if raw is None:
            return pd.DataFrame()
        if isinstance(raw, pd.Series):
            raw = raw.to_frame()
        if not isinstance(raw, pd.DataFrame):
            raise DataProviderError(
                f"price response for {symbol} was {type(raw).__name__}, not a DataFrame"
            )
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            # A one-symbol frame occasionally arrives with the ticker as either level.
            for level in range(frame.columns.nlevels):
                labels = list(frame.columns.get_level_values(level).unique())
                match = next((label for label in labels if normalize_symbol(label) == symbol), None)
                if match is not None:
                    frame = frame.xs(match, axis=1, level=level, drop_level=True)
                    break
            if isinstance(frame.columns, pd.MultiIndex) and frame.columns.nlevels == 1:
                frame.columns = frame.columns.get_level_values(0)

        renamed: dict[Any, str] = {}
        for column in frame.columns:
            key = "".join(character for character in str(column).casefold() if character.isalnum())
            renamed[column] = _PRICE_COLUMN_NAMES.get(key, str(column))
        frame = frame.rename(columns=renamed)
        try:
            frame = frame.sort_index()
        except (TypeError, ValueError):
            pass
        frame.index.name = frame.index.name or "date"
        return frame

    @staticmethod
    def _has_price_observations(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        relevant = [column for column in ("close", "adj_close") if column in frame.columns]
        if relevant:
            return not frame[relevant].dropna(how="all").empty
        return not frame.dropna(how="all").empty

    def fetch_price_histories(
        self,
        symbols: Iterable[str],
        *,
        lookback_years: int = 3,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        batch_size: int = 50,
        force_refresh: bool = False,
    ) -> PriceBatchResult:
        """Fetch prices in yfinance download batches with per-symbol recovery."""

        self._validate_history_request(lookback_years, start, end)
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        ordered = tuple(
            dict.fromkeys(symbol for item in symbols if (symbol := normalize_symbol(item)))
        )
        prices: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}

        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset : offset + batch_size]
            missing: list[str] = []
            for symbol in batch:
                key = self._price_key(symbol, lookback_years, start, end)
                hit = self.cache.lookup(key, ttl=self.price_cache_ttl, force_refresh=force_refresh)
                if hit.found:
                    prices[symbol] = hit.value.copy()
                else:
                    missing.append(symbol)
            if not missing:
                continue

            split: Mapping[str, pd.DataFrame] = {}
            try:
                raw = self._retry_call(
                    lambda batch=tuple(missing): self._download_batch(
                        batch, lookback_years, start, end
                    )
                )
                split = self._split_download_frame(raw, missing)
            except Exception:
                # Individual calls below are both a compatibility fallback and a
                # way to identify exactly which symbol failed in a partial batch.
                split = {}

            for symbol in missing:
                frame = split.get(symbol)
                if frame is not None and self._has_price_observations(frame):
                    key = self._price_key(symbol, lookback_years, start, end)
                    self.cache.set(key, frame, ttl=self.price_cache_ttl)
                    prices[symbol] = frame.copy()
                    continue
                try:
                    prices[symbol] = self.fetch_price_history(
                        symbol,
                        lookback_years=lookback_years,
                        start=start,
                        end=end,
                        force_refresh=force_refresh,
                    )
                except Exception as exc:
                    errors[symbol] = f"{type(exc).__name__}: {exc}"

        retrieved_at = self._now()
        as_of_values = [
            value
            for frame in prices.values()
            if (value := self._latest_frame_timestamp(frame)) is not None
        ]
        metadata = SourceMetadata(
            name=YAHOO_SOURCE_NAME,
            url="https://finance.yahoo.com/",
            retrieved_at=retrieved_at,
            as_of=max(as_of_values) if as_of_values else retrieved_at,
            cached=False,
            details={
                "symbols_requested": len(ordered),
                "symbols_returned": len(prices),
                "batch_size": batch_size,
                "lookback_years": lookback_years,
            },
        )
        return PriceBatchResult(prices=prices, errors=errors, source=metadata)

    def _download_batch(
        self,
        symbols: Sequence[str],
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> Any:
        kwargs = self._history_kwargs(lookback_years, start, end)
        return self.yf.download(
            tickers=list(symbols),
            group_by="ticker",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=True,
            **kwargs,
        )

    @classmethod
    def _split_download_frame(cls, raw: Any, symbols: Sequence[str]) -> Mapping[str, pd.DataFrame]:
        if not isinstance(raw, pd.DataFrame):
            return {}
        if len(symbols) == 1 and not isinstance(raw.columns, pd.MultiIndex):
            return {symbols[0]: cls._normalise_price_frame(raw, symbols[0])}
        if not isinstance(raw.columns, pd.MultiIndex):
            return {}

        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            for level in range(raw.columns.nlevels):
                labels = list(raw.columns.get_level_values(level).unique())
                match = next((label for label in labels if normalize_symbol(label) == symbol), None)
                if match is None:
                    continue
                try:
                    selected = raw.xs(match, axis=1, level=level, drop_level=True)
                    result[symbol] = cls._normalise_price_frame(selected, symbol)
                except (KeyError, TypeError, ValueError):
                    pass
                break
        return result

    def fetch_symbol(
        self,
        symbol: str,
        *,
        lookback_years: int = 3,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        force_refresh: bool = False,
        _provided_prices: pd.DataFrame | None = None,
    ) -> EquitySnapshot:
        """Fetch all available data for one equity, preserving partial failures."""

        self._validate_history_request(lookback_years, start, end)
        normalized = normalize_symbol(symbol)
        if not normalized:
            raise ValueError("symbol cannot be empty")
        cache_key = self._snapshot_key(normalized, lookback_years, start, end)
        hit = self.cache.lookup(cache_key, ttl=self.cache_ttl, force_refresh=force_refresh)
        if hit.found:
            return self._mark_snapshot_cached(hit.value)

        try:
            snapshot = self._fetch_symbol_uncached(
                normalized,
                lookback_years=lookback_years,
                start=start,
                end=end,
                provided_prices=_provided_prices,
            )
            self.cache.set(cache_key, snapshot, ttl=self.cache_ttl)
            return snapshot
        except Exception:
            stale = self.cache.lookup(cache_key, ttl=self.cache_ttl, allow_stale=True)
            if stale.found:
                snapshot = self._mark_snapshot_cached(stale.value)
                return replace(
                    snapshot,
                    warnings=snapshot.warnings
                    + ("Live refresh failed; returned stale cached equity snapshot.",),
                )
            raise

    @staticmethod
    def _mark_snapshot_cached(snapshot: EquitySnapshot) -> EquitySnapshot:
        return replace(snapshot, source=replace(snapshot.source, cached=True))

    def _fetch_symbol_uncached(
        self,
        symbol: str,
        *,
        lookback_years: int,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
        provided_prices: pd.DataFrame | None,
    ) -> EquitySnapshot:
        ticker = self._retry_call(lambda: self.yf.Ticker(symbol))
        errors: list[str] = []
        warnings: list[str] = []

        if provided_prices is not None:
            prices = provided_prices.copy()
        else:
            try:
                prices = self.fetch_price_history(
                    symbol,
                    lookback_years=lookback_years,
                    start=start,
                    end=end,
                    _ticker=ticker,
                )
            except Exception as exc:
                errors.append(f"price_history: {type(exc).__name__}: {exc}")
                prices = pd.DataFrame()

        info = self._capture_mapping("metadata", lambda: self._read_info(ticker), errors)
        fast_info: Mapping[str, Any] = {}
        if not _not_missing(info.get("marketCap")) or not _not_missing(info.get("currency")):
            fast_info = self._capture_mapping(
                "fast_metadata", lambda: self._read_fast_info(ticker), errors
            )
        metadata = self._normalise_metadata(info, fast_info)

        annual_income = self._capture_frame(
            "annual_income",
            lambda: self._read_statement(
                ticker,
                getter="get_income_stmt",
                frequency="yearly",
                properties=("income_stmt", "financials"),
            ),
            errors,
        )
        quarterly_income = self._capture_frame(
            "quarterly_income",
            lambda: self._read_statement(
                ticker,
                getter="get_income_stmt",
                frequency="quarterly",
                properties=("quarterly_income_stmt", "quarterly_financials"),
            ),
            errors,
        )
        annual_balance = self._capture_frame(
            "annual_balance_sheet",
            lambda: self._read_statement(
                ticker,
                getter="get_balance_sheet",
                frequency="yearly",
                properties=("balance_sheet", "balancesheet"),
            ),
            errors,
        )
        quarterly_balance = self._capture_frame(
            "quarterly_balance_sheet",
            lambda: self._read_statement(
                ticker,
                getter="get_balance_sheet",
                frequency="quarterly",
                properties=("quarterly_balance_sheet", "quarterly_balancesheet"),
            ),
            errors,
        )
        annual_cash_flow = self._capture_frame(
            "annual_cash_flow",
            lambda: self._read_statement(
                ticker,
                getter="get_cash_flow",
                frequency="yearly",
                properties=("cash_flow", "cashflow"),
            ),
            errors,
        )
        quarterly_cash_flow = self._capture_frame(
            "quarterly_cash_flow",
            lambda: self._read_statement(
                ticker,
                getter="get_cash_flow",
                frequency="quarterly",
                properties=("quarterly_cash_flow", "quarterly_cashflow"),
            ),
            errors,
        )
        statements = FinancialStatements(
            annual_income=annual_income,
            quarterly_income=quarterly_income,
            annual_balance_sheet=annual_balance,
            quarterly_balance_sheet=quarterly_balance,
            annual_cash_flow=annual_cash_flow,
            quarterly_cash_flow=quarterly_cash_flow,
        )

        if not prices.empty:
            coverage_days = self._coverage_days(prices)
            if coverage_days is not None and start is None and coverage_days < 700:
                warnings.append(
                    f"Only {coverage_days} calendar days of price history were returned; "
                    "the requested screener baseline is at least two years."
                )
        if annual_income.empty:
            warnings.append("Annual income statement is unavailable/empty.")
        if annual_cash_flow.empty:
            warnings.append("Annual cash-flow statement is unavailable/empty.")

        has_metadata = any(
            _not_missing(value) for key, value in metadata.items() if key != "field_sources"
        )
        if prices.empty and not has_metadata and statements.empty:
            detail = "; ".join(errors) or "all returned components were empty"
            raise DataUnavailableError(f"no usable Yahoo Finance data for {symbol}: {detail}")

        retrieved_at = self._now()
        data_dates = [
            candidate
            for candidate in (
                self._latest_frame_timestamp(prices),
                self._latest_statement_timestamp(statements),
            )
            if candidate is not None
        ]
        source = SourceMetadata(
            name=YAHOO_SOURCE_NAME,
            url=f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}",
            retrieved_at=retrieved_at,
            as_of=max(data_dates) if data_dates else retrieved_at,
            cached=False,
            details={
                "lookback_years": lookback_years,
                "start": str(start) if start is not None else None,
                "end": str(end) if end is not None else None,
                "statement_orientation": "line_items_as_rows; reporting_dates_as_columns",
                "component_error_count": len(errors),
            },
        )
        return EquitySnapshot(
            symbol=symbol,
            prices=prices,
            metadata=metadata,
            statements=statements,
            source=source,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _capture_mapping(
        self,
        label: str,
        operation: Callable[[], Mapping[str, Any]],
        errors: list[str],
    ) -> Mapping[str, Any]:
        try:
            result = self._retry_call(operation)
            return result if isinstance(result, Mapping) else dict(result or {})
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return {}

    def _capture_frame(
        self,
        label: str,
        operation: Callable[[], pd.DataFrame],
        errors: list[str],
    ) -> pd.DataFrame:
        try:
            result = self._retry_call(operation)
            if result is None:
                return pd.DataFrame()
            if isinstance(result, pd.Series):
                return result.to_frame()
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"expected DataFrame, received {type(result).__name__}")
            return result.copy()
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _read_info(ticker: Any) -> Mapping[str, Any]:
        getter = getattr(ticker, "get_info", None)
        result = getter() if callable(getter) else ticker.info
        return result or {}

    @staticmethod
    def _read_fast_info(ticker: Any) -> Mapping[str, Any]:
        result = ticker.fast_info
        if isinstance(result, Mapping):
            return result
        try:
            return dict(result)
        except (TypeError, ValueError):
            fields: dict[str, Any] = {}
            for aliases in _FAST_INFO_FIELDS.values():
                for alias in aliases:
                    try:
                        fields[alias] = getattr(result, alias)
                    except (AttributeError, KeyError):
                        continue
            return fields

    @staticmethod
    def _normalise_metadata(
        info: Mapping[str, Any], fast_info: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        field_sources: dict[str, str] = {}
        for target, candidates in _INFO_FIELDS.items():
            value, raw_field = _first_present(info, candidates)
            result[target] = value
            if raw_field is not None:
                field_sources[target] = f"info.{raw_field}"
        for target, candidates in _FAST_INFO_FIELDS.items():
            if target in result and _not_missing(result[target]):
                continue
            value, raw_field = _first_present(fast_info, candidates)
            if target not in result or value is not None:
                result[target] = value
            if raw_field is not None:
                field_sources[target] = f"fast_info.{raw_field}"
        result["field_sources"] = field_sources
        return result

    @staticmethod
    def _read_statement(
        ticker: Any,
        *,
        getter: str,
        frequency: str,
        properties: Sequence[str],
    ) -> pd.DataFrame:
        method = getattr(ticker, getter, None)
        if callable(method):
            try:
                result = method(freq=frequency)
            except TypeError:
                result = method(frequency=frequency)
            if isinstance(result, pd.DataFrame) and not result.empty:
                return result
            if isinstance(result, pd.Series) and not result.empty:
                return result.to_frame()

        for property_name in properties:
            try:
                result = getattr(ticker, property_name)
            except (AttributeError, KeyError):
                continue
            if isinstance(result, pd.DataFrame) and not result.empty:
                return result
            if isinstance(result, pd.Series) and not result.empty:
                return result.to_frame()
        return pd.DataFrame()

    @staticmethod
    def _latest_frame_timestamp(frame: pd.DataFrame) -> datetime | None:
        if frame.empty:
            return None
        try:
            value = pd.Timestamp(frame.index.max())
        except (TypeError, ValueError):
            return None
        if pd.isna(value):
            return None
        parsed = value.to_pydatetime()
        return ensure_utc(parsed)

    @classmethod
    def _latest_statement_timestamp(cls, statements: FinancialStatements) -> datetime | None:
        candidates: list[datetime] = []
        for frame in (
            statements.annual_income,
            statements.quarterly_income,
            statements.annual_balance_sheet,
            statements.quarterly_balance_sheet,
            statements.annual_cash_flow,
            statements.quarterly_cash_flow,
        ):
            if frame.empty:
                continue
            try:
                values = pd.to_datetime(frame.columns, errors="coerce")
                latest = values.max()
            except (TypeError, ValueError):
                continue
            if pd.isna(latest):
                continue
            candidates.append(ensure_utc(pd.Timestamp(latest).to_pydatetime()))
        return max(candidates) if candidates else None

    @staticmethod
    def _coverage_days(frame: pd.DataFrame) -> int | None:
        if frame.empty:
            return None
        try:
            values = pd.to_datetime(frame.index, errors="coerce")
            start, end = values.min(), values.max()
        except (TypeError, ValueError):
            return None
        if pd.isna(start) or pd.isna(end):
            return None
        return int((end - start).days)

    def fetch_many(
        self,
        symbols: Iterable[str],
        *,
        lookback_years: int = 3,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        batch_size: int = 40,
        force_refresh: bool = False,
        stop_on_error: bool = False,
        on_progress: Callable[[int, int, str, str | None], None] | None = None,
    ) -> BatchFetchResult:
        """Fetch many symbols with batched prices and per-symbol checkpoints."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        ordered = tuple(
            dict.fromkeys(symbol for item in symbols if (symbol := normalize_symbol(item)))
        )
        started_at = self._now()
        price_result = self.fetch_price_histories(
            ordered,
            lookback_years=lookback_years,
            start=start,
            end=end,
            batch_size=batch_size,
            force_refresh=force_refresh,
        )
        snapshots: dict[str, EquitySnapshot] = {}
        errors: dict[str, str] = {}

        for completed, symbol in enumerate(ordered, start=1):
            error: str | None = None
            try:
                snapshots[symbol] = self.fetch_symbol(
                    symbol,
                    lookback_years=lookback_years,
                    start=start,
                    end=end,
                    force_refresh=force_refresh,
                    _provided_prices=price_result.prices.get(symbol),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors[symbol] = error
                if stop_on_error:
                    if on_progress:
                        on_progress(completed, len(ordered), symbol, error)
                    raise
            if on_progress:
                on_progress(completed, len(ordered), symbol, error)

        return BatchFetchResult(
            snapshots=snapshots,
            errors=errors,
            started_at=started_at,
            completed_at=self._now(),
        )

    def fetch_benchmark_prices(
        self,
        symbols: Iterable[str],
        *,
        lookback_years: int = 3,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        batch_size: int = 20,
        force_refresh: bool = False,
    ) -> PriceBatchResult:
        """Fetch arbitrary benchmark tickers through the same auditable cache."""

        return self.fetch_price_histories(
            symbols,
            lookback_years=lookback_years,
            start=start,
            end=end,
            batch_size=batch_size,
            force_refresh=force_refresh,
        )
