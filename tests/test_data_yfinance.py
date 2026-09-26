from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from narrative_dislocation.data.cache import DiskCache
from narrative_dislocation.data.yfinance_provider import YFinanceProvider

DATES = pd.to_datetime(["2023-01-03", "2024-01-03", "2026-01-03"], utc=True)


def price_frame(base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [base, base + 1, base + 2],
            "High": [base + 2, base + 3, base + 4],
            "Low": [base - 1, base, base + 1],
            "Close": [base + 1, base + 2, base + 3],
            "Volume": [1000, 1100, 1200],
        },
        index=DATES,
    )


class FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.fast_info = {"market_cap": 12_500_000_000, "currency": "USD"}

    def history(self, **kwargs: object) -> pd.DataFrame:
        assert "repair" not in kwargs
        return price_frame()

    def get_info(self) -> dict[str, object]:
        return {
            "longName": "Example Corp",
            "sector": None,
            "industry": "Software",
            "marketCap": np.nan,
            "sharesOutstanding": 100_000_000,
            "trailingPE": 11.5,
        }

    def get_income_stmt(self, *, freq: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                pd.Timestamp("2025-12-31"): [1_000.0, np.nan],
                pd.Timestamp("2024-12-31"): [900.0, 200.0],
            },
            index=["Total Revenue", "Net Income"],
        )

    def get_balance_sheet(self, *, freq: str) -> pd.DataFrame:
        del freq
        return pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [300.0, 100.0]},
            index=["Cash Cash Equivalents And Short Term Investments", "Total Debt"],
        )

    def get_cash_flow(self, *, freq: str) -> pd.DataFrame:
        del freq
        return pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [250.0, -50.0]},
            index=["Operating Cash Flow", "Capital Expenditure"],
        )


class FakeYFinance:
    def __init__(self) -> None:
        self.ticker_calls: list[str] = []
        self.download_calls: list[tuple[str, ...]] = []

    def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802 - mirrors yfinance API
        self.ticker_calls.append(symbol)
        return FakeTicker(symbol)

    def download(self, *, tickers: list[str], **kwargs: object) -> pd.DataFrame:
        del kwargs
        self.download_calls.append(tuple(tickers))
        data: dict[tuple[str, str], list[float]] = {}
        for offset, symbol in enumerate(tickers):
            frame = price_frame(100.0 + offset * 10)
            data[(symbol, "Close")] = frame["Close"].tolist()
            data[(symbol, "Volume")] = frame["Volume"].tolist()
        result = pd.DataFrame(data, index=DATES)
        result.columns = pd.MultiIndex.from_tuples(result.columns)
        return result


def make_provider(tmp_path, fake: FakeYFinance) -> YFinanceProvider:
    now = datetime(2026, 1, 4, tzinfo=UTC)
    cache = DiskCache(tmp_path / "cache", namespace="yf-test", now_fn=lambda: now)
    return YFinanceProvider(
        cache,
        yf_module=fake,
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
    )


def test_fetch_symbol_preserves_missing_values_and_source_metadata(tmp_path) -> None:
    fake = FakeYFinance()
    provider = make_provider(tmp_path, fake)

    snapshot = provider.fetch_symbol("exm")

    assert snapshot.symbol == "EXM"
    assert list(snapshot.prices.columns) == ["open", "high", "low", "close", "volume"]
    assert snapshot.metadata["market_cap"] == 12_500_000_000
    assert snapshot.metadata["field_sources"]["market_cap"] == "fast_info.market_cap"
    assert snapshot.metadata["sector"] is None
    assert pd.isna(snapshot.statements.annual_income.loc["Net Income", pd.Timestamp("2025-12-31")])
    assert snapshot.source.name == "Yahoo Finance via yfinance"
    assert snapshot.source.cached is False
    assert snapshot.source.as_of is not None
    serialized = snapshot.to_dict()
    assert serialized["statements"]["annual_income"]["data"][1][0] is None

    cached = provider.fetch_symbol("EXM")
    assert cached.source.cached is True
    assert fake.ticker_calls == ["EXM"]


def test_batched_prices_are_split_cached_and_reused_by_full_fetch(tmp_path) -> None:
    fake = FakeYFinance()
    provider = make_provider(tmp_path, fake)

    batch = provider.fetch_many(["aaa", "bbb", "AAA"], batch_size=2)

    assert tuple(batch.snapshots) == ("AAA", "BBB")
    assert batch.errors == {}
    assert fake.download_calls == [("AAA", "BBB")]
    assert fake.ticker_calls == ["AAA", "BBB"]
    assert batch.snapshots["AAA"].prices["close"].iloc[0] == 101.0
    assert batch.snapshots["BBB"].prices["close"].iloc[0] == 111.0

    again = provider.fetch_many(["AAA", "BBB"], batch_size=2)
    assert tuple(again.snapshots) == ("AAA", "BBB")
    assert fake.download_calls == [("AAA", "BBB")]
    assert fake.ticker_calls == ["AAA", "BBB"]
    assert all(snapshot.source.cached for snapshot in again.snapshots.values())


def test_benchmark_interface_returns_partial_errors_without_fabricating(tmp_path) -> None:
    class PartialFake(FakeYFinance):
        def download(self, *, tickers: list[str], **kwargs: object) -> pd.DataFrame:
            good = [symbol for symbol in tickers if symbol != "BAD"]
            return super().download(tickers=good, **kwargs)

        def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802
            if symbol == "BAD":
                raise RuntimeError("not found")
            return super().Ticker(symbol)

    fake = PartialFake()
    provider = make_provider(tmp_path, fake)
    result = provider.fetch_benchmark_prices(["SPY", "BAD"], batch_size=10)

    assert tuple(result.prices) == ("SPY",)
    assert "BAD" in result.errors
    assert result.source.details["symbols_requested"] == 2
