from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd

from narrative_dislocation.adapters import snapshot_to_quant_inputs
from narrative_dislocation.data.models import EquitySnapshot, FinancialStatements, SourceMetadata
from narrative_dislocation.quant import compute_quant_metrics


def _statement(rows):
    return pd.DataFrame(
        rows,
        index=pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"]),
    ).T


def _currency_snapshot(
    *,
    currency: str | None,
    financial_currency: str | None,
) -> EquitySnapshot:
    income = _statement(
        [
            {
                "Total Revenue": 90.0,
                "Gross Profit": 36.0,
                "Net Income": 7.0,
                "Diluted Average Shares": 10.0,
                "Diluted EPS": 0.7,
            },
            {
                "Total Revenue": 95.0,
                "Gross Profit": 38.0,
                "Net Income": 8.0,
                "Diluted Average Shares": 10.2,
                "Diluted EPS": 0.8,
            },
            {
                "Total Revenue": 100.0,
                "Gross Profit": 41.0,
                "Net Income": 9.0,
                "Diluted Average Shares": 10.3,
                "Diluted EPS": 0.9,
            },
        ]
    )
    cash_flow = _statement(
        [
            {"Free Cash Flow": 9.0},
            {"Free Cash Flow": 10.0},
            {"Free Cash Flow": 11.0},
        ]
    )
    balance = _statement(
        [
            {"Total Debt": 20.0, "Cash And Cash Equivalents": 5.0},
            {"Total Debt": 18.0, "Cash And Cash Equivalents": 6.0},
            {"Total Debt": 16.0, "Cash And Cash Equivalents": 7.0},
        ]
    )
    metadata = {
        "market_cap": 10_000_000_000,
        "enterprise_value": 11_000_000_000,
        "last_price": 25.0,
    }
    if currency is not None:
        metadata["currency"] = currency
    if financial_currency is not None:
        metadata["financial_currency"] = financial_currency
    return EquitySnapshot(
        symbol="XYZ",
        prices=pd.DataFrame(
            {"Close": [20.0, 22.0, 24.0]},
            index=pd.to_datetime(["2022-12-30", "2023-12-29", "2024-12-31"]),
        ),
        metadata=metadata,
        statements=FinancialStatements(
            annual_income=income,
            annual_balance_sheet=balance,
            annual_cash_flow=cash_flow,
        ),
        source=SourceMetadata(
            name="test",
            url="https://example.test",
            retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
        ),
    )


def test_snapshot_adapter_preserves_history_and_derives_fcf():
    income = _statement(
        [
            {
                "Total Revenue": 90.0,
                "Gross Profit": 36.0,
                "Diluted Average Shares": 10.0,
                "Diluted EPS": 2.0,
            },
            {
                "Total Revenue": 95.0,
                "Gross Profit": 38.0,
                "Diluted Average Shares": 10.2,
                "Diluted EPS": 2.2,
            },
            {
                "Total Revenue": 100.0,
                "Gross Profit": 41.0,
                "Diluted Average Shares": 10.3,
                "Diluted EPS": 2.5,
            },
        ]
    )
    cash_flow = _statement(
        [
            {"Operating Cash Flow": 12.0, "Capital Expenditure": -3.0},
            {"Operating Cash Flow": 13.0, "Capital Expenditure": -3.0},
            {"Operating Cash Flow": 15.0, "Capital Expenditure": -4.0},
        ]
    )
    balance = _statement(
        [
            {"Total Debt": 20.0, "Cash And Cash Equivalents": 5.0, "Invested Capital": 50.0},
            {"Total Debt": 18.0, "Cash And Cash Equivalents": 6.0, "Invested Capital": 52.0},
            {"Total Debt": 16.0, "Cash And Cash Equivalents": 7.0, "Invested Capital": 54.0},
        ]
    )
    prices = pd.DataFrame(
        {"Close": [20.0, 22.0, 24.0]},
        index=pd.to_datetime(["2022-12-30", "2023-12-29", "2024-12-31"]),
    )
    source = SourceMetadata(
        name="test",
        url="https://example.test",
        retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
    )
    snapshot = EquitySnapshot(
        symbol="XYZ",
        prices=prices,
        metadata={
            "market_cap": 10_000_000_000,
            "enterprise_value": 11_000_000_000,
            "currency": "USD",
            "financial_currency": "USD",
        },
        statements=FinancialStatements(
            annual_income=income,
            annual_balance_sheet=balance,
            annual_cash_flow=cash_flow,
        ),
        source=source,
    )

    result = snapshot_to_quant_inputs(snapshot)

    assert result.fundamentals["free_cash_flow"].tolist() == [9.0, 10.0, 11.0]
    assert result.fundamentals["market_cap"].isna().tolist() == [True, True, False]
    assert len(result.valuation_history) == 3
    assert result.provenance["fundamental_period_basis"].startswith("latest")


def test_snapshot_adapter_valuation_falls_back_from_empty_adjusted_close():
    snapshot = _currency_snapshot(currency="USD", financial_currency="USD")
    prices = snapshot.prices.assign(adj_close=float("nan"))[["adj_close", "Close"]]

    result = snapshot_to_quant_inputs(replace(snapshot, prices=prices))

    assert result.valuation_history["price"].tolist() == [20.0, 22.0, 24.0]


def test_snapshot_adapter_keeps_measured_screening_cap_when_statements_are_missing():
    source = SourceMetadata(
        name="test",
        url="https://example.test",
        retrieved_at=datetime(2025, 1, 10, tzinfo=UTC),
    )
    snapshot = EquitySnapshot(
        symbol="XYZ",
        prices=pd.DataFrame(
            {"close": [100.0, 60.0]},
            index=pd.to_datetime(["2024-01-01", "2025-01-01"]),
        ),
        metadata={
            "market_cap": 8_000_000_000,
            "trailing_pe": 15.0,
            "currency": "USD",
        },
        statements=FinancialStatements(),
        source=source,
    )

    result = snapshot_to_quant_inputs(snapshot)

    assert result.screening_market_cap == 8_000_000_000
    assert pd.isna(result.fundamentals.iloc[-1]["market_cap"])
    assert result.fundamentals.iloc[-1]["pe_ratio"] == 15.0
    assert result.fundamentals.iloc[-1]["period_type"] == "metadata_only"
    assert "No usable annual" in result.warnings[0]


def test_snapshot_adapter_withholds_cross_currency_valuation_inputs():
    result = snapshot_to_quant_inputs(_currency_snapshot(currency="usd", financial_currency="EUR"))

    latest = result.fundamentals.iloc[-1]
    metrics = compute_quant_metrics(
        result.fundamentals,
        valuation_history=result.valuation_history,
    )

    assert result.screening_market_cap == 10_000_000_000
    assert result.quote_currency == "USD"
    assert result.financial_currency == "EUR"
    assert result.ratio_currency_compatible is False
    assert latest["currency"] == "USD"
    assert latest["financial_currency"] == "EUR"
    assert latest["ratio_currency_status"] == "mismatch"
    assert pd.isna(latest["market_cap"])
    assert pd.isna(latest["enterprise_value"])
    assert pd.isna(latest["price"])
    assert result.valuation_history["ev_to_fcf"].isna().all()
    assert result.valuation_history["pe_ratio"].isna().all()
    assert pd.isna(metrics["ev_to_fcf"])
    assert pd.isna(metrics["fcf_yield"])
    assert pd.isna(metrics["pe_ratio"])
    assert result.provenance["quote_currency"] == "USD"
    assert result.provenance["financial_currency"] == "EUR"
    assert result.provenance["ratio_currency_status"] == "mismatch"
    assert any("no FX conversion" in warning for warning in result.warnings)


def test_snapshot_adapter_allows_ratios_when_known_currencies_match():
    result = snapshot_to_quant_inputs(_currency_snapshot(currency="usd", financial_currency="USD"))

    latest = result.fundamentals.iloc[-1]
    metrics = compute_quant_metrics(
        result.fundamentals,
        valuation_history=result.valuation_history,
    )

    assert result.ratio_currency_compatible is True
    assert latest["ratio_currency_status"] == "matched"
    assert latest["market_cap"] == 10_000_000_000
    assert latest["enterprise_value"] == 11_000_000_000
    assert latest["price"] == 25.0
    assert result.valuation_history["ev_to_fcf"].notna().all()
    assert result.valuation_history["pe_ratio"].notna().all()
    assert pd.notna(metrics["ev_to_fcf"])
    assert pd.notna(metrics["fcf_yield"])
    assert pd.notna(metrics["pe_ratio"])


def test_snapshot_adapter_withholds_ratios_when_currency_is_unknown():
    result = snapshot_to_quant_inputs(_currency_snapshot(currency="USD", financial_currency=None))

    latest = result.fundamentals.iloc[-1]
    metrics = compute_quant_metrics(
        result.fundamentals,
        valuation_history=result.valuation_history,
    )

    assert result.quote_currency == "USD"
    assert result.financial_currency is None
    assert result.ratio_currency_compatible is None
    assert latest["currency"] == "USD"
    assert pd.isna(latest["financial_currency"])
    assert latest["ratio_currency_status"] == "unknown"
    assert pd.isna(latest["market_cap"])
    assert pd.isna(latest["enterprise_value"])
    assert result.valuation_history["ev_to_fcf"].isna().all()
    assert result.valuation_history["pe_ratio"].isna().all()
    assert pd.isna(metrics["fcf_yield"])
    assert result.provenance["ratio_currency_status"] == "unknown"
    assert any("compatibility is unknown" in warning for warning in result.warnings)
