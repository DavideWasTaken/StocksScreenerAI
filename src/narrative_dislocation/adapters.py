"""Adapters from provider-native snapshots to the stable quant contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from narrative_dislocation.data.models import EquitySnapshot


@dataclass(slots=True)
class QuantInputs:
    """Normalized inputs for one company's quant calculation."""

    fundamentals: pd.DataFrame
    valuation_history: pd.DataFrame
    prices: pd.DataFrame
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str]
    # Keep the measured quote-currency market cap outside ``fundamentals`` so
    # callers can apply the universe gate after quant ratios have been
    # calculated.  This matters when statement and quote currencies differ:
    # market cap remains valid for the gate, but not as an FCF/net-income
    # denominator without an explicit FX conversion.
    screening_market_cap: float | None = None
    quote_currency: str | None = None
    financial_currency: str | None = None
    # True means both known currencies match, False means both are known and
    # differ, and None means compatibility cannot be established.
    ratio_currency_compatible: bool | None = None


_INCOME_LINES: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income", "EBIT"),
    "pretax_income": ("Pretax Income", "Income Before Tax"),
    "income_tax_expense": ("Tax Provision", "Income Tax Expense"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "diluted_average_shares": ("Diluted Average Shares", "Basic Average Shares"),
    "diluted_eps": ("Diluted EPS", "Basic EPS"),
}

_CASH_FLOW_LINES: dict[str, tuple[str, ...]] = {
    "free_cash_flow": ("Free Cash Flow",),
    "operating_cash_flow": ("Operating Cash Flow", "Total Cash From Operating Activities"),
    "capital_expenditure": ("Capital Expenditure", "Capital Expenditures"),
}

_BALANCE_LINES: dict[str, tuple[str, ...]] = {
    "total_debt": ("Total Debt",),
    "cash_and_equivalents": (
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash",
    ),
    "invested_capital": ("Invested Capital", "Total Invested Capital"),
}


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _metadata_value(metadata: dict[str, Any], *aliases: str) -> float | str | None:
    by_key = {_key(name): value for name, value in metadata.items()}
    for alias in aliases:
        value = by_key.get(_key(alias))
        if isinstance(value, str) and value.strip():
            return value.strip()
        if _finite(value):
            return float(value)
    return None


def _currency_code(metadata: dict[str, Any], *aliases: str) -> str | None:
    value = _metadata_value(metadata, *aliases)
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


def _currency_compatibility(
    quote_currency: str | None,
    financial_currency: str | None,
) -> bool | None:
    if quote_currency is None or financial_currency is None:
        return None
    return quote_currency == financial_currency


def _statement_values(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    rows = {_key(label): label for label in frame.index}
    for alias in aliases:
        label = rows.get(_key(alias))
        if label is None:
            continue
        selected = frame.loc[label]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[0]
        values = pd.to_numeric(selected, errors="coerce")
        values.index = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None)
        return values[~values.index.isna()].sort_index()
    return pd.Series(dtype=float)


def _collect_statement_rows(
    snapshot: EquitySnapshot,
    *,
    quote_currency: str | None,
    financial_currency: str | None,
    ratio_currency_compatible: bool | None,
) -> pd.DataFrame:
    statements = snapshot.statements
    series_by_field: dict[str, pd.Series] = {}
    for field, aliases in _INCOME_LINES.items():
        series_by_field[field] = _statement_values(statements.annual_income, aliases)
    for field, aliases in _CASH_FLOW_LINES.items():
        series_by_field[field] = _statement_values(statements.annual_cash_flow, aliases)
    for field, aliases in _BALANCE_LINES.items():
        series_by_field[field] = _statement_values(statements.annual_balance_sheet, aliases)

    periods = sorted({period for values in series_by_field.values() for period in values.index})
    if not periods:
        known_at = snapshot.source.as_of or snapshot.source.retrieved_at
        rows: list[dict[str, Any]] = [
            {
                "period_end": known_at,
                "available_at": snapshot.source.retrieved_at,
                "period_type": "metadata_only",
            }
        ]
    else:
        rows = []

    for period in periods:
        row: dict[str, Any] = {
            "period_end": period,
            # Yahoo does not expose a dependable filing timestamp through this
            # provider. Treating the rows as known only at retrieval is
            # conservative and prevents their use in historical backtests.
            "available_at": snapshot.source.retrieved_at,
            "period_type": "annual",
        }
        for field, values in series_by_field.items():
            value = values.get(period, np.nan)
            row[field] = float(value) if _finite(value) else np.nan

        if not _finite(row.get("free_cash_flow")):
            operating = row.get("operating_cash_flow")
            capex = row.get("capital_expenditure")
            if _finite(operating) and _finite(capex):
                # Yahoo normally reports capex as a negative cash-flow line.
                row["free_cash_flow"] = float(operating) + (
                    float(capex) if float(capex) <= 0 else -float(capex)
                )
                row["free_cash_flow_is_derived"] = True
        rows.append(row)

    frame = pd.DataFrame.from_records(rows).sort_values("period_end").reset_index(drop=True)
    latest = frame.index[-1]
    metadata = dict(snapshot.metadata)
    frame.at[latest, "currency"] = quote_currency
    frame.at[latest, "financial_currency"] = financial_currency
    frame.at[latest, "ratio_currency_status"] = (
        "matched"
        if ratio_currency_compatible is True
        else "mismatch"
        if ratio_currency_compatible is False
        else "unknown"
    )
    current_values = {
        "market_cap": _metadata_value(metadata, "market_cap", "marketCap"),
        "enterprise_value": _metadata_value(metadata, "enterprise_value", "enterpriseValue"),
        "price": _metadata_value(
            metadata, "current_price", "currentPrice", "regular_market_price", "last_price"
        ),
        "shares_outstanding": _metadata_value(metadata, "shares_outstanding", "sharesOutstanding"),
        "diluted_eps": _metadata_value(metadata, "trailing_eps", "trailingEps"),
        "pe_ratio": _metadata_value(metadata, "trailing_pe", "trailingPE"),
    }
    for field, value in current_values.items():
        if value is not None:
            if ratio_currency_compatible is not True and field in {
                "market_cap",
                "enterprise_value",
                "price",
            }:
                # Quote-based monetary values cannot safely be combined with
                # statement FCF, net income, debt, cash, or EPS when their
                # known currencies differ or compatibility is unknown. No
                # implicit currency equivalence or FX conversion is assumed.
                frame.at[latest, field] = np.nan
                continue
            # Shares/EPS from metadata are trailing/current; use them only as a
            # fallback on the current row, never as fabricated history.
            if (
                field == "shares_outstanding"
                and "diluted_average_shares" in frame
                and _finite(frame.at[latest, "diluted_average_shares"])
            ):
                continue
            if field in {"shares_outstanding", "diluted_eps"} and _finite(
                frame.at[latest, field] if field in frame else np.nan
            ):
                continue
            frame.at[latest, field] = value
    return frame


def _price_series(prices: pd.DataFrame) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    frame = prices.copy()
    columns = {_key(column): column for column in frame.columns}
    column = None
    fallback = None
    for name in ("adjustedclose", "adjclose", "close", "price"):
        candidate = columns.get(name)
        if candidate is None:
            continue
        fallback = fallback or candidate
        values = pd.to_numeric(frame[candidate], errors="coerce")
        if (np.isfinite(values) & (values > 0)).any():
            column = candidate
            break
    column = column or fallback
    if column is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    index = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None)
    result = pd.Series(values.to_numpy(dtype=float), index=index)
    return result[
        ~result.index.isna() & result.notna() & np.isfinite(result) & (result > 0)
    ].sort_index()


def _valuation_history(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    ratio_currency_compatible: bool | None,
) -> pd.DataFrame:
    price_series = _price_series(prices)
    if fundamentals.empty or price_series.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in fundamentals.to_dict(orient="records"):
        period = pd.Timestamp(record["period_end"]).tz_localize(None)
        available_prices = price_series[price_series.index <= period]
        if available_prices.empty or period - available_prices.index[-1] > pd.Timedelta(days=14):
            continue
        price = float(available_prices.iloc[-1])
        shares = record.get("diluted_average_shares")
        fcf = record.get("free_cash_flow")
        debt = record.get("total_debt")
        cash = record.get("cash_and_equivalents")
        eps = record.get("diluted_eps")
        row: dict[str, Any] = {
            "period_end": period,
            "available_at": record.get("available_at"),
            "price": price,
            "valuation_history_basis": "fiscal-period-end price proxy",
            "ev_to_fcf": np.nan,
            "pe_ratio": np.nan,
        }
        # Only a confirmed match allows price * shares (quote currency) to be
        # combined with statement debt/cash/FCF or price with statement EPS.
        if _finite(shares) and ratio_currency_compatible is True:
            historical_market_cap = price * float(shares)
            if _finite(fcf) and float(fcf) > 0 and _finite(debt) and _finite(cash):
                enterprise_value = historical_market_cap + float(debt) - float(cash)
                if enterprise_value > 0:
                    row["ev_to_fcf"] = enterprise_value / float(fcf)
            if _finite(eps) and float(eps) > 0:
                row["pe_ratio"] = price / float(eps)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def snapshot_to_quant_inputs(snapshot: EquitySnapshot) -> QuantInputs:
    """Normalize a provider snapshot without imputing missing statements."""

    metadata = dict(snapshot.metadata)
    quote_currency = _currency_code(metadata, "currency", "quote_currency")
    financial_currency = _currency_code(
        metadata,
        "financial_currency",
        "financialCurrency",
        "statement_currency",
    )
    ratio_currency_compatible = _currency_compatibility(
        quote_currency,
        financial_currency,
    )
    screening_market_cap_value = _metadata_value(metadata, "market_cap", "marketCap")
    screening_market_cap = (
        float(screening_market_cap_value) if _finite(screening_market_cap_value) else None
    )
    fundamentals = _collect_statement_rows(
        snapshot,
        quote_currency=quote_currency,
        financial_currency=financial_currency,
        ratio_currency_compatible=ratio_currency_compatible,
    )
    has_annual_period = not fundamentals.empty and fundamentals["period_type"].eq("annual").any()
    warnings = list(snapshot.warnings)
    if not has_annual_period:
        warnings.append("No usable annual financial-statement periods were returned")
    elif (
        "free_cash_flow_is_derived" in fundamentals
        and fundamentals["free_cash_flow_is_derived"].fillna(False).astype(bool).any()
    ):
        warnings.append(
            "Free cash flow was derived as operating cash flow less capital expenditure "
            "for at least one annual period"
        )
    if snapshot.prices.empty:
        warnings.append("No usable price history was returned")
    if ratio_currency_compatible is False:
        warnings.append(
            f"Quote currency {quote_currency} differs from financial-statement currency "
            f"{financial_currency}; cross-currency valuation inputs and historical "
            "multiples were withheld because no FX conversion was applied"
        )
    elif ratio_currency_compatible is None:
        warnings.append(
            "Quote/financial-statement currency compatibility is unknown; quote-based "
            "valuation inputs and historical multiples were withheld"
        )

    provenance = {
        "financial_data_source": snapshot.source.name,
        "source_url": snapshot.source.url,
        "retrieved_at": snapshot.source.retrieved_at.isoformat(),
        "source_as_of": snapshot.source.as_of.isoformat() if snapshot.source.as_of else None,
        "source_cached": snapshot.source.cached,
        "source_details": dict(snapshot.source.details),
        "metadata_field_sources": dict(snapshot.metadata.get("field_sources", {})),
        "fundamental_period_basis": (
            "latest comparable annual statements"
            if has_annual_period
            else "metadata only; no usable annual statement periods"
        ),
        "free_cash_flow_definition": (
            "Yahoo reported FCF when available; otherwise operating cash flow less capex"
        ),
        "roic_note": "ROIC may be a labelled NOPAT / invested-capital proxy",
        "valuation_history_note": "Historical multiples use fiscal-period-end prices as a proxy",
        "quote_currency": quote_currency,
        "financial_currency": financial_currency,
        "ratio_currency_status": (
            "matched"
            if ratio_currency_compatible is True
            else "mismatch"
            if ratio_currency_compatible is False
            else "unknown"
        ),
        "ratio_currency_policy": (
            "Quote-based monetary inputs and historical multiples require a confirmed "
            "quote/financial currency match; no FX conversion or equivalence is inferred"
        ),
        "screening_market_cap_currency": quote_currency,
    }
    return QuantInputs(
        fundamentals=fundamentals,
        valuation_history=_valuation_history(
            fundamentals,
            snapshot.prices,
            ratio_currency_compatible=ratio_currency_compatible,
        ),
        prices=snapshot.prices,
        metadata=metadata,
        provenance=provenance,
        warnings=warnings + list(snapshot.errors),
        screening_market_cap=screening_market_cap,
        quote_currency=quote_currency,
        financial_currency=financial_currency,
        ratio_currency_compatible=ratio_currency_compatible,
    )


__all__ = ["QuantInputs", "snapshot_to_quant_inputs"]
