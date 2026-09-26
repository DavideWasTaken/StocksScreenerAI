"""Point-in-time-safe quantitative metrics for the dislocation screener.

The functions in this module deliberately use ``NaN`` for unavailable values.
There is no median filling (or other imputation) here: callers can decide how
much data coverage they require before ranking a company.

Ratios and growth rates are decimals (``0.10`` means 10%). Monetary values may
use any currency/unit as long as numerator and denominator use the same unit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NAN = float("nan")


@dataclass(frozen=True)
class QuantConfig:
    """Thresholds and lookbacks used by the first-pass quant screen."""

    minimum_market_cap: float = 5_000_000_000.0
    minimum_drawdown: float = 0.25
    drawdown_lookback_days: int = 730
    relative_return_lookback_days: int = 365
    gross_margin_periods: int = 5
    valuation_history_years: int = 5


_DATE_COLUMNS = (
    "period_end",
    "fiscal_period_end",
    "fiscal_date",
    "date",
    "as_of",
)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float:
    return float(value) if _finite(value) else NAN


def safe_divide(numerator: Any, denominator: Any, *, positive_denominator: bool = False) -> float:
    """Divide finite values, returning NaN for invalid or disallowed ratios."""

    if not (_finite(numerator) and _finite(denominator)):
        return NAN
    denominator = float(denominator)
    if denominator == 0 or (positive_denominator and denominator <= 0):
        return NAN
    return float(numerator) / denominator


def calculate_growth(current: Any, previous: Any) -> float:
    """Return signed growth using ``abs(previous)`` as the denominator.

    Using the absolute prior value makes an improvement from a negative base
    positive, but users should still inspect the underlying FCF when cash flow
    crosses zero. A zero or missing base is undefined rather than infinite.
    """

    if not (_finite(current) and _finite(previous)) or float(previous) == 0:
        return NAN
    return (float(current) - float(previous)) / abs(float(previous))


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lower_to_original = {str(col).lower(): str(col) for col in frame.columns}
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def _series(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    column = _find_column(frame, aliases)
    if column is None:
        return pd.Series(NAN, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _value(row: pd.Series, aliases: Sequence[str]) -> float:
    lower_to_original = {str(key).lower(): key for key in row.index}
    for alias in aliases:
        key = lower_to_original.get(alias.lower())
        if key is not None and _finite(row[key]):
            return float(row[key])
    return NAN


def _as_frame(data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        history = data.get("history")
        if isinstance(history, pd.DataFrame):
            frame = history.copy()
        elif isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
            frame = pd.DataFrame(history)
        else:
            return pd.DataFrame([dict(data)])

        # Flat values such as market cap can supplement every history row.
        for key, value in data.items():
            if (
                key != "history"
                and key not in frame
                and not isinstance(value, (Mapping, list, tuple, pd.Series))
            ):
                frame[key] = value
        return frame
    return pd.DataFrame(list(data))


def point_in_time_frame(
    data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter observations to information known by ``as_of`` and sort periods.

    When an ``available_at`` column exists it is the disclosure timestamp and
    is always enforced. Fiscal/observation dates later than ``as_of`` are also
    removed. Input order is retained when no period column exists.
    """

    frame = _as_frame(data)
    if frame.empty:
        return frame

    cutoff = pd.Timestamp(as_of) if as_of is not None else None
    available_col = _find_column(frame, ("available_at", "filing_date", "published_at"))
    if available_col is not None:
        available = pd.to_datetime(frame[available_col], errors="coerce", utc=True).dt.tz_localize(
            None
        )
        frame = frame.assign(__available_at=available)
        if cutoff is not None:
            cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
            frame = frame[
                frame["__available_at"].notna() & (frame["__available_at"] <= cutoff_naive)
            ]

    date_col = _find_column(frame, _DATE_COLUMNS)
    if date_col is not None:
        periods = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        frame = frame.assign(__period_end=periods)
        if cutoff is not None:
            cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
            frame = frame[frame["__period_end"].notna() & (frame["__period_end"] <= cutoff_naive)]
        frame = frame.sort_values(
            ["__period_end", "__available_at"] if "__available_at" in frame else ["__period_end"]
        )
    return frame.drop(columns=["__available_at", "__period_end"], errors="ignore")


def _usable_price_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    """Prefer the first named price field that has a finite positive observation."""

    fallback: str | None = None
    for alias in aliases:
        column = _find_column(frame, (alias,))
        if column is None:
            continue
        fallback = fallback or column
        values = pd.to_numeric(frame[column], errors="coerce")
        if (np.isfinite(values) & (values > 0)).any():
            return column
    return fallback


def _price_series(data: pd.Series | pd.DataFrame | Mapping[Any, Any]) -> pd.Series:
    if isinstance(data, pd.Series):
        result = pd.to_numeric(data, errors="coerce")
    elif isinstance(data, Mapping):
        result = pd.to_numeric(pd.Series(data), errors="coerce")
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        date_col = _find_column(frame, ("date", "timestamp", "as_of"))
        if date_col is not None:
            frame.index = pd.to_datetime(frame.pop(date_col), errors="coerce")
        price_col = _usable_price_column(
            frame, ("adjusted_close", "adj_close", "adj close", "close", "price", "value")
        )
        if price_col is None:
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.shape[1] != 1:
                raise ValueError(
                    "Price DataFrame needs an adjusted_close/close column or exactly one numeric column"
                )
            price_col = str(numeric.columns[0])
        result = pd.to_numeric(frame[price_col], errors="coerce")
    else:
        raise TypeError("Prices must be a pandas Series/DataFrame or a date-to-price mapping")

    index = pd.to_datetime(result.index, errors="coerce", utc=True).tz_localize(None)
    result = pd.Series(result.to_numpy(dtype=float), index=index, name="price")
    result = result[~result.index.isna() & result.notna() & np.isfinite(result) & (result > 0)]
    return result.groupby(level=0).last().sort_index()


def calculate_drawdown(
    prices: pd.Series | pd.DataFrame | Mapping[Any, Any],
    *,
    as_of: str | pd.Timestamp | None = None,
    lookback_days: int = 730,
) -> float:
    """Positive decline from the highest adjusted close in the lookback."""

    series = _price_series(prices)
    if series.empty:
        return NAN
    cutoff = (
        pd.Timestamp(as_of).tz_localize(None)
        if as_of is not None and pd.Timestamp(as_of).tzinfo
        else pd.Timestamp(as_of)
        if as_of is not None
        else series.index.max()
    )
    series = series[series.index <= cutoff]
    if series.empty:
        return NAN
    series = series[series.index >= cutoff - pd.Timedelta(days=lookback_days)]
    if series.empty:
        return NAN
    high = series.max()
    return safe_divide(high - series.iloc[-1], high, positive_denominator=True)


def calculate_total_return(
    prices: pd.Series | pd.DataFrame | Mapping[Any, Any],
    *,
    as_of: str | pd.Timestamp | None = None,
    lookback_days: int = 365,
) -> float:
    series = _price_series(prices)
    if series.empty:
        return NAN
    cutoff = (
        pd.Timestamp(as_of).tz_localize(None)
        if as_of is not None and pd.Timestamp(as_of).tzinfo
        else pd.Timestamp(as_of)
        if as_of is not None
        else series.index.max()
    )
    series = series[
        (series.index <= cutoff) & (series.index >= cutoff - pd.Timedelta(days=lookback_days))
    ]
    if len(series) < 2:
        return NAN
    return safe_divide(series.iloc[-1], series.iloc[0], positive_denominator=True) - 1.0


def calculate_relative_return(
    stock_prices: pd.Series | pd.DataFrame | Mapping[Any, Any],
    benchmark_prices: pd.Series | pd.DataFrame | Mapping[Any, Any],
    *,
    as_of: str | pd.Timestamp | None = None,
    lookback_days: int = 365,
) -> float:
    """Stock total return minus benchmark return over common observations."""

    stock = _price_series(stock_prices).rename("stock")
    benchmark = _price_series(benchmark_prices).rename("benchmark")
    aligned = pd.concat([stock, benchmark], axis=1, sort=False).sort_index().ffill().dropna()
    if aligned.empty:
        return NAN
    cutoff = (
        pd.Timestamp(as_of).tz_localize(None)
        if as_of is not None and pd.Timestamp(as_of).tzinfo
        else pd.Timestamp(as_of)
        if as_of is not None
        else aligned.index.max()
    )
    aligned = aligned[
        (aligned.index <= cutoff) & (aligned.index >= cutoff - pd.Timedelta(days=lookback_days))
    ]
    if len(aligned) < 2:
        return NAN
    stock_return = aligned["stock"].iloc[-1] / aligned["stock"].iloc[0] - 1.0
    benchmark_return = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1.0
    return float(stock_return - benchmark_return)


def _direct_metric(data: Mapping[str, Any] | pd.DataFrame, aliases: Sequence[str]) -> float:
    if not isinstance(data, Mapping):
        return NAN
    lower = {str(key).lower(): value for key, value in data.items()}
    for alias in aliases:
        if alias.lower() in lower and _finite(lower[alias.lower()]):
            return float(lower[alias.lower()])
    return NAN


def _current_and_previous(series: pd.Series) -> tuple[float, float]:
    """Return values from the latest two rows without skipping missing periods."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.empty:
        return NAN, NAN
    current = _number(numeric.iloc[-1])
    previous = _number(numeric.iloc[-2]) if len(numeric) > 1 else NAN
    return current, previous


def calculate_fundamental_metrics(
    fundamentals: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    as_of: str | pd.Timestamp | None = None,
    gross_margin_periods: int = 5,
) -> dict[str, Any]:
    """Calculate latest fundamental quality, growth, leverage and valuation.

    The rows should represent comparable reporting periods (normally annual or
    trailing-twelve-month observations). ``available_at`` is honored when
    present. Growth is latest-versus-previous comparable row.
    """

    frame = point_in_time_frame(fundamentals, as_of)
    result: dict[str, Any] = {
        "revenue_growth": NAN,
        "fcf_growth": NAN,
        "fcf_per_share_growth": NAN,
        "fcf_margin": NAN,
        "gross_margin": NAN,
        "gross_margin_stability": NAN,
        "gross_margin_change": NAN,
        "share_dilution": NAN,
        "net_debt": NAN,
        "net_debt_to_fcf": NAN,
        "roic": NAN,
        "roic_proxy": NAN,
        "ev_to_fcf": NAN,
        "fcf_yield": NAN,
        "pe_ratio": NAN,
        "market_cap": NAN,
        "enterprise_value": NAN,
        "revenue": NAN,
        "free_cash_flow": NAN,
        "shares_outstanding": NAN,
        "fundamental_observations": int(len(frame)),
        "latest_fundamental_period_end": None,
        "prior_fundamental_period_end": None,
        "latest_fundamental_available_at": None,
        "latest_fundamental_age_days": NAN,
        "fundamental_period_type": None,
    }
    if frame.empty:
        return result

    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else pd.Series(dtype=float)

    period_column = _find_column(frame, _DATE_COLUMNS)
    if period_column is not None:
        latest_period = pd.to_datetime(latest.get(period_column), errors="coerce", utc=True)
        prior_period = (
            pd.to_datetime(previous.get(period_column), errors="coerce", utc=True)
            if not previous.empty
            else pd.NaT
        )
        if not pd.isna(latest_period):
            result["latest_fundamental_period_end"] = latest_period.isoformat()
            if as_of is not None:
                cutoff = pd.Timestamp(as_of)
                cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
                result["latest_fundamental_age_days"] = max(
                    0, int((cutoff - latest_period).total_seconds() // 86_400)
                )
        if not pd.isna(prior_period):
            result["prior_fundamental_period_end"] = prior_period.isoformat()
    available_column = _find_column(frame, ("available_at", "filing_date", "published_at"))
    if available_column is not None:
        available = pd.to_datetime(latest.get(available_column), errors="coerce", utc=True)
        if not pd.isna(available):
            result["latest_fundamental_available_at"] = available.isoformat()
    period_type_column = _find_column(frame, ("period_type",))
    if period_type_column is not None and pd.notna(latest.get(period_type_column)):
        result["fundamental_period_type"] = str(latest.get(period_type_column))

    revenue = _series(frame, ("revenue", "total_revenue", "sales"))
    fcf = _series(frame, ("free_cash_flow", "fcf"))
    shares = _series(frame, ("shares_outstanding", "diluted_average_shares", "diluted_shares"))
    gross_margin = _series(frame, ("gross_margin",))
    gross_profit = _series(frame, ("gross_profit",))
    calculated_margin = gross_profit / revenue.replace(0, np.nan)
    gross_margin = gross_margin.where(gross_margin.notna(), calculated_margin)

    current_revenue, prior_revenue = _current_and_previous(revenue)
    current_fcf, prior_fcf = _current_and_previous(fcf)
    current_shares, prior_shares = _current_and_previous(shares)
    current_margin, prior_margin = _current_and_previous(gross_margin)

    direct = fundamentals if isinstance(fundamentals, Mapping) else {}
    result["revenue_growth"] = _direct_metric(direct, ("revenue_growth",))
    if not _finite(result["revenue_growth"]):
        result["revenue_growth"] = calculate_growth(current_revenue, prior_revenue)
    result["fcf_growth"] = _direct_metric(direct, ("fcf_growth", "free_cash_flow_growth"))
    if not _finite(result["fcf_growth"]):
        result["fcf_growth"] = calculate_growth(current_fcf, prior_fcf)

    current_fcf_per_share = safe_divide(current_fcf, current_shares, positive_denominator=True)
    prior_fcf_per_share = safe_divide(prior_fcf, prior_shares, positive_denominator=True)
    result["fcf_per_share_growth"] = _direct_metric(
        direct, ("fcf_per_share_growth", "fcf_share_growth")
    )
    if not _finite(result["fcf_per_share_growth"]):
        result["fcf_per_share_growth"] = calculate_growth(
            current_fcf_per_share, prior_fcf_per_share
        )
    result["fcf_margin"] = _direct_metric(direct, ("fcf_margin",))
    if not _finite(result["fcf_margin"]):
        result["fcf_margin"] = safe_divide(current_fcf, current_revenue, positive_denominator=True)

    result["gross_margin"] = current_margin
    stable_values = gross_margin.dropna().tail(gross_margin_periods)
    if len(stable_values) >= 2:
        result["gross_margin_stability"] = float(stable_values.std(ddof=0))
    result["gross_margin_change"] = (
        current_margin - prior_margin if _finite(current_margin) and _finite(prior_margin) else NAN
    )
    result["share_dilution"] = _direct_metric(direct, ("share_dilution", "shares_growth"))
    if not _finite(result["share_dilution"]):
        result["share_dilution"] = calculate_growth(current_shares, prior_shares)

    debt = _value(latest, ("total_debt", "debt", "short_and_long_term_debt"))
    cash = _value(latest, ("cash_and_equivalents", "cash", "cash_and_short_term_investments"))
    if _finite(debt) and _finite(cash):
        result["net_debt"] = debt - cash
        if current_fcf > 0:
            result["net_debt_to_fcf"] = (debt - cash) / current_fcf

    direct_roic = _value(latest, ("roic", "return_on_invested_capital"))
    if _finite(direct_roic):
        result["roic"] = direct_roic
    else:
        operating_income = _value(latest, ("operating_income", "ebit"))
        tax_rate = _value(latest, ("effective_tax_rate", "tax_rate"))
        if not _finite(tax_rate):
            tax_expense = _value(latest, ("income_tax_expense", "tax_expense"))
            pretax_income = _value(latest, ("pretax_income", "income_before_tax"))
            tax_rate = safe_divide(tax_expense, pretax_income, positive_denominator=True)
        invested_now = _value(latest, ("invested_capital", "total_invested_capital"))
        invested_prior = (
            _value(previous, ("invested_capital", "total_invested_capital"))
            if not previous.empty
            else NAN
        )
        average_invested = (
            (invested_now + invested_prior) / 2.0
            if _finite(invested_now) and _finite(invested_prior)
            else invested_now
        )
        if (
            _finite(operating_income)
            and _finite(tax_rate)
            and 0 <= tax_rate <= 1
            and _finite(average_invested)
        ):
            nopat = operating_income * (1.0 - tax_rate)
            result["roic_proxy"] = safe_divide(nopat, average_invested, positive_denominator=True)
            result["roic_proxy_label"] = (
                "NOPAT / average invested capital (derived proxy, not reported ROIC)"
            )

    market_cap = _value(latest, ("market_cap", "market_capitalization"))
    enterprise_value = _value(latest, ("enterprise_value", "ev"))
    price = _value(latest, ("price", "share_price", "close", "adjusted_close"))
    eps = _value(latest, ("diluted_eps", "eps"))
    net_income = _value(latest, ("net_income", "net_income_common"))

    result.update(
        {
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "revenue": current_revenue,
            "free_cash_flow": current_fcf,
            "shares_outstanding": current_shares,
        }
    )
    if current_fcf > 0:
        if enterprise_value > 0:
            result["ev_to_fcf"] = safe_divide(
                enterprise_value, current_fcf, positive_denominator=True
            )
        result["fcf_yield"] = safe_divide(current_fcf, market_cap, positive_denominator=True)
    reported_pe = _value(latest, ("pe_ratio", "pe", "trailing_pe"))
    if reported_pe > 0:
        result["pe_ratio"] = reported_pe
    elif eps > 0:
        result["pe_ratio"] = safe_divide(price, eps, positive_denominator=True)
    elif net_income > 0:
        result["pe_ratio"] = safe_divide(market_cap, net_income, positive_denominator=True)
    return result


def calculate_valuation_compression(
    current_metrics: Mapping[str, Any],
    valuation_history: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    as_of: str | pd.Timestamp | None = None,
    history_years: int = 5,
    minimum_observations: int = 3,
) -> dict[str, float]:
    """Compare positive current multiples with their own historical medians."""

    history = point_in_time_frame(valuation_history, as_of)
    if as_of is not None and not history.empty:
        date_col = _find_column(_as_frame(valuation_history), _DATE_COLUMNS)
        if date_col is not None:
            raw = point_in_time_frame(valuation_history, as_of)
            dates = pd.to_datetime(raw[date_col], errors="coerce")
            cutoff = (
                pd.Timestamp(as_of).tz_localize(None)
                if pd.Timestamp(as_of).tzinfo
                else pd.Timestamp(as_of)
            )
            history = raw[dates >= cutoff - pd.DateOffset(years=history_years)]

    result = {
        "historical_ev_to_fcf_median": NAN,
        "historical_pe_median": NAN,
        "ev_to_fcf_compression": NAN,
        "pe_compression": NAN,
        "valuation_compression": NAN,
    }
    pairs = (
        (
            "ev_to_fcf",
            ("ev_to_fcf", "ev_fcf"),
            "historical_ev_to_fcf_median",
            "ev_to_fcf_compression",
        ),
        (
            "pe_ratio",
            ("pe_ratio", "pe", "price_to_earnings"),
            "historical_pe_median",
            "pe_compression",
        ),
    )
    compression_values: list[float] = []
    for current_key, aliases, median_key, compression_key in pairs:
        values = _series(history, aliases)
        values = values[(values > 0) & np.isfinite(values)]
        current = _number(current_metrics.get(current_key))
        if len(values) < minimum_observations or not (current > 0):
            continue
        historical_median = float(values.median())
        compression = (historical_median - current) / historical_median
        result[median_key] = historical_median
        result[compression_key] = compression
        compression_values.append(compression)
    if compression_values:
        result["valuation_compression"] = float(np.mean(compression_values))
    return result


def compute_quant_metrics(
    fundamentals: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    prices: pd.Series | pd.DataFrame | Mapping[Any, Any] | None = None,
    valuation_history: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    sector_prices: pd.Series | pd.DataFrame | Mapping[Any, Any] | None = None,
    index_prices: pd.Series | pd.DataFrame | Mapping[Any, Any] | None = None,
    as_of: str | pd.Timestamp | None = None,
    config: QuantConfig | None = None,
) -> dict[str, Any]:
    """Compute the full per-company metric dictionary used by scoring."""

    cfg = config or QuantConfig()
    metrics = calculate_fundamental_metrics(
        fundamentals,
        as_of=as_of,
        gross_margin_periods=cfg.gross_margin_periods,
    )
    metrics.update(
        {
            "drawdown_2y": NAN,
            "relative_return_sector_1y": NAN,
            "relative_return_index_1y": NAN,
        }
    )
    if prices is not None:
        metrics["drawdown_2y"] = calculate_drawdown(
            prices,
            as_of=as_of,
            lookback_days=cfg.drawdown_lookback_days,
        )
        if sector_prices is not None:
            metrics["relative_return_sector_1y"] = calculate_relative_return(
                prices,
                sector_prices,
                as_of=as_of,
                lookback_days=cfg.relative_return_lookback_days,
            )
        if index_prices is not None:
            metrics["relative_return_index_1y"] = calculate_relative_return(
                prices,
                index_prices,
                as_of=as_of,
                lookback_days=cfg.relative_return_lookback_days,
            )
    if valuation_history is not None:
        metrics.update(
            calculate_valuation_compression(
                metrics,
                valuation_history,
                as_of=as_of,
                history_years=cfg.valuation_history_years,
            )
        )
    return metrics


def evaluate_hard_screen(
    metrics: Mapping[str, Any],
    *,
    minimum_market_cap: float = 5_000_000_000.0,
    minimum_drawdown: float = 0.25,
) -> tuple[bool, list[str]]:
    """Apply strict ``>`` market-cap and drawdown gates with explicit reasons."""

    reasons: list[str] = []
    market_cap = _number(metrics.get("market_cap"))
    drawdown = _number(metrics.get("drawdown_2y"))
    if not _finite(market_cap):
        reasons.append("market_cap_missing")
    elif market_cap <= minimum_market_cap:
        reasons.append("market_cap_not_above_threshold")
    if not _finite(drawdown):
        reasons.append("drawdown_2y_missing")
    elif drawdown <= minimum_drawdown:
        reasons.append("drawdown_not_above_threshold")
    return not reasons, reasons


def annotate_hard_screen(
    candidates: pd.DataFrame,
    *,
    minimum_market_cap: float = 5_000_000_000.0,
    minimum_drawdown: float = 0.25,
) -> pd.DataFrame:
    """Add deterministic eligibility and semicolon-separated failure reasons."""

    output = candidates.copy()
    evaluations = [
        evaluate_hard_screen(
            row,
            minimum_market_cap=minimum_market_cap,
            minimum_drawdown=minimum_drawdown,
        )
        for row in output.to_dict(orient="records")
    ]
    output["screen_pass"] = [passed for passed, _ in evaluations]
    output["screen_reasons"] = [";".join(reasons) for _, reasons in evaluations]
    return output


def hard_screen(
    candidates: pd.DataFrame,
    *,
    minimum_market_cap: float = 5_000_000_000.0,
    minimum_drawdown: float = 0.25,
) -> pd.DataFrame:
    """Return only candidates that pass both non-negotiable quant gates."""

    annotated = annotate_hard_screen(
        candidates,
        minimum_market_cap=minimum_market_cap,
        minimum_drawdown=minimum_drawdown,
    )
    return annotated.loc[annotated["screen_pass"]].reset_index(drop=True)


def build_quant_snapshot(
    universe: pd.DataFrame,
    fundamentals_by_ticker: Mapping[str, pd.DataFrame | Mapping[str, Any]],
    prices_by_ticker: Mapping[str, pd.Series | pd.DataFrame | Mapping[Any, Any]],
    *,
    as_of: str | pd.Timestamp,
    valuations_by_ticker: Mapping[str, pd.DataFrame] | None = None,
    sector_prices_by_ticker: Mapping[str, pd.Series | pd.DataFrame] | None = None,
    index_prices: pd.Series | pd.DataFrame | Mapping[Any, Any] | None = None,
    ticker_column: str = "ticker",
    config: QuantConfig | None = None,
) -> pd.DataFrame:
    """Convenience batch adapter for data-layer dictionaries."""

    if ticker_column not in universe:
        raise ValueError(f"Universe is missing {ticker_column!r}")
    cfg = config or QuantConfig()
    rows: list[dict[str, Any]] = []
    universe_lookup = universe.set_index(ticker_column, drop=False)
    for ticker in universe[ticker_column].astype(str):
        source = fundamentals_by_ticker.get(ticker, {})
        if isinstance(source, Mapping):
            merged_source: Any = dict(source)
            for key, value in universe_lookup.loc[ticker].to_dict().items():
                merged_source.setdefault(key, value)
        elif isinstance(source, pd.DataFrame):
            merged_source = source.copy()
            for key, value in universe_lookup.loc[ticker].to_dict().items():
                if key not in merged_source:
                    merged_source[key] = value
        else:
            merged_source = source
        metrics = compute_quant_metrics(
            merged_source,
            prices=prices_by_ticker.get(ticker),
            valuation_history=(valuations_by_ticker or {}).get(ticker),
            sector_prices=(sector_prices_by_ticker or {}).get(ticker),
            index_prices=index_prices,
            as_of=as_of,
            config=cfg,
        )
        rows.append({ticker_column: ticker, "as_of": pd.Timestamp(as_of), **metrics})
    return annotate_hard_screen(
        pd.DataFrame(rows),
        minimum_market_cap=cfg.minimum_market_cap,
        minimum_drawdown=cfg.minimum_drawdown,
    )


__all__ = [
    "QuantConfig",
    "annotate_hard_screen",
    "build_quant_snapshot",
    "calculate_drawdown",
    "calculate_fundamental_metrics",
    "calculate_growth",
    "calculate_relative_return",
    "calculate_total_return",
    "calculate_valuation_compression",
    "compute_quant_metrics",
    "evaluate_hard_screen",
    "hard_screen",
    "point_in_time_frame",
    "safe_divide",
]
