"""Separate point-in-time portfolio backtester for historical screen snapshots.

The engine consumes already-computed signal snapshots. It never reconstructs
historical fundamentals from today's values. Both ``as_of`` (snapshot date) and
``available_at`` (when the snapshot could have been acted upon) are mandatory,
and both must be no later than the rebalance cutoff.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

DEFAULT_BENCHMARK_TICKERS = ("VALL", "^GSPC")


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a periodic equal-weight top-N simulation."""

    top_n: int = 20
    rebalance_frequency: str = "M"
    transaction_cost_bps: float = 10.0
    execution_lag_sessions: int = 1
    score_column: str = "dislocation_score"
    ticker_column: str = "ticker"
    as_of_column: str = "as_of"
    available_at_column: str = "available_at"
    eligible_column: str | None = "screen_pass"
    minimum_score: float | None = None
    snapshot_mode: Literal["latest_snapshot", "per_ticker"] = "latest_snapshot"
    max_signal_age_days: int | None = None
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    benchmark_tickers: tuple[str, ...] = DEFAULT_BENCHMARK_TICKERS

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.execution_lag_sessions < 1:
            raise ValueError("execution_lag_sessions must be at least one for daily close data")
        if self.snapshot_mode not in {"latest_snapshot", "per_ticker"}:
            raise ValueError("snapshot_mode must be 'latest_snapshot' or 'per_ticker'")
        if self.max_signal_age_days is not None and self.max_signal_age_days < 0:
            raise ValueError("max_signal_age_days cannot be negative")


@dataclass
class BacktestResult:
    """Backtest outputs kept as data structures suitable for CSV/JSON export."""

    equity_curve: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    benchmark_curves: pd.DataFrame
    summary: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _naive_datetime(values: Any) -> Any:
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(converted, pd.Series):
        return converted.dt.tz_localize(None)
    if isinstance(converted, pd.DatetimeIndex):
        return converted.tz_localize(None)
    return converted.tz_localize(None) if not pd.isna(converted) else converted


def _timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    converted = pd.Timestamp(value)
    return converted.tz_localize(None) if converted.tzinfo is not None else converted


def _explicit_true(value: Any) -> bool:
    """Accept common CSV truth values without treating non-empty strings as true."""

    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return False


def _price_column(frame: pd.DataFrame) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    fallback: str | None = None
    for alias in ("adjusted_close", "adj_close", "adj close", "close", "price", "value"):
        if alias in lower:
            column = lower[alias]
            fallback = fallback or column
            values = pd.to_numeric(frame[column], errors="coerce")
            if (np.isfinite(values) & (values > 0)).any():
                return column
    return fallback


def normalize_price_frame(
    prices: pd.DataFrame | Mapping[str, pd.Series | pd.DataFrame | Mapping[Any, Any]],
) -> pd.DataFrame:
    """Normalize wide, long, or ticker-to-series inputs to date x ticker prices."""

    if isinstance(prices, Mapping):
        columns: dict[str, pd.Series] = {}
        for ticker, values in prices.items():
            if isinstance(values, pd.Series):
                series = pd.to_numeric(values, errors="coerce")
            elif isinstance(values, Mapping):
                series = pd.to_numeric(pd.Series(values), errors="coerce")
            elif isinstance(values, pd.DataFrame):
                local = values.copy()
                date_col = next(
                    (
                        column
                        for column in local
                        if str(column).lower() in {"date", "timestamp", "as_of"}
                    ),
                    None,
                )
                if date_col is not None:
                    local.index = _naive_datetime(local.pop(date_col))
                value_col = _price_column(local)
                if value_col is None:
                    numeric = local.select_dtypes(include=[np.number])
                    if numeric.shape[1] != 1:
                        raise ValueError(
                            f"Price data for {ticker} needs one identifiable price column"
                        )
                    value_col = str(numeric.columns[0])
                series = pd.to_numeric(local[value_col], errors="coerce")
            else:
                raise TypeError(f"Unsupported prices for {ticker}: {type(values)!r}")
            series.index = _naive_datetime(series.index)
            columns[str(ticker)] = series
        frame = pd.DataFrame(columns)
    elif isinstance(prices, pd.DataFrame):
        local = prices.copy()
        lower = {str(column).lower(): str(column) for column in local.columns}
        date_col = next(
            (lower[key] for key in ("date", "timestamp", "as_of") if key in lower), None
        )
        ticker_col = next((lower[key] for key in ("ticker", "symbol") if key in lower), None)
        value_col = _price_column(local)
        if date_col is not None and ticker_col is not None and value_col is not None:
            local[date_col] = _naive_datetime(local[date_col])
            local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
            frame = local.pivot_table(
                index=date_col, columns=ticker_col, values=value_col, aggfunc="last"
            )
        else:
            if date_col is not None:
                local.index = _naive_datetime(local.pop(date_col))
            else:
                local.index = _naive_datetime(local.index)
            frame = local.apply(pd.to_numeric, errors="coerce")
    else:
        raise TypeError("prices must be a DataFrame or ticker-to-price mapping")

    frame.index = _naive_datetime(frame.index)
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.groupby(level=0).last()
    frame.columns = frame.columns.astype(str)
    frame = frame.replace([np.inf, -np.inf], np.nan).where(lambda values: values > 0)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("No valid price observations")
    return frame


def validate_signal_snapshots(signals: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Validate mandatory PIT fields and normalize their types."""

    required = {
        config.ticker_column,
        config.as_of_column,
        config.available_at_column,
        config.score_column,
    }
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(
            "Signal snapshots must include point-in-time columns; missing "
            + ", ".join(sorted(missing))
        )
    result = signals.copy()
    raw_tickers = result[config.ticker_column]
    invalid_ticker = raw_tickers.isna() | raw_tickers.map(lambda value: not str(value).strip())
    if invalid_ticker.any():
        raise ValueError(f"Every signal needs a non-blank {config.ticker_column}")
    result[config.ticker_column] = raw_tickers.map(lambda value: str(value).strip())
    result[config.as_of_column] = _naive_datetime(result[config.as_of_column])
    result[config.available_at_column] = _naive_datetime(result[config.available_at_column])
    result[config.score_column] = pd.to_numeric(
        result[config.score_column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    if config.eligible_column is not None and config.eligible_column in result:
        result[config.eligible_column] = result[config.eligible_column].map(_explicit_true)
    if result[config.as_of_column].isna().any():
        raise ValueError(f"Every signal needs a valid {config.as_of_column}")
    if result[config.available_at_column].isna().any():
        raise ValueError(f"Every signal needs a valid {config.available_at_column}")
    return result


def select_point_in_time_signals(
    signals: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    config: BacktestConfig | None = None,
    *,
    apply_top_n: bool = True,
) -> pd.DataFrame:
    """Select a top-N cross-section using only information known at ``cutoff``."""

    cfg = config or BacktestConfig()
    normalized = validate_signal_snapshots(signals, cfg)
    timestamp = _timestamp(cutoff)
    assert timestamp is not None
    eligible = normalized[
        (normalized[cfg.as_of_column] <= timestamp)
        & (normalized[cfg.available_at_column] <= timestamp)
        & normalized[cfg.score_column].notna()
    ].copy()
    if cfg.eligible_column is not None and cfg.eligible_column in eligible:
        eligible = eligible[eligible[cfg.eligible_column]]
    if cfg.minimum_score is not None:
        eligible = eligible[eligible[cfg.score_column] >= cfg.minimum_score]
    if cfg.max_signal_age_days is not None:
        eligible = eligible[
            (timestamp - eligible[cfg.as_of_column]) <= pd.Timedelta(days=cfg.max_signal_age_days)
        ]
    if eligible.empty:
        return eligible

    if cfg.snapshot_mode == "latest_snapshot":
        latest_snapshot = eligible[cfg.as_of_column].max()
        eligible = eligible[eligible[cfg.as_of_column] == latest_snapshot]
    else:
        eligible = (
            eligible.sort_values(
                [cfg.ticker_column, cfg.as_of_column, cfg.available_at_column],
                kind="mergesort",
            )
            .groupby(cfg.ticker_column, as_index=False, sort=False)
            .tail(1)
        )

    # A correction to the same ticker/snapshot can have a later availability.
    eligible = (
        eligible.sort_values(
            [cfg.ticker_column, cfg.as_of_column, cfg.available_at_column],
            kind="mergesort",
        )
        .groupby(cfg.ticker_column, as_index=False, sort=False)
        .tail(1)
    )
    eligible = eligible.sort_values(
        [cfg.score_column, cfg.ticker_column],
        ascending=[False, True],
        kind="mergesort",
    )
    if apply_top_n:
        eligible = eligible.head(cfg.top_n)
    return eligible.reset_index(drop=True)


def _period_alias(frequency: str) -> str:
    aliases = {"ME": "M", "QE": "Q", "YE": "Y"}
    return aliases.get(frequency.upper(), frequency)


def rebalance_cutoffs(price_dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Last observed trading session in each configured calendar period."""

    if price_dates.empty:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(price_dates).sort_values().unique()
    periods = dates.to_period(_period_alias(frequency))
    cutoffs = pd.Series(dates, index=periods).groupby(level=0).max()
    return pd.DatetimeIndex(cutoffs.to_numpy()).sort_values()


def _performance_statistics(
    values: pd.Series, returns: pd.Series | None = None
) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_zero_rate": float("nan"),
            "max_drawdown": float("nan"),
            "observations": 0,
        }
    daily_returns = (
        pd.to_numeric(returns, errors="coerce").dropna()
        if returns is not None
        else values.pct_change(fill_method=None).dropna()
    )
    ending_value = float(values.iloc[-1])
    total_return = ending_value - 1.0
    elapsed_days = max((values.index[-1] - values.index[0]).days, 1)
    cagr = ending_value ** (365.25 / elapsed_days) - 1.0 if ending_value > 0 else -1.0
    volatility = (
        float(daily_returns.std(ddof=1) * np.sqrt(252.0)) if len(daily_returns) >= 2 else NAN
    )
    sharpe = (
        float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252.0))
        if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0
        else NAN
    )
    drawdown = values / values.cummax() - 1.0
    return {
        "total_return": total_return,
        "cagr": float(cagr),
        "annualized_volatility": volatility,
        "sharpe_zero_rate": sharpe,
        "max_drawdown": float(drawdown.min()),
        "observations": int(len(values)),
    }


def _benchmark_output(
    benchmark_prices: pd.DataFrame
    | Mapping[str, pd.Series | pd.DataFrame | Mapping[Any, Any]]
    | None,
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if benchmark_prices is None or dates.empty:
        return pd.DataFrame(index=dates), {}
    normalized = normalize_price_frame(benchmark_prices)
    requested = list(tickers) if tickers else list(normalized.columns)
    curves: dict[str, pd.Series] = {}
    summaries: dict[str, Any] = {}
    for ticker in requested:
        if ticker not in normalized:
            continue
        source = normalized[ticker].dropna()
        combined_dates = source.index.union(dates).sort_values()
        series = source.reindex(combined_dates).ffill().reindex(dates).dropna()
        if series.empty:
            continue
        curve = series / series.iloc[0]
        curves[ticker] = curve.reindex(dates)
        summaries[ticker] = _performance_statistics(curve)
    return pd.DataFrame(curves, index=dates), summaries


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame | Mapping[str, pd.Series | pd.DataFrame | Mapping[Any, Any]],
    *,
    benchmark_prices: pd.DataFrame
    | Mapping[str, pd.Series | pd.DataFrame | Mapping[Any, Any]]
    | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a periodic, equal-weight, next-session top-N simulation.

    Rebalance cutoffs are the last price date in each calendar period. With the
    default one-session lag, signals known by that cutoff trade on the following
    observed session. Transaction costs are charged on absolute weight traded;
    a full A-to-B rotation therefore has turnover 2.0.
    """

    cfg = config or BacktestConfig()
    normalized_signals = validate_signal_snapshots(signals, cfg)
    price_frame = normalize_price_frame(prices)
    start = _timestamp(cfg.start)
    end = _timestamp(cfg.end)
    if start is not None:
        price_frame = price_frame[price_frame.index >= start]
    if end is not None:
        price_frame = price_frame[price_frame.index <= end]
    if len(price_frame.index) < 2:
        raise ValueError("Backtest needs at least two price dates in the selected range")

    cutoffs = rebalance_cutoffs(price_frame.index, cfg.rebalance_frequency)
    events: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
    for cutoff in cutoffs:
        cutoff_location = int(price_frame.index.searchsorted(cutoff, side="left"))
        execution_location = cutoff_location + cfg.execution_lag_sessions
        if execution_location >= len(price_frame.index):
            continue
        execution_date = price_frame.index[execution_location]
        # Calendar-date snapshots are treated as available through cutoff EOD.
        signal_cutoff = cutoff + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        # Keep the complete ranked cross-section until execution-price
        # availability is known, then backfill the requested Top N.
        selection = select_point_in_time_signals(
            normalized_signals,
            signal_cutoff,
            cfg,
            apply_top_n=False,
        )
        events[execution_date] = (cutoff, selection)

    if not events:
        raise ValueError(
            "No executable rebalance dates; extend the price range or reduce execution lag"
        )

    first_execution = min(events)
    simulation_prices = price_frame.loc[first_execution:]
    asset_returns = simulation_prices.pct_change(fill_method=None)
    current_weights: dict[str, float] = {}
    nav = 1.0
    curve_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    missing_held_returns: list[dict[str, str]] = []
    total_transaction_cost = 0.0

    for date in simulation_prices.index:
        raw_returns = asset_returns.loc[date]
        market_return = 0.0
        for ticker, weight in current_weights.items():
            ticker_return = raw_returns.get(ticker, NAN)
            if not np.isfinite(ticker_return):
                ticker_return = 0.0
                missing_held_returns.append({"date": str(date.date()), "ticker": ticker})
            market_return += weight * float(ticker_return)

        nav_before = nav
        nav *= 1.0 + market_return
        if current_weights and 1.0 + market_return > 0:
            drifted = {}
            for ticker, weight in current_weights.items():
                ticker_return = raw_returns.get(ticker, NAN)
                ticker_return = float(ticker_return) if np.isfinite(ticker_return) else 0.0
                drifted[ticker] = weight * (1.0 + ticker_return) / (1.0 + market_return)
            current_weights = drifted

        turnover = 0.0
        cost_fraction = 0.0
        if date in events:
            cutoff, selection = events[date]
            tradable = selection[
                selection[cfg.ticker_column].isin(simulation_prices.columns)
            ].copy()
            if not tradable.empty:
                available_prices = simulation_prices.loc[date].map(np.isfinite)
                has_price = tradable[cfg.ticker_column].map(available_prices).fillna(False)
                tradable = tradable[has_price].head(cfg.top_n).copy()
            tickers = tradable[cfg.ticker_column].astype(str).tolist()
            target_weights = {ticker: 1.0 / len(tickers) for ticker in tickers} if tickers else {}
            all_tickers = sorted(set(current_weights) | set(target_weights))
            turnover = float(
                sum(
                    abs(target_weights.get(ticker, 0.0) - current_weights.get(ticker, 0.0))
                    for ticker in all_tickers
                )
            )
            cost_fraction = turnover * cfg.transaction_cost_bps / 10_000.0
            cost_amount = nav * cost_fraction
            nav *= max(1.0 - cost_fraction, 0.0)
            total_transaction_cost += cost_amount

            lookup = tradable.set_index(cfg.ticker_column) if not tradable.empty else pd.DataFrame()
            for ticker in all_tickers:
                old_weight = current_weights.get(ticker, 0.0)
                new_weight = target_weights.get(ticker, 0.0)
                if abs(new_weight - old_weight) <= 1e-15:
                    continue
                trade_rows.append(
                    {
                        "rebalance_cutoff": cutoff,
                        "execution_date": date,
                        "ticker": ticker,
                        "old_weight": old_weight,
                        "target_weight": new_weight,
                        "weight_traded": abs(new_weight - old_weight),
                        "estimated_cost": cost_amount * abs(new_weight - old_weight) / turnover
                        if turnover > 0
                        else 0.0,
                    }
                )
            for ticker, weight in target_weights.items():
                signal = lookup.loc[ticker]
                holdings_rows.append(
                    {
                        "rebalance_cutoff": cutoff,
                        "execution_date": date,
                        "ticker": ticker,
                        "target_weight": weight,
                        "score": float(signal[cfg.score_column]),
                        "signal_as_of": signal[cfg.as_of_column],
                        "signal_available_at": signal[cfg.available_at_column],
                    }
                )
            current_weights = target_weights

        portfolio_return = nav / nav_before - 1.0 if nav_before > 0 else NAN
        curve_rows.append(
            {
                "date": date,
                "portfolio_value": nav,
                "portfolio_return": portfolio_return,
                "market_return_before_cost": market_return,
                "turnover": turnover,
                "transaction_cost_fraction": cost_fraction,
                "number_of_holdings": len(current_weights),
            }
        )

    equity_curve = pd.DataFrame(curve_rows).set_index("date")
    holdings = pd.DataFrame(holdings_rows)
    trades = pd.DataFrame(trade_rows)
    benchmark_curves, benchmark_summary = _benchmark_output(
        benchmark_prices,
        equity_curve.index,
        cfg.benchmark_tickers,
    )
    portfolio_summary = _performance_statistics(
        equity_curve["portfolio_value"],
        equity_curve["portfolio_return"],
    )
    portfolio_summary.update(
        {
            "total_transaction_cost": float(total_transaction_cost),
            "rebalance_count": int(len(events)),
        }
    )
    summary = {
        "portfolio": portfolio_summary,
        "benchmarks": benchmark_summary,
        "config": asdict(cfg),
    }
    diagnostics = {
        "lookahead_guard": "Signals require as_of <= cutoff and available_at <= cutoff; default execution is next observed session.",
        "missing_held_return_count": len(missing_held_returns),
        "missing_held_returns": missing_held_returns,
        "price_assumption": "Adjusted prices are assumed point-in-time accurate; delisting-return reconstruction is outside v1.",
    }
    return BacktestResult(
        equity_curve=equity_curve,
        holdings=holdings,
        trades=trades,
        benchmark_curves=benchmark_curves,
        summary=summary,
        diagnostics=diagnostics,
    )


NAN = float("nan")


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DEFAULT_BENCHMARK_TICKERS",
    "normalize_price_frame",
    "rebalance_cutoffs",
    "run_backtest",
    "select_point_in_time_signals",
    "validate_signal_snapshots",
]
