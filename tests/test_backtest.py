from __future__ import annotations

import pandas as pd
import pytest

from narrative_dislocation.backtest import (
    BacktestConfig,
    normalize_price_frame,
    run_backtest,
    select_point_in_time_signals,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-31", "2025-02-01", "2025-02-28", "2025-03-01", "2025-03-31"]
            ),
            "AAA": [100.0, 100.0, 110.0, 111.0, 112.0],
            "BBB": [100.0, 100.0, 100.0, 100.0, 100.0],
            "CCC": [100.0, 100.0, 90.0, 90.0, 99.0],
        }
    )


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "as_of": "2025-01-31",
                "available_at": "2025-01-31",
                "dislocation_score": 90.0,
                "screen_pass": True,
            },
            {
                "ticker": "BBB",
                "as_of": "2025-01-31",
                "available_at": "2025-01-31",
                "dislocation_score": 80.0,
                "screen_pass": True,
            },
            # Same snapshot, but not known at the January cutoff.
            {
                "ticker": "CCC",
                "as_of": "2025-01-31",
                "available_at": "2025-02-05",
                "dislocation_score": 100.0,
                "screen_pass": True,
            },
        ]
    )


def test_signal_selection_guards_both_as_of_and_available_at() -> None:
    config = BacktestConfig(top_n=2)
    selected_january = select_point_in_time_signals(_signals(), "2025-01-31 23:59:59", config)
    selected_february = select_point_in_time_signals(_signals(), "2025-02-28 23:59:59", config)

    assert selected_january["ticker"].tolist() == ["AAA", "BBB"]
    assert selected_february["ticker"].tolist() == ["CCC", "AAA"]


def test_missing_point_in_time_columns_are_rejected() -> None:
    bad = _signals().drop(columns="available_at")

    with pytest.raises(ValueError, match="available_at"):
        run_backtest(bad, _prices(), config=BacktestConfig(top_n=1))


def test_periodic_top_n_rebalance_uses_next_session_and_charges_turnover() -> None:
    config = BacktestConfig(top_n=1, transaction_cost_bps=10.0)

    result = run_backtest(_signals(), _prices(), config=config)

    holdings = result.holdings.sort_values("execution_date")
    assert holdings["ticker"].tolist() == ["AAA", "CCC"]
    assert holdings["execution_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-02-01",
        "2025-03-01",
    ]
    # Cash-to-AAA is 1x turnover; AAA-to-CCC is 2x turnover.
    assert result.equity_curve.loc[pd.Timestamp("2025-02-01"), "turnover"] == pytest.approx(1.0)
    assert result.equity_curve.loc[pd.Timestamp("2025-03-01"), "turnover"] == pytest.approx(2.0)
    assert result.summary["portfolio"]["total_transaction_cost"] > 0
    assert "lookahead_guard" in result.diagnostics


def test_benchmarks_are_configurable_and_reported_separately() -> None:
    benchmarks = pd.DataFrame(
        {
            "date": _prices()["date"],
            "WORLD": [100.0, 100.0, 101.0, 101.0, 103.0],
            "SPX": [100.0, 100.0, 102.0, 102.0, 104.0],
        }
    )
    config = BacktestConfig(top_n=1, benchmark_tickers=("WORLD", "SPX"))

    result = run_backtest(_signals(), _prices(), benchmark_prices=benchmarks, config=config)

    assert result.benchmark_curves.columns.tolist() == ["WORLD", "SPX"]
    assert set(result.summary["benchmarks"]) == {"WORLD", "SPX"}
    assert result.benchmark_curves.iloc[0].tolist() == pytest.approx([1.0, 1.0])


def test_long_price_frame_normalizes_to_wide() -> None:
    long = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
            "ticker": ["A", "A", "B", "B"],
            "adjusted_close": [10.0, 11.0, 20.0, 18.0],
        }
    )

    wide = normalize_price_frame(long)

    assert wide.columns.tolist() == ["A", "B"]
    assert wide.loc[pd.Timestamp("2025-01-02"), "B"] == pytest.approx(18.0)


def test_long_prices_fall_back_to_close_when_adjusted_close_is_empty() -> None:
    long = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "ticker": ["A", "A"],
            "adjusted_close": [float("nan"), float("nan")],
            "close": [10.0, 11.0],
        }
    )

    wide = normalize_price_frame(long)

    assert wide["A"].tolist() == [10.0, 11.0]


def test_execution_filter_backfills_next_ranked_tradable_asset() -> None:
    prices = _prices()
    prices.loc[prices["date"] == pd.Timestamp("2025-02-01"), "AAA"] = float("nan")

    result = run_backtest(_signals(), prices, config=BacktestConfig(top_n=1))

    first = result.holdings.sort_values("execution_date").iloc[0]
    assert first["ticker"] == "BBB"


def test_benchmark_forward_fill_uses_observations_outside_portfolio_calendar() -> None:
    benchmarks = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-30", "2025-02-27", "2025-03-28"]),
            "WORLD": [100.0, 105.0, 110.0],
        }
    )

    result = run_backtest(
        _signals(),
        _prices(),
        benchmark_prices=benchmarks,
        config=BacktestConfig(top_n=1, benchmark_tickers=("WORLD",)),
    )

    assert "WORLD" in result.benchmark_curves
    assert result.benchmark_curves["WORLD"].dropna().iloc[-1] == pytest.approx(1.10)


def test_zero_session_execution_lag_is_rejected_for_close_data() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BacktestConfig(execution_lag_sessions=0)


def test_invalid_scores_and_false_csv_strings_are_not_eligible() -> None:
    signals = pd.DataFrame(
        [
            {
                "ticker": "INF",
                "as_of": "2025-01-31",
                "available_at": "2025-01-31",
                "dislocation_score": float("inf"),
                "screen_pass": "true",
            },
            {
                "ticker": "FALSE",
                "as_of": "2025-01-31",
                "available_at": "2025-01-31",
                "dislocation_score": 99.0,
                "screen_pass": "false",
            },
            {
                "ticker": " PASS ",
                "as_of": "2025-01-31",
                "available_at": "2025-01-31",
                "dislocation_score": 80.0,
                "screen_pass": "1",
            },
        ]
    )

    selected = select_point_in_time_signals(signals, "2025-01-31 23:59:59")

    assert selected["ticker"].tolist() == ["PASS"]


@pytest.mark.parametrize("ticker", [None, "", "   "])
def test_blank_signal_tickers_are_rejected(ticker) -> None:
    signals = _signals().copy()
    signals.loc[0, "ticker"] = ticker

    with pytest.raises(ValueError, match="non-blank ticker"):
        select_point_in_time_signals(signals, "2025-01-31 23:59:59")
