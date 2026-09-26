from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from narrative_dislocation.quant import (
    annotate_hard_screen,
    calculate_drawdown,
    calculate_fundamental_metrics,
    calculate_relative_return,
    calculate_valuation_compression,
    compute_quant_metrics,
    evaluate_hard_screen,
    hard_screen,
)


def test_drawdown_is_positive_decline_and_honors_as_of_and_lookback() -> None:
    prices = pd.Series(
        [500.0, 200.0, 160.0, 100.0, 50.0],
        index=pd.to_datetime(
            ["2022-01-01", "2024-01-01", "2025-01-01", "2025-12-31", "2026-01-02"]
        ),
    )

    value = calculate_drawdown(prices, as_of="2025-12-31", lookback_days=730)

    assert value == pytest.approx(0.50)


def test_drawdown_falls_back_to_close_when_adjusted_close_has_no_observations() -> None:
    prices = pd.DataFrame(
        {
            "adjusted_close": [np.nan, np.nan],
            "close": [100.0, 60.0],
        },
        index=pd.to_datetime(["2024-01-01", "2025-01-01"]),
    )

    assert calculate_drawdown(prices) == pytest.approx(0.40)


def test_fundamental_metrics_use_only_disclosed_rows_and_label_roic_proxy() -> None:
    history = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            "available_at": pd.to_datetime(["2024-02-01", "2025-02-01", "2026-02-01"]),
            "revenue": [100.0, 110.0, 999.0],
            "free_cash_flow": [10.0, 12.0, -100.0],
            "shares_outstanding": [10.0, 11.0, 100.0],
            "gross_margin": [0.40, 0.41, 0.10],
            "total_debt": [18.0, 20.0, 200.0],
            "cash_and_equivalents": [5.0, 5.0, 0.0],
            "operating_income": [18.0, 20.0, -50.0],
            "effective_tax_rate": [0.25, 0.25, 0.25],
            "invested_capital": [50.0, 60.0, 500.0],
            "market_cap": [180.0, 200.0, 20.0],
            "enterprise_value": [108.0, 120.0, 220.0],
            "price": [18.0, 20.0, 0.2],
            "diluted_eps": [1.8, 2.0, -1.0],
        }
    )

    metrics = calculate_fundamental_metrics(history, as_of="2025-12-31")

    assert metrics["fundamental_observations"] == 2
    assert metrics["latest_fundamental_period_end"].startswith("2024-12-31")
    assert metrics["prior_fundamental_period_end"].startswith("2023-12-31")
    assert metrics["latest_fundamental_available_at"].startswith("2025-02-01")
    assert metrics["latest_fundamental_age_days"] == 365
    assert metrics["revenue_growth"] == pytest.approx(0.10)
    assert metrics["fcf_growth"] == pytest.approx(0.20)
    assert metrics["fcf_per_share_growth"] == pytest.approx(1.0 / 11.0)
    assert metrics["fcf_margin"] == pytest.approx(12.0 / 110.0)
    assert metrics["gross_margin_stability"] == pytest.approx(0.005)
    assert metrics["share_dilution"] == pytest.approx(0.10)
    assert metrics["net_debt"] == pytest.approx(15.0)
    assert metrics["net_debt_to_fcf"] == pytest.approx(1.25)
    assert math.isnan(metrics["roic"])
    assert metrics["roic_proxy"] == pytest.approx(15.0 / 55.0)
    assert "derived proxy" in metrics["roic_proxy_label"]
    assert metrics["ev_to_fcf"] == pytest.approx(10.0)
    assert metrics["fcf_yield"] == pytest.approx(0.06)
    assert metrics["pe_ratio"] == pytest.approx(10.0)


def test_fundamental_metrics_do_not_backfill_missing_latest_values() -> None:
    history = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            "revenue": [100.0, 120.0, np.nan],
            "free_cash_flow": [10.0, 15.0, np.nan],
            "shares_outstanding": [10.0, 11.0, np.nan],
            "gross_margin": [0.40, 0.41, np.nan],
            "total_debt": [20.0, 21.0, 22.0],
            "cash": [5.0, 5.0, 5.0],
            "market_cap": [600.0, 700.0, 800.0],
            "enterprise_value": [620.0, 720.0, 820.0],
        }
    )

    metrics = calculate_fundamental_metrics(history)

    for key in (
        "revenue",
        "free_cash_flow",
        "shares_outstanding",
        "revenue_growth",
        "fcf_growth",
        "fcf_per_share_growth",
        "fcf_margin",
        "gross_margin",
        "gross_margin_change",
        "share_dilution",
        "net_debt_to_fcf",
        "ev_to_fcf",
        "fcf_yield",
    ):
        assert math.isnan(metrics[key]), key
    assert metrics["gross_margin_stability"] == pytest.approx(0.005)
    assert metrics["net_debt"] == pytest.approx(17.0)
    assert metrics["market_cap"] == pytest.approx(800.0)
    assert metrics["enterprise_value"] == pytest.approx(820.0)


def test_growth_does_not_skip_an_immediately_preceding_missing_period() -> None:
    history = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            "revenue": [100.0, np.nan, 130.0],
            "free_cash_flow": [10.0, np.nan, 13.0],
            "shares_outstanding": [10.0, np.nan, 11.0],
            "gross_margin": [0.40, np.nan, 0.42],
            "market_cap": [600.0, 700.0, 800.0],
            "enterprise_value": [620.0, 720.0, 820.0],
        }
    )

    metrics = calculate_fundamental_metrics(history)

    assert metrics["revenue"] == pytest.approx(130.0)
    assert metrics["free_cash_flow"] == pytest.approx(13.0)
    assert metrics["shares_outstanding"] == pytest.approx(11.0)
    assert metrics["fcf_margin"] == pytest.approx(0.10)
    assert metrics["gross_margin"] == pytest.approx(0.42)
    assert metrics["gross_margin_stability"] == pytest.approx(0.01)
    for key in (
        "revenue_growth",
        "fcf_growth",
        "fcf_per_share_growth",
        "gross_margin_change",
        "share_dilution",
    ):
        assert math.isnan(metrics[key]), key


def test_explicit_mapping_metrics_override_aligned_derived_values() -> None:
    history = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2024-12-31", "2025-12-31"]),
            "revenue": [100.0, np.nan],
            "free_cash_flow": [10.0, np.nan],
            "shares_outstanding": [10.0, np.nan],
        }
    )
    fundamentals = {
        "history": history,
        "revenue_growth": 0.20,
        "fcf_growth": 0.15,
        "fcf_per_share_growth": 0.12,
        "fcf_margin": 0.08,
        "share_dilution": -0.01,
    }

    metrics = calculate_fundamental_metrics(fundamentals)

    assert metrics["revenue_growth"] == pytest.approx(0.20)
    assert metrics["fcf_growth"] == pytest.approx(0.15)
    assert metrics["fcf_per_share_growth"] == pytest.approx(0.12)
    assert metrics["fcf_margin"] == pytest.approx(0.08)
    assert metrics["share_dilution"] == pytest.approx(-0.01)
    assert math.isnan(metrics["revenue"])
    assert math.isnan(metrics["free_cash_flow"])
    assert math.isnan(metrics["shares_outstanding"])


def test_sequence_history_preserves_flat_scalar_supplements_without_backfilling() -> None:
    fundamentals = {
        "history": [
            {
                "period_end": "2024-12-31",
                "revenue": 100.0,
                "free_cash_flow": 10.0,
                "enterprise_value": 900.0,
            },
            {
                "period_end": "2025-12-31",
                "revenue": 120.0,
                "free_cash_flow": 12.0,
                "enterprise_value": None,
            },
        ],
        "market_cap": 600.0,
        "enterprise_value": 1_000.0,
    }

    metrics = calculate_fundamental_metrics(fundamentals)

    assert metrics["market_cap"] == pytest.approx(600.0)
    assert metrics["fcf_yield"] == pytest.approx(0.02)
    assert math.isnan(metrics["enterprise_value"])
    assert math.isnan(metrics["ev_to_fcf"])


def test_negative_fcf_does_not_create_misleading_positive_valuation_ratios() -> None:
    metrics = calculate_fundamental_metrics(
        {
            "revenue": 100.0,
            "free_cash_flow": -5.0,
            "market_cap": 10_000.0,
            "enterprise_value": 11_000.0,
            "total_debt": 2_000.0,
            "cash": 100.0,
        }
    )

    assert math.isnan(metrics["ev_to_fcf"])
    assert math.isnan(metrics["fcf_yield"])
    assert math.isnan(metrics["net_debt_to_fcf"])
    assert metrics["net_debt"] == pytest.approx(1_900.0)


def test_negative_enterprise_value_does_not_create_ev_to_fcf_multiple() -> None:
    metrics = calculate_fundamental_metrics(
        {
            "revenue": 5_000.0,
            "free_cash_flow": 500.0,
            "market_cap": 10_000.0,
            "enterprise_value": -1_000.0,
        }
    )

    assert math.isnan(metrics["ev_to_fcf"])
    assert metrics["fcf_yield"] == pytest.approx(0.05)


def test_valuation_compression_requires_real_history_and_excludes_future_disclosures() -> None:
    current = {"ev_to_fcf": 10.0, "pe_ratio": 12.0}
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]),
            "available_at": pd.to_datetime(
                ["2022-01-02", "2023-01-02", "2024-01-02", "2026-01-02"]
            ),
            "ev_to_fcf": [20.0, 18.0, 16.0, 3.0],
            "pe_ratio": [24.0, 20.0, 16.0, 2.0],
        }
    )

    metrics = calculate_valuation_compression(current, history, as_of="2025-01-01")

    assert metrics["historical_ev_to_fcf_median"] == pytest.approx(18.0)
    assert metrics["ev_to_fcf_compression"] == pytest.approx(8.0 / 18.0)
    assert metrics["historical_pe_median"] == pytest.approx(20.0)
    assert metrics["pe_compression"] == pytest.approx(0.40)
    assert metrics["valuation_compression"] == pytest.approx(((8.0 / 18.0) + 0.40) / 2.0)


def test_relative_return_uses_common_observation_window() -> None:
    stock = pd.Series([100.0, 80.0], index=pd.to_datetime(["2025-01-01", "2025-12-31"]))
    benchmark = pd.Series([100.0, 110.0], index=pd.to_datetime(["2025-01-01", "2025-12-31"]))

    assert calculate_relative_return(stock, benchmark, as_of="2025-12-31") == pytest.approx(-0.30)


def test_compute_quant_metrics_keeps_unavailable_values_as_nan() -> None:
    metrics = compute_quant_metrics(
        {"market_cap": 7_000_000_000.0, "revenue": 100.0},
        prices=pd.Series([100.0, 70.0], index=pd.to_datetime(["2024-01-01", "2025-01-01"])),
        as_of="2025-01-01",
    )

    assert metrics["drawdown_2y"] == pytest.approx(0.30)
    assert math.isnan(metrics["revenue_growth"])
    assert math.isnan(metrics["fcf_margin"])
    assert math.isnan(metrics["relative_return_sector_1y"])


def test_hard_screen_is_strict_and_missing_values_fail_with_reasons() -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "PASS", "market_cap": 5_000_000_001.0, "drawdown_2y": 0.2501},
            {"ticker": "CAP", "market_cap": 5_000_000_000.0, "drawdown_2y": 0.50},
            {"ticker": "DD", "market_cap": 6_000_000_000.0, "drawdown_2y": 0.25},
            {"ticker": "MISS", "market_cap": np.nan, "drawdown_2y": np.nan},
        ]
    )

    annotated = annotate_hard_screen(frame)
    screened = hard_screen(frame)

    assert screened["ticker"].tolist() == ["PASS"]
    assert (
        annotated.set_index("ticker").at["MISS", "screen_reasons"]
        == "market_cap_missing;drawdown_2y_missing"
    )
    passed, reasons = evaluate_hard_screen(frame.iloc[1])
    assert not passed
    assert reasons == ["market_cap_not_above_threshold"]
