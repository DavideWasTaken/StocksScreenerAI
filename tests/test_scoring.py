from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from narrative_dislocation.scoring import score_candidate, score_candidates


def _complete_metrics() -> dict[str, float]:
    return {
        "drawdown_2y": 0.50,
        "relative_return_sector_1y": -0.25,
        "relative_return_index_1y": -0.30,
        "revenue_growth": 0.08,
        "fcf_growth": 0.20,
        "fcf_per_share_growth": 0.18,
        "fcf_margin": 0.16,
        "gross_margin_stability": 0.015,
        "share_dilution": 0.01,
        "roic": 0.14,
        "net_debt_to_fcf": 1.0,
        "ev_to_fcf": 12.0,
        "fcf_yield": 0.07,
        "pe_ratio": 18.0,
        "valuation_compression": 0.35,
    }


def test_all_four_scores_and_composite_are_bounded_and_fully_covered() -> None:
    scored = score_candidate(_complete_metrics())

    score_names = [
        "price_dislocation_score",
        "fundamental_resilience_score",
        "valuation_compression_score",
        "balance_sheet_quality_score",
        "quant_score",
    ]
    assert all(0.0 <= scored[name] <= 100.0 for name in score_names)
    assert scored["quant_coverage"] == pytest.approx(100.0)
    assert scored["quant_confidence"] == pytest.approx(100.0)


def test_missing_metrics_receive_no_neutral_points_and_reduce_coverage() -> None:
    scored = score_candidate({"drawdown_2y": 0.40})

    # At 40% drawdown the published breakpoint is 55; no absent metric adds 50.
    assert scored["price_dislocation_score"] == pytest.approx(55.0)
    assert scored["price_dislocation_coverage"] == pytest.approx(55.0)
    assert math.isnan(scored["fundamental_resilience_score"])
    assert math.isnan(scored["valuation_compression_score"])
    assert math.isnan(scored["balance_sheet_quality_score"])
    assert scored["quant_score"] == pytest.approx(55.0)
    assert scored["quant_coverage"] == pytest.approx(16.5)


def test_roic_proxy_is_used_only_as_explicit_fallback() -> None:
    proxy_scored = score_candidate({"roic_proxy": 0.10})
    reported_scored = score_candidate({"roic": 0.20, "roic_proxy": 0.01})

    proxy_explanation = proxy_scored["score_explanations"]["fundamental_resilience"]["metrics"][-1]
    reported_explanation = reported_scored["score_explanations"]["fundamental_resilience"][
        "metrics"
    ][-1]
    assert proxy_explanation["source_metric"] == "roic_proxy"
    assert reported_explanation["source_metric"] == "roic"
    assert (
        reported_scored["fundamental_resilience_score"]
        > proxy_scored["fundamental_resilience_score"]
    )


def test_dataframe_scoring_is_deterministic_and_json_explanations_are_strict() -> None:
    candidates = pd.DataFrame(
        [{"ticker": "A", **_complete_metrics()}, {"ticker": "B", "drawdown_2y": 0.40}]
    )

    first = score_candidates(candidates)
    second = score_candidates(candidates)

    pd.testing.assert_series_equal(first["quant_score"], second["quant_score"])
    parsed = json.loads(first.loc[0, "score_explanations"])
    assert parsed["price_dislocation"]["metrics"][0]["available"] is True
    parsed_missing = json.loads(first.loc[1, "score_explanations"])
    assert parsed_missing["valuation_compression"]["score"] is None
