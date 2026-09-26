"""Transparent, missing-data-aware quant scoring.

All input ratios/growth rates are decimals. Each metric is mapped to 0--100 by
published piecewise-linear breakpoints. Missing inputs have no score and add no
points; coverage reports how much of the configured evidence was actually
available.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SCORING_MODEL_VERSION = "quant-v1"


@dataclass(frozen=True)
class MetricRule:
    """A weighted piecewise-linear scoring rule.

    ``aliases`` are tried in order. This is useful for reported ROIC first and
    the clearly labelled NOPAT/invested-capital proxy second.
    """

    metric: str
    weight: float
    points: tuple[tuple[float, float], ...]
    aliases: tuple[str, ...] = ()
    interpretation: str = ""


@dataclass(frozen=True)
class ComponentScore:
    name: str
    score: float
    coverage: float
    available_weight: float
    total_weight: float
    explanations: tuple[dict[str, Any], ...]


PRICE_DISLOCATION_RULES = (
    MetricRule(
        "drawdown_2y",
        0.55,
        ((0.0, 0.0), (0.25, 25.0), (0.40, 55.0), (0.60, 85.0), (0.80, 100.0)),
        interpretation="Larger decline from the two-year high means greater price dislocation.",
    ),
    MetricRule(
        "relative_return_sector_1y",
        0.25,
        ((-0.50, 100.0), (-0.30, 80.0), (-0.15, 55.0), (0.0, 25.0), (0.15, 0.0)),
        interpretation="Worse one-year return than the sector means greater dislocation.",
    ),
    MetricRule(
        "relative_return_index_1y",
        0.20,
        ((-0.50, 100.0), (-0.30, 80.0), (-0.15, 55.0), (0.0, 25.0), (0.15, 0.0)),
        interpretation="Worse one-year return than the broad index means greater dislocation.",
    ),
)


FUNDAMENTAL_RESILIENCE_RULES = (
    MetricRule(
        "revenue_growth",
        0.14,
        ((-0.25, 0.0), (-0.10, 20.0), (0.0, 50.0), (0.10, 75.0), (0.20, 100.0)),
        interpretation="Positive latest-period revenue growth indicates demand resilience.",
    ),
    MetricRule(
        "fcf_growth",
        0.16,
        ((-0.50, 0.0), (-0.20, 25.0), (0.0, 50.0), (0.25, 75.0), (0.50, 100.0)),
        interpretation="Positive latest-period free-cash-flow growth indicates resilience.",
    ),
    MetricRule(
        "fcf_per_share_growth",
        0.20,
        ((-0.50, 0.0), (-0.20, 25.0), (0.0, 50.0), (0.20, 75.0), (0.40, 100.0)),
        interpretation="Per-share FCF growth incorporates both cash generation and dilution.",
    ),
    MetricRule(
        "fcf_margin",
        0.18,
        ((-0.05, 0.0), (0.0, 10.0), (0.05, 40.0), (0.15, 75.0), (0.25, 100.0)),
        interpretation="Higher free-cash-flow margin provides an operating cushion.",
    ),
    MetricRule(
        "gross_margin_stability",
        0.12,
        ((0.0, 100.0), (0.01, 90.0), (0.03, 65.0), (0.07, 20.0), (0.10, 0.0)),
        interpretation="Lower standard deviation of recent gross margins is more resilient.",
    ),
    MetricRule(
        "share_dilution",
        0.10,
        ((-0.05, 100.0), (0.0, 90.0), (0.03, 60.0), (0.08, 10.0), (0.10, 0.0)),
        interpretation="Lower share-count growth preserves each owner's claim on FCF.",
    ),
    MetricRule(
        "roic",
        0.10,
        ((-0.05, 0.0), (0.0, 10.0), (0.05, 35.0), (0.10, 65.0), (0.20, 100.0)),
        aliases=("roic", "roic_proxy"),
        interpretation="Reported ROIC is preferred; otherwise the labelled NOPAT/invested-capital proxy is used.",
    ),
)


VALUATION_COMPRESSION_RULES = (
    MetricRule(
        "ev_to_fcf",
        0.20,
        ((5.0, 100.0), (8.0, 90.0), (15.0, 65.0), (25.0, 25.0), (40.0, 0.0)),
        interpretation="Lower positive EV/FCF is cheaper.",
    ),
    MetricRule(
        "fcf_yield",
        0.25,
        ((0.0, 0.0), (0.02, 10.0), (0.05, 50.0), (0.08, 75.0), (0.12, 100.0)),
        interpretation="Higher positive FCF yield is cheaper.",
    ),
    MetricRule(
        "pe_ratio",
        0.10,
        ((7.0, 100.0), (10.0, 90.0), (20.0, 55.0), (35.0, 10.0), (50.0, 0.0)),
        interpretation="Lower positive earnings multiple is cheaper.",
    ),
    MetricRule(
        "valuation_compression",
        0.45,
        ((-0.20, 0.0), (0.0, 20.0), (0.20, 50.0), (0.40, 80.0), (0.60, 100.0)),
        interpretation="Current positive multiples below their own historical medians score higher.",
    ),
)


BALANCE_SHEET_QUALITY_RULES = (
    MetricRule(
        "net_debt_to_fcf",
        1.0,
        ((-2.0, 100.0), (0.0, 95.0), (1.0, 80.0), (2.0, 60.0), (3.0, 35.0), (5.0, 0.0)),
        interpretation="Net cash or fewer years of positive FCF needed to repay net debt is stronger.",
    ),
)


COMPONENT_RULES: dict[str, tuple[MetricRule, ...]] = {
    "price_dislocation": PRICE_DISLOCATION_RULES,
    "fundamental_resilience": FUNDAMENTAL_RESILIENCE_RULES,
    "valuation_compression": VALUATION_COMPRESSION_RULES,
    "balance_sheet_quality": BALANCE_SHEET_QUALITY_RULES,
}


DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "price_dislocation": 0.30,
    "fundamental_resilience": 0.35,
    "valuation_compression": 0.25,
    "balance_sheet_quality": 0.10,
}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _metric_value(metrics: Mapping[str, Any], rule: MetricRule) -> tuple[str | None, float]:
    keys = rule.aliases or (rule.metric,)
    for key in keys:
        value = metrics.get(key)
        if _finite(value):
            return key, float(value)
    return None, float("nan")


def score_metric(value: Any, rule: MetricRule) -> float:
    """Score one finite value using a rule's clipped linear interpolation."""

    if not _finite(value):
        return float("nan")
    points = sorted(rule.points, key=lambda pair: pair[0])
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    return float(np.clip(np.interp(float(value), x, y), 0.0, 100.0))


def score_component(
    metrics: Mapping[str, Any],
    name: str,
    rules: tuple[MetricRule, ...] | None = None,
) -> ComponentScore:
    """Score available evidence only and separately report weighted coverage."""

    selected_rules = rules if rules is not None else COMPONENT_RULES[name]
    total_weight = float(sum(rule.weight for rule in selected_rules))
    available_weight = 0.0
    weighted_points = 0.0
    explanations: list[dict[str, Any]] = []

    for rule in selected_rules:
        source_metric, raw_value = _metric_value(metrics, rule)
        metric_score = score_metric(raw_value, rule)
        available = _finite(metric_score)
        if available:
            available_weight += rule.weight
            weighted_points += rule.weight * metric_score
        explanations.append(
            {
                "metric": rule.metric,
                "source_metric": source_metric,
                "raw_value": raw_value if _finite(raw_value) else None,
                "metric_score": metric_score if available else None,
                "configured_weight": rule.weight,
                "available": available,
                "interpretation": rule.interpretation,
            }
        )

    score = weighted_points / available_weight if available_weight > 0 else float("nan")
    coverage = available_weight / total_weight if total_weight > 0 else 0.0
    return ComponentScore(
        name=name,
        score=float(score),
        coverage=float(coverage),
        available_weight=float(available_weight),
        total_weight=float(total_weight),
        explanations=tuple(explanations),
    )


def score_candidate(
    metrics: Mapping[str, Any],
    *,
    component_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return four scores, a deterministic quant composite and confidence.

    The composite is a weighted average of component scores that have at least
    one observed input. ``quant_coverage``/``quant_confidence`` (0--100) retain
    the penalty information instead of silently assigning neutral points to
    missing components.
    """

    weights = dict(component_weights or DEFAULT_COMPONENT_WEIGHTS)
    unknown = set(weights).difference(COMPONENT_RULES)
    if unknown:
        raise ValueError(f"Unknown scoring components: {sorted(unknown)}")
    if any((not _finite(weight) or float(weight) < 0) for weight in weights.values()):
        raise ValueError("Component weights must be finite and non-negative")

    components = {name: score_component(metrics, name) for name in weights}
    available_component_weight = sum(
        weights[name] for name, component in components.items() if _finite(component.score)
    )
    quant_score = (
        sum(
            weights[name] * component.score
            for name, component in components.items()
            if _finite(component.score)
        )
        / available_component_weight
        if available_component_weight > 0
        else float("nan")
    )
    total_component_weight = sum(weights.values())
    covered_weight = sum(
        weights[name] * component.coverage for name, component in components.items()
    )
    coverage = covered_weight / total_component_weight if total_component_weight > 0 else 0.0

    explanations = {
        name: {
            "score": component.score if _finite(component.score) else None,
            "coverage": component.coverage,
            "metrics": list(component.explanations),
        }
        for name, component in components.items()
    }
    result: dict[str, Any] = {
        "price_dislocation_score": components["price_dislocation"].score,
        "price_dislocation_coverage": components["price_dislocation"].coverage * 100.0,
        "fundamental_resilience_score": components["fundamental_resilience"].score,
        "fundamental_resilience_coverage": components["fundamental_resilience"].coverage * 100.0,
        "valuation_compression_score": components["valuation_compression"].score,
        "valuation_compression_coverage": components["valuation_compression"].coverage * 100.0,
        "balance_sheet_quality_score": components["balance_sheet_quality"].score,
        "balance_sheet_quality_coverage": components["balance_sheet_quality"].coverage * 100.0,
        "quant_score": float(quant_score),
        "quant_coverage": float(coverage * 100.0),
        "quant_confidence": float(coverage * 100.0),
        "scoring_model_version": SCORING_MODEL_VERSION,
        "score_explanations": explanations,
    }
    return result


def score_candidates(
    candidates: pd.DataFrame,
    *,
    component_weights: Mapping[str, float] | None = None,
    explanations_as_json: bool = True,
) -> pd.DataFrame:
    """Append quant scores to a candidate DataFrame without imputing metrics."""

    output = candidates.copy()
    scored = [
        score_candidate(row, component_weights=component_weights)
        for row in output.to_dict(orient="records")
    ]
    score_frame = pd.DataFrame(scored, index=output.index)
    if explanations_as_json and "score_explanations" in score_frame:
        score_frame["score_explanations"] = score_frame["score_explanations"].map(
            lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        )
    for column in score_frame:
        output[column] = score_frame[column]
    return output


__all__ = [
    "BALANCE_SHEET_QUALITY_RULES",
    "COMPONENT_RULES",
    "DEFAULT_COMPONENT_WEIGHTS",
    "FUNDAMENTAL_RESILIENCE_RULES",
    "MetricRule",
    "PRICE_DISLOCATION_RULES",
    "SCORING_MODEL_VERSION",
    "VALUATION_COMPRESSION_RULES",
    "score_candidate",
    "score_candidates",
    "score_component",
    "score_metric",
]
