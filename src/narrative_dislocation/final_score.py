"""Final, explicit combination of quant evidence and narrative assessment."""

from __future__ import annotations

import math
from typing import Any

FINAL_SCORE_VERSION = "dislocation-v1"
QUANT_WEIGHT = 0.70
NARRATIVE_GAP_WEIGHT = 0.24
INVERSE_VALUE_TRAP_WEIGHT = 0.06


def _score(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return min(100.0, max(0.0, parsed))


def quant_evidence_score(quant_score: Any, quant_coverage: Any) -> float | None:
    """Coverage-adjust a quant score without pretending missing inputs are neutral.

    The observed-only quant score retains at least half its weight; complete
    evidence receives full weight. Both the raw score and coverage remain in
    output so this ranking adjustment is auditable.
    """

    score = _score(quant_score)
    coverage = _score(quant_coverage)
    if score is None or coverage is None:
        return None
    coverage_multiplier = 0.5 + 0.5 * coverage / 100.0
    return round(score * coverage_multiplier, 4)


def combine_dislocation_score(
    *,
    quant_score: Any,
    quant_coverage: Any,
    narrative_gap_score: Any = None,
    value_trap_probability: Any = None,
) -> dict[str, Any]:
    """Return the final 0--100 score and its calculation metadata.

    With completed AI research the weights are 70% coverage-adjusted quant,
    24% Narrative Gap, and 6% inverse value-trap probability. A quant-only run
    receives a clearly labelled provisional score, not an invented AI value.
    """

    quant_evidence = quant_evidence_score(quant_score, quant_coverage)
    gap = _score(narrative_gap_score)
    trap = _score(value_trap_probability)
    if quant_evidence is None:
        return {
            "dislocation_score": None,
            "quant_evidence_score": None,
            "ai_score": None,
            "score_status": "insufficient_quant_data",
            "final_score_version": FINAL_SCORE_VERSION,
        }
    if gap is None or trap is None:
        return {
            "dislocation_score": quant_evidence,
            "quant_evidence_score": quant_evidence,
            "ai_score": None,
            "score_status": "quant_only",
            "final_score_version": FINAL_SCORE_VERSION,
        }

    inverse_trap = 100.0 - trap
    ai_score = (NARRATIVE_GAP_WEIGHT * gap + INVERSE_VALUE_TRAP_WEIGHT * inverse_trap) / (
        NARRATIVE_GAP_WEIGHT + INVERSE_VALUE_TRAP_WEIGHT
    )
    final = (
        QUANT_WEIGHT * quant_evidence
        + NARRATIVE_GAP_WEIGHT * gap
        + INVERSE_VALUE_TRAP_WEIGHT * inverse_trap
    )
    return {
        "dislocation_score": round(final, 4),
        "quant_evidence_score": quant_evidence,
        "ai_score": round(ai_score, 4),
        "score_status": "quant_plus_ai",
        "final_score_version": FINAL_SCORE_VERSION,
    }


__all__ = [
    "FINAL_SCORE_VERSION",
    "combine_dislocation_score",
    "quant_evidence_score",
]
