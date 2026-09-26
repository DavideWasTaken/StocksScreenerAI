from narrative_dislocation.final_score import (
    combine_dislocation_score,
    quant_evidence_score,
)


def test_quant_evidence_score_penalizes_missing_coverage_transparently():
    assert quant_evidence_score(80, 100) == 80
    assert quant_evidence_score(80, 50) == 60
    assert quant_evidence_score(80, 0) == 40


def test_final_score_combines_quant_gap_and_inverse_value_trap():
    result = combine_dislocation_score(
        quant_score=80,
        quant_coverage=100,
        narrative_gap_score=90,
        value_trap_probability=20,
    )

    assert result["dislocation_score"] == 82.4
    assert result["ai_score"] == 88.0
    assert result["score_status"] == "quant_plus_ai"


def test_missing_ai_is_labelled_quant_only():
    result = combine_dislocation_score(quant_score=75, quant_coverage=80)

    assert result["dislocation_score"] == 67.5
    assert result["ai_score"] is None
    assert result["score_status"] == "quant_only"
