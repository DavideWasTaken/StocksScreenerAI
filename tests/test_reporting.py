from __future__ import annotations

import json

import pandas as pd

from narrative_dislocation.reporting import (
    build_markdown_report,
    ranked_dataframe,
    write_detailed_json,
    write_reports,
)


def result(ticker: str, score: float, gap: int, *, missing_fcf: bool = False):
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Incorporated",
        "dislocation_score": score,
        "price_dislocation_score": score - 2,
        "narrative_analysis": {
            "measured_facts": {
                "ticker": ticker,
                "drawdown_from_2y_high": -0.35,
                "revenue_growth": 0.04,
                "fcf_growth": None if missing_fcf else 0.02,
                "fcf_margin": 0.15,
            },
            "ai_inference": {
                "why_it_fell": f"{ticker} lowered guidance.",
                "dominant_bearish_narrative": "Demand weakness is structural.",
                "expected_damage_if_bear_thesis_is_correct": [
                    "Revenue contracts",
                    "FCF margin falls",
                ],
                "observed_fundamental_damage": ["Supplied revenue_growth remains positive"],
                "catalysts": ["Guidance stabilization"],
                "structural_risks": ["Permanent market-share loss"],
                "value_trap_probability": 30,
                "narrative_gap_score": gap,
                "concise_thesis": "Observed damage is smaller than the selloff implies.",
                "invalidation_conditions": ["Sustained negative revenue growth"],
                "ai_inferences": ["Weak demand may be cyclical"],
                "evidence": [],
            },
            "evidence_sources": [
                {
                    "title": "Earnings release",
                    "url": f"https://example.com/{ticker.lower()}/earnings",
                    "supports": "The guidance revision",
                    "source_type": "url_citation",
                }
            ],
            "ai_metadata": {"status": "completed", "cache_hit": False},
        },
    }


def sample_results():
    return [
        result("LOW", 55, 50),
        result("HIGH", 91, 84, missing_fcf=True),
        result("MID", 73, 68),
    ]


def test_ranked_dataframe_sorts_descending_flattens_and_limits():
    frame = ranked_dataframe(pd.DataFrame(sample_results()), top_n=2)

    assert list(frame["ticker"]) == ["HIGH", "MID"]
    assert list(frame["rank"]) == [1, 2]
    assert "narrative_analysis.ai_inference.narrative_gap_score" in frame.columns
    assert frame.loc[0, "narrative_analysis.ai_inference.narrative_gap_score"] == 84


def test_detailed_json_preserves_nested_fact_and_inference_blocks(tmp_path):
    path = write_detailed_json(
        sample_results(),
        tmp_path / "detailed.json",
        run_metadata={"run_id": "test-run"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.is_absolute()
    assert payload["count"] == 3
    assert payload["run_metadata"]["run_id"] == "test-run"
    assert payload["candidates"][0]["ticker"] == "HIGH"
    assert payload["candidates"][0]["narrative_analysis"]["measured_facts"]["fcf_growth"] is None
    assert (
        payload["candidates"][0]["narrative_analysis"]["ai_inference"]["narrative_gap_score"] == 84
    )


def test_markdown_has_requested_sections_labels_missing_data_and_links():
    markdown = build_markdown_report(sample_results(), top_n=1)

    assert "## 1. HIGH — HIGH Incorporated" in markdown
    assert "### Why it fell" in markdown
    assert "### What the market believes" in markdown
    assert "### What fundamentals show" in markdown
    assert "### Why it may be mispriced" in markdown
    assert "### What invalidates the thesis" in markdown
    assert "**Measured facts supplied by the quant pipeline:**" in markdown
    assert "**AI inference:** HIGH lowered guidance." in markdown
    assert "`fcf_growth`: not supplied" in markdown
    assert "[Earnings release](https://example.com/high/earnings)" in markdown
    assert "LOW Incorporated" not in markdown


def test_write_reports_returns_complete_bundle(tmp_path):
    paths = write_reports(sample_results(), tmp_path / "run-1")

    assert set(paths) == {"ranked_csv", "detailed_json", "markdown_report"}
    assert paths["ranked_csv"].name == "top_20.csv"
    assert paths["detailed_json"].name == "detailed_results.json"
    assert paths["markdown_report"].name == "top_10_report.md"
    assert all(path.exists() for path in paths.values())


def test_missing_score_preserves_caller_order():
    frame = ranked_dataframe([{"ticker": "B"}, {"ticker": "A"}], top_n=None)
    assert list(frame["ticker"]) == ["B", "A"]
