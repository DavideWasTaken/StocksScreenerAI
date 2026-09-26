from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from narrative_dislocation.ai import (
    ClaimEvidence,
    MissingOpenAIKeyError,
    NarrativeAnalysisError,
    NarrativeAnalyzer,
    NarrativeAssessment,
    configured_model,
    narrative_results_to_dataframe,
)


def assessment(*, gap: int = 78) -> NarrativeAssessment:
    return NarrativeAssessment(
        why_it_fell="A guidance cut and weaker near-term demand changed sentiment.",
        dominant_bearish_narrative="The slowdown is structural rather than cyclical.",
        expected_damage_if_bear_thesis_is_correct=[
            "Sustained negative revenue growth",
            "Material and persistent FCF-margin erosion",
        ],
        observed_fundamental_damage=[
            "Supplied revenue_growth is 0.03",
            "Supplied fcf_margin is 0.18",
        ],
        catalysts=["Demand stabilization", "Execution against cost targets"],
        structural_risks=["Permanent share loss"],
        value_trap_probability=34,
        narrative_gap_score=gap,
        concise_thesis="The price reaction appears larger than measured deterioration.",
        invalidation_conditions=["Revenue growth turns persistently negative"],
        ai_inferences=["The market appears to extrapolate the weak quarter."],
        evidence=[
            ClaimEvidence(
                title="Company earnings release",
                url="https://investor.example.com/results",
                supports="The guidance change and reported quarter",
            )
        ],
    )


def fake_response(*, gap: int = 78):
    return SimpleNamespace(
        id="resp_test",
        output_parsed=assessment(gap=gap),
        usage=SimpleNamespace(input_tokens=100, output_tokens=200, total_tokens=300),
        output=[
            {
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "sources": [
                        {
                            "type": "url",
                            "title": "SEC filing",
                            "url": "https://www.sec.gov/Archives/example",
                        }
                    ],
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "{}",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Company earnings release",
                                "url": "https://investor.example.com/results",
                            }
                        ],
                    }
                ],
            },
        ],
    )


class RecordingResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


def test_analyze_candidate_uses_responses_parse_web_search_and_cache(tmp_path):
    responses = RecordingResponses(fake_response())
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses),
        model="test-model",
        cache_dir=tmp_path,
        retry_base_delay=0,
    )
    candidate = {
        "ticker": "acme",
        "company_name": "ACME Corp",
        "drawdown_from_2y_high": -0.41,
        "revenue_growth": 0.03,
        "fcf_margin": 0.18,
        "pe": float("nan"),
    }

    first = analyzer.analyze_candidate(candidate)
    cache_path = tmp_path / "ACME.json"
    cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    cached_payload["result"]["ai_metadata"]["used_web_search"] = False
    cache_path.write_text(json.dumps(cached_payload), encoding="utf-8")
    second = analyzer.analyze_candidate(candidate)

    assert len(responses.calls) == 1
    request = responses.calls[0]
    assert request["model"] == "test-model"
    assert request["text_format"] is NarrativeAssessment
    assert request["tools"] == [{"type": "web_search"}]
    assert request["include"] == ["web_search_call.action.sources"]
    assert first["ticker"] == "ACME"
    assert first["measured_facts"]["pe"] is None
    assert first["ai_inference"]["narrative_gap_score"] == 78
    assert first["ai_metadata"]["cache_hit"] is False
    assert first["ai_metadata"]["web_search_enabled"] is True
    assert first["ai_metadata"]["used_web_search"] is True
    assert second["ai_metadata"]["cache_hit"] is True
    assert second["ai_metadata"]["used_web_search"] is True
    assert {item["url"] for item in first["evidence_sources"]} == {
        "https://investor.example.com/results",
        "https://www.sec.gov/Archives/example",
    }
    assert cache_path.exists()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["result"]["measured_facts"]["pe"] is None


def test_cache_ignores_run_bookkeeping_and_refreshes_current_measured_facts(tmp_path):
    responses = RecordingResponses(fake_response())
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses),
        model="test-model",
        cache_dir=tmp_path,
        retry_base_delay=0,
    )
    first_candidate = {
        "ticker": "ACME",
        "as_of": "2026-08-29T10:00:00+00:00",
        "revenue_growth": 0.03,
        "data_provenance": {
            "retrieved_at": "2026-08-29T09:55:00+00:00",
            "source_cached": False,
        },
    }
    resumed_candidate = {
        **first_candidate,
        "as_of": "2026-08-29T10:05:00+00:00",
        "data_provenance": {
            **first_candidate["data_provenance"],
            "source_cached": True,
        },
    }

    first = analyzer.analyze_candidate(first_candidate)
    resumed = analyzer.analyze_candidate(resumed_candidate)

    assert len(responses.calls) == 1
    assert first["ai_metadata"]["cache_hit"] is False
    assert resumed["ai_metadata"]["cache_hit"] is True
    assert resumed["measured_facts"]["as_of"] == resumed_candidate["as_of"]
    assert resumed["measured_facts"]["data_provenance"]["source_cached"] is True


def test_web_search_can_be_disabled(tmp_path):
    responses = RecordingResponses(fake_response())
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses),
        cache_dir=tmp_path,
        use_web_search=False,
    )

    result = analyzer.analyze_candidate({"ticker": "NOWEB"})

    request = responses.calls[0]
    assert "tools" not in request
    assert "tool_choice" not in request
    assert "include" not in request
    assert result["ai_metadata"]["web_search_enabled"] is False
    assert result["ai_metadata"]["used_web_search"] is False


def test_zero_cache_ttl_forces_fresh_analysis(tmp_path):
    responses = RecordingResponses(fake_response())
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses),
        cache_dir=tmp_path,
        cache_ttl_hours=0,
        use_web_search=False,
    )

    analyzer.analyze_candidate({"ticker": "FRESH"})
    analyzer.analyze_candidate({"ticker": "FRESH"})

    assert len(responses.calls) == 2


def test_older_sdk_create_fallback_stays_on_responses_api(tmp_path):
    class CreateOnlyResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="resp_create",
                output_text=assessment().model_dump_json(),
                output=[],
                usage=None,
            )

    responses = CreateOnlyResponses()
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses), cache_dir=tmp_path, use_web_search=False
    )

    result = analyzer.analyze_candidate({"ticker": "LEGACY"})

    assert result["ai_inference"]["narrative_gap_score"] == 78
    assert "text_format" not in responses.calls[0]
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert responses.calls[0]["text"]["format"]["strict"] is True


def test_retries_transient_failures_without_sleep(tmp_path):
    class FlakyResponses:
        def __init__(self):
            self.attempts = 0

        def parse(self, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise RuntimeError("temporary rate limit")
            return fake_response(gap=65)

    responses = FlakyResponses()
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses),
        cache_dir=tmp_path,
        max_retries=2,
        retry_base_delay=0,
    )

    result = analyzer.analyze_candidate({"ticker": "RETRY"})

    assert responses.attempts == 3
    assert result["ai_inference"]["narrative_gap_score"] == 65


def test_exhausted_retries_raise_clear_error(tmp_path):
    class BrokenResponses:
        def parse(self, **kwargs):
            raise RuntimeError("offline")

    analyzer = NarrativeAnalyzer(
        client=FakeClient(BrokenResponses()),
        cache_dir=tmp_path,
        max_retries=1,
        retry_base_delay=0,
    )

    with pytest.raises(NarrativeAnalysisError, match="failed for FAIL after 2 attempt"):
        analyzer.analyze_candidate({"ticker": "FAIL"})


def test_dataframe_batch_analyzes_only_requested_limit_and_flattens(tmp_path):
    responses = RecordingResponses(fake_response())
    analyzer = NarrativeAnalyzer(
        client=FakeClient(responses), cache_dir=tmp_path, use_web_search=False
    )
    candidates = pd.DataFrame(
        [
            {"ticker": "ONE", "fcf_margin": 0.1},
            {"ticker": "TWO", "fcf_margin": 0.2},
        ]
    )

    results = analyzer.analyze_candidates(candidates, limit=1)
    frame = narrative_results_to_dataframe(results)

    assert len(responses.calls) == 1
    assert list(frame["ticker"]) == ["ONE"]
    assert frame.loc[0, "narrative_gap_score"] == 78
    assert frame.loc[0, "fcf_margin"] == pytest.approx(0.1)


def test_configured_model_reads_environment_at_call_time(monkeypatch):
    monkeypatch.delenv("NARRATIVE_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert configured_model() == "env-model"
    monkeypatch.setenv("NARRATIVE_MODEL", "narrative-model")
    assert configured_model() == "narrative-model"


def test_missing_api_key_fails_without_retry_sleep(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "narrative_dislocation.ai.time.sleep",
        lambda _: pytest.fail("missing credentials must not be retried"),
    )
    analyzer = NarrativeAnalyzer(cache_dir=tmp_path, max_retries=2)

    with pytest.raises(MissingOpenAIKeyError, match="OPENAI_API_KEY"):
        analyzer.analyze_candidate({"ticker": "NOKEY"})
