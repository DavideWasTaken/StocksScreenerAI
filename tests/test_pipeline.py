from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from narrative_dislocation.ai import MissingOpenAIKeyError
from narrative_dislocation.config import AppConfig
from narrative_dislocation.data.models import (
    BatchFetchResult,
    EquitySnapshot,
    FinancialStatements,
    PriceBatchResult,
    SourceMetadata,
)
from narrative_dislocation.pipeline import ScreenerPipeline, ScreenOptions

AS_OF = datetime(2025, 1, 10, tzinfo=UTC)


def _source() -> SourceMetadata:
    return SourceMetadata(
        name="offline fixture",
        url="https://example.test/data",
        retrieved_at=AS_OF,
        as_of=AS_OF,
    )


def _prices(last: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0, 90.0, last]},
        index=pd.to_datetime(["2023-01-11", "2024-06-01", "2025-01-09"]),
    )


def _statements() -> FinancialStatements:
    periods = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
    income = pd.DataFrame(
        {
            periods[0]: [4.0e9, 1.6e9, 1.0e8, 2.0],
            periods[1]: [4.5e9, 1.9e9, 1.0e8, 2.5],
            periods[2]: [5.0e9, 2.2e9, 9.8e7, 3.0],
        },
        index=["Total Revenue", "Gross Profit", "Diluted Average Shares", "Diluted EPS"],
    )
    cash = pd.DataFrame(
        {
            periods[0]: [5.0e8],
            periods[1]: [6.0e8],
            periods[2]: [7.0e8],
        },
        index=["Free Cash Flow"],
    )
    balance = pd.DataFrame(
        {
            periods[0]: [1.5e9, 3.0e8],
            periods[1]: [1.3e9, 4.0e8],
            periods[2]: [1.1e9, 5.0e8],
        },
        index=["Total Debt", "Cash And Cash Equivalents"],
    )
    return FinancialStatements(
        annual_income=income,
        annual_cash_flow=cash,
        annual_balance_sheet=balance,
    )


class FakeProvider:
    def __init__(
        self,
        *,
        quote_currency: str = "USD",
        financial_currency: str = "USD",
    ) -> None:
        self.price_map = {"AAA": _prices(60.0), "BBB": _prices(95.0)}
        self.full_fetch_symbols: list[str] = []
        self.quote_currency = quote_currency
        self.financial_currency = financial_currency

    def fetch_price_histories(self, symbols, **kwargs):
        selected = {symbol: self.price_map[symbol] for symbol in symbols}
        return PriceBatchResult(prices=selected, errors={}, source=_source())

    def fetch_many(self, symbols, **kwargs):
        self.full_fetch_symbols = list(symbols)
        snapshots = {
            symbol: EquitySnapshot(
                symbol=symbol,
                prices=self.price_map[symbol],
                metadata={
                    "company_name": "Alpha Corp",
                    "sector": "Technology",
                    "market_cap": 10.0e9,
                    "enterprise_value": 10.6e9,
                    "trailing_pe": 12.0,
                    "currency": self.quote_currency,
                    "financial_currency": self.financial_currency,
                },
                statements=_statements(),
                source=_source(),
            )
            for symbol in symbols
        }
        callback = kwargs.get("on_progress")
        if callback:
            for number, symbol in enumerate(symbols, start=1):
                callback(number, len(symbols), symbol, None)
        return BatchFetchResult(
            snapshots=snapshots,
            errors={},
            started_at=AS_OF,
            completed_at=AS_OF,
        )

    def fetch_benchmark_prices(self, symbols, **kwargs):
        prices = {
            symbol: pd.DataFrame(
                {"close": [100.0, 105.0]},
                index=pd.to_datetime(["2024-01-10", "2025-01-09"]),
            )
            for symbol in symbols
        }
        return PriceBatchResult(prices=prices, errors={}, source=_source())


class FakeAnalyzer:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def analyze_candidates(self, records, **kwargs):
        output = []
        for record in records:
            self.seen.append(record["ticker"])
            output.append(
                {
                    "ticker": record["ticker"],
                    "company_name": record["company_name"],
                    "measured_facts": record,
                    "ai_inference": {
                        "why_it_fell": "Fixture event",
                        "dominant_bearish_narrative": "Fixture bear case",
                        "expected_damage_if_bear_thesis_is_correct": ["Revenue declines"],
                        "observed_fundamental_damage": ["No decline in supplied revenue"],
                        "catalysts": ["Fixture catalyst"],
                        "structural_risks": ["Fixture risk"],
                        "value_trap_probability": 20,
                        "narrative_gap_score": 80,
                        "concise_thesis": "Fixture thesis",
                        "invalidation_conditions": ["Revenue declines"],
                        "ai_inferences": ["Fixture inference"],
                        "evidence": [],
                    },
                    "evidence_sources": [],
                    "ai_metadata": {"status": "completed", "error": None},
                }
            )
        return output


def test_pipeline_runs_two_stage_screen_ai_and_outputs(tmp_path):
    provider = FakeProvider()
    analyzer = FakeAnalyzer()
    config = AppConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        final_top=20,
        ai_candidates=20,
        report_top=10,
    )
    pipeline = ScreenerPipeline(
        config,
        data_provider=provider,
        narrative_analyzer=analyzer,
    )

    result = pipeline.run(ScreenOptions(tickers=("AAA", "BBB"), run_id="offline-test"))

    assert provider.full_fetch_symbols == ["AAA"]
    assert analyzer.seen == ["AAA"]
    assert len(result.candidates) == 1
    assert result.candidates[0]["score_status"] == "quant_plus_ai"
    assert result.candidates[0]["share_dilution"] < 0
    assert result.candidates[0]["pe_ratio"] == 12.0
    assert result.candidates[0]["latest_fundamental_period_end"].startswith("2024-12-31")
    assert result.candidates[0]["latest_fundamental_age_days"] == 10
    assert all(path.is_file() for path in result.output_paths.values())


def test_pipeline_fails_before_data_fetch_when_ai_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = FakeProvider()
    pipeline = ScreenerPipeline(
        AppConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out"),
        data_provider=provider,
    )

    with pytest.raises(MissingOpenAIKeyError, match="--no-ai"):
        pipeline.run(ScreenOptions(tickers=("AAA",)))

    assert provider.full_fetch_symbols == []


def test_ai_run_never_compares_unresearched_quant_score_with_final_score():
    candidates = [
        {"ticker": "AAA", "quant_score": 90.0, "quant_coverage": 100.0},
        {"ticker": "BBB", "quant_score": 88.0, "quant_coverage": 100.0},
    ]
    analyses = {
        "AAA": {
            "ticker": "AAA",
            "ai_inference": {
                "narrative_gap_score": 0,
                "value_trap_probability": 100,
            },
        }
    }

    scored = ScreenerPipeline._add_final_scores(
        candidates,
        analyses,
        require_ai=True,
        selected_tickers={"AAA"},
    )
    scored.sort(key=lambda row: (row["score_status"] != "quant_plus_ai", row["ticker"]))

    assert scored[0]["ticker"] == "AAA"
    assert scored[0]["dislocation_score"] == 63.0
    assert scored[1]["score_status"] == "not_ai_analyzed"
    assert scored[1]["dislocation_score"] is None
    assert scored[1]["quant_evidence_score"] == 88.0


def test_pipeline_uses_usd_cap_gate_but_withholds_cross_currency_ratios(tmp_path):
    provider = FakeProvider(financial_currency="EUR")
    pipeline = ScreenerPipeline(
        AppConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out"),
        data_provider=provider,
    )

    result = pipeline.run(ScreenOptions(tickers=("AAA",), enable_ai=False, run_id="currency-test"))

    candidate = result.candidates[0]
    assert candidate["market_cap"] == 10.0e9
    assert candidate["market_cap_currency"] == "USD"
    assert candidate["financial_currency"] == "EUR"
    assert candidate["ratio_currency_status"] == "mismatch"
    assert candidate["ev_to_fcf"] is None
    assert candidate["fcf_yield"] is None


@pytest.mark.parametrize(
    "run_id",
    ("../escape", "/tmp/escape", "nested/run", r"nested\run", ".", "A.", "CON", ""),
)
def test_run_id_must_be_one_portable_safe_path_component(run_id):
    with pytest.raises(ValueError, match="run_id"):
        ScreenOptions(run_id=run_id).validate()


def test_safe_run_id_is_accepted():
    ScreenOptions(run_id="smoke-2026.08_29").validate()
