"""End-to-end orchestration for a current Narrative Dislocation screen."""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from narrative_dislocation.adapters import snapshot_to_quant_inputs
from narrative_dislocation.ai import MissingOpenAIKeyError, NarrativeAnalyzer
from narrative_dislocation.config import AppConfig
from narrative_dislocation.data.universe import UniverseLoader, normalize_symbol
from narrative_dislocation.data.yfinance_provider import YFinanceProvider
from narrative_dislocation.domain import json_safe
from narrative_dislocation.final_score import (
    combine_dislocation_score,
    quant_evidence_score,
)
from narrative_dislocation.quant import QuantConfig, calculate_drawdown, compute_quant_metrics
from narrative_dislocation.reporting import write_reports
from narrative_dislocation.scoring import score_candidates

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(slots=True)
class ScreenOptions:
    """Per-run controls; stable policy defaults live in :class:`AppConfig`."""

    tickers: tuple[str, ...] = ()
    include_russell: bool = True
    russell_source: str | Path | None = None
    enable_ai: bool = True
    force_refresh: bool = False
    universe_limit: int | None = None
    run_id: str | None = None

    def validate(self) -> None:
        if self.universe_limit is not None and self.universe_limit <= 0:
            raise ValueError("universe_limit must be positive")
        if self.run_id is not None:
            if (
                not isinstance(self.run_id, str)
                or not _SAFE_RUN_ID.fullmatch(self.run_id)
                or self.run_id.endswith(".")
            ):
                raise ValueError(
                    "run_id must be one safe path component (1-128 letters, digits, dots, "
                    "underscores, or hyphens; starting with a letter or digit)"
                )
            if self.run_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
                raise ValueError(f"run_id uses a reserved filename: {self.run_id}")


@dataclass(slots=True)
class ScreenRunResult:
    run_id: str
    output_dir: Path
    candidates: list[dict[str, Any]]
    output_paths: dict[str, Path]
    run_metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class ScreenerPipeline:
    """Cache-first current screener with injectable external dependencies."""

    def __init__(
        self,
        config: AppConfig,
        *,
        universe_loader: UniverseLoader | None = None,
        data_provider: YFinanceProvider | None = None,
        narrative_analyzer: NarrativeAnalyzer | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        cache_ttl = timedelta(hours=config.cache_ttl_hours)
        self.universe_loader = universe_loader or UniverseLoader(
            cache_dir=config.cache_dir,
            cache_ttl=max(cache_ttl, timedelta(days=1)),
        )
        self.data_provider = data_provider or YFinanceProvider(
            cache_dir=config.cache_dir,
            cache_ttl=cache_ttl,
        )
        self.narrative_analyzer = narrative_analyzer
        self.progress = progress or (lambda phase, message: LOGGER.info("[%s] %s", phase, message))

    def run(self, options: ScreenOptions | None = None) -> ScreenRunResult:
        options = options or ScreenOptions()
        options.validate()
        self.config.validate()
        if (
            options.enable_ai
            and self.narrative_analyzer is None
            and not os.getenv("OPENAI_API_KEY")
        ):
            raise MissingOpenAIKeyError(
                "OPENAI_API_KEY is required for AI analysis; set it in the environment "
                "or rerun with --no-ai"
            )
        self.config.prepare_directories()
        started_at = datetime.now(UTC)
        run_id = options.run_id or _new_run_id(started_at)
        output_root = self.config.output_dir.resolve()
        output_dir = (output_root / run_id).resolve()
        if output_dir.parent != output_root:
            raise ValueError("run_id resolves outside the configured output directory")
        warnings: list[str] = []

        self.progress("universe", "Loading and deduplicating index constituents")
        symbols, member_metadata, universe_meta, universe_warnings = self._load_universe(options)
        warnings.extend(universe_warnings)
        if options.universe_limit is not None:
            symbols = symbols[: options.universe_limit]
        if not symbols:
            raise RuntimeError("The selected universe is empty")

        self.progress(
            "prices",
            f"Fetching two-year price histories for {len(symbols)} symbols (cache first)",
        )
        pre_prices = self.data_provider.fetch_price_histories(
            symbols,
            lookback_years=2,
            force_refresh=options.force_refresh,
        )
        collapsed: list[str] = []
        for symbol in symbols:
            prices = pre_prices.prices.get(symbol)
            if prices is None:
                continue
            drawdown = calculate_drawdown(prices, lookback_days=730)
            if pd.notna(drawdown) and float(drawdown) > self.config.min_drawdown:
                collapsed.append(symbol)
        warnings.extend(f"{symbol}: {error}" for symbol, error in pre_prices.errors.items())
        self.progress(
            "pre-screen",
            f"{len(collapsed)} of {len(symbols)} symbols exceed the drawdown gate",
        )

        records: list[dict[str, Any]] = []
        fetch_errors: dict[str, str] = {}
        completed_at = datetime.now(UTC)
        if collapsed:
            self.progress(
                "fundamentals",
                f"Fetching statements and longer price history for {len(collapsed)} symbols",
            )
            batch = self.data_provider.fetch_many(
                collapsed,
                lookback_years=self.config.price_history_years,
                force_refresh=options.force_refresh,
                on_progress=self._data_progress,
            )
            fetch_errors = dict(batch.errors)
            completed_at = batch.completed_at
            records = self._quant_records(
                batch.snapshots,
                member_metadata,
                as_of=completed_at,
                force_refresh=options.force_refresh,
            )
        warnings.extend(f"{symbol}: {error}" for symbol, error in fetch_errors.items())

        candidates = self._score_and_rank(records)
        quant_candidate_count = len(candidates)
        if options.enable_ai and candidates:
            candidates = self._add_ai(candidates, force_refresh=options.force_refresh)
        else:
            candidates = self._add_final_scores(candidates, {})

        candidates.sort(key=_final_rank_key)
        for rank, record in enumerate(candidates, start=1):
            record["rank"] = rank

        completed_at = datetime.now(UTC)
        run_metadata = {
            "run_id": run_id,
            "status": "completed",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "universe": universe_meta,
            "universe_size": len(symbols),
            "drawdown_pre_screen_count": len(collapsed),
            "fundamentals_fetched_count": len(records),
            "hard_screen_excluded_count": max(0, len(records) - quant_candidate_count),
            "quant_candidate_count": quant_candidate_count,
            "ai_analyzed_count": sum(
                1 for row in candidates if row.get("score_status") == "quant_plus_ai"
            ),
            "configuration": json_safe(asdict(self.config)),
            "warnings": warnings,
            "data_policy": (
                "Missing values remain null; current Yahoo snapshots are not claimed to be "
                "point-in-time historical fundamentals."
            ),
        }
        self.progress("output", f"Writing ranked outputs to {output_dir}")
        paths = write_reports(
            candidates,
            output_dir,
            ranked_top_n=self.config.final_top,
            report_top_n=self.config.report_top,
            score_column="dislocation_score",
            run_metadata=run_metadata,
        )
        return ScreenRunResult(
            run_id=run_id,
            output_dir=output_dir.resolve(),
            candidates=candidates,
            output_paths=paths,
            run_metadata=run_metadata,
            warnings=warnings,
        )

    def _load_universe(
        self, options: ScreenOptions
    ) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any], list[str]]:
        if options.tickers:
            symbols = list(
                dict.fromkeys(
                    symbol for raw in options.tickers if (symbol := normalize_symbol(raw))
                )
            )
            return (
                symbols,
                {symbol: {"memberships": ["user_supplied"]} for symbol in symbols},
                {"kind": "user_supplied", "symbols": len(symbols)},
                [],
            )

        snapshot = self.universe_loader.load(
            include_russell=options.include_russell,
            russell_source=options.russell_source,
            force_refresh=options.force_refresh,
        )
        members = {
            member.symbol: {
                "company_name": member.company_name,
                "sector": member.sector,
                "industry": member.industry,
                "memberships": list(member.memberships),
                "universe_source_urls": list(member.source_urls),
            }
            for member in snapshot.members
        }
        meta = {
            "kind": "index_constituents",
            "as_of": snapshot.as_of.isoformat(),
            "sources": [source.to_dict() for source in snapshot.sources],
        }
        return list(snapshot.symbols), members, meta, list(snapshot.warnings)

    def _quant_records(
        self,
        snapshots: Mapping[str, Any],
        member_metadata: Mapping[str, Mapping[str, Any]],
        *,
        as_of: datetime,
        force_refresh: bool,
    ) -> list[dict[str, Any]]:
        adapted = {
            symbol: snapshot_to_quant_inputs(snapshot) for symbol, snapshot in snapshots.items()
        }
        sectors = {
            str(
                inputs.metadata.get("sector") or member_metadata.get(symbol, {}).get("sector") or ""
            )
            for symbol, inputs in adapted.items()
        }
        sector_tickers = {
            benchmark
            for sector in sectors
            if (benchmark := self.config.sector_benchmarks.get(sector))
        }
        benchmark_symbols = {self.config.benchmark_ticker, *sector_tickers}
        benchmarks = self.data_provider.fetch_benchmark_prices(
            sorted(benchmark_symbols),
            lookback_years=2,
            force_refresh=force_refresh,
        )
        index_prices = benchmarks.prices.get(self.config.benchmark_ticker)
        quant_config = QuantConfig(
            minimum_market_cap=self.config.min_market_cap,
            minimum_drawdown=self.config.min_drawdown,
        )

        rows: list[dict[str, Any]] = []
        for symbol, inputs in adapted.items():
            member = dict(member_metadata.get(symbol, {}))
            sector = str(inputs.metadata.get("sector") or member.get("sector") or "") or None
            sector_benchmark = self.config.sector_benchmarks.get(sector or "")
            data_warnings = list(inputs.warnings)
            market_cap_currency = inputs.quote_currency
            if inputs.screening_market_cap is not None and market_cap_currency != "USD":
                data_warnings.append(
                    "The measured market-cap currency is not confirmed as USD; the $5B "
                    "hard gate cannot be verified"
                )
            for benchmark in filter(None, (self.config.benchmark_ticker, sector_benchmark)):
                if benchmark in benchmarks.errors:
                    data_warnings.append(
                        f"Benchmark {benchmark} unavailable: {benchmarks.errors[benchmark]}"
                    )
                elif benchmark not in benchmarks.prices:
                    data_warnings.append(f"Benchmark {benchmark} returned no price history")
            metrics = compute_quant_metrics(
                inputs.fundamentals,
                prices=inputs.prices,
                valuation_history=inputs.valuation_history,
                sector_prices=benchmarks.prices.get(sector_benchmark) if sector_benchmark else None,
                index_prices=index_prices,
                as_of=as_of,
                config=quant_config,
            )
            # The universe threshold is explicitly denominated in US dollars.
            # Keep the raw measured cap for audit, but only expose it as the
            # hard-screen market_cap when Yahoo confirms USD quote currency.
            metrics["market_cap"] = (
                inputs.screening_market_cap
                if inputs.screening_market_cap is not None and market_cap_currency == "USD"
                else float("nan")
            )
            market_name = inputs.metadata.get("company_name") or member.get("company_name")
            row = {
                "ticker": symbol,
                "company_name": market_name,
                "sector": sector,
                "industry": inputs.metadata.get("industry") or member.get("industry"),
                "memberships": member.get("memberships", []),
                "as_of": as_of.isoformat(),
                "sector_benchmark": sector_benchmark,
                "screening_market_cap": inputs.screening_market_cap,
                "market_cap_currency": market_cap_currency,
                "financial_currency": inputs.financial_currency,
                "ratio_currency_status": (
                    "matched"
                    if inputs.ratio_currency_compatible is True
                    else "mismatch"
                    if inputs.ratio_currency_compatible is False
                    else "unknown"
                ),
                **metrics,
                "data_provenance": inputs.provenance,
                "data_warnings": data_warnings,
            }
            passed, reasons = _hard_screen(row, self.config)
            row["screen_pass"] = passed
            row["screen_reasons"] = ";".join(reasons)
            rows.append(row)
        return rows

    def _score_and_rank(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [record for record in records if record.get("screen_pass")]
        if not eligible:
            return []
        scored = score_candidates(pd.DataFrame.from_records(eligible), explanations_as_json=False)
        scored["quant_evidence_score"] = scored.apply(
            lambda row: quant_evidence_score(row.get("quant_score"), row.get("quant_coverage")),
            axis=1,
        )
        scored = scored.sort_values(
            ["quant_evidence_score", "quant_coverage", "ticker"],
            ascending=[False, False, True],
            na_position="last",
        )
        return [json_safe(record) for record in scored.to_dict(orient="records")]

    def _add_ai(
        self, candidates: list[dict[str, Any]], *, force_refresh: bool
    ) -> list[dict[str, Any]]:
        analyzer = self.narrative_analyzer
        if analyzer is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise MissingOpenAIKeyError(
                    "OPENAI_API_KEY is required for AI analysis; set it in the environment "
                    "or rerun with --no-ai"
                )
            analyzer = NarrativeAnalyzer(
                model=self.config.openai_model,
                cache_dir=self.config.cache_dir / "ai",
                use_web_search=self.config.ai_web_search,
                cache_ttl_hours=self.config.cache_ttl_hours,
            )
        selected = candidates[: self.config.ai_candidates]
        facts = [_facts_for_ai(candidate) for candidate in selected]
        self.progress("ai", f"Researching the top {len(facts)} quant candidates")
        results = analyzer.analyze_candidates(
            facts,
            force_refresh=force_refresh,
            continue_on_error=True,
        )
        by_ticker = {str(result.get("ticker")): result for result in results}
        return self._add_final_scores(
            candidates,
            by_ticker,
            require_ai=True,
            selected_tickers={str(candidate.get("ticker")) for candidate in selected},
        )

    @staticmethod
    def _add_final_scores(
        candidates: list[dict[str, Any]],
        analyses: Mapping[str, Mapping[str, Any]],
        *,
        require_ai: bool = False,
        selected_tickers: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for candidate in candidates:
            record = dict(candidate)
            analysis = analyses.get(str(record.get("ticker")))
            inference: Mapping[str, Any] = {}
            if analysis:
                record.update(json_safe(dict(analysis)))
                raw_inference = analysis.get("ai_inference")
                if isinstance(raw_inference, Mapping):
                    inference = raw_inference
            record.update(
                combine_dislocation_score(
                    quant_score=record.get("quant_score"),
                    quant_coverage=record.get("quant_coverage"),
                    narrative_gap_score=inference.get("narrative_gap_score"),
                    value_trap_probability=inference.get("value_trap_probability"),
                )
            )
            if require_ai and not inference:
                selected = str(record.get("ticker")) in (selected_tickers or set())
                record["dislocation_score"] = None
                record["ai_score"] = None
                record["score_status"] = "ai_error" if selected else "not_ai_analyzed"
            output.append(record)
        return output

    def _data_progress(self, completed: int, total: int, symbol: str, error: str | None) -> None:
        suffix = f" (error: {error})" if error else ""
        self.progress("fundamentals", f"{completed}/{total} {symbol}{suffix}")


def _hard_screen(record: Mapping[str, Any], config: AppConfig) -> tuple[bool, list[str]]:
    from narrative_dislocation.quant import evaluate_hard_screen

    return evaluate_hard_screen(
        record,
        minimum_market_cap=config.min_market_cap,
        minimum_drawdown=config.min_drawdown,
    )


_AI_FACT_FIELDS = (
    "ticker",
    "company_name",
    "sector",
    "industry",
    "memberships",
    "as_of",
    "latest_fundamental_period_end",
    "prior_fundamental_period_end",
    "latest_fundamental_available_at",
    "latest_fundamental_age_days",
    "fundamental_period_type",
    "drawdown_2y",
    "revenue_growth",
    "fcf_growth",
    "fcf_per_share_growth",
    "fcf_margin",
    "gross_margin",
    "gross_margin_stability",
    "gross_margin_change",
    "share_dilution",
    "net_debt_to_fcf",
    "roic",
    "roic_proxy",
    "ev_to_fcf",
    "fcf_yield",
    "pe_ratio",
    "valuation_compression",
    "relative_return_sector_1y",
    "relative_return_index_1y",
    "market_cap",
    "screening_market_cap",
    "market_cap_currency",
    "financial_currency",
    "ratio_currency_status",
    "revenue",
    "free_cash_flow",
    "price_dislocation_score",
    "fundamental_resilience_score",
    "valuation_compression_score",
    "balance_sheet_quality_score",
    "quant_score",
    "quant_coverage",
    "quant_evidence_score",
    "data_provenance",
    "data_warnings",
)


def _facts_for_ai(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return json_safe({field: candidate.get(field) for field in _AI_FACT_FIELDS})


def _new_run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def _sortable_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return parsed if pd.notna(parsed) else float("-inf")


def _final_rank_key(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Keep full AI scores comparable; rank provisional evidence only afterward."""

    status = str(record.get("score_status") or "")
    status_order = {
        "quant_plus_ai": 0.0,
        "quant_only": 0.0,
        "ai_error": 1.0,
        "not_ai_analyzed": 2.0,
        "insufficient_quant_data": 3.0,
    }.get(status, 3.0)
    primary = (
        record.get("dislocation_score")
        if status in {"quant_plus_ai", "quant_only"}
        else record.get("quant_evidence_score")
    )
    return (
        status_order,
        -_sortable_score(primary),
        -_sortable_score(record.get("quant_coverage")),
        str(record.get("ticker", "")),
    )


def parse_tickers(values: Iterable[str]) -> tuple[str, ...]:
    """Parse repeated or comma-separated ticker arguments."""

    flattened: list[str] = []
    for value in values:
        flattened.extend(part for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(symbol for item in flattened if (symbol := normalize_symbol(item))))


__all__ = ["ScreenOptions", "ScreenRunResult", "ScreenerPipeline", "parse_tickers"]
