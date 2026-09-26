# Narrative Dislocation

**A research screener for stocks whose prices have fallen further than their reported fundamentals might suggest.**

This project turns a broad stock universe into a short, inspectable research queue. It combines a deterministic quantitative screen with optional AI research into the reasons behind a selloff. It is a tool for asking better questions about a company, **not a buy list or a proven trading strategy**.

```text
Current universe → Drawdown filter → Financial data → Quant ranking
                                                   ↘ Optional AI research → Reports
```

## What it does

1. Loads the current S&P 500 and Nasdaq-100 constituents, with an optional user-supplied Russell 1000 CSV. You can also provide your own tickers.
2. Keeps stocks down **more than 25%** from their two-year high and with measured USD market capitalization **above $5 billion**. Both thresholds are configurable.
3. Fetches prices, company metadata, and annual financial statements. Missing values stay missing; currency mismatches block unsafe valuation ratios.
4. Scores four dimensions: price dislocation, fundamental resilience, valuation compression, and balance-sheet quality. Every metric, weight, and coverage value is available for inspection.
5. Optionally researches the highest-ranked quant candidates with the OpenAI Responses API, separating measured facts from narrative inference and source links.
6. Writes a ranked CSV, detailed JSON, and readable Markdown briefs.

## Quick start

Requires **Python 3.11+**.

```bash
python -m pip install -e .
python -m narrative_dislocation doctor
python -m narrative_dislocation screen --tickers PYPL,DIS,NKE --no-ai
```

The ticker list above is only a small functional check. To screen the default index universe without AI:

```bash
python -m narrative_dislocation screen --no-russell --no-ai
```

For optional AI research, set `OPENAI_API_KEY` in your environment or in a local `.env` based on [`.env.example`](.env.example), then run:

```bash
python -m narrative_dislocation screen --no-russell --ai-candidates 10
```

AI runs use billable API calls and may use web search. Start with a small universe or candidate limit. See `screen --help` for threshold, cache, model, output, and universe options.

## Reading the results

| Output | What to inspect |
| --- | --- |
| `outputs/<run-id>/top_20.csv` | Ranked candidates and flattened metrics. |
| `outputs/<run-id>/detailed_results.json` | Full measurements, missing-data coverage, scoring explanations, fiscal dates, source provenance, warnings, AI inference, and evidence links. |
| `outputs/<run-id>/top_10_report.md` | Short research briefs, including bear thesis, possible catalysts, risks, and thesis invalidators when AI research is available. |

The quant score weighs **price dislocation 30%**, **fundamental resilience 35%**, **valuation compression 25%**, and **balance-sheet quality 10%**. Missing inputs are not invented or awarded neutral points: coverage is reported separately and reduces the ranking score.

With completed AI research, the final Dislocation Score combines **70% coverage-adjusted quant evidence**, **24% AI-assessed Narrative Gap**, and **6% inverse AI-assessed value-trap probability**. AI-disabled results are marked `quant_only`; their provisional scores should not be compared as though they included narrative research. The formulas, breakpoints, and assumptions are documented in [the methodology](docs/methodology.md).

## Backtesting

The repository includes a separate point-in-time backtest engine. It requires **historical signal snapshots and adjusted prices supplied by the user**; today's screener output is not a historical signal dataset. Signals must include both `as_of` and `available_at`, and execution is delayed by at least one observed trading session. The engine can model transaction costs and configurable benchmarks.

There is **no historical performance claim** bundled with this repository. See the [backtest specification](docs/methodology.md#point-in-time-backtest-module) for input schemas, commands, and limitations.

## Data boundaries

- Index membership comes from current public Wikipedia tables; Russell 1000 membership requires your own CSV or direct CSV URL.
- Prices and statements come from Yahoo Finance through `yfinance`. These free sources can be incomplete, revised, stale, or rate-limited.
- The v1 quant model uses **annual** statement rows. It reports the latest fiscal period and its age, but does not treat a current snapshot as point-in-time historical data.
- AI narrative scores are interpretations, not measured accounting facts. Open the cited filings and other sources before relying on a claim.
- The backtest does not reconstruct delisting returns, spreads, market impact, taxes, or corporate actions. Missing returns for held assets are treated as zero and disclosed in diagnostics.

This is research software, not investment advice. A high score means **investigate further**, not **buy**.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

The package is split into data providers, metric calculations, scoring, AI research, reporting, and backtesting under `src/narrative_dislocation/`. Offline tests use fixtures and fake external clients; they do not establish investment performance.

## License

[MIT](LICENSE).
