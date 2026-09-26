# Methodology and data contracts

This document describes the implemented `quant-v1` and `dislocation-v1` models. It is a specification of the current code, not a claim that these weights are optimal.

All rates and growth values are stored as decimals: `0.10` means 10%, and `-0.10` means −10%. Component and final scores are on 0–100 scales. Monetary values are used only when numerator and denominator are in compatible units. Unavailable or invalid measurements remain `null` in JSON and blank in CSV.

## Current-screen sequence

1. Load and deduplicate current S&P 500 and Nasdaq-100 constituents. Include a user-supplied Russell 1000 CSV when configured, or replace discovery entirely with `--tickers`.
2. Fetch two years of daily prices for every discovered symbol.
3. Keep names with a measured `drawdown_2y > 0.25`.
4. Fetch longer price history, company metadata, and statements only for those collapsed names.
5. Calculate the quant metrics below and enforce `market_cap > 5_000_000_000` plus the drawdown gate again. The market-cap gate is USD-denominated and passes only when Yahoo confirms the quote currency is `USD`; missing or unconfirmed gate values fail. Neither boundary is inclusive.
6. Calculate four component scores, the observed-only quant score, and quant coverage.
7. Rank by coverage-adjusted quant evidence and send only the first `ai_candidates` rows to narrative research.
8. Calculate the final Dislocation Score and rerank the complete eligible set.
9. Export the default Top 20 CSV, all eligible details in JSON, and default Top 10 Markdown briefs.

The default threshold values are configurable through `ND_MIN_DRAWDOWN` and `ND_MIN_MARKET_CAP`. The formulas and score breakpoints do not change with those gates.

## Data basis

### Universe

- S&P 500: current public constituent table on Wikipedia.
- Nasdaq-100: current public constituent table on Wikipedia.
- Russell 1000: current CSV supplied by the user as a local path or direct HTTP(S) URL. No Russell list is inferred when this is absent.
- Custom run: `--tickers` bypasses all three index sources.

Symbols are trimmed, uppercased, and converted from dotted to dashed share-class form for Yahoo Finance. Duplicate membership is merged. V1 trusts constituent membership and does not add a separate US domicile, exchange, security-type, liquidity, or primary-listing test.

### Prices and fundamentals

Yahoo Finance is accessed through `yfinance` with `auto_adjust=False`. The quant price selector prefers adjusted close, then close. Metadata includes market capitalization, enterprise value, shares, quote currency, financial-statement currency, sector, and other descriptive fields when returned. Both annual and quarterly statements are cached, but the v1 quant adapter uses comparable annual rows.

The `$5B` threshold is explicitly in US dollars. The raw measured value remains in `screening_market_cap` for audit, but it is exposed as the hard-gate `market_cap` only when quote currency is confirmed as `USD`; there is no implicit FX conversion. Quote-based valuation inputs that combine market data with statement values—market cap, enterprise value, FCF yield, EV/FCF, and fiscal-period historical proxies—are used only when quote and statement currencies are both present and equal. A vendor-reported trailing P/E can still be retained because it is already dimensionless. The record reports `market_cap_currency`, `financial_currency`, and `ratio_currency_status` as `matched`, `mismatch`, or `unknown`.

When Yahoo does not return a Free Cash Flow line but provides operating cash flow and capital expenditure, FCF is derived as:

```text
FCF = operating cash flow + capex                  when capex <= 0
FCF = operating cash flow - capex                  when capex > 0
```

The derived row is labeled. No other statement value is imputed.

Yahoo does not expose a dependable filing timestamp through this adapter. Every annual row is therefore conservatively marked `available_at = retrieval time`. This makes the current screen usable while preventing those rows from masquerading as historically available fundamentals.

### Benchmarks for the current screen

Broad-index relative return uses `SPY` by default; `ND_BENCHMARK` changes it. Sector-relative return uses the following metadata-to-ETF mapping:

| Yahoo/index sector label | Benchmark |
| --- | --- |
| Basic Materials | XLB |
| Communication Services | XLC |
| Consumer Cyclical; Consumer Discretionary | XLY |
| Consumer Defensive; Consumer Staples | XLP |
| Energy | XLE |
| Financial Services; Financials | XLF |
| Healthcare; Health Care | XLV |
| Industrials | XLI |
| Real Estate | XLRE |
| Technology; Information Technology | XLK |
| Utilities | XLU |

An absent or unrecognized sector produces a missing sector-relative return, not a guessed comparison.

## Quant metric definitions

Unless noted otherwise, “current” and “previous” mean values from the latest two comparable annual rows available to the current run. Missing values in either of those rows remain missing; the code does not skip a recent gap and substitute an older period.

### Price dislocation and relative returns

#### `drawdown_2y`

Use positive prices no later than the cutoff and within the preceding 730 calendar days:

```text
drawdown_2y = (maximum adjusted close - latest adjusted close)
              / maximum adjusted close
```

The output is a positive decline. The hard gate requires `drawdown_2y > minimum_drawdown`, which defaults to `0.25`.

#### One-year total return

For observations within the preceding 365 calendar days:

```text
total return = last price / first price - 1
```

At least two observations are required. “One year” uses the first and last available observations in that calendar window; it does not require an observation exactly 365 days earlier.

#### `relative_return_sector_1y` and `relative_return_index_1y`

Stock and benchmark series are aligned on their combined date index, forward-filled, and rows without both values are removed. Over the common 365-day window:

```text
relative return = stock total return - benchmark total return
```

A negative number means the stock underperformed. The sector field uses the mapped ETF above; the index field uses `SPY` unless configured otherwise.

### Growth, margins, and share count

#### Signed growth convention

Revenue, FCF, FCF/share, and shares use:

```text
growth(current, previous) = (current - previous) / abs(previous)
```

A zero or missing previous value makes growth undefined. The absolute prior denominator makes an improvement from a negative base positive; a cash-flow sign change still requires inspection of the underlying values.

#### `revenue_growth`

```text
revenue_growth = growth(current revenue, previous revenue)
```

#### `fcf_growth`

```text
fcf_growth = growth(current FCF, previous FCF)
```

#### `fcf_per_share_growth`

```text
FCF/share = FCF / positive share count
fcf_per_share_growth = growth(current FCF/share, previous FCF/share)
```

This incorporates both cash generation and dilution. Comparable annual diluted average shares (falling back to basic average shares) are used when Yahoo supplies them. Current shares-outstanding metadata is only a latest-row fallback; it is not copied backward, so the growth metric remains missing when no comparable prior share count exists. The metric is missing if either per-share value cannot be calculated.

#### `fcf_margin`

```text
fcf_margin = current FCF / current revenue
```

Revenue must be positive. FCF may be negative.

#### `gross_margin`

Use a reported gross margin if present; otherwise:

```text
gross_margin = gross profit / revenue
```

#### `gross_margin_stability`

```text
gross_margin_stability = population standard deviation
                         of the latest up to 5 gross margins
```

At least two non-missing margins are required. Lower is better. This is an absolute margin standard deviation, so `0.03` means three percentage points of dispersion.

#### `gross_margin_change`

```text
gross_margin_change = current gross margin - previous gross margin
```

This measured diagnostic is exported but is not independently scored in `quant-v1`.

#### `share_dilution`

```text
share_dilution = growth(current shares, previous shares)
```

Negative values indicate net share-count reduction. For Yahoo snapshots, comparable annual diluted average shares (falling back to basic average shares) are preferred; current shares-outstanding metadata is used only when the latest statement share count is unavailable and is never fabricated for earlier rows.

### Balance sheet and returns on capital

#### `net_debt`

```text
net_debt = total debt - cash and equivalents
```

Both inputs are required. A negative value represents net cash.

#### `net_debt_to_fcf`

```text
net_debt_to_fcf = net debt / current FCF
```

Current FCF must be positive. Negative FCF is not assigned an artificial leverage multiple.

#### `roic` and `roic_proxy`

Reported ROIC is used when supplied. Otherwise the labeled proxy is:

```text
effective tax rate = income tax expense / pretax income  # when no rate is supplied
NOPAT              = operating income × (1 - effective tax rate)
average invested capital = (current + previous invested capital) / 2
roic_proxy         = NOPAT / average invested capital
```

If previous invested capital is missing, current invested capital is used rather than inventing a prior value. A derived tax rate requires positive pretax income; the resulting tax rate must be between zero and one, and invested capital must be positive. Output includes the label `NOPAT / average invested capital (derived proxy, not reported ROIC)` when this proxy is calculated.

### Valuation

#### `ev_to_fcf`

```text
ev_to_fcf = current enterprise value / current FCF
```

FCF and enterprise value must both be positive. The code does not score a
non-positive or undefined EV/FCF multiple.

#### `fcf_yield`

```text
fcf_yield = current FCF / current market capitalization
```

Both FCF and market capitalization must be positive.

#### `pe_ratio`

The first available valid value or calculation is used:

```text
pe_ratio = positive reported trailing P/E

# First fallback:
pe_ratio = current price / positive diluted EPS

# Second fallback when positive EPS is unavailable:
pe_ratio = market capitalization / positive net income
```

Negative or zero earnings do not receive a P/E value.

#### Historical multiple proxies

For each available annual fiscal period, the adapter finds the latest price no later than the period end and no more than 14 days old. With period share count and statement values it derives:

```text
historical market cap = fiscal-period-end price × diluted average shares
historical EV         = historical market cap + total debt - cash
historical EV/FCF     = historical EV / positive FCF
historical P/E        = fiscal-period-end price / positive diluted EPS
```

Historical EV must also be positive. These cross-source proxies are calculated only when quote and financial-statement currencies are confirmed equal. They are labeled fiscal-period-end price proxies and are not claimed to reproduce a vendor's as-reported historical factor series.

#### `ev_to_fcf_compression` and `pe_compression`

Use positive proxy observations from the preceding five years. At least three observations for a multiple are required:

```text
multiple compression = (historical median - current multiple)
                       / historical median
```

Current and historical median multiples must be positive. Positive compression means the current multiple is lower than its own history.

#### `valuation_compression`

```text
valuation_compression = arithmetic mean of available
                        EV/FCF compression and P/E compression
```

If only one meets the observation requirements, that one is used. The individual historical medians and compression values are exported.

### Other exported measured fields

The detailed record also retains `market_cap`, raw `screening_market_cap`, currency status, `enterprise_value`, `revenue`, `free_cash_flow`, `shares_outstanding`, and `fundamental_observations`. It reports the actual `latest_fundamental_period_end`, `prior_fundamental_period_end`, retrieval-based `latest_fundamental_available_at`, `latest_fundamental_age_days`, and `fundamental_period_type`. It also includes company identity, sector, industry, index memberships, sector benchmark, observation timestamp, source provenance, and provider warnings. These fields provide context but do not add separate points beyond the scoring rules described below.

## Quant scoring model

Every finite metric is mapped to 0–100 with clipped linear interpolation between its listed `(raw value → score)` points. A raw value outside the outer breakpoints receives the nearest endpoint score. For example, drawdown `0.50` lies halfway between `(0.40 → 55)` and `(0.60 → 85)`, so it scores 70.

The “metric weight” below is within its component.

### Price Dislocation Score

| Metric | Weight | Piecewise-linear raw value → score points |
| --- | ---: | --- |
| `drawdown_2y` | 55% | `0.00→0`, `0.25→25`, `0.40→55`, `0.60→85`, `0.80→100` |
| `relative_return_sector_1y` | 25% | `-0.50→100`, `-0.30→80`, `-0.15→55`, `0.00→25`, `0.15→0` |
| `relative_return_index_1y` | 20% | `-0.50→100`, `-0.30→80`, `-0.15→55`, `0.00→25`, `0.15→0` |

Larger drawdown and worse benchmark-relative performance represent greater price dislocation. A stock at the 25% hard-gate boundary would score 25 on the drawdown metric, but the strict gate requires a value above that boundary.

### Fundamental Resilience Score

| Metric | Weight | Piecewise-linear raw value → score points |
| --- | ---: | --- |
| `revenue_growth` | 14% | `-0.25→0`, `-0.10→20`, `0.00→50`, `0.10→75`, `0.20→100` |
| `fcf_growth` | 16% | `-0.50→0`, `-0.20→25`, `0.00→50`, `0.25→75`, `0.50→100` |
| `fcf_per_share_growth` | 20% | `-0.50→0`, `-0.20→25`, `0.00→50`, `0.20→75`, `0.40→100` |
| `fcf_margin` | 18% | `-0.05→0`, `0.00→10`, `0.05→40`, `0.15→75`, `0.25→100` |
| `gross_margin_stability` | 12% | `0.00→100`, `0.01→90`, `0.03→65`, `0.07→20`, `0.10→0` |
| `share_dilution` | 10% | `-0.05→100`, `0.00→90`, `0.03→60`, `0.08→10`, `0.10→0` |
| `roic`, falling back to `roic_proxy` | 10% | `-0.05→0`, `0.00→10`, `0.05→35`, `0.10→65`, `0.20→100` |

The source metric records whether reported `roic` or derived `roic_proxy` was used.

### Valuation Compression Score

| Metric | Weight | Piecewise-linear raw value → score points |
| --- | ---: | --- |
| `ev_to_fcf` | 20% | `5→100`, `8→90`, `15→65`, `25→25`, `40→0` |
| `fcf_yield` | 25% | `0.00→0`, `0.02→10`, `0.05→50`, `0.08→75`, `0.12→100` |
| `pe_ratio` | 10% | `7→100`, `10→90`, `20→55`, `35→10`, `50→0` |
| `valuation_compression` | 45% | `-0.20→0`, `0.00→20`, `0.20→50`, `0.40→80`, `0.60→100` |

Cash-flow and earnings multiples are calculated only under the denominator rules above. Valuation compression itself requires positive current and historical multiples.

### Balance Sheet Quality Score

| Metric | Weight | Piecewise-linear raw value → score points |
| --- | ---: | --- |
| `net_debt_to_fcf` | 100% | `-2→100`, `0→95`, `1→80`, `2→60`, `3→35`, `5→0` |

Net cash and fewer years of positive FCF required to repay debt score better.

## Missing data and coverage

Missing evidence is not given a neutral score. For component `c`, using only observed metrics `O`:

```text
component_score_c = sum(metric_weight_m × metric_score_m, m in O)
                    / sum(metric_weight_m, m in O)

component_coverage_c = sum(metric_weight_m, m in O)
                       / sum(all configured metric weights in c)
```

If no metric is observed, the component score is missing and its coverage is zero. The four component weights are:

| Component | Quant composite weight |
| --- | ---: |
| Price Dislocation | 30% |
| Fundamental Resilience | 35% |
| Valuation Compression | 25% |
| Balance Sheet Quality | 10% |

The observed-only quant score renormalizes across components with a finite score:

```text
quant_score = sum(component_weight_c × component_score_c, available c)
              / sum(component_weight_c, available c)
```

Quant coverage preserves the missing-data penalty separately:

```text
quant_coverage = 100 ×
    sum(component_weight_c × component_coverage_c, all c)
    / sum(all component weights)
```

`quant_confidence` is currently the same value as `quant_coverage`. The detailed JSON includes a score explanation for every metric: raw value, interpolated score, configured weight, availability, source alias, and interpretation.

The initial candidate ranking uses:

```text
quant_evidence_score = quant_score × (0.5 + 0.5 × quant_coverage / 100)
```

Thus 100% coverage retains the full score; 50% coverage multiplies it by 0.75; 0% coverage would multiply it by 0.50, although a fully uncovered candidate cannot have a quant score. The 50% floor keeps incomplete but real observed evidence visible while making coverage explicit.

## AI narrative layer

Only the first `ND_AI_CANDIDATES` candidates from the quant-evidence ranking are sent to OpenAI. The default is 20. `OPENAI_API_KEY` is read from the environment, and `OPENAI_MODEL` defaults to `gpt-5-mini`.

AI-enabled runs make billable API and, when invoked, web-search tool calls. Use a small custom ticker set or low `--ai-candidates` value first, and review the limits and billing controls for the OpenAI account that owns the key.

The integration uses the OpenAI Responses API with strict Pydantic structured output. When enabled, it offers the built-in `web_search` tool with automatic tool choice and retains response source metadata. `--no-web-search` disables only that tool; it does not disable the AI call. `--no-ai` skips the whole layer.

Successful AI responses are cached for the configured `ND_CACHE_TTL_HOURS` and must match a fingerprint of stable analytical facts, model, prompt version, and web-search setting. The fingerprint excludes only the per-run top-level `as_of` value and `data_provenance.source_cached`; on a hit, the current run's complete measured-fact envelope replaces the cached envelope. A changed analytical input or expired entry triggers a new request unless the run is AI-disabled.

### Measured input to AI

The model receives a `measured_facts` object containing available identity, metrics, four components, quant score and coverage, provenance, and warnings. Missing facts are `null`. The system instructions forbid estimating, interpolating, or filling missing financial values from web material.

### AI inference schema

Everything inside `ai_inference` is model interpretation:

| Field | Type | Meaning |
| --- | --- | --- |
| `why_it_fell` | string | Concise researched explanation of the decline |
| `dominant_bearish_narrative` | string | Main market bear thesis |
| `expected_damage_if_bear_thesis_is_correct` | string list | Observable damage expected if the thesis is correct |
| `observed_fundamental_damage` | string list | Interpretation grounded in supplied measured facts, including unavailable evidence |
| `catalysts` | string list | Events that could close the apparent gap |
| `structural_risks` | string list | Durable competitive, financial, regulatory, or industry risks |
| `value_trap_probability` | integer 0–100 | Estimated probability that cheapness reflects lasting deterioration |
| `narrative_gap_score` | integer 0–100 | Estimated gap between market-implied and observed deterioration |
| `concise_thesis` | string | Why the security may or may not be mispriced |
| `invalidation_conditions` | string list | Specific, observable conditions that invalidate the contrarian thesis |
| `ai_inferences` | string list | Other explicitly labeled inferences |
| `evidence` | object list | Model-structured title, HTTP(S) URL, and supported claim |

The result envelope also contains normalized `evidence_sources` and `ai_metadata` with status, model, prompt version, analysis time, `web_search_enabled`, actual `used_web_search`, cache hit, response ID, usage when returned, and error when applicable. `used_web_search` becomes true only when the response contains a web-search tool call; enabling the tool alone is not reported as use. Response citations/search sources are deduplicated. A plausible model-written URL is not promoted when authoritative response citation metadata contradicts it.

### Narrative Gap

The core concept is:

```text
Narrative Gap = market-implied deterioration
                - observed fundamental deterioration
```

This is a research framework, not an accounting subtraction with common units. The model is instructed to award a high score only when the magnitude and durability of the selloff appear to imply much more damage than supplied measurements and credible evidence currently show. A low score means observed damage meets or exceeds the narrative, evidence is weak, or the apparent gap is unsupported.

The expected-damage list is particularly important: it converts a broad bear story into conditions that can be monitored in later statements. It does not make the probability estimate a measured fact.

## Final Dislocation Score

When both AI values are valid:

```text
inverse_value_trap = 100 - value_trap_probability

ai_score = 80% × narrative_gap_score
         + 20% × inverse_value_trap

dislocation_score = 70% × quant_evidence_score
                  + 24% × narrative_gap_score
                  +  6% × inverse_value_trap
```

The final weights therefore allocate 70% to coverage-adjusted measured quant evidence and 30% to AI inference. Scores are clipped to 0–100 before combination and the output is rounded to four decimal places.

Status behavior is explicit:

| `score_status` | Behavior |
| --- | --- |
| `quant_plus_ai` | Full 70/24/6 formula; `ai_score` is populated |
| `quant_only` | Used for an explicitly AI-disabled run: `dislocation_score = quant_evidence_score`; no AI value is invented |
| `ai_error` | Ticker was selected for AI but no valid inference was returned; final and AI scores are missing, while quant evidence and error metadata remain |
| `not_ai_analyzed` | Ticker was outside the configured AI candidate budget; final and AI scores are missing, while quant evidence remains |
| `insufficient_quant_data` | Final score is missing because quant score or coverage is unavailable |

Within an AI-enabled run, completed `quant_plus_ai` rows rank first by their comparable combined score. `ai_error` rows then rank by quant evidence, followed by `not_ai_analyzed` rows; their missing final score prevents a provisional quant score from being compared directly with a full 70/24/6 score. An AI-disabled run and a completed AI run are not methodologically identical even though both use the `dislocation_score` column. Filter or compare `score_status` when combining exports from different runs.

## Screen output contracts

### `top_20.csv`

The default CSV contains the first 20 candidates after final reranking. `--top` changes the row count while the v1 filename remains `top_20.csv`.

Nested JSON is flattened with dotted names, such as `ai_inference.narrative_gap_score` and `ai_metadata.status`. Lists and nested objects are serialized as JSON strings within cells. Missing values are empty cells. Preferred columns begin with rank, ticker, company, final score, four components, Narrative Gap, and value-trap probability; all remaining candidate fields follow.

### `detailed_results.json`

Top-level schema:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Report schema, currently `1` |
| `generated_at_utc` | ISO-8601 string | Export time |
| `count` | integer | Number of eligible candidate records |
| `run_metadata` | object | Run status, UTC times, universe sources/counts, configuration, warnings, and data policy |
| `candidates` | object list | Complete ranked candidate records |

`run_metadata` contains `run_id`, `status`, `started_at_utc`, `completed_at_utc`, `universe`, `universe_size`, `drawdown_pre_screen_count`, `fundamentals_fetched_count`, `hard_screen_excluded_count`, `quant_candidate_count`, `ai_analyzed_count`, `configuration`, `warnings`, and `data_policy`.

Each candidate contains:

- identity and provenance: `rank`, `ticker`, company, sector, industry, memberships, `as_of`, sector benchmark, `data_provenance`, and `data_warnings`;
- all measured metric fields described above, actual fundamental-period dates/age, currency status, hard-gate status/reasons, historical median/compression fields, and observation count;
- four component scores and coverages, `quant_score`, `quant_coverage`, `quant_confidence`, `quant_evidence_score`, scoring model/version, and complete score explanations;
- final `dislocation_score`, `ai_score`, `score_status`, and final-score version;
- for researched rows, nested `measured_facts`, `ai_inference`, `evidence_sources`, and `ai_metadata`.

The JSON contains all candidates that passed both hard gates, not every failed or unfetched universe member. Fetch failures and universe warnings are summarized in run metadata. All NaN/infinite values are converted to JSON `null`.

### `top_10_report.md`

The default Markdown file contains ten briefs; `--report-top` changes the count while the v1 filename remains `top_10_report.md`. Each brief includes:

- final score, Narrative Gap, and value-trap probability when available;
- why it fell;
- dominant market belief and expected damage;
- selected scalar measured facts, explicitly labeled;
- AI interpretation of observed damage;
- concise mispricing thesis and catalysts;
- invalidation conditions and structural risks;
- deduplicated evidence links.

Unavailable AI sections say so instead of synthesizing a narrative from missing data.

## Point-in-time backtest module

The backtester is a separate engine in `backtest.py`. It consumes historical signal snapshots that already existed at their stated availability times. It never fetches or reconstructs historical fundamentals.

### CLI command

```bash
narrative-dislocation backtest \
  --signals data/pit_signals.csv \
  --prices data/adjusted_prices.csv \
  --benchmark-prices data/benchmark_prices.csv \
  --benchmarks "VALL,^GSPC" \
  --top-n 20 \
  --rebalance M \
  --transaction-cost-bps 10 \
  --execution-lag-sessions 1 \
  --snapshot-mode latest_snapshot \
  --max-signal-age-days 45 \
  --start 2015-01-01 \
  --end 2025-12-31
```

PowerShell accepts the same command on one line. Its multiline continuation character is a backtick rather than `\`.

Supported CLI controls:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--signals` | required | Long point-in-time signal CSV |
| `--prices` | required | Wide or long asset-price CSV |
| `--benchmark-prices` | none | Optional wide or long benchmark-price CSV |
| `--output-dir` | `outputs/backtest` | Root; a UTC timestamp directory is appended |
| `--top-n` | `20` | Maximum holdings after each selection |
| `--rebalance` | `M` | Pandas calendar period frequency, for example `M` or `Q` |
| `--transaction-cost-bps` | `10` | One-way cost charged to absolute weight traded |
| `--execution-lag-sessions` | `1` | Observed price sessions between cutoff and execution event; minimum `1` for daily-close data |
| `--score-column` | `dislocation_score` | Signal column to rank descending |
| `--snapshot-mode` | `latest_snapshot` | Cross-section selection policy; see below |
| `--max-signal-age-days` | none | Optional maximum calendar age of `as_of` at cutoff |
| `--start`, `--end` | none | Inclusive price-range filters |
| `--benchmarks` | `VALL,^GSPC` | Comma-separated columns to include from benchmark prices |

### Signal CSV schema

| Column | Required | Contract |
| --- | --- | --- |
| `ticker` | yes | Non-blank asset identifier matching a price column; surrounding whitespace is removed; configurable only through the Python API |
| `as_of` | yes | Valid date/time for the signal snapshot's measurement date |
| `available_at` | yes | Valid date/time at which the complete signal could actually have been used |
| `dislocation_score` | yes by default | Numeric score; use `--score-column` for another name |
| `screen_pass` | no | When present, only explicit true values (`true`, `1`, `yes`, `y`, or `on`, case-insensitive) are eligible |

Additional columns are allowed and retained during selection. Both timestamps are parsed in UTC and normalized for comparison. Every row must have valid `as_of` and `available_at`; every selected row must satisfy:

```text
as_of <= rebalance cutoff end-of-day
available_at <= rebalance cutoff end-of-day
score is finite, numeric, and non-missing
screen_pass is true, when that column exists
signal age <= configured maximum, when supplied
```

Two snapshot modes are available:

- `latest_snapshot`: find the latest eligible `as_of` across the dataset, use only that cross-section, then rank. This is the default and is appropriate for complete batch snapshots.
- `per_ticker`: select each ticker's latest eligible row independently, then rank. This supports asynchronous updates but can mix measurement dates; use `--max-signal-age-days` to limit staleness.

For duplicate corrections to the same ticker and `as_of`, the row with the latest eligible `available_at` wins. Scores sort descending, with ticker ascending as a deterministic tie-breaker. The Python API also supports `minimum_score` and custom column names through `BacktestConfig`; these do not currently have separate CLI flags.

The current screener's output is not a ready-made historical signal CSV. It does not contain a history of original `available_at` timestamps or historical constituent snapshots. Build and archive genuine point-in-time signals before running this module.

### Asset and benchmark price schemas

Wide form:

```csv
date,AAA,BBB
2024-01-31,100.0,50.0
2024-02-01,101.0,49.5
```

Long form:

```csv
date,ticker,adjusted_close
2024-01-31,AAA,100.0
2024-02-01,AAA,101.0
2024-01-31,BBB,50.0
2024-02-01,BBB,49.5
```

Recognized date aliases are `date`, `timestamp`, and `as_of`; ticker aliases are `ticker` and `symbol`; value aliases are `adjusted_close`, `adj_close`, `Adj Close`, `close`, `price`, and `value`. For a wide CLI CSV, include an explicit date column. Duplicate dates use the last observation. Non-numeric, non-positive, NaN, and infinite prices become missing.

Benchmark prices use the same schema. `--benchmarks` selects exact normalized price-frame column names; missing requested columns are skipped. The module does not download, identify, currency-convert, or validate benchmark series.

The defaults `VALL` and `^GSPC` are therefore only configurable identifiers. `VALL` is **not asserted to be the FTSE Global All Cap index**. A valid comparison with FTSE Global All Cap requires a correctly sourced index total-return series or a clearly disclosed investable proxy. Label the CSV column truthfully and pass it explicitly, for example:

```bash
narrative-dislocation backtest --signals data/pit.csv --prices data/assets.csv --benchmark-prices data/benchmarks.csv --benchmarks "FTSE_Global_All_Cap_TR,^GSPC"
```

### Portfolio mechanics

1. Filter the price range with optional `start` and `end`.
2. Choose the last observed price session in each configured calendar period as the rebalance cutoff.
3. Select only signals known by cutoff end-of-day.
4. Move forward by `execution_lag_sessions`; skip a cutoff when no execution session remains.
5. At the execution event, walk the full ranked eligible list and discard tickers without a matching price column or finite positive execution price.
6. Take the first remaining Top N names and equal-weight them. This backfills a missing-price name with the next tradable candidate; an empty selection is cash with zero modeled interest.
7. Let weights drift with asset returns until the next event.
8. Charge transaction cost on absolute weight change.

For old weight `w_old` and target `w_target`:

```text
turnover = sum(abs(w_target - w_old)) over union of old and new tickers
cost fraction = turnover × transaction_cost_bps / 10,000
new NAV = pre-cost NAV × (1 - cost fraction)
```

Cash-to-fully-invested has turnover 1.0. A full rotation from one disjoint fully invested portfolio to another has turnover 2.0. Target weights booked on an execution date drive the following return interval. If a held asset lacks a return, v1 uses 0% for that interval and records the date/ticker in diagnostics.

Portfolio NAV begins at `1.0`. Daily portfolio return is the weighted asset return net of any event-day transaction cost. Benchmark observations are first forward-filled on the union of benchmark and portfolio calendars, then sampled on portfolio dates and normalized to `1.0` at their first available aligned observation. This preserves an off-calendar benchmark observation for the next portfolio session.

### Backtest output schemas

Outputs are written to a unique `<output-dir>/<UTC timestamp with microseconds>-<random suffix>/` directory.

#### `equity_curve.csv`

| Column | Meaning |
| --- | --- |
| `date` | Observed simulation date |
| `portfolio_value` | NAV starting from 1.0 |
| `portfolio_return` | Net change in NAV for the row |
| `market_return_before_cost` | Drifted portfolio's asset return before transaction cost |
| `turnover` | Absolute target-weight change at an event; zero otherwise |
| `transaction_cost_fraction` | NAV fraction charged at an event |
| `number_of_holdings` | Holdings after any event on that date |

#### `holdings.csv`

One row per target holding and rebalance:

| Column | Meaning |
| --- | --- |
| `rebalance_cutoff` | Signal-selection cutoff |
| `execution_date` | Scheduled execution event |
| `ticker` | Selected asset |
| `target_weight` | Equal target weight |
| `score` | Selected score |
| `signal_as_of` | Measurement date used |
| `signal_available_at` | Availability timestamp used |

#### `trades.csv`

One row per changed asset weight:

| Column | Meaning |
| --- | --- |
| `rebalance_cutoff` | Signal-selection cutoff |
| `execution_date` | Scheduled execution event |
| `ticker` | Asset bought, sold, or resized |
| `old_weight` | Drifted pre-event weight |
| `target_weight` | Post-event weight |
| `weight_traded` | Absolute difference between target and old weight |
| `estimated_cost` | Pro-rata share of modeled event cost in NAV units |

#### `benchmark_curves.csv`

The first column is `date`; each returned benchmark has one normalized wealth-index column. Requested but absent/empty benchmark columns do not appear.

#### `summary.json`

The envelope contains:

- `configuration`: parsed CLI options;
- `summary.portfolio`: `total_return`, `cagr`, `annualized_volatility`, `sharpe_zero_rate`, `max_drawdown`, `observations`, `total_transaction_cost`, and `rebalance_count`;
- `summary.benchmarks.<ticker>`: the same performance fields except portfolio transaction cost/rebalance count;
- `summary.config`: effective `BacktestConfig`;
- `diagnostics`: look-ahead guard statement, missing-held-return count/details, and price assumption.

Performance formulas are:

```text
total return = ending normalized value - 1
CAGR = ending value ** (365.25 / elapsed calendar days) - 1
annualized volatility = sample standard deviation(daily returns) × sqrt(252)
zero-rate Sharpe = mean(daily returns) / sample std(daily returns) × sqrt(252)
max drawdown = minimum(value / cumulative maximum value - 1)
```

No risk-free rate is deducted. `total_transaction_cost` is the sum of modeled NAV amounts charged, with initial NAV equal to one.

### Python API

```python
import pandas as pd

from narrative_dislocation.backtest import BacktestConfig, run_backtest

signals = pd.read_csv("data/pit_signals.csv")
prices = pd.read_csv("data/adjusted_prices.csv")
benchmarks = pd.read_csv("data/benchmark_prices.csv")

result = run_backtest(
    signals,
    prices,
    benchmark_prices=benchmarks,
    config=BacktestConfig(
        top_n=20,
        rebalance_frequency="M",
        transaction_cost_bps=10,
        execution_lag_sessions=1,
        benchmark_tickers=("VALL", "^GSPC"),
    ),
)
```

The API also accepts a mapping from ticker to a pandas Series, DataFrame, or date-to-price mapping.

### Backtest limitations and required controls

- Point-in-time safety of the signals is the caller's responsibility. Both timestamps are enforced, but the module cannot verify how a score was created.
- The daily-close engine rejects an execution lag of zero to avoid acting on a cutoff-close signal at that same close. Use at least one observed price session and model a more detailed execution convention outside v1 when needed.
- Do not reconstruct historical signals from today's Yahoo statements. Archive original filings/data releases and their actual availability times.
- Use historical constituent membership to avoid survivorship bias. Today's S&P 500, Nasdaq-100, or Russell 1000 membership is not valid for old rebalance dates.
- Historical AI research must exclude later articles, revised guidance, and other future information. The current live AI workflow is not automatically an historical research engine.
- Supply adjusted or total-return-consistent price series and audit splits, distributions, mergers, spin-offs, symbol changes, and delistings. V1 assumes prices are point-in-time accurate and does not reconstruct delisting returns.
- Benchmark identity, licensing, dividends, currency, timezone, and total-return convention are not validated. Portfolio and benchmark series must be economically comparable.
- Missing held returns are modeled as zero and disclosed; this can materially bias a result, especially around delistings.
- Equal weights, close-to-close returns, and a single bps cost model omit bid/ask spreads, slippage, market impact, liquidity/capacity, borrow availability and fees, taxes, and execution constraints.
- Cash earns zero. There is no leverage or shorting in v1.
- Multiple-testing, parameter tuning, and model-version changes can overfit results. Keep immutable signal snapshots and score-version fields.

Treat backtest output as a diagnostic of a precisely documented historical dataset, not evidence that the current live screen would have achieved the same return.

## Operational limitations of the current screener

- Constituent pages, Yahoo endpoints, and OpenAI services can be unavailable or change schema.
- Market cap can be missing or lack confirmed USD quote currency; the strict USD-denominated gate excludes that row. Cross-source valuation ratios are withheld unless quote and statement currencies are confirmed equal; v1 performs no FX conversion.
- Fundamental periods can be revised or restated and are not a licensed as-reported history.
- Fiscal-period-end valuation proxies mix statement periods with nearby market prices and can differ materially from trailing valuation as perceived on that date.
- Relative-return benchmarks are broad ETFs, not exact peer portfolios. Forward-filling aligned calendars is a simplifying assumption.
- A high drawdown can reflect fraud, insolvency, disruption, dilution, litigation, regulation, or permanent impairment. The screen is designed to surface the question, not answer it.
- Missing data reduces coverage but does not always reduce an observed-only component score. Always read coverage and explanations alongside the score.
- AI scores are subjective, sensitive to source availability and prompting, and can be wrong despite structured output and citations.

All results require human verification. This is research triage, not investment advice.
