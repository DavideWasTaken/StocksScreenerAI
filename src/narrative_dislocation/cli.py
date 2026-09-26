"""Dependency-light command line interface for screening and backtesting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from narrative_dislocation import __version__
from narrative_dislocation.ai import MissingOpenAIKeyError
from narrative_dislocation.backtest import BacktestConfig, run_backtest
from narrative_dislocation.config import AppConfig, load_dotenv_if_present
from narrative_dislocation.domain import json_safe
from narrative_dislocation.pipeline import ScreenerPipeline, ScreenOptions, parse_tickers

LOGGER = logging.getLogger("narrative_dislocation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narrative-dislocation",
        description="Find stocks whose prices fell much more than observed fundamentals.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser("screen", help="Run the current stock screener")
    screen.add_argument(
        "--tickers",
        action="append",
        default=[],
        metavar="AAPL,MSFT",
        help="Override index discovery; repeat or use comma-separated tickers",
    )
    screen.add_argument(
        "--russell-source",
        help="Optional Russell 1000 constituent CSV path or URL",
    )
    screen.add_argument(
        "--no-russell",
        action="store_true",
        help="Do not attempt the optional Russell 1000 source",
    )
    screen.add_argument("--no-ai", action="store_true", help="Run quant scoring only")
    screen.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable the Responses API web_search tool",
    )
    screen.add_argument(
        "--force-refresh", action="store_true", help="Ignore fresh data and AI cache entries"
    )
    screen.add_argument(
        "--universe-limit",
        type=int,
        help="Limit symbols for a development smoke test (not a full screen)",
    )
    screen.add_argument("--top", type=int, help="Number of rows in the ranked CSV")
    screen.add_argument("--ai-candidates", type=int, help="Top quant names researched by AI")
    screen.add_argument("--report-top", type=int, help="Number of detailed Markdown briefs")
    screen.add_argument("--model", help="Responses API model (or set OPENAI_MODEL)")
    screen.add_argument("--output-dir", type=Path, help="Root output directory")
    screen.add_argument("--cache-dir", type=Path, help="Persistent cache directory")
    screen.add_argument("--cache-ttl-hours", type=float, help="Live-data cache TTL")
    screen.add_argument("--run-id", help="Stable name for this output run directory")

    backtest = subparsers.add_parser(
        "backtest", help="Backtest precomputed point-in-time signal snapshots"
    )
    backtest.add_argument("--signals", type=Path, required=True, help="Long signal CSV")
    backtest.add_argument("--prices", type=Path, required=True, help="Asset price CSV")
    backtest.add_argument("--benchmark-prices", type=Path, help="Optional benchmark price CSV")
    backtest.add_argument("--output-dir", type=Path, default=Path("outputs/backtest"))
    backtest.add_argument("--top-n", type=int, default=20)
    backtest.add_argument("--rebalance", default="M", help="Pandas period frequency, e.g. M or Q")
    backtest.add_argument("--transaction-cost-bps", type=float, default=10.0)
    backtest.add_argument("--execution-lag-sessions", type=int, default=1)
    backtest.add_argument("--score-column", default="dislocation_score")
    backtest.add_argument(
        "--snapshot-mode",
        choices=("latest_snapshot", "per_ticker"),
        default="latest_snapshot",
    )
    backtest.add_argument("--max-signal-age-days", type=int)
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument(
        "--benchmarks",
        default="VALL,^GSPC",
        help="Comma-separated benchmark columns/tickers in benchmark-prices",
    )

    subparsers.add_parser("doctor", help="Check dependencies and local configuration")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        if args.command == "screen":
            return _run_screen(args)
        if args.command == "backtest":
            return _run_backtest(args)
        if args.command == "doctor":
            return _run_doctor()
    except (ValueError, FileNotFoundError, MissingOpenAIKeyError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


def main(argv: list[str] | None = None) -> None:
    try:
        code = run_cli(argv)
    except KeyboardInterrupt:
        print("\nInterrupted; completed per-symbol cache entries are preserved.", file=sys.stderr)
        code = 130
    if code:
        raise SystemExit(code)


def _run_screen(args: argparse.Namespace) -> int:
    config = AppConfig.from_env().with_overrides(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        cache_ttl_hours=args.cache_ttl_hours,
        final_top=args.top,
        ai_candidates=args.ai_candidates,
        report_top=args.report_top,
        openai_model=args.model,
        ai_web_search=False if args.no_web_search else None,
    )
    options = ScreenOptions(
        tickers=parse_tickers(args.tickers),
        include_russell=not args.no_russell,
        russell_source=args.russell_source,
        enable_ai=not args.no_ai,
        force_refresh=args.force_refresh,
        universe_limit=args.universe_limit,
        run_id=args.run_id,
    )
    pipeline = ScreenerPipeline(config, progress=_print_progress)
    result = pipeline.run(options)
    print(f"Completed run {result.run_id}: {len(result.candidates)} eligible candidates")
    for label, path in result.output_paths.items():
        print(f"  {label}: {path}")
    if result.warnings:
        print(f"  warnings: {len(result.warnings)} (see detailed_results.json)")
    return 0


def _run_backtest(args: argparse.Namespace) -> int:
    signals = _read_csv(args.signals, "signals")
    prices = _read_csv(args.prices, "prices")
    benchmark_prices = (
        _read_csv(args.benchmark_prices, "benchmark prices") if args.benchmark_prices else None
    )
    benchmarks = tuple(item.strip() for item in args.benchmarks.split(",") if item.strip())
    config = BacktestConfig(
        top_n=args.top_n,
        rebalance_frequency=args.rebalance,
        transaction_cost_bps=args.transaction_cost_bps,
        execution_lag_sessions=args.execution_lag_sessions,
        score_column=args.score_column,
        snapshot_mode=args.snapshot_mode,
        max_signal_age_days=args.max_signal_age_days,
        start=args.start,
        end=args.end,
        benchmark_tickers=benchmarks,
    )
    result = run_backtest(
        signals,
        prices,
        benchmark_prices=benchmark_prices,
        config=config,
    )
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:6]}"
    output_dir = (args.output_dir.expanduser() / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.holdings.to_csv(output_dir / "holdings.csv", index=False)
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.benchmark_curves.to_csv(output_dir / "benchmark_curves.csv")
    summary = {
        "configuration": json_safe(vars(args)),
        "summary": json_safe(result.summary),
        "diagnostics": json_safe(result.diagnostics),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Backtest completed: {output_dir}")
    for key, value in result.summary.items():
        print(f"  {key}: {value}")
    return 0


def _run_doctor() -> int:
    modules = ("pandas", "numpy", "yfinance", "openai", "pydantic", "lxml")
    print(f"narrative-dislocation {__version__}")
    print(f"Python: {sys.version.split()[0]}")
    missing: list[str] = []
    for module in modules:
        installed = importlib.util.find_spec(module) is not None
        print(f"{module}: {'ok' if installed else 'missing'}")
        if not installed:
            missing.append(module)
    print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set (use --no-ai)'}")
    return 1 if missing else 0


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    return pd.read_csv(path)


def _print_progress(phase: str, message: str) -> None:
    print(f"[{phase}] {message}", file=sys.stderr, flush=True)


def _configure_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity >= 2 else logging.INFO if verbosity == 1 else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


__all__ = ["build_parser", "main", "run_cli"]
