from __future__ import annotations

import json

import pandas as pd

from narrative_dislocation.cli import build_parser, run_cli


def test_screen_parser_accepts_comma_separated_tickers():
    args = build_parser().parse_args(
        ["screen", "--tickers", "AAA,BBB", "--no-ai", "--universe-limit", "2"]
    )

    assert args.command == "screen"
    assert args.tickers == ["AAA,BBB"]
    assert args.no_ai is True


def test_backtest_cli_writes_bundle(tmp_path):
    dates = pd.bdate_range("2024-01-02", "2024-05-15")
    prices = pd.DataFrame(
        {
            "date": dates,
            "AAA": [100 + index * 0.2 for index in range(len(dates))],
            "BBB": [100 + index * 0.1 for index in range(len(dates))],
        }
    )
    benchmarks = pd.DataFrame(
        {
            "date": dates,
            "VALL": [100 + index * 0.12 for index in range(len(dates))],
            "^GSPC": [100 + index * 0.15 for index in range(len(dates))],
        }
    )
    signals = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "as_of": as_of,
                "available_at": as_of,
                "dislocation_score": score,
                "screen_pass": True,
            }
            for as_of in ("2024-01-30", "2024-02-27", "2024-03-28", "2024-04-29")
            for ticker, score in (("AAA", 80), ("BBB", 70))
        ]
    )
    signals_path = tmp_path / "signals.csv"
    prices_path = tmp_path / "prices.csv"
    benchmarks_path = tmp_path / "benchmarks.csv"
    output_path = tmp_path / "out"
    signals.to_csv(signals_path, index=False)
    prices.to_csv(prices_path, index=False)
    benchmarks.to_csv(benchmarks_path, index=False)

    code = run_cli(
        [
            "backtest",
            "--signals",
            str(signals_path),
            "--prices",
            str(prices_path),
            "--benchmark-prices",
            str(benchmarks_path),
            "--output-dir",
            str(output_path),
            "--top-n",
            "1",
        ]
    )

    assert code == 0
    run_dirs = list(output_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "equity_curve.csv").is_file()
    summary = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["portfolio"]["rebalance_count"] >= 1
    assert "VALL" in summary["summary"]["benchmarks"]
