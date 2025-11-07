"""Command-line interface for running sample backtests."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from backtest.engine import BacktestEngine
from backtest.report import render_report
from data.historical import HistoricalPriceLoader
from strategies.moving_average import MovingAverageCrossover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a moving average crossover backtest on daily price data.",
    )
    parser.add_argument("ticker", help="Ticker symbol to download, e.g. AAPL or msft")
    parser.add_argument(
        "--fast",
        type=int,
        default=12,
        help="Fast EMA window length (default: 12)",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=26,
        help="Slow EMA window length (default: 26)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial capital for the backtest (default: 10000)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional path to store downloaded price data (defaults to ~/.cache/jonas_pinigai)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Path to save a visual performance report (defaults to ./<ticker>_backtest.png)"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the generated report interactively after saving",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fast <= 0 or args.slow <= 0:
        parser.error("EMA window lengths must be positive integers")
    if args.fast >= args.slow:
        parser.error("Fast EMA window should be smaller than slow EMA window")

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    loader_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    loader = HistoricalPriceLoader(**loader_kwargs)
    prices = loader.load_daily(args.ticker)

    strategy = MovingAverageCrossover(fast_window=args.fast, slow_window=args.slow)
    engine = BacktestEngine(initial_capital=args.capital)

    result = engine.run(prices, strategy)

    print(f"Ticker: {args.ticker.upper()}")
    print(f"Bars evaluated: {len(prices)}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Annualized return: {result.annualized_return:.2%}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")

    if result.trades.empty:
        print("No completed trades recorded.")
    else:
        print("\nTrades:")
        print(result.trades.to_string(index=False))

    output_report = args.report or Path.cwd() / f"{args.ticker.lower()}_backtest.png"
    report_path = render_report(
        prices,
        result,
        fast_window=args.fast,
        slow_window=args.slow,
        output=output_report,
        ticker=args.ticker.upper(),
        show=args.show,
    )
    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
