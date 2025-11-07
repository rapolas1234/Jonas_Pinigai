"""Tools for turning backtest results into visual reports."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.figure import Figure

from backtest.engine import BacktestResult


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _build_summary_frame(result: BacktestResult) -> pd.DataFrame:
    summary = {
        "Total return": _format_percent(result.total_return),
        "Annualized return": _format_percent(result.annualized_return),
        "Max drawdown": _format_percent(result.max_drawdown),
        "Final equity": f"{result.equity_curve.iloc[-1]:,.2f}",
        "Trades": str(len(result.trades)),
    }
    return pd.DataFrame(summary, index=["Value"]).T


def _format_trades(trades: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"message": ["No completed trades"]})
    truncated = trades.tail(limit).copy()
    display_cols: Iterable[str]
    display_cols = [
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "pnl",
    ]
    truncated = truncated[display_cols]
    for column in ("entry_price", "exit_price", "pnl"):
        truncated[column] = truncated[column].map(lambda x: f"{x:,.2f}")
    return truncated


def create_report_figure(
    prices: pd.DataFrame,
    result: BacktestResult,
    *,
    fast_window: int,
    slow_window: int,
    ticker: str | None = None,
) -> Figure:
    """Create a matplotlib figure summarising the strategy performance."""
    if "date" not in prices.columns:
        raise ValueError("prices must contain a 'date' column for plotting")

    indexed_prices = prices.set_index("date")
    close = indexed_prices["close"]
    fast = close.ewm(span=fast_window, adjust=False).mean()
    slow = close.ewm(span=slow_window, adjust=False).mean()

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Backtest report{f' - {ticker}' if ticker else ''}", fontsize=14)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1.6], figure=fig)

    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(close.index, close.values, label="Close", color="#1f77b4")
    ax_price.plot(fast.index, fast.values, label=f"EMA {fast_window}", color="#ff7f0e")
    ax_price.plot(slow.index, slow.values, label=f"EMA {slow_window}", color="#2ca02c")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    ax_equity = fig.add_subplot(gs[1], sharex=ax_price)
    ax_equity.plot(result.equity_curve.index, result.equity_curve.values, color="#9467bd")
    ax_equity.set_ylabel("Equity")
    ax_equity.grid(True, alpha=0.3)

    summary = _build_summary_frame(result)
    trades = _format_trades(result.trades)

    ax_table = fig.add_subplot(gs[2])
    ax_table.axis("off")

    summary_table = ax_table.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        loc="upper left",
        colWidths=[0.35],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)

    if "message" in trades.columns:
        ax_table.text(0.02, 0.25, trades.loc[0, "message"], fontsize=10)
    else:
        trade_table = ax_table.table(
            cellText=trades.values,
            colLabels=trades.columns,
            loc="lower left",
            cellLoc="center",
        )
        trade_table.auto_set_font_size(False)
        trade_table.set_fontsize(9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def render_report(
    prices: pd.DataFrame,
    result: BacktestResult,
    *,
    fast_window: int,
    slow_window: int,
    output: Path,
    ticker: str | None = None,
    show: bool = False,
) -> Path:
    """Save a visual report to ``output`` and optionally display it."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = create_report_figure(
        prices,
        result,
        fast_window=fast_window,
        slow_window=slow_window,
        ticker=ticker,
    )
    fig.savefig(output, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output


__all__ = ["create_report_figure", "render_report"]
