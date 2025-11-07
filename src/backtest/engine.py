"""Simple backtesting engine for daily bar data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        ...


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    total_return: float
    annualized_return: float
    max_drawdown: float


@dataclass
class BacktestEngine:
    initial_capital: float = 10_000.0

    def run(self, prices: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if prices.empty:
            raise ValueError("Price data is empty")

        prices = prices.sort_values("date").reset_index(drop=True)
        signals = strategy.generate_signals(prices)
        if len(signals) != len(prices):
            raise ValueError("Signals length must match prices length")

        returns = prices["close"].pct_change().fillna(0.0)
        positions = signals.shift(1).fillna(0.0)  # enter at next bar open
        strategy_returns = positions * returns
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital

        trades = self._build_trades(signals, prices)
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        annualized_return = self._annualized_return(strategy_returns)
        max_drawdown = self._max_drawdown(equity_curve)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
        )

    def _build_trades(self, signals: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
        positions = signals.astype(int)
        changes = positions.diff().fillna(positions)
        entries = prices.loc[changes > 0, ["date", "close"]].copy()
        exits = prices.loc[changes < 0, ["date", "close"]].copy()
        entries.rename(columns={"close": "entry_price", "date": "entry_date"}, inplace=True)
        exits.rename(columns={"close": "exit_price", "date": "exit_date"}, inplace=True)

        trades = entries.reset_index(drop=True)
        trades = trades.join(exits.reset_index(drop=True))
        if not trades.empty:
            trades["pnl"] = trades["exit_price"].fillna(trades["entry_price"]) - trades["entry_price"]
        return trades

    def _annualized_return(self, daily_returns: pd.Series) -> float:
        compounded = (1 + daily_returns).prod()
        num_days = len(daily_returns)
        if num_days == 0:
            return 0.0
        return compounded ** (252 / num_days) - 1

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        cumulative_max = equity_curve.cummax()
        drawdowns = equity_curve / cumulative_max - 1
        return drawdowns.min()


__all__ = ["BacktestEngine", "BacktestResult", "Strategy"]
