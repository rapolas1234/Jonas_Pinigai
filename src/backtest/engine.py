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
    signals: pd.Series
    daily_returns: pd.Series
    volatility: float
    sharpe_ratio: float


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

        if "date" in prices.columns:
            signals.index = prices["date"]

        returns = prices["close"].pct_change().fillna(0.0)
        positions = signals.shift(1).fillna(0.0)  # enter at next bar open
        strategy_returns = positions * returns

        if "date" in prices.columns:
            index = prices["date"]
            returns.index = index
            positions.index = index
            strategy_returns.index = index

        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital

        trades = self._build_trades(positions, prices)
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        annualized_return = self._annualized_return(strategy_returns)
        max_drawdown = self._max_drawdown(equity_curve)
        volatility = self._annualized_volatility(strategy_returns)
        sharpe_ratio = self._sharpe_ratio(strategy_returns, volatility)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            signals=signals,
            daily_returns=strategy_returns,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
        )

    def _build_trades(self, positions: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
        if positions.empty:
            return pd.DataFrame()

        trade_frame = prices.copy()
        if "date" not in trade_frame.columns:
            trade_frame = trade_frame.copy()
            trade_frame["date"] = range(len(trade_frame))

        trade_frame["position"] = positions.to_numpy()
        trade_frame["position_change"] = trade_frame["position"].diff().fillna(trade_frame["position"])

        entry_price_col = "open" if "open" in trade_frame.columns else "close"
        exit_price_col = entry_price_col

        entries = trade_frame.loc[trade_frame["position_change"] > 0, ["date", entry_price_col]]
        exits = trade_frame.loc[trade_frame["position_change"] < 0, ["date", exit_price_col]]

        trades = entries.rename(columns={"date": "entry_date", entry_price_col: "entry_price"}).reset_index(drop=True)
        exit_frame = exits.rename(columns={"date": "exit_date", exit_price_col: "exit_price"}).reset_index(drop=True)
        exit_frame["is_open"] = False

        if len(exit_frame) < len(trades):
            remaining = len(trades) - len(exit_frame)
            last_row = trade_frame.iloc[-1]
            last_date = last_row["date"]
            last_price = last_row[exit_price_col]
            open_exits = pd.DataFrame(
                {
                    "exit_date": [last_date] * remaining,
                    "exit_price": [last_price] * remaining,
                    "is_open": [True] * remaining,
                }
            )
            exit_frame = pd.concat([exit_frame, open_exits], ignore_index=True)

        trades = trades.join(exit_frame)
        if trades.empty:
            return trades

        trades["return_pct"] = trades["exit_price"] / trades["entry_price"] - 1

        if pd.api.types.is_datetime64_any_dtype(trades["entry_date"]):
            trades["holding_period"] = (trades["exit_date"] - trades["entry_date"]).dt.days
        else:
            trades["holding_period"] = trades["exit_date"] - trades["entry_date"]

        trades["status"] = trades.pop("is_open").map({True: "OPEN", False: "CLOSED"})
        return trades.sort_values("entry_date").reset_index(drop=True)

    def _annualized_return(self, daily_returns: pd.Series) -> float:
        compounded = (1 + daily_returns).prod()
        num_days = len(daily_returns)
        if num_days == 0:
            return 0.0
        return compounded ** (252 / num_days) - 1

    def _annualized_volatility(self, daily_returns: pd.Series) -> float:
        if daily_returns.empty:
            return 0.0
        daily_vol = daily_returns.std(ddof=0)
        return float(daily_vol * (252 ** 0.5)) if daily_vol != 0 else 0.0

    def _sharpe_ratio(self, daily_returns: pd.Series, volatility: float) -> float:
        if daily_returns.empty or volatility == 0:
            return 0.0
        mean_return = daily_returns.mean()
        return float(mean_return * 252 / volatility)

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        cumulative_max = equity_curve.cummax()
        drawdowns = equity_curve / cumulative_max - 1
        return drawdowns.min()


__all__ = ["BacktestEngine", "BacktestResult", "Strategy"]
