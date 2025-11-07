"""Configuration helpers for the trading package."""
from __future__ import annotations

from dataclasses import dataclass


def default_ticker() -> str:
    return "AAPL"


@dataclass(frozen=True)
class BacktestConfig:
    ticker: str = default_ticker()
    fast_window: int = 12
    slow_window: int = 26
    start_cash: float = 10_000.0
