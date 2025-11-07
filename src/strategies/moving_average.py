"""Technical analysis strategies."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MovingAverageCrossover:
    """Exponential moving average crossover strategy."""

    fast_window: int = 12
    slow_window: int = 26

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return target position (1 long / 0 flat) per bar."""
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must include a 'close' column")
        close = prices["close"]
        fast_ema = close.ewm(span=self.fast_window, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_window, adjust=False).mean()
        signal = (fast_ema > slow_ema).astype(int)
        return signal


__all__ = ["MovingAverageCrossover"]
