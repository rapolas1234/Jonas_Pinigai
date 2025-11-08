import math
import pytest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    pytest.skip("pandas is required for backtest tests", allow_module_level=True)

from backtest.engine import BacktestEngine


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    steps = pd.Series(range(len(dates)), dtype=float)
    close = 100 + steps * 0.3 + steps.apply(lambda x: math.sin(x / 3.0) * 2)
    open_ = close - 0.2
    high = close + 0.5
    low = close - 0.5
    volume = (1_000_000 + steps * 1_000).astype(int)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def closed_strategy():
    class ClosedStrategy:
        def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
            signals = pd.Series(0.0, index=prices.index)
            signals.iloc[5:20] = 1.0
            signals.iloc[20:] = 0.0
            return signals

    return ClosedStrategy()


@pytest.fixture
def open_strategy():
    class OpenStrategy:
        def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
            signals = pd.Series(0.0, index=prices.index)
            signals.iloc[5:] = 1.0
            return signals

    return OpenStrategy()


def test_backtest_result_includes_trade_metrics(sample_prices, closed_strategy):
    engine = BacktestEngine(initial_capital=10_000.0)
    result = engine.run(sample_prices, closed_strategy)

    assert not result.trades.empty
    assert {"entry_price", "exit_price", "return_pct", "holding_period", "status"}.issubset(
        result.trades.columns
    )
    assert len(result.daily_returns) == len(sample_prices)
    assert result.volatility >= 0
    assert isinstance(result.sharpe_ratio, float)

    entry_date_expected = sample_prices.loc[6, "date"]
    exit_date_expected = sample_prices.loc[21, "date"]
    assert result.trades.iloc[0]["entry_date"] == entry_date_expected
    assert result.trades.iloc[0]["exit_date"] == exit_date_expected
    assert result.trades.iloc[0]["holding_period"] > 0


def test_backtest_marks_open_trades(sample_prices, open_strategy):
    engine = BacktestEngine(initial_capital=10_000.0)
    result = engine.run(sample_prices, open_strategy)

    assert not result.trades.empty
    last_trade = result.trades.iloc[-1]
    assert last_trade["status"] == "OPEN"
    assert last_trade["exit_date"] == sample_prices.iloc[-1]["date"]
    assert pd.notna(last_trade["return_pct"])
