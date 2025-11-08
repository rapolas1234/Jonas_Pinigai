import math
import pytest

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    pytest.skip("matplotlib is required for report tests", allow_module_level=True)

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    pytest.skip("pandas is required for report tests", allow_module_level=True)

from backtest.engine import BacktestEngine
from backtest.report import create_report_figure, render_report


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
def closed_result(sample_prices: pd.DataFrame, closed_strategy):
    engine = BacktestEngine(initial_capital=10_000.0)
    return engine.run(sample_prices, closed_strategy)


def test_create_report_figure_has_three_axes(sample_prices, closed_result):
    fig = create_report_figure(
        sample_prices,
        closed_result,
        fast_window=5,
        slow_window=12,
        ticker="TEST",
    )
    try:
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_render_report_writes_file(tmp_path, sample_prices, closed_result):
    output = tmp_path / "report.png"
    path = render_report(
        sample_prices,
        closed_result,
        fast_window=5,
        slow_window=12,
        output=output,
        ticker="TEST",
        show=False,
    )
    assert path == output.resolve()
    assert path.exists()
    assert path.is_file()
