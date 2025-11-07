# Jonas Pinigai Backtesting Toolkit

A lightweight Python package for downloading historical equity data, running
technical strategies, and evaluating their performance with a simple backtester.

## Installation

Create and activate a virtual environment, then install the package in editable
mode together with optional testing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

## Usage

Run the following snippet to execute a moving average crossover backtest:

```python
from data.historical import HistoricalPriceLoader
from strategies.moving_average import MovingAverageCrossover
from backtest.engine import BacktestEngine

loader = HistoricalPriceLoader()
prices = loader.load_daily("AAPL")

strategy = MovingAverageCrossover(fast_window=12, slow_window=26)
engine = BacktestEngine(initial_capital=10_000)

result = engine.run(prices, strategy)
print("Total return:", f"{result.total_return:.2%}")
print("Annualized return:", f"{result.annualized_return:.2%}")
print("Max drawdown:", f"{result.max_drawdown:.2%}")
print(result.trades.tail())
```

The price loader caches requests under `~/.cache/jonas_pinigai` by default to
avoid repeated network calls. Delete files in that directory or set the
`PRICE_DATA_CACHE` environment variable to change the cache location.

## Testing

```bash
pytest
```
