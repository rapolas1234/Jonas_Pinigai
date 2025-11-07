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

### Command-line execution

You can also launch the backtest from the command line once the package is
installed:

```bash
python -m backtest.cli AAPL --fast 12 --slow 26 --capital 10000 --report reports/aapl.png
```

In addition to the text summary, the CLI now generates a visual report with the
price action, equity curve, and performance tables. By default the report is
saved to `./<ticker>_backtest.png`, and you can override the location via the
`--report` option. Pass `--show` to open the figure interactively after saving.
Use `python -m backtest.cli --help` to list all available options.

## Testing

```bash
pytest
```
