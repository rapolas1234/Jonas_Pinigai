"""Historical price data loading utilities."""
from __future__ import annotations

import datetime as dt
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


DEFAULT_CACHE_DIR = Path(os.getenv("PRICE_DATA_CACHE", Path.home() / ".cache" / "jonas_pinigai"))


@dataclass
class HistoricalPriceLoader:
    """Loader for historical daily price data with simple file-based caching."""

    cache_dir: Path = DEFAULT_CACHE_DIR
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.session is None:
            self.session = requests.Session()

    def _cache_path(self, ticker: str) -> Path:
        safe_ticker = ticker.lower().replace("/", "-")
        return self.cache_dir / f"{safe_ticker}.csv"

    def _is_cache_fresh(self, cache_path: Path, max_age: dt.timedelta) -> bool:
        if not cache_path.exists():
            return False
        modified = dt.datetime.fromtimestamp(cache_path.stat().st_mtime, tz=dt.timezone.utc)
        return dt.datetime.now(tz=dt.timezone.utc) - modified < max_age

    def load_daily(self, ticker: str, max_age: dt.timedelta = dt.timedelta(hours=12)) -> pd.DataFrame:
        """Load daily historical data for *ticker*.

        Data is sourced from Stooq (https://stooq.com) which provides free historical quotes.
        Results are cached locally for ``max_age`` to limit network requests.
        """

        cache_path = self._cache_path(ticker)
        if self._is_cache_fresh(cache_path, max_age):
            return self._read_cached(cache_path)

        df = self._download_daily_bars(ticker)
        if not df.empty:
            df.to_csv(cache_path, index=False)
        return df

    def _read_cached(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _download_daily_bars(self, ticker: str) -> pd.DataFrame:
        url_ticker = ticker.lower()
        if not url_ticker.endswith(".us") and not url_ticker.endswith(".uk"):
            # Assume US exchange by default
            url_ticker = f"{url_ticker}.us"

        url = f"https://stooq.com/q/d/l/?s={url_ticker}&i=d"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        if "No data" in response.text:
            raise ValueError(f"No data available for ticker '{ticker}'")
        csv_bytes = response.content
        df = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=["Date"])
        df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        return df.sort_values("date").reset_index(drop=True)


__all__ = ["HistoricalPriceLoader"]
