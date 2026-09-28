"""
Dynamic ART-DRL — Alpaca Historical Data Client
=================================================
Uses the alpaca-py SDK (``StockHistoricalDataClient``) to fetch OHLCV bars
for US-listed ETFs (GLD, USO).  Results are cached to ``data_cache/`` as
Parquet files so that repeated calls never re-download the same data.

Key features
------------
* Daily and hourly timeframes.
* Transparent caching with cache-key = ``{symbol}_{timeframe}_{start}_{end}.parquet``.
* Canonical DataFrame output: ``open, high, low, close, volume`` (lowercase).
* Fully typed, production-quality, with loguru logging.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DATA_CACHE_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
)

# ---------------------------------------------------------------------------
# Lazy-import alpaca-py so module can be imported even without the lib installed
# (e.g. in unit-test stubs).  The import guard is at function level.
# ---------------------------------------------------------------------------

_ALPACA_AVAILABLE: bool = True
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning(
        "alpaca-py not installed — AlpacaClient will raise on data requests."
    )

# Canonical column order returned by every public method.
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class AlpacaClient:
    """Thin wrapper around the Alpaca v2 Market-Data SDK.

    Parameters
    ----------
    api_key : str, optional
        Alpaca API key.  Falls back to ``config.settings.ALPACA_API_KEY``.
    secret_key : str, optional
        Alpaca secret key.  Falls back to ``config.settings.ALPACA_SECRET_KEY``.
    cache_dir : Path | str, optional
        Directory for parquet caches. Defaults to ``config.settings.DATA_CACHE_DIR``.
    """

    # Mapping from our canonical timeframe strings to alpaca-py TimeFrame
    _TF_MAP: dict[str, "TimeFrame"] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for AlpacaClient. "
                "Install with: pip install alpaca-py"
            )

        self._api_key: str = api_key or ALPACA_API_KEY
        self._secret_key: str = secret_key or ALPACA_SECRET_KEY
        self._cache_dir: Path = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Build client (no auth needed for free historical data, but we
        # supply keys when available for higher rate limits)
        if self._api_key and self._secret_key:
            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            logger.info("AlpacaClient initialised with API key auth.")
        else:
            self._client = StockHistoricalDataClient()
            logger.info(
                "AlpacaClient initialised without auth (anonymous — lower rate limits)."
            )

        # Populate timeframe map now that the import is guaranteed
        if not AlpacaClient._TF_MAP:
            AlpacaClient._TF_MAP = {
                "1d": TimeFrame.Day,
                "1h": TimeFrame.Hour,
            }

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> Path:
        """Build a deterministic cache path for the given request params."""
        tag = f"{symbol}_{timeframe}_{start}_{end}"
        safe_name = hashlib.md5(tag.encode()).hexdigest()[:12]
        filename = f"alpaca_{symbol}_{timeframe}_{safe_name}.parquet"
        return self._cache_dir / filename

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Return cached DataFrame or None."""
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug("Cache hit: {}", path.name)
                return df
            except Exception as exc:
                logger.warning("Corrupt cache file {} — re-downloading. Error: {}", path, exc)
                path.unlink(missing_ok=True)
        return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        """Persist DataFrame to parquet."""
        try:
            df.to_parquet(path, engine="pyarrow")
            logger.debug("Cached {} rows → {}", len(df), path.name)
        except Exception as exc:
            logger.error("Failed to write cache {}: {}", path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpaca (or cache).

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. ``"GLD"`` or ``"USO"``.
        timeframe : str
            One of ``"1d"`` (daily) or ``"1h"`` (hourly).
        start : str, optional
            ISO date string ``"YYYY-MM-DD"``.  Defaults to ``DATA_START_DATE``.
        end : str, optional
            ISO date string ``"YYYY-MM-DD"``.  Defaults to ``DATA_END_DATE``.
        use_cache : bool
            If ``True`` (default), return cached data when available.

        Returns
        -------
        pd.DataFrame
            Index: ``DatetimeIndex`` (UTC-aware), columns: ``open, high, low, close, volume``.

        Raises
        ------
        ValueError
            If ``timeframe`` is unsupported.
        RuntimeError
            If the API call fails.
        """
        start = start or DATA_START_DATE
        end = end or DATA_END_DATE
        symbol = symbol.upper()

        if timeframe not in self._TF_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Choose from: {list(self._TF_MAP)}"
            )

        # Check cache
        cache_path = self._cache_key(symbol, timeframe, start, end)
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        logger.info(
            "Fetching {} {} bars from Alpaca: {} → {}",
            symbol,
            timeframe,
            start,
            end,
        )

        try:
            feed_param = DataFeed.IEX if _ALPACA_AVAILABLE else "iex"
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._TF_MAP[timeframe],
                start=datetime.fromisoformat(start),
                end=datetime.fromisoformat(end),
                feed=feed_param,
            )
            bars = self._client.get_stock_bars(request)
        except Exception as exc:
            logger.error("Alpaca API error for {} {}: {}", symbol, timeframe, exc)
            raise RuntimeError(
                f"Failed to fetch {symbol} data from Alpaca: {exc}"
            ) from exc

        # Convert to DataFrame
        df = bars.df
        if df.empty:
            logger.warning("Alpaca returned 0 bars for {} {} ({} → {}).", symbol, timeframe, start, end)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # alpaca-py returns a MultiIndex (symbol, timestamp) for multi-symbol
        # requests.  Flatten if needed.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only canonical OHLCV columns (alpaca may return extras like
        # trade_count, vwap)
        available = [c for c in _OHLCV_COLUMNS if c in df.columns]
        df = df[available]

        # Ensure numeric types
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort chronologically
        df.sort_index(inplace=True)

        # Cache the result
        self._write_cache(cache_path, df)

        logger.info(
            "Fetched {} bars for {} ({}) — date range {} to {}.",
            len(df),
            symbol,
            timeframe,
            df.index.min(),
            df.index.max(),
        )
        return df

    def fetch_multi(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from symbol → OHLCV DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                results[sym] = self.fetch_ohlcv(
                    symbol=sym,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    use_cache=use_cache,
                )
            except Exception as exc:
                logger.error("Skipping {} due to error: {}", sym, exc)
        return results

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """Delete cached parquet files.

        Parameters
        ----------
        symbol : str, optional
            If given, only delete caches for that symbol.  Otherwise delete
            all Alpaca caches.

        Returns
        -------
        int
            Number of files deleted.
        """
        pattern = f"alpaca_{symbol}_*.parquet" if symbol else "alpaca_*.parquet"
        files = list(self._cache_dir.glob(pattern))
        for f in files:
            f.unlink(missing_ok=True)
        logger.info("Cleared {} cached file(s) (pattern={}).", len(files), pattern)
        return len(files)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_alpaca_client(**kwargs) -> AlpacaClient:
    """Factory for the default AlpacaClient with project config."""
    return AlpacaClient(**kwargs)


if __name__ == "__main__":
    # Quick smoke test
    client = get_alpaca_client()
    for sym in ("GLD", "USO"):
        df = client.fetch_ohlcv(sym, timeframe="1d")
        print(f"{sym}: {len(df)} rows, columns={list(df.columns)}")
        print(df.head(3))
        print()
