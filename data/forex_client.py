"""
Dynamic ART-DRL — Forex Data Client (Yahoo Finance)
===================================================
Fetches historical OHLCV data for currency pairs (specifically USDEUR=X or EURUSD=X)
using yfinance. Caches data locally to parquet to prevent redundant API calls.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from config.settings import (
    DATA_CACHE_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
)

# Canonical column order
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class ForexClient:
    """Forex data downloader using Yahoo Finance and Alpaca Forex API.

    Parameters
    ----------
    cache_dir : Path | str, optional
        Directory for parquet caches. Defaults to ``config.settings.DATA_CACHE_DIR``.
    """

    def __init__(self, cache_dir: Optional[Path | str] = None) -> None:
        self._cache_dir: Path = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
        self._api_key = ALPACA_API_KEY
        self._secret_key = ALPACA_SECRET_KEY
        logger.info("ForexClient initialised (Yahoo Finance + Alpaca).")

    def _cache_key(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> Path:
        """Build a deterministic cache path."""
        tag = f"forex_{symbol}_{timeframe}_{start}_{end}"
        safe_name = hashlib.md5(tag.encode()).hexdigest()[:12]
        filename = f"forex_{symbol}_{timeframe}_{safe_name}.parquet"
        return self._cache_dir / filename

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Return cached DataFrame or None."""
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug("Forex Cache hit: {}", path.name)
                return df
            except Exception as exc:
                logger.warning("Corrupt Forex cache {} — re-downloading: {}", path, exc)
                path.unlink(missing_ok=True)
        return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        """Persist DataFrame to parquet."""
        try:
            df.to_parquet(path, engine="pyarrow")
            logger.debug("Forex Cached {} rows → {}", len(df), path.name)
        except Exception as exc:
            logger.error("Failed to write Forex cache {}: {}", path, exc)

    def _fetch_from_alpaca(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch historical Forex rates from Alpaca REST API."""
        import requests
        
        # Map timeframe to Alpaca format: "5min" -> "5Min", "15min" -> "15Min"
        tf_map = {
            "5min": "5Min",
            "15min": "15Min",
        }
        alpaca_tf = tf_map.get(timeframe, "5Min")
        
        # Alpaca uses slash in currency symbols (e.g. USD/EUR)
        alpaca_symbol = symbol
        if "/" not in alpaca_symbol and len(alpaca_symbol) == 6:
            alpaca_symbol = f"{alpaca_symbol[:3]}/{alpaca_symbol[3:]}"
            
        url = "https://data.alpaca.markets/v1beta1/forex/rates"
        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }
        
        # Format dates to RFC3339 strings
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
            end_dt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y-%m-%dT23:59:59Z")
        except ValueError:
            # Fallback if already formatted or different
            start_dt = start
            end_dt = end
            
        all_bars = []
        page_token = None
        
        logger.info("Requesting Forex data from Alpaca for symbol={} interval={} start={} end={}", 
                    alpaca_symbol, alpaca_tf, start_dt, end_dt)
        
        while True:
            params = {
                "currency_pairs": alpaca_symbol,
                "timeframe": alpaca_tf,
                "start": start_dt,
                "end": end_dt,
                "limit": 10000,
            }
            if page_token:
                params["page_token"] = page_token
                
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                logger.error(
                    "Alpaca Forex API error: HTTP {} - {}",
                    response.status_code,
                    response.text,
                )
                break
                
            data = response.json()
            rates = data.get("rates", {})
            symbol_bars = rates.get(alpaca_symbol, [])
            
            if not symbol_bars:
                break
                
            all_bars.extend(symbol_bars)
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
        if not all_bars:
            logger.warning("No bars returned from Alpaca Forex API for symbol {}", alpaca_symbol)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
            
        # Parse bars
        records = []
        for bar in all_bars:
            records.append({
                "timestamp": pd.to_datetime(bar["t"]),
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar["v"]),
            })
            
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        # Ensure UTC conversion is timezone-naive to match yfinance output
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.sort_index(inplace=True)
        
        return df

    def _fetch_from_yfinance_subhourly(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch historical sub-hourly Forex rates from Yahoo Finance (last 60 days limit)."""
        yf_symbol = symbol
        if not yf_symbol.endswith("=X"):
            yf_symbol = f"{yf_symbol}=X"
            
        interval = "5m" if timeframe == "5min" else "15m"
        
        # Yahoo Finance limits 5m/15m data to the last 60 days.
        limit_start = (datetime.utcnow() - pd.Timedelta(days=58)).strftime("%Y-%m-%d")
        if start < limit_start:
            logger.warning(
                "Yahoo Finance limits sub-hourly data to 60 days. "
                "Adjusting start date from {} to {} for {} request.",
                start,
                limit_start,
                timeframe,
            )
            start = limit_start
            
        logger.info(
            "Fetching {} {} bars from Yahoo Finance: {} → {}",
            yf_symbol,
            timeframe,
            start,
            end,
        )
        
        try:
            df = yf.download(
                tickers=yf_symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error("yfinance API error for {}: {}", yf_symbol, exc)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
            
        if df.empty:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
            
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.columns = [c.lower() for c in df.columns]
        df = df[[c for c in _OHLCV_COLUMNS if c in df.columns]]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.sort_index(inplace=True)
        
        # Clean numeric data
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        return df

    def fetch_ohlcv(
        self,
        symbol: str = "USDEUR=X",
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch Forex OHLCV bars from Yahoo Finance or Alpaca Forex API.

        Parameters
        ----------
        symbol : str
            Yahoo Finance ticker or currency pair, e.g. ``"USDEUR=X"`` or ``"USDEUR"``.
        timeframe : str
            One of ``"1d"``, ``"1h"``, ``"5min"``, or ``"15min"``.
        start : str, optional
            ISO date string ``"YYYY-MM-DD"``.
        end : str, optional
            ISO date string ``"YYYY-MM-DD"``.
        use_cache : bool
            Whether to use cache.

        Returns
        -------
        pd.DataFrame
            Columns: ``open, high, low, close, volume``.
        """
        start = start or DATA_START_DATE
        end = end or DATA_END_DATE
        symbol = symbol.upper()

        # If Alpaca format or sub-hourly, handle symbol matching
        clean_symbol = symbol.replace("=X", "")

        # Check cache
        cache_path = self._cache_key(clean_symbol, timeframe, start, end)
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        # Route sub-hourly timeframes to Alpaca
        if timeframe in ("5min", "15min"):
            try:
                df = self._fetch_from_alpaca(clean_symbol, timeframe, start, end)
                if df.empty:
                    logger.warning("Alpaca Forex API returned empty data. Falling back to yfinance...")
                    df = self._fetch_from_yfinance_subhourly(clean_symbol, timeframe, start, end)
            except Exception as exc:
                logger.warning("Alpaca Forex API failed ({}). Falling back to yfinance...", exc)
                df = self._fetch_from_yfinance_subhourly(clean_symbol, timeframe, start, end)
        elif timeframe in ("1d", "1h"):
            # Map clean symbol back to Yahoo Finance format
            yf_symbol = clean_symbol
            if not yf_symbol.endswith("=X"):
                yf_symbol = f"{yf_symbol}=X"

            # Map timeframe to yfinance interval
            if timeframe == "1d":
                interval = "1d"
            else:
                interval = "1h"
                # Yahoo Finance limits hourly data to the last 730 days.
                limit_start = (datetime.utcnow() - pd.Timedelta(days=725)).strftime("%Y-%m-%d")
                if start < limit_start:
                    logger.warning(
                        "Yahoo Finance limits hourly data to 730 days. "
                        "Adjusting start date from {} to {} for hourly request.",
                        start,
                        limit_start,
                    )
                    start = limit_start

            logger.info(
                "Fetching {} {} bars from Yahoo Finance: {} → {}",
                yf_symbol,
                timeframe,
                start,
                end,
            )

            try:
                # Download from yfinance
                df = yf.download(
                    tickers=yf_symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
            except Exception as exc:
                logger.error("yfinance API error for {}: {}", yf_symbol, exc)
                raise RuntimeError(f"Failed to fetch Forex data: {exc}") from exc

            if df.empty:
                logger.warning("yfinance returned 0 bars for {}.", yf_symbol)
                return pd.DataFrame(columns=_OHLCV_COLUMNS)

            # Handle multi-level columns if returned (yfinance sometimes does this)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Filter out to canonical columns
            df = df[[c for c in _OHLCV_COLUMNS if c in df.columns]]

            # Ensure index is datetime and sorted
            df.index = pd.to_datetime(df.index)
            # Ensure UTC timezone alignment
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df.sort_index(inplace=True)

            # Clean NaNs
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.ffill(inplace=True)
            df.bfill(inplace=True)
        else:
            raise ValueError(f"Unsupported Forex timeframe '{timeframe}'. Choose '1d', '1h', '5min', or '15min'.")

        # Cache the result
        if not df.empty:
            self._write_cache(cache_path, df)
            logger.info(
                "Fetched {} Forex bars for {} ({}) — date range {} to {}.",
                len(df),
                clean_symbol,
                timeframe,
                df.index.min(),
                df.index.max(),
            )
        return df


def get_forex_client(**kwargs) -> ForexClient:
    """Factory for default ForexClient."""
    return ForexClient(**kwargs)


if __name__ == "__main__":
    # Test fetch
    client = get_forex_client()
    df = client.fetch_ohlcv("USDEUR=X", "1d", "2024-01-01", "2024-12-31")
    print(df.head())
