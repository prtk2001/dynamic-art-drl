"""
Dynamic ART-DRL — Upstox Market Data Client
=============================================
Direct REST-based client for the Upstox API v2.  Fetches historical OHLCV
candles and real-time quotes for MCX futures (Gold, Crude Oil).

Key design decisions
--------------------
* **No SDK dependency** — uses ``requests`` directly for full control over
  URL-path encoding (the Upstox v2 historical endpoint uses URL path params
  *not* query params).
* **Active-contract resolution** — MCX futures expire monthly; the client
  resolves the current near-month instrument key automatically.
* **Parquet caching** — identical to :pymod:`data.alpaca_client`.

API reference:  https://upstox.com/developer/api-documentation/
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd
import requests
from loguru import logger

from config.settings import (
    DATA_CACHE_DIR,
    DATA_END_DATE,
    DATA_START_DATE,
    UPSTOX_ACCESS_TOKEN,
    UPSTOX_API_KEY,
    UPSTOX_API_SECRET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.upstox.com/v2"

# Supported candle intervals for the historical-candle endpoint
_VALID_INTERVALS = {"1d", "1minute", "30minute", "week", "month"}

# Map our canonical timeframe strings to Upstox interval strings
_TF_TO_INTERVAL: dict[str, str] = {
    "1d": "day",
    "1h": "30minute",   # Upstox doesn't have 1h; use 30min as closest
    "30m": "30minute",
    "1m": "1minute",
    "1w": "week",
    "1M": "month",
    "day": "day",
    "30minute": "30minute",
    "1minute": "1minute",
    "week": "week",
    "month": "month",
}

# Month abbreviations for contract code generation
_MONTH_ABBR = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

# Canonical output columns
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Maximum date span per single API call (Upstox limits to ~1 year in some
# intervals).  We chunk requests to stay within limits.
_MAX_DAYS_PER_REQUEST = 365


class UpstoxClient:
    """Client for the Upstox REST API v2.

    Parameters
    ----------
    access_token : str, optional
        Bearer token for API auth.  Falls back to
        ``config.settings.UPSTOX_ACCESS_TOKEN``.
    cache_dir : Path | str, optional
        Cache directory.  Defaults to ``config.settings.DATA_CACHE_DIR``.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        self._token: str = access_token or UPSTOX_ACCESS_TOKEN
        self._cache_dir: Path = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._session: requests.Session = self._build_session()
        logger.info("UpstoxClient initialised (token={}…).", self._token[:8] if self._token else "EMPTY")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Create a ``requests.Session`` with auth header."""
        sess = requests.Session()
        if self._token:
            sess.headers.update({
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            })
        else:
            logger.warning("No Upstox access token set — API calls will fail.")
        return sess

    def _get(self, url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Perform a GET request and return parsed JSON.

        Raises
        ------
        RuntimeError
            On non-2xx status or network errors.
        """
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            logger.error("Upstox HTTP {} — {}", exc.response.status_code if exc.response else "?", body[:500])
            raise RuntimeError(f"Upstox API error: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Upstox network error: {}", exc)
            raise RuntimeError(f"Upstox network error: {exc}") from exc

    # ------------------------------------------------------------------
    # Instrument-key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def build_instrument_key(
        symbol: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> str:
        """Build an MCX futures instrument key for the given contract month.

        Parameters
        ----------
        symbol : str
            Base symbol, e.g. ``"GOLD"`` or ``"CRUDEOIL"``.
        year : int, optional
            Two-digit or four-digit year.  Defaults to current year.
        month : int, optional
            Month number (1-12).  Defaults to current month.

        Returns
        -------
        str
            e.g. ``"MCX_FO|GOLD25JUNFUT"``
        """
        now = datetime.utcnow()
        year = year or now.year
        month = month or now.month

        yy = year % 100
        mon_str = _MONTH_ABBR[month - 1]
        return f"MCX_FO|{symbol.upper()}{yy}{mon_str}FUT"

    @staticmethod
    def resolve_active_instrument_key(
        symbol: str,
        reference_date: Optional[datetime] = None,
    ) -> str:
        """Resolve the *near-month active* contract instrument key.

        MCX futures typically expire on the last trading day of the month.
        If we are past the 25th of the current month, we roll to next month's
        contract.

        Parameters
        ----------
        symbol : str
            ``"GOLD"`` or ``"CRUDEOIL"``.
        reference_date : datetime, optional
            Override for 'today'.  Defaults to ``datetime.utcnow()``.

        Returns
        -------
        str
            Instrument key for the near-month contract.
        """
        ref = reference_date or datetime.utcnow()
        # Roll to next month if past the 25th (expiry zone)
        if ref.day >= 25:
            # Move to 1st of next month
            if ref.month == 12:
                target_year = ref.year + 1
                target_month = 1
            else:
                target_year = ref.year
                target_month = ref.month + 1
        else:
            target_year = ref.year
            target_month = ref.month

        key = UpstoxClient.build_instrument_key(symbol, target_year, target_month)
        logger.debug("Resolved active contract for {}: {}", symbol, key)
        return key

    def search_instruments(
        self,
        query: str,
        exchange: str = "MCX",
    ) -> list[dict[str, Any]]:
        """Search for instruments matching a query string via the Upstox API.

        Parameters
        ----------
        query : str
            Search query, e.g. ``"GOLD"`` or ``"CRUDEOIL"``.
        exchange : str
            Exchange segment.  Default ``"MCX"``.

        Returns
        -------
        list[dict]
            List of instrument dicts returned by the API.
        """
        url = f"{_BASE_URL}/search/instruments"
        params = {"q": query, "exchange": exchange}
        try:
            data = self._get(url, params=params)
            instruments = data.get("data", [])
            logger.info("Found {} instruments for query='{}' on {}.", len(instruments), query, exchange)
            return instruments
        except RuntimeError:
            logger.warning("Instrument search failed for '{}' on {}.", query, exchange)
            return []

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        instrument_key: str,
        interval: str,
        start: str,
        end: str,
    ) -> Path:
        """Deterministic cache path."""
        # Sanitise the instrument key (contains '|')
        safe_ik = re.sub(r"[^A-Za-z0-9_]", "_", instrument_key)
        tag = f"{safe_ik}_{interval}_{start}_{end}"
        h = hashlib.md5(tag.encode()).hexdigest()[:12]
        filename = f"upstox_{safe_ik}_{interval}_{h}.parquet"
        return self._cache_dir / filename

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug("Cache hit: {}", path.name)
                return df
            except Exception as exc:
                logger.warning("Corrupt cache {} — re-downloading: {}", path, exc)
                path.unlink(missing_ok=True)
        return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(path, engine="pyarrow")
            logger.debug("Cached {} rows → {}", len(df), path.name)
        except Exception as exc:
            logger.error("Failed to write cache {}: {}", path, exc)

    # ------------------------------------------------------------------
    # Historical candles
    # ------------------------------------------------------------------

    def _fetch_candle_chunk(
        self,
        instrument_key: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """Fetch a single chunk of historical candles (≤ 1 year).

        Upstox v2 endpoint:
            GET /historical-candle/{instrumentKey}/{interval}/{to_date}/{from_date}

        NOTE: The instrument key in the URL must be **URL-encoded**
        (``MCX_FO%7CGOLD25JUNFUT``).
        """
        encoded_key = quote(instrument_key, safe="")
        url = (
            f"{_BASE_URL}/historical-candle/"
            f"{encoded_key}/{interval}/{to_date}/{from_date}"
        )

        data = self._get(url)
        candles = data.get("data", {}).get("candles", [])

        if not candles:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # Each candle: [timestamp, open, high, low, close, volume, oi]
        rows = []
        for c in candles:
            rows.append(
                {
                    "timestamp": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": int(c[5]),
                }
            )

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df[_OHLCV_COLUMNS]

    def fetch_historical(
        self,
        instrument_key: str,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV candles with automatic chunking.

        Parameters
        ----------
        instrument_key : str
            Full Upstox instrument key, e.g. ``"MCX_FO|GOLD25JUNFUT"``.
        interval : str
            Candle interval.  Accepts canonical strings (``"1d"``, ``"1h"``)
            which are mapped internally, or raw Upstox values
            (``"day"``, ``"30minute"``, etc.).
        start : str, optional
            ``"YYYY-MM-DD"`` start date.  Defaults to ``DATA_START_DATE``.
        end : str, optional
            ``"YYYY-MM-DD"`` end date.  Defaults to ``DATA_END_DATE``.
        use_cache : bool
            Whether to use / update the parquet cache.

        Returns
        -------
        pd.DataFrame
            Columns: ``open, high, low, close, volume``.
        """
        start = start or DATA_START_DATE
        end = end or DATA_END_DATE

        # Resolve canonical timeframe
        resolved_interval = _TF_TO_INTERVAL.get(interval, interval)
        if resolved_interval not in _VALID_INTERVALS and resolved_interval != "day":
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Valid: {_VALID_INTERVALS | set(_TF_TO_INTERVAL)}"
            )

        # Check cache
        cache_path = self._cache_key(instrument_key, resolved_interval, start, end)
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        logger.info(
            "Fetching {} {} candles from Upstox: {} → {}",
            instrument_key,
            resolved_interval,
            start,
            end,
        )

        # Chunk by _MAX_DAYS_PER_REQUEST to respect API limits
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        chunks: list[pd.DataFrame] = []

        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=_MAX_DAYS_PER_REQUEST), end_dt)
            from_str = cursor.strftime("%Y-%m-%d")
            to_str = chunk_end.strftime("%Y-%m-%d")

            try:
                chunk_df = self._fetch_candle_chunk(
                    instrument_key=instrument_key,
                    interval=resolved_interval,
                    from_date=from_str,
                    to_date=to_str,
                )
                if not chunk_df.empty:
                    chunks.append(chunk_df)
                    logger.debug(
                        "Chunk {} → {}: {} rows",
                        from_str,
                        to_str,
                        len(chunk_df),
                    )
            except RuntimeError as exc:
                logger.warning(
                    "Chunk {} → {} failed for {}: {}",
                    from_str,
                    to_str,
                    instrument_key,
                    exc,
                )

            cursor = chunk_end + timedelta(days=1)

        if not chunks:
            logger.warning("No data returned for {} ({} → {}).", instrument_key, start, end)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df = pd.concat(chunks).sort_index()
        # Remove potential duplicates from overlapping chunks
        df = df[~df.index.duplicated(keep="first")]

        # Cache
        if use_cache:
            self._write_cache(cache_path, df)

        logger.info(
            "Fetched {} candles for {} ({}): {} → {}.",
            len(df),
            instrument_key,
            resolved_interval,
            df.index.min(),
            df.index.max(),
        )
        return df

    def fetch_mcx(
        self,
        symbol: str,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Convenience method: resolve the active MCX contract and fetch data.

        Parameters
        ----------
        symbol : str
            ``"GOLD"`` or ``"CRUDEOIL"``.
        interval, start, end, use_cache
            See :meth:`fetch_historical`.
        reference_date : datetime, optional
            Override for contract resolution date.

        Returns
        -------
        pd.DataFrame
        """
        key = self.resolve_active_instrument_key(symbol, reference_date)
        return self.fetch_historical(
            instrument_key=key,
            interval=interval,
            start=start,
            end=end,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Real-time quotes
    # ------------------------------------------------------------------

    def get_ohlc_quote(self, instrument_key: str) -> dict[str, Any]:
        """Fetch real-time OHLC quote for an instrument.

        Endpoint
        --------
        ``GET /market-quote/ohlc?instrument_key={instrumentKey}``

        Returns
        -------
        dict
            Raw OHLC quote data from the API.
        """
        encoded_key = quote(instrument_key, safe="")
        url = f"{_BASE_URL}/market-quote/ohlc"
        params = {"instrument_key": encoded_key}

        data = self._get(url, params=params)
        quote_data = data.get("data", {})
        logger.debug("OHLC quote for {}: {}", instrument_key, quote_data)
        return quote_data

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Delete cached parquet files.

        Parameters
        ----------
        pattern : str, optional
            Glob pattern within cache dir.  Defaults to ``upstox_*.parquet``.

        Returns
        -------
        int
            Number of files deleted.
        """
        pat = pattern or "upstox_*.parquet"
        files = list(self._cache_dir.glob(pat))
        for f in files:
            f.unlink(missing_ok=True)
        logger.info("Cleared {} cached file(s) (pattern={}).", len(files), pat)
        return len(files)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_upstox_client(**kwargs) -> UpstoxClient:
    """Factory for the default UpstoxClient with project config."""
    return UpstoxClient(**kwargs)


if __name__ == "__main__":
    client = get_upstox_client()
    for sym in ("GOLD", "CRUDEOIL"):
        key = client.resolve_active_instrument_key(sym)
        print(f"Active contract for {sym}: {key}")
