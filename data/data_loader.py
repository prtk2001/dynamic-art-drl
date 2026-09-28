"""
Dynamic ART-DRL — Unified Market Data Loader
=============================================
Centralizes data loading for all assets (US ETFs, MCX futures, Nifty, Forex).
Handles local Parquet caches, Upstox API queries, and Yahoo Finance fallbacks.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from config.assets import get_asset
from config.settings import (
    DATA_CACHE_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
)
from data.alpaca_client import AlpacaClient
from data.forex_client import ForexClient
from data.upstox_client import UpstoxClient


def load_asset_data(
    asset_name: str,
    timeframe: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load, stitch, and cache historical OHLCV data for the given asset.

    Parameters
    ----------
    asset_name : str
        Asset name in ASSET_REGISTRY, e.g. 'GLD', 'NIFTY_FUT', 'USDEUR'.
    timeframe : str
        '1d' (daily) or '1h' (hourly).
    start : str, optional
        ISO start date 'YYYY-MM-DD'.
    end : str, optional
        ISO end date 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume (and options columns if NIFTY_OPT).
    """
    start = start or DATA_START_DATE
    end = end or DATA_END_DATE
    asset_config = get_asset(asset_name)

    logger.info(
        "Loading data for {} ({}): {} → {}",
        asset_name,
        timeframe,
        start,
        end,
    )

    # 1. USD/EUR Forex Pair
    if asset_name == "USDEUR":
        client = ForexClient()
        df = client.fetch_ohlcv("USDEUR=X", timeframe, start, end)
        return df

    # 2. Indian Index Futures & Options (Nifty, Bank Nifty, Sensex)
    elif asset_name in ["NIFTY_FUT", "NIFTY_OPT", "BANKNIFTY_FUT", "BANKNIFTY_OPT", "SENSEX_FUT", "SENSEX_OPT"]:
        # Resolve index specific properties
        if "BANKNIFTY" in asset_name:
            index_name = "BANKNIFTY"
            yf_ticker = "^NSEBANK"
            strike_interval = 100.0
        elif "SENSEX" in asset_name:
            index_name = "SENSEX"
            yf_ticker = "^BSESN"
            strike_interval = 100.0
        else:
            index_name = "NIFTY"
            yf_ticker = "^NSEI"
            strike_interval = 50.0

        is_opt = "_OPT" in asset_name
        cache_name = f"{index_name.lower()}_local_{timeframe}.parquet"
        cache_path = DATA_CACHE_DIR / cache_name
        
        if cache_path.exists():
            logger.info("Loading {} local data from cache...", index_name)
            df_local = pd.read_parquet(cache_path)
            if df_local.index.tz is not None:
                df_local.index = df_local.index.tz_localize(None)
            # Filter range
            df_local = df_local[(df_local.index >= start) & (df_local.index <= end)]
        else:
            logger.warning("Local {} compiled parquet cache not found at {}!", index_name, cache_path)
            df_local = pd.DataFrame()

        # If data range extends into 2025/2026, stitch it from Upstox (only supported for 1d and 1h)
        if end > "2024-12-31" and timeframe in ["1d", "1h"]:
            logger.info("Fetching {} 2025-2026 data...", index_name)
            upstox_start = max(start, "2025-01-01")
            
            try:
                # 2a. Fetch from Upstox
                client = UpstoxClient()
                df_upstox = pd.DataFrame()
                
                # Resolve and download Index Future
                ref_dt = datetime.strptime(upstox_start, "%Y-%m-%d")
                fut_key, fut_symbol, fut_expiry = client.resolve_index_future_key(index_name, ref_dt)
                
                logger.info("Upstox resolved {} Future contract: {} (expiry={})", index_name, fut_symbol, fut_expiry)
                df_fut = client.fetch_historical(
                    instrument_key=fut_key,
                    interval=timeframe,
                    start=upstox_start,
                    end=end,
                )
                
                if not df_fut.empty:
                    if not is_opt:
                        df_upstox = df_fut
                    else:
                        # For options, we fetch the ATM premiums
                        logger.info("Upstox fetching ATM option premiums for {}...", index_name)
                        processed_rows = []
                        
                        # Loop through future prices to resolve options
                        for ts, row in df_fut.iterrows():
                            close_p = row["close"]
                            strike = int(round(close_p / strike_interval) * strike_interval)
                            
                            # Resolve option keys
                            try:
                                ce_key, ce_sym = client.resolve_index_option_key(index_name, ts, fut_expiry, strike, "CE")
                                pe_key, pe_sym = client.resolve_index_option_key(index_name, ts, fut_expiry, strike, "PE")
                                
                                # Fetch option premiums (1 bar on that timestamp)
                                df_ce = client.fetch_historical(
                                    instrument_key=ce_key,
                                    interval=timeframe,
                                    start=ts.strftime("%Y-%m-%d"),
                                    end=ts.strftime("%Y-%m-%d"),
                                )
                                df_pe = client.fetch_historical(
                                    instrument_key=pe_key,
                                    interval=timeframe,
                                    start=ts.strftime("%Y-%m-%d"),
                                    end=ts.strftime("%Y-%m-%d"),
                                )
                                
                                ce_val = df_ce.loc[ts, "close"] if ts in df_ce.index else 100.0
                                pe_val = df_pe.loc[ts, "close"] if ts in df_pe.index else 100.0
                            except Exception:
                                # Fallback approximation if option retrieval fails
                                ce_val = max(5.0, close_p - strike)
                                pe_val = max(5.0, strike - close_p)
                                
                            processed_rows.append({
                                "timestamp": ts,
                                "open": row["open"],
                                "high": row["high"],
                                "low": row["low"],
                                "close": close_p,
                                "volume": row["volume"],
                                "open_interest": row.get("open_interest", 0.0),
                                "ce_close": ce_val,
                                "pe_close": pe_val,
                                "ce_close_next": ce_val, # Simple MTM placeholder
                                "pe_close_next": pe_val,
                                "atm_strike": strike,
                                "expiry": pd.to_datetime(fut_expiry)
                            })
                            
                        df_upstox = pd.DataFrame(processed_rows)
                        if not df_upstox.empty:
                            df_upstox.set_index("timestamp", inplace=True)
                            
                # Merge local and Upstox F&O
                if not df_upstox.empty:
                    if df_upstox.index.tz is not None:
                        df_upstox.index = df_upstox.index.tz_localize(None)
                    if df_local.empty:
                        df_local = df_upstox
                    else:
                        # Drop overlaps and concatenate
                        df_local = df_local[df_local.index < df_upstox.index.min()]
                        df_local = pd.concat([df_local, df_upstox])
                    
                if len(df_local) < 100:
                    raise ValueError("Merged dataframe from local + Upstox has insufficient data (under 100 rows).")
                        
            except Exception as e:
                logger.error("Upstox {} F&O retrieval failed: {}. Falling back to Yahoo Finance index proxy.", index_name, e)
                # 2b. Fallback to Yahoo Finance for index proxy
                try:
                    df_yf = yf.download(yf_ticker, start=upstox_start, end=end, interval="1d" if timeframe == "1d" else "1h")
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.droplevel(1)
                    df_yf.columns = [c.lower() for c in df_yf.columns]
                    # Format matching
                    df_yf = df_yf[["open", "high", "low", "close", "volume"]]
                    
                    if is_opt:
                        # Approximate premium columns if Options requested but yfinance used
                        df_yf["atm_strike"] = (df_yf["close"] / strike_interval).round() * strike_interval
                        df_yf["ce_close"] = (df_yf["close"] * 0.015).round() # 1.5% premium approx
                        df_yf["pe_close"] = (df_yf["close"] * 0.015).round()
                        df_yf["ce_close_next"] = df_yf["ce_close"].shift(-1).fillna(df_yf["ce_close"])
                        df_yf["pe_close_next"] = df_yf["pe_close"].shift(-1).fillna(df_yf["pe_close"])
                        df_yf["expiry"] = df_yf.index
                        
                    if df_local.empty:
                        df_local = df_yf
                    else:
                        df_local = df_local[df_local.index < df_yf.index.min()]
                        df_local = pd.concat([df_local, df_yf])
                except Exception as yf_err:
                    logger.critical("All {} data fallbacks failed: {}", index_name, yf_err)

        # Standardize columns to lowercase
        df_local.columns = [c.lower() for c in df_local.columns]
        return df_local

    # 3. Gold & Crude Oil (Alpaca / Upstox MCX)
    else:
        if asset_config.exchange == "MCX":
            try:
                client = UpstoxClient()
                df = client.fetch_mcx(symbol=asset_config.symbol, interval=timeframe, start=start, end=end)
                if len(df) > 100:
                    return df
                else:
                    raise ValueError("Insufficient data returned from Upstox (likely a single contract with no history).")
            except Exception as e:
                logger.warning("MCX data fetch from Upstox failed or was insufficient: {}. Falling back to GLD/USO USD proxy scaled to INR.", e)
                proxy_symbol = "GLD" if asset_config.symbol == "GOLD" else "USO"
                df_proxy = load_asset_data(proxy_symbol, timeframe, start, end)
                
                # Make a copy to avoid mutating cache
                df_proxy = df_proxy.copy()
                # Multiply prices by 83.0 to approximate MCX INR prices
                for col in ["open", "high", "low", "close"]:
                    if col in df_proxy.columns:
                        df_proxy[col] = df_proxy[col] * 83.0
                return df_proxy
        else:
            client = AlpacaClient()
            df = client.fetch_ohlcv(symbol=asset_config.symbol, timeframe=timeframe, start=start, end=end)
            return df
