"""
Dynamic ART-DRL — Nifty Local Data Parser
==========================================
Parses local 1-minute bar CSV files for Nifty futures and options (2020–2024).
Reconstructs continuous futures (NIFTY-I.NFO) and ATM Call/Put option series,
resamples them to 1d and 1h intervals, and caches them as Parquet.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    DATA_CACHE_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
)

# Column names in output
_OUTPUT_COLS_FUT = ["open", "high", "low", "close", "volume", "open_interest"]
_OUTPUT_COLS_OPT = ["ce_close", "pe_close", "atm_strike", "expiry"]

# Regex to parse option details: e.g. NIFTY02JAN2011000CE.NFO
# Group 1: Expiry date (e.g. 02JAN20)
# Group 2: Strike price (e.g. 11000)
# Group 3: Option type (e.g. CE or PE)
OPTION_REGEX = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)\.NFO$")


class NiftyLocalParser:
    """Parser for the local Nifty futures and options 1-minute CSV dataset."""

    def __init__(
        self,
        data_dir: str = "/Users/prateek/Documents/nifty_options/data/nifty_bank_nifty",
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("NiftyLocalParser initialised with data_dir={}", self.data_dir)

    def parse_expiry_date(self, expiry_str: str) -> Optional[datetime]:
        """Convert a 3-letter month expiry string like '02JAN20' to a datetime."""
        try:
            return datetime.strptime(expiry_str, "%d%b%y")
        except ValueError:
            return None

    def process_day_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single day's 1-minute CSV file.

        Returns a DataFrame with columns:
        - future_open, future_high, future_low, future_close, future_volume, future_oi
        - ce_close (ATM CE option price)
        - pe_close (ATM PE option price)
        - atm_strike
        - expiry (datetime)
        """
        try:
            # Load file (can be large, so filter immediately)
            df_raw = pd.read_csv(filepath)
        except Exception as e:
            logger.error("Failed to read {}: {}", filepath.name, e)
            return None

        if df_raw.empty:
            return None

        # Standardise column names
        df_raw.columns = [c.strip() for c in df_raw.columns]

        # 1. Filter for Nifty tickers
        df_nifty = df_raw[df_raw["Ticker"].str.startswith("NIFTY", na=False)].copy()
        if df_nifty.empty:
            return None

        # Clean strings
        df_nifty["Ticker"] = df_nifty["Ticker"].str.strip()

        # Parse trading date
        sample_date_str = df_nifty["Date"].iloc[0]
        try:
            trading_date = datetime.strptime(sample_date_str, "%d/%m/%Y")
        except ValueError:
            try:
                trading_date = datetime.strptime(sample_date_str, "%Y-%m-%d")
            except ValueError:
                logger.error("Unknown date format in file {}: {}", filepath.name, sample_date_str)
                return None

        # 2. Extract Nifty Future (NIFTY-I.NFO)
        df_fut = df_nifty[df_nifty["Ticker"] == "NIFTY-I.NFO"].copy()
        if df_fut.empty:
            logger.warning("No NIFTY-I.NFO found in {}", filepath.name)
            return None

        # 3. Determine nearest option expiry date
        # Parse all option tickers to extract expiry dates
        expiries = []
        option_tickers = []
        
        for ticker in df_nifty["Ticker"].unique():
            match = OPTION_REGEX.match(ticker)
            if match:
                expiry_str, strike_str, opt_type = match.groups()
                expiry_dt = self.parse_expiry_date(expiry_str)
                if expiry_dt and expiry_dt >= trading_date:
                    expiries.append(expiry_dt)
                    option_tickers.append((ticker, expiry_dt, int(strike_str), opt_type))

        if not expiries:
            logger.warning("No valid option expiries found for {}", filepath.name)
            return None

        # Find the minimum expiry >= trading date
        nearest_expiry = min(expiries)
        nearest_expiry_str = nearest_expiry.strftime("%d%b%y").upper()

        # Keep only options that match this nearest expiry
        # Re-build option tickers list with only selected expiry
        valid_options = [t for t in option_tickers if t[1] == nearest_expiry]
        if not valid_options:
            return None

        # Pivot the option contracts so we can easily query Close by Time, Strike, and OptionType
        df_opts = df_nifty[df_nifty["Ticker"].isin([v[0] for v in valid_options])].copy()
        if df_opts.empty:
            return None

        # Extract strike and type from ticker in the pivoted dataframe
        def extract_strike_type(ticker: str) -> Tuple[int, str]:
            match = OPTION_REGEX.match(ticker)
            if match:
                return int(match.group(2)), match.group(3)
            return 0, ""

        strike_type_map = {t[0]: extract_strike_type(t[0]) for t in valid_options}
        df_opts["strike"] = df_opts["Ticker"].map(lambda x: strike_type_map.get(x, (0, ""))[0])
        df_opts["opt_type"] = df_opts["Ticker"].map(lambda x: strike_type_map.get(x, (0, ""))[1])

        # Create time-indexed futures series
        df_fut["timestamp"] = pd.to_datetime(df_fut["Date"] + " " + df_fut["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        # Fallback for alternative date format
        if df_fut["timestamp"].isna().any():
            df_fut["timestamp"] = pd.to_datetime(df_fut["Date"] + " " + df_fut["Time"], errors="coerce")
        
        df_fut.dropna(subset=["timestamp"], inplace=True)
        df_fut.set_index("timestamp", inplace=True)
        df_fut.sort_index(inplace=True)

        # Build options lookup: Time -> Strike -> Type -> Close
        df_opts["timestamp"] = pd.to_datetime(df_opts["Date"] + " " + df_opts["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if df_opts["timestamp"].isna().any():
            df_opts["timestamp"] = pd.to_datetime(df_opts["Date"] + " " + df_opts["Time"], errors="coerce")
        
        df_opts.dropna(subset=["timestamp"], inplace=True)
        
        # Create a pivoted table for options close prices: Index=timestamp, Columns=MultiIndex(strike, opt_type)
        df_opts_pivot = df_opts.pivot(index="timestamp", columns=["strike", "opt_type"], values="Close")
        # Forward fill and backward fill to handle missing ticks in individual strikes
        df_opts_pivot.ffill(inplace=True)
        df_opts_pivot.bfill(inplace=True)

        # 4. Construct ATM series aligned with future timestamps
        processed_rows = []
        for ts, fut_row in df_fut.iterrows():
            fut_close = fut_row["Close"]
            atm_strike = int(round(fut_close / 50.0) * 50)
            
            # Retrieve premiums for ATM CE and PE
            ce_price = np.nan
            pe_price = np.nan
            
            if ts in df_opts_pivot.index:
                row_options = df_opts_pivot.loc[ts]
                if (atm_strike, "CE") in row_options.index:
                    ce_price = row_options.loc[(atm_strike, "CE")]
                if (atm_strike, "PE") in row_options.index:
                    pe_price = row_options.loc[(atm_strike, "PE")]
                    
            processed_rows.append({
                "timestamp": ts,
                "open": fut_row["Open"],
                "high": fut_row["High"],
                "low": fut_row["Low"],
                "close": fut_close,
                "volume": fut_row["Volume"],
                "open_interest": fut_row["Open Interest"],
                "ce_close": ce_price,
                "pe_close": pe_price,
                "atm_strike": atm_strike,
                "expiry": nearest_expiry
            })

        df_day = pd.DataFrame(processed_rows)
        df_day.set_index("timestamp", inplace=True)
        # Forward-fill any missing option premiums during the trading day
        df_day.ffill(inplace=True)
        df_day.bfill(inplace=True)
        
        return df_day

    def run_compilation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scan all years (2020-2024), compile them, and save to Parquet.

        Returns
        -------
        df_1d : pd.DataFrame
            Daily resampled Nifty series.
        df_1h : pd.DataFrame
            Hourly resampled Nifty series.
        df_5min : pd.DataFrame
            5-minute resampled Nifty series.
        df_15min : pd.DataFrame
            15-minute resampled Nifty series.
        """
        logger.info("Scanning local directories for Nifty CSV files...")
        all_files = []
        for year_dir in sorted(self.data_dir.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                for month_dir in sorted(year_dir.iterdir()):
                    if month_dir.is_dir() and "_" in month_dir.name:
                        for csv_file in sorted(month_dir.glob("GFDLNFO_BACKADJUSTED_*.csv")):
                            all_files.append(csv_file)

        logger.info("Found {} daily CSV files to compile.", len(all_files))
        if not all_files:
            raise FileNotFoundError(f"No Nifty CSV files found at {self.data_dir}")

        compiled_dfs = []
        # Process files
        for i, filepath in enumerate(all_files):
            if i % 50 == 0 or i == len(all_files) - 1:
                logger.info("Processing file {}/{} ({})...", i + 1, len(all_files), filepath.name)
            df_day = self.process_day_file(filepath)
            if df_day is not None and not df_day.empty:
                compiled_dfs.append(df_day)

        if not compiled_dfs:
            raise RuntimeError("No data was successfully parsed from any file.")

        df_all = pd.concat(compiled_dfs).sort_index()
        # Clean duplicates
        df_all = df_all[~df_all.index.duplicated(keep="first")]

        logger.info("Raw compilation completed. Resampling to 1d, 1h, 5min, and 15min intervals...")

        # ── Resample to Daily (1d) ───────────────────────────────────────────
        # Note: Options expire on specific dates; we preserve the last available premium
        df_1d = df_all.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
            "ce_close": "last",
            "pe_close": "last",
            "atm_strike": "last",
            "expiry": "last"
        }).dropna(subset=["close"])  # Keep only active trading days
        df_1d.ffill(inplace=True)
        df_1d.bfill(inplace=True)

        # ── Resample to Hourly (1h) ──────────────────────────────────────────
        df_1h = df_all.resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
            "ce_close": "last",
            "pe_close": "last",
            "atm_strike": "last",
            "expiry": "last"
        }).dropna(subset=["close"])
        df_1h = df_1h.between_time("09:15", "15:30")
        df_1h.ffill(inplace=True)
        df_1h.bfill(inplace=True)

        # ── Resample to 5-Minute (5min) ──────────────────────────────────────
        df_5min = df_all.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
            "ce_close": "last",
            "pe_close": "last",
            "atm_strike": "last",
            "expiry": "last"
        }).dropna(subset=["close"])
        df_5min = df_5min.between_time("09:15", "15:30")
        df_5min.ffill(inplace=True)
        df_5min.bfill(inplace=True)

        # ── Resample to 15-Minute (15min) ────────────────────────────────────
        df_15min = df_all.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
            "ce_close": "last",
            "pe_close": "last",
            "atm_strike": "last",
            "expiry": "last"
        }).dropna(subset=["close"])
        df_15min = df_15min.between_time("09:15", "15:30")
        df_15min.ffill(inplace=True)
        df_15min.bfill(inplace=True)

        # Save to Parquet
        path_1d = self.cache_dir / "nifty_local_1d.parquet"
        path_1h = self.cache_dir / "nifty_local_1h.parquet"
        path_5min = self.cache_dir / "nifty_local_5min.parquet"
        path_15min = self.cache_dir / "nifty_local_15min.parquet"

        df_1d.to_parquet(path_1d, engine="pyarrow")
        df_1h.to_parquet(path_1h, engine="pyarrow")
        df_5min.to_parquet(path_5min, engine="pyarrow")
        df_15min.to_parquet(path_15min, engine="pyarrow")

        logger.success("Saved compiled daily data to {}", path_1d)
        logger.success("Saved compiled hourly data to {}", path_1h)
        logger.success("Saved compiled 5-minute data to {}", path_5min)
        logger.success("Saved compiled 15-minute data to {}", path_15min)

        return df_1d, df_1h, df_5min, df_15min


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse local 1-minute Nifty CSV files.")
    parser.add_argument("--verify", action="store_true", help="Verify compilation on a subset of files.")
    args = parser.parse_args()

    parser_obj = NiftyLocalParser()
    if args.verify:
        logger.info("Running parser verification on first 5 files...")
        # Get first 5 files
        files = []
        for year_dir in sorted(parser_obj.data_dir.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                for month_dir in sorted(year_dir.iterdir()):
                    if month_dir.is_dir() and "_" in month_dir.name:
                        files.extend(list(month_dir.glob("GFDLNFO_BACKADJUSTED_*.csv")))
                        break
                break
        
        test_files = sorted(files)[:5]
        logger.info("Test files: {}", [f.name for f in test_files])
        results = []
        for f in test_files:
            df = parser_obj.process_day_file(f)
            if df is not None:
                results.append(df)
                logger.info("Parsed {} rows from {}", len(df), f.name)
        
        if results:
            df_test = pd.concat(results).sort_index()
            print("\nVerify Head:")
            print(df_test.head(3))
            print("\nVerify Tail:")
            print(df_test.tail(3))
        else:
            logger.error("No test files could be processed.")
    else:
        # Full compilation
        parser_obj.run_compilation()
