"""
Dynamic ART-DRL — Nifty Options Post-processor (Highly Optimized)
=================================================================
Computes the exact next-period marked-to-market premium (ce_close_next, pe_close_next)
for options contracts held from step t to step t+1.
Uses a target-matching approach to only parse needed option contracts, reducing
the processing time of the raw 1-minute files from hours to minutes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    DATA_CACHE_DIR,
)


class NiftyOptionsPreprocessor:
    """Computes exact mark-to-market returns for Nifty options backtesting."""

    def __init__(
        self,
        data_dir: str = "/Users/prateek/Documents/nifty_options/data/nifty_bank_nifty",
        cache_dir: str = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
        logger.info("NiftyOptionsPreprocessor initialised with data_dir={}", self.data_dir)

    def run_postprocessing(self) -> None:
        """Load resampled Parquets, identify target contracts, parse raw CSVs, and compute next-step premiums."""
        # 1. Load compiled daily, hourly, 5min, and 15min parquet files
        path_1d = self.cache_dir / "nifty_local_1d.parquet"
        path_1h = self.cache_dir / "nifty_local_1h.parquet"
        path_5min = self.cache_dir / "nifty_local_5min.parquet"
        path_15min = self.cache_dir / "nifty_local_15min.parquet"

        if not all(p.exists() for p in [path_1d, path_1h, path_5min, path_15min]):
            logger.error("Missing resampled parquets nifty_local_1d, 1h, 5min, or 15min in cache. Please run nifty_local_parser.py first!")
            return

        df_1d = pd.read_parquet(path_1d)
        df_1h = pd.read_parquet(path_1h)
        df_5min = pd.read_parquet(path_5min)
        df_15min = pd.read_parquet(path_15min)
        logger.info("Loaded resampled data | Daily shape: {} | Hourly shape: {} | 5min shape: {} | 15min shape: {}",
                    df_1d.shape, df_1h.shape, df_5min.shape, df_15min.shape)

        # 2. Build target dictionary of needed option contracts to extract
        # Key: date_str -> set of option tickers needed on that day
        needed_daily: Dict[str, Set[str]] = {}
        dates_1d = df_1d.index.strftime("%Y-%m-%d").tolist()
        
        for i in range(len(df_1d) - 1):
            next_date_str = dates_1d[i+1]
            strike = int(df_1d["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_1d["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%d%b%y").upper()
            
            ce_ticker = f"NIFTY{expiry_str}{strike}CE.NFO"
            pe_ticker = f"NIFTY{expiry_str}{strike}PE.NFO"
            
            if next_date_str not in needed_daily:
                needed_daily[next_date_str] = set()
            needed_daily[next_date_str].add(ce_ticker)
            needed_daily[next_date_str].add(pe_ticker)

        # Key: hour_str -> set of option tickers needed on that hour
        needed_hourly: Dict[str, Set[str]] = {}
        hours_1h = df_1h.index.strftime("%Y-%m-%d %H:00:00").tolist()
        
        for i in range(len(df_1h) - 1):
            next_hour_str = hours_1h[i+1]
            strike = int(df_1h["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_1h["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%d%b%y").upper()
            
            ce_ticker = f"NIFTY{expiry_str}{strike}CE.NFO"
            pe_ticker = f"NIFTY{expiry_str}{strike}PE.NFO"
            
            if next_hour_str not in needed_hourly:
                needed_hourly[next_hour_str] = set()
            needed_hourly[next_hour_str].add(ce_ticker)
            needed_hourly[next_hour_str].add(pe_ticker)

        # Key: ts_str -> set of option tickers needed for 5min
        needed_5min: Dict[str, Set[str]] = {}
        times_5min = df_5min.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        for i in range(len(df_5min) - 1):
            next_ts_str = times_5min[i+1]
            strike = int(df_5min["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_5min["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%d%b%y").upper()
            
            ce_ticker = f"NIFTY{expiry_str}{strike}CE.NFO"
            pe_ticker = f"NIFTY{expiry_str}{strike}PE.NFO"
            
            if next_ts_str not in needed_5min:
                needed_5min[next_ts_str] = set()
            needed_5min[next_ts_str].add(ce_ticker)
            needed_5min[next_ts_str].add(pe_ticker)

        # Key: ts_str -> set of option tickers needed for 15min
        needed_15min: Dict[str, Set[str]] = {}
        times_15min = df_15min.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        for i in range(len(df_15min) - 1):
            next_ts_str = times_15min[i+1]
            strike = int(df_15min["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_15min["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%d%b%y").upper()
            
            ce_ticker = f"NIFTY{expiry_str}{strike}CE.NFO"
            pe_ticker = f"NIFTY{expiry_str}{strike}PE.NFO"
            
            if next_ts_str not in needed_15min:
                needed_15min[next_ts_str] = set()
            needed_15min[next_ts_str].add(ce_ticker)
            needed_15min[next_ts_str].add(pe_ticker)

        # 3. Scan and locate all daily CSV files
        logger.info("Scanning local directories for Nifty CSV files...")
        all_files = []
        for year_dir in sorted(self.data_dir.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                for month_dir in sorted(year_dir.iterdir()):
                    if month_dir.is_dir() and "_" in month_dir.name:
                        for csv_file in sorted(month_dir.glob("GFDLNFO_BACKADJUSTED_*.csv")):
                            all_files.append(csv_file)

        logger.info("Found {} daily CSV files to parse.", len(all_files))

        # 4. Parse the CSV files and collect target contract prices
        # DB Keys: (date, strike, expiry, opt_type) -> close
        opt_db_daily: Dict[Tuple[str, int, str, str], float] = {}
        opt_db_hourly: Dict[Tuple[str, int, str, str], float] = {}
        opt_db_5min: Dict[Tuple[str, int, str, str], float] = {}
        opt_db_15min: Dict[Tuple[str, int, str, str], float] = {}
        future_closes_daily: Dict[str, float] = {}

        logger.info("Extracting option premiums via target matching...")
        t0 = datetime.now()
        
        # Cache for parsed expiry dates
        expiry_cache: Dict[str, str] = {}
        def parse_expiry_cached(exp_str: str) -> str | None:
            if exp_str not in expiry_cache:
                try:
                    exp_dt = datetime.strptime(exp_str, "%d%b%y")
                    expiry_cache[exp_str] = exp_dt.strftime("%Y-%m-%d")
                except Exception:
                    expiry_cache[exp_str] = None
            return expiry_cache[exp_str]

        for idx, filepath in enumerate(all_files):
            if (idx + 1) % 100 == 0 or idx == len(all_files) - 1:
                logger.info("Processing file {}/{}...", idx + 1, len(all_files))
            
            # Find the date of the file from the first line matching NIFTY
            date_str = None
            try:
                with open(filepath, 'r') as f:
                    f.readline()  # Skip header
                    for line in f:
                        if line.startswith('NIFTY'):
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                date_str = parts[1]
                                break
            except Exception as e:
                logger.error("Failed to read header of {}: {}", filepath.name, e)
                continue

            if date_str is None:
                continue

            try:
                dt = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue
            trading_date_str = dt.strftime("%Y-%m-%d")

            # Collect target sets for this day
            daily_targets = needed_daily.get(trading_date_str, set())
            
            hourly_targets_by_hour = {}
            for hour_val in ["09", "10", "11", "12", "13", "14", "15"]:
                hour_str = f"{trading_date_str} {hour_val}:00:00"
                hourly_targets_by_hour[hour_val] = needed_hourly.get(hour_str, set())

            all_targets = daily_targets.copy()
            for targets in hourly_targets_by_hour.values():
                all_targets.update(targets)

            # Add 5min and 15min targets for this day
            for ts_str, targets in needed_5min.items():
                if ts_str.startswith(trading_date_str):
                    all_targets.update(targets)
            for ts_str, targets in needed_15min.items():
                if ts_str.startswith(trading_date_str):
                    all_targets.update(targets)

            # We also always track NIFTY-I.NFO to get the daily future close
            all_targets.add("NIFTY-I.NFO")

            # Scan the file line-by-line matching targets
            daily_closes = {}
            hourly_closes = {}
            closes_5min = {}
            closes_15min = {}

            try:
                with open(filepath, 'r') as f:
                    f.readline()  # Skip header
                    for line in f:
                        if not line.startswith('NIFTY'):
                            continue
                        parts = line.strip().split(',')
                        if len(parts) < 7:
                            continue
                        
                        ticker = parts[0]
                        if ticker in all_targets:
                            time_val = parts[2]
                            close_val = parts[6]
                            try:
                                close = float(close_val)
                            except ValueError:
                                continue
                            
                            if ticker == "NIFTY-I.NFO" or ticker in daily_targets:
                                daily_closes[ticker] = close
                            
                            hour_val = time_val[:2]
                            if ticker in hourly_targets_by_hour.get(hour_val, set()):
                                hourly_closes[(ticker, hour_val)] = close

                            # 5min close
                            try:
                                minute_5 = int(time_val[3:5])
                                minute_5_base = minute_5 - (minute_5 % 5)
                                ts_5min_str = f"{trading_date_str} {time_val[:3]}{minute_5_base:02d}:00"
                                if ts_5min_str in needed_5min and ticker in needed_5min[ts_5min_str]:
                                    closes_5min[(ticker, ts_5min_str)] = close
                            except Exception:
                                pass
                                
                            # 15min close
                            try:
                                minute_15 = int(time_val[3:5])
                                minute_15_base = minute_15 - (minute_15 % 15)
                                ts_15min_str = f"{trading_date_str} {time_val[:3]}{minute_15_base:02d}:00"
                                if ts_15min_str in needed_15min and ticker in needed_15min[ts_15min_str]:
                                    closes_15min[(ticker, ts_15min_str)] = close
                            except Exception:
                                pass
            except Exception as e:
                logger.error("Failed parsing lines of {}: {}", filepath.name, e)
                continue

            # Save future close
            if "NIFTY-I.NFO" in daily_closes:
                future_closes_daily[trading_date_str] = daily_closes["NIFTY-I.NFO"]

            # Save daily options closes
            for ticker, close in daily_closes.items():
                if ticker == "NIFTY-I.NFO":
                    continue
                body = ticker[5:-4]
                expiry_str = body[:7]
                strike_type = body[7:]
                opt_type = strike_type[-2:]
                try:
                    strike = int(strike_type[:-2])
                    expiry_date_str = parse_expiry_cached(expiry_str)
                    if expiry_date_str:
                        opt_db_daily[(trading_date_str, strike, expiry_date_str, opt_type)] = close
                except Exception:
                    continue

            # Save hourly options closes
            for (ticker, hour_val), close in hourly_closes.items():
                body = ticker[5:-4]
                expiry_str = body[:7]
                strike_type = body[7:]
                opt_type = strike_type[-2:]
                try:
                    strike = int(strike_type[:-2])
                    expiry_date_str = parse_expiry_cached(expiry_str)
                    if expiry_date_str:
                        hour_str = f"{trading_date_str} {hour_val}:00:00"
                        opt_db_hourly[(hour_str, strike, expiry_date_str, opt_type)] = close
                except Exception:
                    continue

            # Save 5min options closes
            for (ticker, ts_5min_str), close in closes_5min.items():
                body = ticker[5:-4]
                expiry_str = body[:7]
                strike_type = body[7:]
                opt_type = strike_type[-2:]
                try:
                    strike = int(strike_type[:-2])
                    expiry_date_str = parse_expiry_cached(expiry_str)
                    if expiry_date_str:
                        opt_db_5min[(ts_5min_str, strike, expiry_date_str, opt_type)] = close
                except Exception:
                    continue

            # Save 15min options closes
            for (ticker, ts_15min_str), close in closes_15min.items():
                body = ticker[5:-4]
                expiry_str = body[:7]
                strike_type = body[7:]
                opt_type = strike_type[-2:]
                try:
                    strike = int(strike_type[:-2])
                    expiry_date_str = parse_expiry_cached(expiry_str)
                    if expiry_date_str:
                        opt_db_15min[(ts_15min_str, strike, expiry_date_str, opt_type)] = close
                except Exception:
                    continue

        logger.info(
            "Options database compiled in {} | Daily DB size: {} | Hourly DB size: {} | 5min DB size: {} | 15min DB size: {}",
            datetime.now() - t0,
            len(opt_db_daily),
            len(opt_db_hourly),
            len(opt_db_5min),
            len(opt_db_15min),
        )

        # 5. Process Daily Parquet file
        logger.info("Post-processing daily parquet file...")
        ce_next_vals = []
        pe_next_vals = []
        
        for i in range(len(df_1d)):
            date_str = dates_1d[i]
            strike = int(df_1d["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_1d["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%Y-%m-%d")
            
            is_expiry_day = (date_str >= expiry_str)
            
            if i == len(df_1d) - 1:
                ce_next_vals.append(df_1d["ce_close"].iloc[i])
                pe_next_vals.append(df_1d["pe_close"].iloc[i])
            elif is_expiry_day:
                future_close = df_1d["close"].iloc[i]
                ce_next = max(0.0, future_close - strike)
                pe_next = max(0.0, strike - future_close)
                ce_next_vals.append(ce_next)
                pe_next_vals.append(pe_next)
            else:
                next_date_str = dates_1d[i+1]
                ce_next = opt_db_daily.get((next_date_str, strike, expiry_str, "CE"), np.nan)
                pe_next = opt_db_daily.get((next_date_str, strike, expiry_str, "PE"), np.nan)
                
                # Fallback to intrinsic value if lookup fails
                if np.isnan(ce_next):
                    ce_next = max(0.0, future_closes_daily.get(next_date_str, df_1d["close"].iloc[i+1]) - strike)
                if np.isnan(pe_next):
                    pe_next = max(0.0, strike - future_closes_daily.get(next_date_str, df_1d["close"].iloc[i+1]))
                    
                ce_next_vals.append(ce_next)
                pe_next_vals.append(pe_next)
                
        df_1d["ce_close_next"] = ce_next_vals
        df_1d["pe_close_next"] = pe_next_vals
        df_1d.to_parquet(path_1d, engine="pyarrow")
        logger.success("Updated daily dataset: {}", path_1d.name)

        # 6. Process Hourly Parquet file
        logger.info("Post-processing hourly parquet file...")
        ce_next_vals_h = []
        pe_next_vals_h = []
        
        for i in range(len(df_1h)):
            hour_str = hours_1h[i]
            strike = int(df_1h["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_1h["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%Y-%m-%d")
            
            is_expiry_hour = (hour_str.split(" ")[0] >= expiry_str and hour_str.split(" ")[1] >= "15:00:00")
            
            if i == len(df_1h) - 1:
                ce_next_vals_h.append(df_1h["ce_close"].iloc[i])
                pe_next_vals_h.append(df_1h["pe_close"].iloc[i])
            elif is_expiry_hour:
                future_close = df_1h["close"].iloc[i]
                ce_next = max(0.0, future_close - strike)
                pe_next = max(0.0, strike - future_close)
                ce_next_vals_h.append(ce_next)
                pe_next_vals_h.append(pe_next)
            else:
                next_hour_str = hours_1h[i+1]
                ce_next = opt_db_hourly.get((next_hour_str, strike, expiry_str, "CE"), np.nan)
                pe_next = opt_db_hourly.get((next_hour_str, strike, expiry_str, "PE"), np.nan)
                
                # Fallback to intrinsic value if lookup fails
                if np.isnan(ce_next):
                    ce_next = max(0.0, df_1h["close"].iloc[i+1] - strike)
                if np.isnan(pe_next):
                    pe_next = max(0.0, strike - df_1h["close"].iloc[i+1])
                    
                ce_next_vals_h.append(ce_next)
                pe_next_vals_h.append(pe_next)
                
        df_1h["ce_close_next"] = ce_next_vals_h
        df_1h["pe_close_next"] = pe_next_vals_h
        df_1h.to_parquet(path_1h, engine="pyarrow")
        logger.success("Updated hourly dataset: {}", path_1h.name)

        # 7. Process 5min Parquet file
        logger.info("Post-processing 5min parquet file...")
        ce_next_vals_5m = []
        pe_next_vals_5m = []
        
        for i in range(len(df_5min)):
            ts_str = times_5min[i]
            strike = int(df_5min["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_5min["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%Y-%m-%d")
            
            is_expiry_5m = (ts_str.split(" ")[0] >= expiry_str and ts_str.split(" ")[1] >= "15:25:00")
            
            if i == len(df_5min) - 1:
                ce_next_vals_5m.append(df_5min["ce_close"].iloc[i])
                pe_next_vals_5m.append(df_5min["pe_close"].iloc[i])
            elif is_expiry_5m:
                future_close = df_5min["close"].iloc[i]
                ce_next = max(0.0, future_close - strike)
                pe_next = max(0.0, strike - future_close)
                ce_next_vals_5m.append(ce_next)
                pe_next_vals_5m.append(pe_next)
            else:
                next_ts_str = times_5min[i+1]
                ce_next = opt_db_5min.get((next_ts_str, strike, expiry_str, "CE"), np.nan)
                pe_next = opt_db_5min.get((next_ts_str, strike, expiry_str, "PE"), np.nan)
                
                # Fallback to intrinsic value if lookup fails
                if np.isnan(ce_next):
                    ce_next = max(0.0, df_5min["close"].iloc[i+1] - strike)
                if np.isnan(pe_next):
                    pe_next = max(0.0, strike - df_5min["close"].iloc[i+1])
                    
                ce_next_vals_5m.append(ce_next)
                pe_next_vals_5m.append(pe_next)
                
        df_5min["ce_close_next"] = ce_next_vals_5m
        df_5min["pe_close_next"] = pe_next_vals_5m
        df_5min.to_parquet(path_5min, engine="pyarrow")
        logger.success("Updated 5min dataset: {}", path_5min.name)

        # 8. Process 15min Parquet file
        logger.info("Post-processing 15min parquet file...")
        ce_next_vals_15m = []
        pe_next_vals_15m = []
        
        for i in range(len(df_15min)):
            ts_str = times_15min[i]
            strike = int(df_15min["atm_strike"].iloc[i])
            expiry_dt = pd.to_datetime(df_15min["expiry"].iloc[i])
            expiry_str = expiry_dt.strftime("%Y-%m-%d")
            
            is_expiry_15m = (ts_str.split(" ")[0] >= expiry_str and ts_str.split(" ")[1] >= "15:15:00")
            
            if i == len(df_15min) - 1:
                ce_next_vals_15m.append(df_15min["ce_close"].iloc[i])
                pe_next_vals_15m.append(df_15min["pe_close"].iloc[i])
            elif is_expiry_15m:
                future_close = df_15min["close"].iloc[i]
                ce_next = max(0.0, future_close - strike)
                pe_next = max(0.0, strike - future_close)
                ce_next_vals_15m.append(ce_next)
                pe_next_vals_15m.append(pe_next)
            else:
                next_ts_str = times_15min[i+1]
                ce_next = opt_db_15min.get((next_ts_str, strike, expiry_str, "CE"), np.nan)
                pe_next = opt_db_15min.get((next_ts_str, strike, expiry_str, "PE"), np.nan)
                
                # Fallback to intrinsic value if lookup fails
                if np.isnan(ce_next):
                    ce_next = max(0.0, df_15min["close"].iloc[i+1] - strike)
                if np.isnan(pe_next):
                    pe_next = max(0.0, strike - df_15min["close"].iloc[i+1])
                    
                ce_next_vals_15m.append(ce_next)
                pe_next_vals_15m.append(pe_next)
                
        df_15min["ce_close_next"] = ce_next_vals_15m
        df_15min["pe_close_next"] = pe_next_vals_15m
        df_15min.to_parquet(path_15min, engine="pyarrow")
        logger.success("Updated 15min dataset: {}", path_15min.name)


if __name__ == "__main__":
    preprocessor = NiftyOptionsPreprocessor()
    preprocessor.run_postprocessing()
