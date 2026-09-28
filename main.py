"""
Dynamic ART-DRL — Adaptive Risk-sensitive Transformer-based DRL System
======================================================================
Command Line Interface (CLI) Entry Point.

Usage
-----
Train models:
    python main.py train --asset GOLD_MCX --timesteps 50000

Run out-of-sample backtest & generate Plotly report:
    python main.py backtest --asset GOLD_MCX --timeframe 1d
    python main.py backtest --asset CRUDE_MCX --timeframe 1d
    python main.py backtest --asset GLD --timeframe 1h

Run live/simulated trading:
    python main.py live --asset GOLD_MCX --loops 3
"""

import argparse
import sys
from loguru import logger

from train import run_training_pipeline
from backtest_runner import run_backtest
from live_runner import run_live_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic ART-DRL Portfolio Trading Framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Task to run")
    
    # ── Command: Train ───────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train heterogeneous DRL agents sequentially.")
    train_parser.add_argument(
        "--asset", type=str, default="GOLD_MCX",
        help="Asset config registry key (GOLD_MCX, CRUDE_MCX, GLD, USO)."
    )
    train_parser.add_argument(
        "--timeframe", type=str, default="1d", choices=["1d", "1h", "5min", "15min"],
        help="Trading timeframe interval."
    )
    train_parser.add_argument(
        "--timesteps", type=int, default=10000,
        help="Number of training steps per agent."
    )
    
    # ── Command: Backtest ────────────────────────────────────────────────────
    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward rolling backtests.")
    backtest_parser.add_argument(
        "--asset", type=str, default="GOLD_MCX",
        help="Asset to backtest."
    )
    backtest_parser.add_argument(
        "--timeframe", type=str, default="1d", choices=["1d", "1h", "5min", "15min"],
        help="Trading interval timeframe."
    )
    backtest_parser.add_argument(
        "--capital", type=float, default=None,
        help="Initial starting capital override."
    )
    
    # ── Command: Live ────────────────────────────────────────────────────────
    live_parser = subparsers.add_parser("live", help="Execute live/simulated paper trading.")
    live_parser.add_argument(
        "--asset", type=str, default="GOLD_MCX",
        help="Target asset for execution."
    )
    live_parser.add_argument(
        "--timeframe", type=str, default="1d", choices=["1d", "1h", "5min", "15min"],
        help="Trading timeframe interval."
    )
    live_parser.add_argument(
        "--loops", type=int, default=3,
        help="Demo cycle loop count."
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Dynamic ART-DRL CLI manager...")
    
    try:
        if args.command == "train":
            run_training_pipeline(
                asset_name=args.asset,
                timeframe=args.timeframe,
                timesteps=args.timesteps
            )
        elif args.command == "backtest":
            run_backtest(
                asset_name=args.asset,
                timeframe=args.timeframe,
                capital=args.capital
            )
        elif args.command == "live":
            run_live_pipeline(
                asset_name=args.asset,
                timeframe=args.timeframe,
                loop_count=args.loops
            )
    except KeyboardInterrupt:
        logger.warning("Pipeline terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pipeline failed with exception: {}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
