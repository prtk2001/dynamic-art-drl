#!/usr/bin/env python3
"""
Dynamic ART-DRL — Indian Markets (MCX Futures) Paper Trading Runner
==================================================================
Executes concurrent real-time simulated paper trading for:
1. MCX Gold Futures (1h timeframe / 30m candles)
2. MCX Crude Oil Futures (1d timeframe / day candles)

Under a joint portfolio with a static 50/50 capital split (e.g., ₹10 Lakhs starting capital).
Maintains state locally in config/indian_paper_portfolio.json.
Calculates detailed Indian brokerage, GST, stamp duty, exchange charges, and CTT.
Uses Groq Llama 3.3 for joint Indian market risk advisory.
"""

import argparse
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

# Add root dir to path
sys.path.append(str(Path(__file__).resolve().parent))

from config.assets import get_asset
from data.upstox_client import UpstoxClient
from data.kalman_filter import create_adaptive_filter
from agents.groq_analyst import GroqAnalyst

PORTFOLIO_PATH = Path(__file__).resolve().parent / "config" / "indian_paper_portfolio.json"

def calculate_mcx_transaction_costs(symbol: str, price: float, qty: int, side: str) -> float:
    """Calculate realistic Indian MCX F&O Transaction Costs and Taxes.
    
    Formula based on standard SEBI / Exchange rules:
    - Brokerage: Flat ₹20 per executed order.
    - Exchange Transaction Charges: 0.0026% of turnover.
    - GST: 18% of (Brokerage + Exchange Transaction Charges).
    - SEBI Turnover Fees: 0.0001% of turnover (₹10 per crore).
    - Stamp Duty: 0.002% of turnover (applicable on BUY side only).
    - Commodity Transaction Tax (CTT): 0.01% of turnover (applicable on SELL side only).
    """
    turnover = price * qty
    brokerage = 20.0
    exchange_charges = turnover * 0.000026
    gst = 0.18 * (brokerage + exchange_charges)
    sebi_fees = turnover * 0.000001
    
    stamp_duty = turnover * 0.00002 if side.upper() == "BUY" else 0.0
    ctt = turnover * 0.0001 if side.upper() == "SELL" else 0.0
    
    total_costs = brokerage + exchange_charges + gst + sebi_fees + stamp_duty + ctt
    return total_costs

def load_portfolio(initial_capital: float = 1000000.0) -> dict:
    """Load or initialize local paper portfolio state."""
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if PORTFOLIO_PATH.exists():
        try:
            with open(PORTFOLIO_PATH, "r") as f:
                state = json.load(f)
            # Standardize keys
            if "cash" not in state:
                state["cash"] = state.get("portfolio_value", initial_capital)
            if "holdings" not in state:
                state["holdings"] = {"GOLD": {"shares": 0.0, "average_cost": 0.0}, "CRUDEOIL": {"shares": 0.0, "average_cost": 0.0}}
            if "trade_history" not in state:
                state["trade_history"] = []
            return state
        except Exception as e:
            logger.warning(f"Failed to read portfolio JSON: {e}. Reinitializing.")
            
    # Default initial state
    return {
        "portfolio_value": initial_capital,
        "cash": initial_capital,
        "holdings": {
            "GOLD": {"shares": 0.0, "average_cost": 0.0, "peak_price": 0.0, "in_pos": False, "prev_sig": 0.0},
            "CRUDEOIL": {"shares": 0.0, "average_cost": 0.0, "peak_price": 0.0, "in_pos": False, "prev_sig": 0.0}
        },
        "trade_history": []
    }

def save_portfolio(state: dict) -> None:
    """Save paper portfolio state to JSON file."""
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(state, f, indent=4)

def run_joint_indian_trading(
    capital: float = 1000000.0,
    loop_count: int = 1,
    sleep_seconds: int = 10
) -> None:
    logger.info("==================================================================")
    logger.info("  DYNAMIC ART-DRL CONCURRENT INDIAN PAPER TRADING RUNNER STARTED  ")
    logger.info("==================================================================")
    logger.info("Joint Capital: ₹{:,} | Max Loops: {} | Sleep: {}s", capital, loop_count, sleep_seconds)
    
    # 1. Initialize Clients
    client = UpstoxClient()
    groq_analyst = GroqAnalyst()
    
    # Load portfolio state
    portfolio = load_portfolio(capital)
    logger.success("Portfolio Initialized! Total Cash: ₹{:.2f} | Value: ₹{:.2f}", 
                   portfolio["cash"], portfolio["portfolio_value"])
    
    # Configuration assets
    assets = {
        "GOLD": {
            "config": get_asset("GOLD_MCX"),
            "tf": "1h",      # Maps to 30minute candles in Upstox
            "fast_w": 420,
            "slow_w": 560,
            "stop_pct": 0.08,
            "leverage": 4.0,
            "required_candles": 600
        },
        "CRUDEOIL": {
            "config": get_asset("CRUDE_MCX"),
            "tf": "1d",      # Maps to day candles in Upstox
            "fast_w": 20,
            "slow_w": 25,
            "stop_pct": 0.02,
            "leverage": 3.0,
            "required_candles": 50
        }
    }
    
    # Load state references
    for key, item in assets.items():
        state_hold = portfolio["holdings"].get(key, {})
        item["shares"] = state_hold.get("shares", 0.0)
        item["average_cost"] = state_hold.get("average_cost", 0.0)
        item["peak_price"] = state_hold.get("peak_price", 0.0)
        item["in_pos"] = state_hold.get("in_pos", False)
        item["prev_sig"] = state_hold.get("prev_sig", 0.0)
        
    for loop in range(1, loop_count + 1):
        logger.info(f"==================== INDIAN PORTFOLIO LOOP {loop}/{loop_count} ====================")
        
        # Static 50/50 capital split
        allocated_capital = portfolio["portfolio_value"] * 0.5
        
        joint_status = {}
        target_signals = {}
        current_ltps = {}
        
        # Step 1: Fetch data & evaluate signals
        for key, item in assets.items():
            logger.info(f"--- Signal Evaluation for MCX {key} ({item['config'].name}) ---")
            
            # Resolve instrument key
            try:
                inst_key = client.resolve_active_instrument_key(key)
                logger.info(f"Resolved active instrument key for {key}: {inst_key}")
            except Exception as e:
                logger.error(f"Failed to resolve active instrument key for {key}: {e}")
                continue
                
            # Fetch latest data to compute indicator
            # Fetch start is reference date back to ensure enough candles
            now_dt = datetime.now()
            # Calculate a safe starting date to fetch enough history
            if item["tf"] == "1d":
                days_back = item["required_candles"] * 2
            else:
                # 30-minute interval has 30 bars a day, so fetch last 30 days
                days_back = int(item["required_candles"] / 15) + 5
                
            start_str = (now_dt - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
            end_str = now_dt.strftime("%Y-%m-%d")
            
            try:
                df = client.fetch_historical(
                    instrument_key=inst_key,
                    interval=item["tf"],
                    start=start_str,
                    end=end_str,
                    use_cache=False # Get fresh candles
                )
            except Exception as e:
                logger.error(f"Upstox fetch historical failed for {key}: {e}")
                continue
                
            if len(df) < item["slow_w"] + 5:
                logger.error(f"Insufficient history fetched for {key}. Got {len(df)} rows, need {item['slow_w'] + 5}. Skipping.")
                continue
                
            # Denoise prices
            kf = create_adaptive_filter(
                process_noise=item['config'].kalman_process_noise,
                measurement_noise=item['config'].kalman_measurement_noise
            )
            df_denoised = kf.filter_ohlcv(df)
            
            # Fetch LTP (Last Trade Price)
            ltp = df["close"].values[-1]
            current_ltps[key] = ltp
            
            # SMA Crossovers on Kalman price
            kalman_price_series = df_denoised["kalman_price"].values
            fast_sma = pd.Series(kalman_price_series).rolling(item["fast_w"]).mean().values[-1]
            slow_sma = pd.Series(kalman_price_series).rolling(item["slow_w"]).mean().values[-1]
            
            sig = 1.0 if fast_sma > slow_sma else 0.0
            
            # Trailing stop monitor
            if item["in_pos"]:
                item["peak_price"] = max(item["peak_price"], ltp)
                unrealized_loss = (item["peak_price"] - ltp) / item["peak_price"]
                logger.info(f"{key} Stop Monitor: Peak=₹{item['peak_price']:.2f} | Loss={unrealized_loss*100:.2f}% (Limit={item['stop_pct']*100:.2f}%)")
                if unrealized_loss > item["stop_pct"]:
                    logger.warning(f"🚨 Trailing Stop-Loss triggered for {key}! Exit signal generated.")
                    sig = 0.0
                    
            target_signals[key] = sig
            
            joint_status[key] = {
                "ltp": ltp,
                "kalman_price": kalman_price_series[-1],
                "fast_sma": fast_sma,
                "slow_sma": slow_sma,
                "signal": sig,
                "shares_held": item["shares"],
                "average_cost": item["average_cost"],
                "leverage": item["leverage"],
                "stop_pct": item["stop_pct"],
                "in_position": item["in_pos"]
            }
            logger.info(f"{key} LTP: ₹{ltp:.2f} | Kalman Price: ₹{kalman_price_series[-1]:.2f} | Fast SMA: ₹{fast_sma:.2f} | Slow SMA: ₹{slow_sma:.2f}")

        # Step 2: Trade executions
        for key, item in assets.items():
            if key not in current_ltps:
                continue
                
            sig = target_signals[key]
            ltp = current_ltps[key]
            current_shares = item["shares"]
            
            # Calculate target shares
            if sig > 0.0:
                # 50% split per asset, leverage factor applied
                target_alloc = allocated_capital * item["leverage"]
                target_shares = int(target_alloc / ltp)
            else:
                target_shares = 0
                
            trade_diff = target_shares - current_shares
            joint_status[key]["target_shares"] = target_shares
            
            if trade_diff != 0 and (sig != item["prev_sig"] or target_shares == 0):
                side = "BUY" if trade_diff > 0 else "SELL"
                trade_qty = abs(trade_diff)
                
                # Compute Transaction Costs
                tx_cost = calculate_mcx_transaction_costs(key, ltp, trade_qty, side)
                trade_value = ltp * trade_qty
                
                logger.warning(f"Executing paper {side} of {trade_qty} lots for {key} at ₹{ltp:.2f} (Tx Cost: ₹{tx_cost:.2f})")
                
                # Update cash and holdings
                if side == "BUY":
                    cost_basis = trade_value + tx_cost
                    portfolio["cash"] -= cost_basis
                    
                    # Update average cost
                    total_cost = (item["shares"] * item["average_cost"]) + cost_basis
                    item["shares"] += trade_qty
                    item["average_cost"] = total_cost / item["shares"]
                    item["in_pos"] = True
                    item["peak_price"] = ltp
                else:
                    proceeds = trade_value - tx_cost
                    portfolio["cash"] += proceeds
                    
                    # Log trade realized PnL
                    realized_pnl = (ltp - item["average_cost"]) * trade_qty - tx_cost
                    logger.success(f"Realized PnL for {key}: ₹{realized_pnl:+.2f}")
                    
                    item["shares"] = 0.0
                    item["average_cost"] = 0.0
                    item["in_pos"] = False
                    item["peak_price"] = 0.0
                
                item["prev_sig"] = sig
                
                # Log trade history
                portfolio["trade_history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": key,
                    "side": side,
                    "qty": trade_qty,
                    "price": ltp,
                    "tx_cost": tx_cost,
                    "pnl": realized_pnl if side == "SELL" else 0.0
                })
            else:
                logger.info(f"No trading action required for {key}.")
        
        # Step 3: Compute current portfolio value
        holdings_value = 0.0
        for key, item in assets.items():
            if key in current_ltps:
                holdings_value += item["shares"] * current_ltps[key]
                
        portfolio["portfolio_value"] = portfolio["cash"] + holdings_value
        
        # Save state back to portfolio JSON
        for key, item in assets.items():
            portfolio["holdings"][key] = {
                "shares": item["shares"],
                "average_cost": item["average_cost"],
                "peak_price": item["peak_price"],
                "in_pos": item["in_pos"],
                "prev_sig": item["prev_sig"]
            }
        save_portfolio(portfolio)
        
        logger.info("Updated Portfolio Cash: ₹{:.2f} | Holdings Value: ₹{:.2f} | Net Portfolio Value: ₹{:.2f}", 
                    portfolio["cash"], holdings_value, portfolio["portfolio_value"])
        
        # Step 4: Groq AI Risk Advisory
        try:
            logger.info("Synthesizing joint Indian market portfolio advisory report from Groq Llama 3.3...")
            market_context = {
                "portfolio_value_inr": portfolio["portfolio_value"],
                "cash_inr": portfolio["cash"],
                "gold_status": joint_status.get("GOLD", {}),
                "crude_status": joint_status.get("CRUDEOIL", {})
            }
            commentary = groq_analyst.generate_trade_commentary(1.0, market_context)
            print(f"\n📢 [Groq AI Joint Indian Portfolio Advisory (INR)]:\n{commentary}\n")
        except Exception as e:
            logger.warning(f"Groq joint commentary generation failed: {e}")
            
        if loop < loop_count:
            logger.info(f"Sleeping {sleep_seconds} seconds before next iteration...")
            time.sleep(sleep_seconds)
            
    logger.success("==================================================================")
    logger.success("  DYNAMIC ART-DRL CONCURRENT INDIAN PAPER TRADING COMPLETED        ")
    logger.success("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run concurrent simulated MCX Gold + Crude Oil paper trading on Upstox.")
    parser.add_argument("--capital", type=float, default=1000000.0,
                        help="Initial capital in INR. Default is 10 Lakhs.")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of execution iterations. Default is 1.")
    parser.add_argument("--sleep", type=int, default=10,
                        help="Sleep time in seconds between loops. Default is 10.")
    args = parser.parse_args()
    
    run_joint_indian_trading(
        capital=args.capital,
        loop_count=args.loops,
        sleep_seconds=args.sleep
    )
