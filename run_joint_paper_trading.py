#!/usr/bin/env python3
"""
Dynamic ART-DRL — Multi-Asset Joint Paper Trading Runner
==========================================================
Executes concurrent real-time paper trading for BOTH Gold (GLD) and Crude Oil (USO)
as a single Joint Portfolio (static 50/50 allocation split).

Applies:
- Dynamic portfolio value split (50% buying power to GLD, 50% to USO).
- Recursive Kalman Filter price denoising.
- Champion Crossover SMA signals + active trailing stops for both assets.
- Live order execution via Alpaca Paper API.
- Integrated Joint Groq LLM risk analysis.
"""

import argparse
import sys
import time
import json
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from alpaca.trading.enums import OrderSide

# Add root dir to path
sys.path.append(str(Path(__file__).resolve().parent))

from config.assets import get_asset
from data.alpaca_client import AlpacaClient
from data.kalman_filter import create_adaptive_filter
from execution.alpaca_executor import AlpacaExecutor
from agents.groq_analyst import GroqAnalyst

SAVED_CONFIGS_PATH = Path(__file__).resolve().parent / "config" / "saved_champion_configs.json"

def run_joint_live_trading(
    mode: str = "swing_monthly",
    loop_count: int = 5,
    sleep_seconds: int = 10
) -> None:
    logger.info("==================================================================")
    logger.info("    DYNAMIC ART-DRL CONCURRENT JOINT PAPER TRADING LOOP STARTED   ")
    logger.info("==================================================================")
    logger.info("Portfolio Mode: {} | Max Loops: {} | Sleep: {}s", mode, loop_count, sleep_seconds)
    
    # 1. Initialize API clients and services
    executor = AlpacaExecutor()
    data_client = AlpacaClient()
    groq_analyst = GroqAnalyst()
    
    # Query account
    acct = executor.get_account_info()
    portfolio_value = acct["portfolio_value"] if acct["portfolio_value"] > 0.0 else 200000.0
    cash = acct["cash"]
    logger.success("Connected to Alpaca Paper Sandbox! Total Buying Power Portfolio Value: USD {:.2f} (Cash: {:.2f})", 
                   portfolio_value, cash)
    
    # 2. Load configurations for both GLD and USO
    assets = {
        "GLD": {
            "config": get_asset("GLD"),
            "fast_w": 3,
            "slow_w": 50,
            "leverage": 5.0,
            "stop_pct": 0.08,
            "in_pos": False,
            "peak_price": 0.0,
            "prev_sig": 0.0,
            "prices_list": [],
            "opens_list": [],
            "highs_list": [],
            "lows_list": [],
            "vols_list": [],
            "dates_list": []
        },
        "USO": {
            "config": get_asset("USO"),
            "fast_w": 10,
            "slow_w": 40,
            "leverage": 2.0,
            "stop_pct": 0.10,
            "in_pos": False,
            "peak_price": 0.0,
            "prev_sig": 0.0,
            "prices_list": [],
            "opens_list": [],
            "highs_list": [],
            "lows_list": [],
            "vols_list": [],
            "dates_list": []
        }
    }
    
    # Load optimal saved parameters from JSON configs
    logger.info(f"Loading champion parameters from config/saved_champion_configs.json for {mode}...")
    try:
        with open(SAVED_CONFIGS_PATH, "r") as f:
            saved_data = json.load(f)
            
        cfg_group = "monthly_swing_champions" if mode == "swing_monthly" else "low_turnover_swing_champions"
        
        for asset_key in ["GLD", "USO"]:
            params = saved_data[cfg_group][asset_key]
            assets[asset_key]["fast_w"] = params["fast_window"]
            assets[asset_key]["slow_w"] = params["slow_window"]
            assets[asset_key]["leverage"] = params["leverage_factor"]
            assets[asset_key]["stop_pct"] = params["stop_loss_pct"]
            
        logger.success("Configurations successfully parsed:")
        for k, v in assets.items():
            logger.info("  --> {}: SMA({}, {}) | Leverage={}x | Trailing Stop={}%", 
                        k, v["fast_w"], v["slow_w"], v["leverage"], v["stop_pct"]*100)
    except Exception as e:
        logger.warning(f"Error loading saved JSON configs: {e}. Using robust defaults.")
        
    # 3. Bootstrap historical data for both symbols
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    for asset_key, item in assets.items():
        logger.info(f"Bootstrapping historical prices for {asset_key} ({item['config'].symbol})...")
        bootstrap_df = data_client.fetch_ohlcv(symbol=item['config'].symbol, timeframe="1d", end=today_str)
        if bootstrap_df.empty:
            logger.error(f"Could not bootstrap data for {asset_key}. Aborting.")
            return
            
        item["prices_list"] = list(bootstrap_df["close"].values)
        item["opens_list"] = list(bootstrap_df["open"].values)
        item["highs_list"] = list(bootstrap_df["high"].values)
        item["lows_list"] = list(bootstrap_df["low"].values)
        item["vols_list"] = list(bootstrap_df["volume"].values)
        item["dates_list"] = list(bootstrap_df.index)
        
        # Check active position
        current_shares = executor.get_position_for_symbol(item['config'].symbol)
        if current_shares > 0:
            item["in_pos"] = True
            item["peak_price"] = item["prices_list"][-1]
            item["prev_sig"] = 1.0
            logger.info(f"Detected active holding of {current_shares} shares in {asset_key}. Initial Peak price set to {item['peak_price']:.2f}")

    # 4. Real-time multi-asset execution loop
    for loop in range(1, loop_count + 1):
        logger.info(f"==================== JOINT PORTFOLIO LOOP {loop}/{loop_count} ====================")
        
        # Re-fetch latest account metrics
        acct = executor.get_account_info()
        current_portfolio_value = acct["portfolio_value"] if acct["portfolio_value"] > 0.0 else portfolio_value
        buying_power = acct["buying_power"] if acct["buying_power"] > 0.0 else current_portfolio_value * 2.0 # default to 2x stock margin
        logger.info("Active Account Value: USD {:.2f} | Buying Power: {:.2f} (Available Cash: {:.2f})", 
                    current_portfolio_value, buying_power, acct["cash"])
        
        # Static 50/50 capital split
        allocated_capital_per_asset = current_portfolio_value * 0.5
        
        joint_status = {}
        target_signals = {}
        current_ltps = {}
        
        # Stage A: Evaluate signals for both assets
        for asset_key, item in assets.items():
            logger.info(f"--- Signal Evaluation for {asset_key} ({item['config'].symbol}) ---")
            
            # Fetch current price (LTP)
            ltp = executor._get_asset_price_estimate(item['config'].symbol)
            if ltp <= 0 or ltp == 100.0 or ltp == 75.0 or ltp == 220.0:
                last_price = item["prices_list"][-1]
                ltp = last_price
                logger.info(f"Session closed or fallback triggered. Using high-fidelity historical LTP for {asset_key}: {ltp:.2f}")
            else:
                logger.info(f"Fetched LTP for {asset_key}: {ltp:.2f}")
                
            current_ltps[asset_key] = ltp
            
            # Append latest price tick to sliding buffers
            current_time = pd.Timestamp.now(tz="UTC")
            item["dates_list"].append(current_time)
            item["prices_list"].append(ltp)
            item["opens_list"].append(item["prices_list"][-2])
            item["highs_list"].append(max(item["prices_list"][-2], ltp))
            item["lows_list"].append(min(item["prices_list"][-2], ltp))
            item["vols_list"].append(np.random.uniform(500, 2000))
            
            if len(item["prices_list"]) > 300:
                item["prices_list"].pop(0)
                item["opens_list"].pop(0)
                item["highs_list"].pop(0)
                item["lows_list"].pop(0)
                item["vols_list"].pop(0)
                item["dates_list"].pop(0)
                
            # Preprocess loop DataFrame
            loop_df = pd.DataFrame(index=item["dates_list"])
            loop_df["open"] = item["opens_list"]
            loop_df["high"] = item["highs_list"]
            loop_df["low"] = item["lows_list"]
            loop_df["close"] = item["prices_list"]
            loop_df["volume"] = item["vols_list"]
            
            # Denoise prices
            kf = create_adaptive_filter(
                process_noise=item['config'].kalman_process_noise,
                measurement_noise=item['config'].kalman_measurement_noise
            )
            df_denoised = kf.filter_ohlcv(loop_df)
            kalman_price = df_denoised["kalman_price"].values[-1]
            
            # Crossover calculations
            fast_sma = pd.Series(df_denoised["kalman_price"].values).rolling(item["fast_w"]).mean().values[-1]
            slow_sma = pd.Series(df_denoised["kalman_price"].values).rolling(item["slow_w"]).mean().values[-1]
            
            sig = 1.0 if fast_sma > slow_sma else 0.0
            logger.info(f"{asset_key} Kalman price: {kalman_price:.2f} | Fast SMA={fast_sma:.2f} | Slow SMA={slow_sma:.2f}")
            
            # Trailing stop monitor
            if item["in_pos"]:
                item["peak_price"] = max(item["peak_price"], ltp)
                unrealized_loss = (item["peak_price"] - ltp) / item["peak_price"]
                logger.info(f"{asset_key} Stop Monitor: Peak={item['peak_price']:.2f} | Loss={unrealized_loss*100:.2f}% (Limit={item['stop_pct']*100:.2f}%)")
                if unrealized_loss > item["stop_pct"]:
                    logger.warning(f"🚨 Trailing Stop triggered for {asset_key}! Liquidating.")
                    sig = 0.0
                    
            target_signals[asset_key] = sig
            
            joint_status[asset_key] = {
                "ltp": ltp,
                "kalman_price": kalman_price,
                "fast_sma": fast_sma,
                "slow_sma": slow_sma,
                "signal": sig,
                "shares_held": executor.get_position_for_symbol(item['config'].symbol),
                "leverage": item["leverage"],
                "stop_pct": item["stop_pct"],
                "in_position": item["in_pos"]
            }

        # Stage B: Dynamic Margin Scaling (Broker Safety Check)
        required_capital = {}
        total_required_capital = 0.0
        for asset_key, sig in target_signals.items():
            req = sig * allocated_capital_per_asset * assets[asset_key]["leverage"]
            required_capital[asset_key] = req
            total_required_capital += req
            
        margin_safety_cap = buying_power * 0.95 # keep 5% margin buffer
        scaling_factor = 1.0
        if total_required_capital > margin_safety_cap and total_required_capital > 0:
            scaling_factor = margin_safety_cap / total_required_capital
            logger.warning(f"⚠️ Total required buying power (USD {total_required_capital:.2f}) exceeds paper trading limit (USD {margin_safety_cap:.2f})!")
            logger.warning(f"   Applying active scaling factor of {scaling_factor:.4f} to fit within Alpaca leverage limit.")
        
        # Stage C: Execution orders placement
        order_plan = []
        for asset_key, item in assets.items():
            sig = target_signals[asset_key]
            ltp = current_ltps[asset_key]
            current_shares = joint_status[asset_key]["shares_held"]
            
            # Size position with scaling factor applied
            scaled_capital = required_capital[asset_key] * scaling_factor
            if sig > 0.0:
                target_shares = int(scaled_capital / ltp)
            else:
                target_shares = 0
                
            trade_diff = target_shares - current_shares
            joint_status[asset_key]["target_shares"] = target_shares
            
            logger.info(f"[{asset_key} EXECUTION] Signal={sig} | Scaled Capital=USD {scaled_capital:.2f} | TargetShares={target_shares} | SharesHeld={current_shares} | Order Qty={trade_diff}")
            
            # Submit order if there's a signal change, position exit, or meaningful rebalance (>$200)
            should_trade = False
            if trade_diff != 0:
                if sig != item["prev_sig"] or target_shares == 0:
                    should_trade = True
                elif abs(trade_diff * ltp) >= 200.0:
                    should_trade = True
                    
            if should_trade:
                side = OrderSide.BUY if trade_diff > 0 else OrderSide.SELL
                trade_qty = abs(trade_diff)
                order_plan.append({
                    "asset_key": asset_key,
                    "item": item,
                    "side": side,
                    "trade_qty": trade_qty,
                    "sig": sig,
                    "ltp": ltp
                })
            else:
                logger.info(f"No trading action required for {asset_key}.")

        # Execute SELL orders first to free up buying power, then BUY orders
        order_plan.sort(key=lambda x: 0 if x["side"] == OrderSide.SELL else 1)
        for order_info in order_plan:
            asset_key = order_info["asset_key"]
            item = order_info["item"]
            side = order_info["side"]
            trade_qty = order_info["trade_qty"]
            sig = order_info["sig"]
            ltp = order_info["ltp"]
            try:
                order_id = executor.place_order(symbol=item['config'].symbol, side=side, qty=trade_qty)
                if order_id:
                    logger.success(f"Order submitted successfully for {asset_key}! Side={side.value} | Qty={trade_qty} | Order ID={order_id}")
                    item["prev_sig"] = sig
                    if sig > 0:
                        item["in_pos"] = True
                        item["peak_price"] = ltp
                    else:
                        item["in_pos"] = False
                        item["peak_price"] = 0.0
            except Exception as e:
                logger.error(f"Order placement failed for {asset_key}: {e}")
                
        # 5. Generate unified joint advisory report from Groq AI Analyst
        try:
            logger.info("Synthesizing joint portfolio advisory report from Groq Llama 3.3...")
            market_context = {
                "portfolio_value": current_portfolio_value,
                "buying_power": buying_power,
                "mode": mode,
                "gld_status": joint_status["GLD"],
                "uso_status": joint_status["USO"]
            }
            commentary = groq_analyst.generate_trade_commentary(1.0, market_context)
            print(f"\n📢 [Groq AI Joint Portfolio Meta-Analyst Advisory]:\n{commentary}\n")
        except Exception as e:
            logger.warning(f"Groq joint commentary generation failed: {e}")

        logger.info(f"Sleeping {sleep_seconds} seconds before next iteration...")
        time.sleep(sleep_seconds)
        
    logger.success("==================================================================")
    logger.success("  DYNAMIC ART-DRL CONCURRENT JOINT PAPER TRADING COMPLETED        ")
    logger.success("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run concurrent multi-asset Gold + Oil paper trading on Alpaca.")
    parser.add_argument("--mode", type=str, default="swing_monthly", choices=["swing_monthly", "swing_low"],
                        help="Swing mode: 'swing_monthly' (default) or 'swing_low'.")
    parser.add_argument("--loops", type=int, default=3,
                        help="Number of execution iterations. Default is 3.")
    parser.add_argument("--sleep", type=int, default=10,
                        help="Sleep time in seconds between loops. Default is 10.")
    args = parser.parse_args()
    
    run_joint_live_trading(
        mode=args.mode,
        loop_count=args.loops,
        sleep_seconds=args.sleep
    )
