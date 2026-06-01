"""
Dynamic ART-DRL — Live Trading Execution Runner
===============================================
Demonstrates the live trading pipeline. Uses WebSocket or polling to fetch the latest
prices, routes execution control based on rolling volatility, generates Groq LLM
meta-analysis trade commentary, and submits orders using Upstox/Alpaca executors.
"""

import argparse
import sys
import time
import pandas as pd
from loguru import logger

from config.assets import get_asset
from data.upstox_client import UpstoxClient
from data.alpaca_client import AlpacaClient
from data.kalman_filter import create_adaptive_filter
from data.preprocessor import Preprocessor
from router.dynamic_router import DynamicRouter
from execution.upstox_executor import UpstoxExecutor
from execution.alpaca_executor import AlpacaExecutor
from agents.groq_analyst import GroqAnalyst
from train import generate_synthetic_data


def run_live_pipeline(
    asset_name: str,
    timeframe: str,
    loop_count: int = 5
) -> None:
    """Executes the live trading strategy loop for a single asset.
    
    1. Initialize the volatility strategy switcher and execution broker
    2. Start the real-time quote loop
    3. Update price series, calculate regime classification
    4. Execute active agent target allocation
    5. Invoke Groq LLM agent to provide natural language commentary
    """
    logger.info("Starting live Dynamic ART-DRL pipeline for {} ({})", asset_name, timeframe)
    asset_config = get_asset(asset_name)
    
    # ── 1. Setup Controllers & Broker Executors ──────────────────────────────
    router = DynamicRouter()
    groq_analyst = GroqAnalyst()
    
    if asset_config.exchange == "MCX":
        logger.info("Initialising Upstox India futures executor...")
        executor = UpstoxExecutor()
        data_client = UpstoxClient()
    else:
        logger.info("Initialising Alpaca US paper trading executor...")
        executor = AlpacaExecutor()
        data_client = AlpacaClient()
        
    # Get portfolio balance to size orders
    if asset_config.exchange == "MCX":
        portfolio_value = asset_config.initial_capital
    else:
        info = executor.get_account_info()
        portfolio_value = info["portfolio_value"] if info["portfolio_value"] > 0 else asset_config.initial_capital
        
    logger.info("Starting portfolio valuation: INR/USD {:.2f}", portfolio_value)
    
    # ── 2. Load historical baseline series ───────────────────────────────────
    # The router requires at least 30-60 bars of historical prices to calculate volatility
    logger.info("Pre-loading historical baseline price series...")
    try:
        if asset_config.exchange == "MCX":
            raw_df = data_client.fetch_mcx(symbol=asset_config.symbol, interval=timeframe)
        else:
            raw_df = data_client.fetch_ohlcv(symbol=asset_config.symbol, timeframe=timeframe)
    except Exception as e:
        logger.warning("Market pre-load failed: {}. Generating historical bootstrap series.", e)
        raw_df = generate_synthetic_data(asset_config, timeframe)
        
    prices_list = list(raw_df["close"].values)
    
    # ── 3. Start Live Quote Loop ─────────────────────────────────────────────
    logger.success("=== Dynamic ART-DRL Real-Time Execution Loop Started ===")
    
    for i in range(1, loop_count + 1):
        logger.info("-------------------- Iteration {}/{} --------------------", i, loop_count)
        
        # 1. Fetch current price tick (LTP)
        # In actual live, we fetch via WebSocket or live HTTP.
        # For this demonstration, we query the broker LTP or fall back to simulated tick.
        if asset_config.exchange == "MCX":
            active_key = executor.resolve_active_future_contract(asset_config.symbol)
            ltp = executor.get_last_price(active_key) if active_key else 0.0
        else:
            ltp = executor._get_asset_price_estimate(asset_config.symbol)
            
        if ltp <= 0:
            # Generate simulated price tick for continuous demonstration
            last_price = prices_list[-1]
            ltp = last_price * (1.0 + np.random.normal(0.0001, 0.005))
            logger.info("Simulating market tick LTP: {:.2f}", ltp)
        else:
            logger.info("Fetched Live Last Traded Price (LTP): {:.2f}", ltp)
            
        # Append latest price to our sliding buffer
        prices_list.append(ltp)
        if len(prices_list) > 500:
            prices_list.pop(0)
            
        # 2. Strategy Routing Regime switch calculation
        active_agent, regime, vol_met = router.route(prices_list)
        logger.info(
            "Regime Switch: Volatility={:.5f} | Confirmed Regime={} | Selected Agent={}",
            vol_met["current_vol"], regime.value, active_agent
        )
        
        # 3. Simulate continuous RL action (predict target allocation)
        # DQN = cautious, PPO = trend, etc.
        # Action determines percentage of capital long or short (range -1 to 1)
        sma_fast = sum(prices_list[-10:]) / 10.0
        sma_slow = sum(prices_list[-30:]) / 30.0
        trend = 1.0 if sma_fast > sma_slow else -1.0
        
        if active_agent == "DQN":
            action = 0.5 * trend   # Conservative discrete anchor
        elif active_agent == "PPO":
            action = 0.95 * trend  # Strong trend allocation
        elif active_agent == "DDPG":
            action = 0.75 * trend  # Continuous fine sizing
        else:
            action = 0.8 * trend   # Fast transition A2C
            
        # 4. Submit Order to Broker
        logger.info("Translating strategy allocation ({:.2f}) into order execution...", action)
        try:
            order_id = executor.execute_rl_action(asset_config, action, portfolio_value)
            if order_id:
                logger.success("Live order successfully filled! Order ID={}", order_id)
            else:
                logger.info("No trading order executed (flat/already matches).")
        except Exception as e:
            logger.error("Failed to submit broker order: {}", e)
            
        # 5. Invoke Groq LLM Agent for trade commentary & risk analysis
        # Construct current market context
        market_context = {
            "symbol": asset_config.symbol,
            "ltp": ltp,
            "volatility": vol_met["current_vol"],
            "regime": regime.value,
            "selected_agent": active_agent,
            "target_allocation": action,
            "portfolio_value": portfolio_value,
            "fast_sma": sma_fast,
            "slow_sma": sma_slow
        }
        
        try:
            logger.info("Requesting Groq LLM (Llama 3.3 70B) live market report...")
            commentary = groq_analyst.generate_trade_commentary(action, market_context)
            print(f"\n📢 [Groq Analyst Live Commentary]:\n{commentary}\n")
            
            # Risk check
            risk_assessment = groq_analyst.assess_risk(market_context, regime.value)
            logger.info("Risk Assessment Score: {}/100", risk_assessment.get("risk_score", 0))
        except Exception as e:
            logger.warning("Groq commentary failed: {}. Continuing loop.", e)
            
        # Cooldown interval (in live this would be hourly/daily)
        logger.info("Sleeping 5 seconds before next market cycle...")
        time.sleep(5)
        
    logger.success("=== Live Runner Demonstration Completed ===")


import numpy as np  # locally needed for synthetic simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live execution runner.")
    parser.add_argument("--asset", type=str, default="GOLD_MCX", help="Asset name from registry (GOLD_MCX, CRUDE_MCX, GLD, USO).")
    parser.add_argument("--timeframe", type=str, default="1d", help="Trading interval.")
    parser.add_argument("--loops", type=int, default=3, help="Demo loop count iterations.")
    args = parser.parse_args()
    
    run_live_pipeline(args.asset, args.timeframe, args.loops)
