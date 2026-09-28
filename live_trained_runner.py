"""
Dynamic ART-DRL — Live Paper Trading Runner
============================================
Executes real-time paper trading by supporting three modes:
1. 'drl'          — Runs neural network inference across trained PPO/DQN/DDPG/A2C models.
2. 'swing_low'    — Runs the low-turnover swing champion crossover strategy (~1.5 trades/yr).
3. 'swing_monthly'— Runs the monthly swing champion crossover strategy (~11 trades/yr).

Applies recursive Kalman Filter price denoising, manages dynamic trailing stop-losses,
checks positions, and executes orders via Alpaca Paper API. Also integrates Groq LLM advisory.
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

from config.assets import get_asset, AssetConfig
from data.alpaca_client import AlpacaClient
from data.kalman_filter import create_adaptive_filter
from data.preprocessor import Preprocessor
from router.dynamic_router import DynamicRouter, MarketRegime
from execution.alpaca_executor import AlpacaExecutor
from agents.groq_analyst import GroqAnalyst

# Import actual trained agents for DRL mode
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.ddpg_agent import DDPGAgent
from agents.a2c_agent import A2CAgent

SAVED_CONFIGS_PATH = Path(__file__).resolve().parent / "config" / "saved_champion_configs.json"

def run_live_trading(
    asset_name: str = "GLD",
    timeframe: str = "1d",
    mode: str = "swing_low",
    loop_count: int = 5,
    sleep_seconds: int = 5
) -> None:
    """Runs the real-time execution loop.
    
    1. Synchronises with Alpaca Paper API.
    2. Loads active configurations (DRL checkpoints or Swing SMA/Stop parameters).
    3. Bootstraps historical data to seed Kalman filters and technical indicators.
    4. Enters real-time market iteration loop.
    5. Computes denoised price, crossovers, or DRL actions.
    6. Submits rebalancing orders to Alpaca Sandbox.
    7. Formulates risk advisory report via Groq.
    """
    logger.info("==================================================================")
    logger.info("       DYNAMIC ART-DRL LIVE PAPERS EXECUTION LOOP STARTED         ")
    logger.info("==================================================================")
    logger.info("Asset: {} | Timeframe: {} | Mode: {} | Max Loops: {}", asset_name, timeframe, mode, loop_count)
    
    # ── 1. Setup Controllers & Broker Executors ──────────────────────────────
    asset_config = get_asset(asset_name)
    router = DynamicRouter()
    groq_analyst = GroqAnalyst()
    
    logger.info("Initialising Alpaca US paper trading executor...")
    executor = AlpacaExecutor()
    data_client = AlpacaClient()
    
    # Verify account balance
    acct = executor.get_account_info()
    portfolio_value = acct["portfolio_value"] if acct["portfolio_value"] > 0 else asset_config.initial_capital
    logger.success("Connected to Alpaca Paper Sandbox! Portfolio Value: USD {:.2f} (Cash: {:.2f})", portfolio_value, acct["cash"])
    
    # ── 2. Configure Trading Parameters based on Mode ───────────────────────
    leverage = 1.0
    fast_w, slow_w = 10, 50
    stop_pct = 0.05
    
    dqn, ppo, ddpg, a2c = None, None, None, None
    
    if mode == "drl":
        logger.info("Loading trained PyTorch DRL checkpoints from checkpoints/ ...")
        checkpoint_dir = Path("checkpoints")
        
        dqn = DQNAgent(); dqn.load(checkpoint_dir / f"dqn_{asset_name}_{timeframe}")
        ppo = PPOAgent(); ppo.load(checkpoint_dir / f"ppo_{asset_name}_{timeframe}")
        ddpg = DDPGAgent(); ddpg.load(checkpoint_dir / f"ddpg_{asset_name}_{timeframe}")
        a2c = A2CAgent(); a2c.load(checkpoint_dir / f"a2c_{asset_name}_{timeframe}")
        
        leverage = asset_config.leverage_factor
        logger.info("DRL Model leverage default set to: {}x", leverage)
        
    else:
        # Load from saved JSON configs
        logger.info("Loading Swing parameters from config/saved_champion_configs.json...")
        try:
            with open(SAVED_CONFIGS_PATH, "r") as f:
                saved_data = json.load(f)
            
            if mode == "swing_low":
                cfg_group = "low_turnover_swing_champions"
            elif mode == "swing_monthly":
                cfg_group = "monthly_swing_champions"
            else:
                cfg_group = "ultra_hf_swing_champions"
            asset_key = "GLD" if "GLD" in asset_name else "USO"
            
            params = saved_data[cfg_group][asset_key]
            fast_w = params["fast_window"]
            slow_w = params["slow_window"]
            leverage = params["leverage_factor"]
            stop_pct = params["stop_loss_pct"]
            
            logger.success("Loaded Swing Champion Parameters:")
            logger.info("  --> Trend Crossover: SMA({}, {})", fast_w, slow_w)
            logger.info("  --> Position Sizing: {}x leverage", leverage)
            logger.info("  --> Trailing Stop:   {}%", stop_pct * 100)
        except Exception as e:
            logger.warning("Could not parse saved configs JSON: {}. Using AssetConfig defaults.", e)
            fast_w, slow_w = 10, 40
            leverage = 3.0
            stop_pct = 0.10
            
    # ── 3. Bootstrap Historical Series ──────────────────────────────────────
    logger.info("Bootstrapping historical prices for feature engineering sliding window...")
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    bootstrap_df = data_client.fetch_ohlcv(symbol=asset_config.symbol, timeframe=timeframe, end=today_str)
    if bootstrap_df.empty:
        logger.error("Could not fetch bootstrap data from Alpaca client. Aborting loop.")
        return
        
    prices_list = list(bootstrap_df["close"].values)
    highs_list = list(bootstrap_df["high"].values)
    lows_list = list(bootstrap_df["low"].values)
    opens_list = list(bootstrap_df["open"].values)
    vols_list = list(bootstrap_df["volume"].values)
    dates_list = list(bootstrap_df.index)
    
    # ── 4. Start Live Quote / Execution Loop ───────────────────────────────────
    prev_action = 0.0
    in_pos = False
    peak_price = 0.0
    
    # Retrieve current active position if any
    current_shares = executor.get_position_for_symbol(asset_config.symbol)
    if current_shares > 0:
        in_pos = True
        peak_price = prices_list[-1]
        prev_action = 1.0
        logger.info("Detected existing position of {} shares. Initialized Trailing Peak price: {:.2f}", current_shares, peak_price)
        
    for i in range(1, loop_count + 1):
        logger.info("-------------------- Iteration {}/{} --------------------", i, loop_count)
        
        # 1. Fetch current price tick (LTP)
        ltp = executor._get_asset_price_estimate(asset_config.symbol)
        
        # If market session is closed, simulate a small price tick around last close
        if ltp <= 0 or ltp == 100.0:
            last_price = prices_list[-1]
            ltp = last_price * (1.0 + np.random.normal(0.0, 0.002))
            logger.info("Market session closed or unavailable. Simulating LTP price tick: {:.2f}", ltp)
        else:
            logger.info("Fetched Real-Time Last Traded Price (LTP) from Alpaca: {:.2f}", ltp)
            
        # Append latest tick to sliding buffers
        current_time = pd.Timestamp.now(tz="UTC")
        dates_list.append(current_time)
        prices_list.append(ltp)
        opens_list.append(prices_list[-2])
        highs_list.append(max(prices_list[-2], ltp))
        lows_list.append(min(prices_list[-2], ltp))
        vols_list.append(np.random.uniform(500, 2000))
        
        if len(prices_list) > 300:
            prices_list.pop(0); opens_list.pop(0); highs_list.pop(0); lows_list.pop(0); vols_list.pop(0); dates_list.pop(0)
            
        # Construct temporary DataFrame for preprocessing
        loop_df = pd.DataFrame(index=dates_list)
        loop_df["open"] = opens_list
        loop_df["high"] = highs_list
        loop_df["low"] = lows_list
        loop_df["close"] = prices_list
        loop_df["volume"] = vols_list
        
        # 2. Volatility Regime switching decision
        active_agent, regime, vol_met = router.route(np.array(prices_list))
        logger.info(
            "Regime Router: Volatility={:.5f} | Confirmed Regime={} | Selected DRL Agent={}",
            vol_met["current_vol"], regime.value, active_agent
        )
        
        # 3. Kalman Price Denoising
        kf = create_adaptive_filter(
            process_noise=asset_config.kalman_process_noise,
            measurement_noise=asset_config.kalman_measurement_noise
        )
        df_denoised = kf.filter_ohlcv(loop_df)
        
        # 4. Generate Execution Signals
        sig = 0.0
        
        if mode == "drl":
            # DRL Mode: variable selection + Neural Net policy execution
            preprocessor = Preprocessor()
            feature_df = preprocessor.preprocess(df_denoised)
            
            feature_cols = ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_width', 'cci_30', 'dx_30', 'atr_14', 'return_simple', 'return_log', 'vol_rolling_30', 'vol_ewma']
            latest_features = feature_df[feature_cols].values[-1]
            
            # Query Portfolio Balance
            acct = executor.get_account_info()
            p_val = acct["portfolio_value"] if acct["portfolio_value"] > 0 else portfolio_value
            cash = acct["cash"]
            shares = executor.get_position_for_symbol(asset_config.symbol)
            pv_safe = max(p_val, 1e-8)
            portfolio_state = np.array([
                cash / pv_safe,
                (shares * ltp) / pv_safe,
                (p_val - asset_config.initial_capital) / asset_config.initial_capital
            ], dtype=np.float32)
            
            obs = np.concatenate([latest_features, portfolio_state])
            
            # Model inference
            if active_agent == "DQN":
                raw_act, _ = dqn.model.predict(obs, deterministic=True)
                action_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
                sig = action_map.get(int(raw_act), 0.0)
            elif active_agent == "PPO":
                raw_act, _ = ppo.model.predict(obs, deterministic=True)
                sig = float(raw_act[0])
            elif active_agent == "DDPG":
                raw_act, _ = ddpg.model.predict(obs, deterministic=True)
                sig = float(raw_act[0])
            else:
                raw_act, _ = a2c.model.predict(obs, deterministic=True)
                sig = float(raw_act[0])
                
            if sig < 0.0:
                sig = 0.0 # long only
                
        else:
            # Swing Mode: Kalman SMA Crossover Signal
            kalman_prices = df_denoised["kalman_price"].values
            
            fast_sma = pd.Series(kalman_prices).rolling(fast_w).mean().values[-1]
            slow_sma = pd.Series(kalman_prices).rolling(slow_w).mean().values[-1]
            
            sig = 1.0 if fast_sma > slow_sma else 0.0
            logger.info("Swing Crossover Denoised Trend: Fast SMA={:.2f} | Slow SMA={:.2f}", fast_sma, slow_sma)
            
            # Trailing Stop-Loss active containment
            if in_pos:
                peak_price = max(peak_price, ltp)
                unrealized_loss = (peak_price - ltp) / peak_price
                logger.info("Position Active Stop Monitor: Peak Price={:.2f} | Current LTP={:.2f} | Loss={:.2f}% (Stop Limit={:.2f}%)", 
                            peak_price, ltp, unrealized_loss*100, stop_pct*100)
                if unrealized_loss > stop_pct:
                    logger.warning("🚨 Dynamic Trailing Stop-Loss Triggered! Forcing Exit Action.")
                    sig = 0.0 # Force liquidation
                    
        logger.info("Target Execution Position Sizing: sig = {:.4f}", sig)
        
        # 5. Position Sizing
        acct = executor.get_account_info()
        p_val = acct["portfolio_value"] if acct["portfolio_value"] > 0 else portfolio_value
        shares = executor.get_position_for_symbol(asset_config.symbol)
        
        if sig > 0.0:
            target_shares = int((sig * p_val * leverage) / ltp)
        else:
            target_shares = 0
            
        trade_diff = target_shares - shares
        
        # 6. Submit Paper orders to Alpaca Sandbox
        if trade_diff != 0 and (sig != prev_action or target_shares == 0):
            side = OrderSide.BUY if trade_diff > 0 else OrderSide.SELL
            trade_qty = abs(trade_diff)
            logger.info("Executing Paper Order Side={} | Qty={} shares | Action={:.2f}", side.value, trade_qty, sig)
            try:
                order_id = executor.place_order(symbol=asset_config.symbol, side=side, qty=trade_qty)
                if order_id:
                    logger.success("Alpaca Paper Order successfully submitted! Order ID={}", order_id)
                    prev_action = sig
                    if sig > 0:
                        in_pos = True
                        peak_price = ltp
                    else:
                        in_pos = False
                        peak_price = 0.0
            except Exception as e:
                logger.error("Alpaca Order execution failed: {}", e)
        else:
            logger.info("No trading order executed. Target allocation matches existing position.")
            
        # 7. Invoke Groq Metadata Risk Analyst
        market_context = {
            "symbol": asset_config.symbol,
            "ltp": ltp,
            "volatility": vol_met["current_vol"],
            "regime": regime.value,
            "selected_agent": "SWING_CROSSOVER" if mode != "drl" else active_agent,
            "target_allocation": sig,
            "leverage": leverage,
            "shares_held": shares,
            "target_shares": target_shares,
            "portfolio_value": p_val,
            "kalman_price": df_denoised["kalman_price"].values[-1],
            "mode": mode
        }
        
        try:
            logger.info("Requesting advisory report from Groq Llama 3.3 Analyst...")
            commentary = groq_analyst.generate_trade_commentary(sig, market_context)
            print(f"\n📢 [Groq AI Meta-Analyst Advisory]:\n{commentary}\n")
        except Exception as e:
            logger.warning("Groq metadata analyst failed: {}. Continuing.", e)
            
        # Cooldown sleep
        logger.info("Sleeping {} seconds before next market tick...", sleep_seconds)
        time.sleep(sleep_seconds)
        
    logger.success("==================================================================")
    logger.success("   DYNAMIC ART-DRL LIVE PAPERS execution DEMO LOOP COMPLETED      ")
    logger.success("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Paper Trading execution engine.")
    parser.add_argument("--asset", type=str, default="GLD", help="Asset config name from registry.")
    parser.add_argument("--mode", type=str, default="swing_low", choices=["drl", "swing_low", "swing_monthly", "swing_high"], 
                        help="Execution mode: drl checkpoint policy, swing_low, swing_monthly, or swing_high (2-3 trades/month).")
    parser.add_argument("--loops", type=int, default=3, help="Number of real-time trading loops.")
    parser.add_argument("--sleep", type=int, default=5, help="Seconds to sleep between ticks.")
    args = parser.parse_args()
    
    run_live_trading(
        asset_name=args.asset,
        timeframe="1d",
        mode=args.mode,
        loop_count=args.loops,
        sleep_seconds=args.sleep
    )
