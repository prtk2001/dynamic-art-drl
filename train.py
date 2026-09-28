"""
Dynamic ART-DRL — DRL Training Pipeline
=======================================
Ingests raw historical data from Upstox or Alpaca, applies recursive Kalman Filter
noise reduction, engineers features, and trains all 4 agents (DQN, PPO, DDPG, A2C)
sequentially using Stable-Baselines3.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from config.settings import (
    TRAINING_TIMESTEPS,
    CHECKPOINT_DIR,
    TENSORBOARD_DIR
)
from config.assets import get_asset
from data.data_loader import load_asset_data
from data.kalman_filter import create_adaptive_filter
from data.preprocessor import Preprocessor
from environment.trading_env import DiscreteTradingEnv, ContinuousTradingEnv
from environment.options_trading_env import DiscreteOptionsTradingEnv, ContinuousOptionsTradingEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.ddpg_agent import DDPGAgent
from agents.a2c_agent import A2CAgent


def run_training_pipeline(
    asset_name: str,
    timeframe: str,
    timesteps: int = TRAINING_TIMESTEPS
) -> None:
    """Sequential training pipeline for the heterogeneous DRL agent pool.
    
    1. Fetch historical OHLCV data for asset
    2. Denoisify price series using recursive Kalman Filter
    3. Generate technical indicators and volatility inputs
    4. Initialize custom Gymnasium trading environments
    5. Train DQN, PPO, DDPG, and A2C agents
    """
    logger.info("Starting DRL Training Pipeline for {} ({})", asset_name, timeframe)
    asset_config = get_asset(asset_name)
    
    # ── 1. Fetch historical data ─────────────────────────────────────────────
    logger.info("Fetching historical data using unified data loader...")
    try:
        raw_df = load_asset_data(asset_name, timeframe)
    except Exception as e:
        logger.warning("Data loading failed: {}. Generating synthetic market data...", e)
        raw_df = generate_synthetic_data(asset_config, timeframe)
            
    # ── 2. Apply Kalman Filter ────────────────────────────────────────────────
    logger.info("Applying recursive Kalman Filter price denoising...")
    kf = create_adaptive_filter(
        process_noise=asset_config.kalman_process_noise,
        measurement_noise=asset_config.kalman_measurement_noise
    )
    df_denoised = kf.filter_ohlcv(raw_df)
    
    # ── 3. Feature engineering ────────────────────────────────────────────────
    logger.info("Engineering technical indicators and rolling volatilities...")
    preprocessor = Preprocessor()
    feature_df = preprocessor.preprocess(df_denoised)
    
    # ── 4. Setup Environments ────────────────────────────────────────────────
    logger.info("Initialising Gymnasium trading environments...")
    feature_cols = ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_width', 'cci_30', 'dx_30', 'atr_14', 'return_simple', 'return_log', 'vol_rolling_30', 'vol_ewma']
    features = feature_df[feature_cols].values
    prices = feature_df["close"].values
    
    # Discrete Env for DQN
    if asset_name in ["NIFTY_OPT", "BANKNIFTY_OPT", "SENSEX_OPT"]:
        logger.info("Initializing options-specific trading environments (ATM Rolling Premium)...")
        ce_close = feature_df["ce_close"].values
        pe_close = feature_df["pe_close"].values
        ce_close_next = feature_df["ce_close_next"].values
        pe_close_next = feature_df["pe_close_next"].values
        
        env_discrete = DiscreteOptionsTradingEnv(
            features=features,
            prices=prices,
            ce_premiums=ce_close,
            pe_premiums=pe_close,
            ce_premiums_next=ce_close_next,
            pe_premiums_next=pe_close_next,
            asset_config=asset_config,
            initial_capital=asset_config.initial_capital,
            max_position_pct=0.15,
            dates=feature_df.index,
            expiries=feature_df["expiry"].values,
            timeframe=timeframe
        )
        env_continuous = ContinuousOptionsTradingEnv(
            features=features,
            prices=prices,
            ce_premiums=ce_close,
            pe_premiums=pe_close,
            ce_premiums_next=ce_close_next,
            pe_premiums_next=pe_close_next,
            asset_config=asset_config,
            initial_capital=asset_config.initial_capital,
            max_position_pct=0.15,
            dates=feature_df.index,
            expiries=feature_df["expiry"].values,
            timeframe=timeframe
        )
    else:
        env_discrete = DiscreteTradingEnv(
            features=features,
            prices=prices,
            asset_config=asset_config,
            initial_capital=asset_config.initial_capital
        )
        
        env_continuous = ContinuousTradingEnv(
            features=features,
            prices=prices,
            asset_config=asset_config,
            initial_capital=asset_config.initial_capital
        )
    
    # ── 5. Train heterogenous agents ─────────────────────────────────────────
    logger.info("=== Starting Heterogenous DRL Agent Training Pool ===")
    
    # DQN (Discrete Action space)
    logger.info("Training DQN Agent (Conservative discrete anchor)...")
    dqn = DQNAgent()
    dqn_model_path = CHECKPOINT_DIR / f"dqn_{asset_name}_{timeframe}"
    dqn.train(env_discrete, total_timesteps=timesteps)
    dqn.save(dqn_model_path)
    
    # PPO (Continuous)
    logger.info("Training PPO Agent (Continuous on-policy trend-following)...")
    ppo = PPOAgent()
    ppo_model_path = CHECKPOINT_DIR / f"ppo_{asset_name}_{timeframe}"
    ppo.train(env_continuous, total_timesteps=timesteps)
    ppo.save(ppo_model_path)
    
    # DDPG (Continuous)
    logger.info("Training DDPG Agent (Continuous off-policy fine positioning)...")
    ddpg = DDPGAgent()
    ddpg_model_path = CHECKPOINT_DIR / f"ddpg_{asset_name}_{timeframe}"
    ddpg.train(env_continuous, total_timesteps=timesteps)
    ddpg.save(ddpg_model_path)
    
    # A2C (Continuous)
    logger.info("Training A2C Agent (Fast-transition responder)...")
    a2c = A2CAgent()
    a2c_model_path = CHECKPOINT_DIR / f"a2c_{asset_name}_{timeframe}"
    a2c.train(env_continuous, total_timesteps=timesteps)
    a2c.save(a2c_model_path)
    
    logger.success("=== DRL Training Pipeline Completed Successfully ===")


def generate_synthetic_data(asset_config: get_asset, timeframe: str) -> pd.DataFrame:
    """Generate mock price data to enable standalone offline pipeline execution."""
    np.random.seed(42)
    n = 2000 if timeframe == "1h" else 500
    dates = pd.date_range(end="2026-05-30", periods=n, freq="H" if timeframe == "1h" else "D")
    
    # Create random walk with slight upward drift
    base_price = 72000.0 if asset_config.currency == "INR" else 150.0
    daily_vol = 0.015 if asset_config.asset_class == "crude_oil" else 0.008
    
    returns = np.random.normal(0.0001, daily_vol, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = df["close"] * (1.0 + np.random.normal(0, 0.001, n))
    df["high"] = df[["open", "close"]].max(axis=1) * (1.0 + abs(np.random.normal(0, 0.002, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1.0 - abs(np.random.normal(0, 0.002, n)))
    df["volume"] = np.random.randint(100, 10000, n).astype(float)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heterogenous DRL agents sequentially.")
    parser.add_argument("--asset", type=str, default="GOLD_MCX", help="Asset config name from registry.")
    parser.add_argument("--timeframe", type=str, default="1d", choices=["1d", "1h", "5min", "15min"], help="Trading timeframe.")
    parser.add_argument("--timesteps", type=str, default="20000", help="Training steps.")
    args = parser.parse_args()
    
    run_training_pipeline(args.asset, args.timeframe, int(args.timesteps))
