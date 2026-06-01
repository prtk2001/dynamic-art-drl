"""
Dynamic ART-DRL — Dynamic Strategy Router (Method 3)
===================================================
Implements the core volatility-regime switching controller. Monitors
rolling market volatility and shifts execution control to the best-suited
DRL agent, applying hysteresis buffers to prevent rapid whipsawing.
"""

from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, Tuple

from config.settings import (
    VOLATILITY_LOOKBACK,
    HISTORICAL_VOL_WINDOW,
    REGIME_HYSTERESIS_BARS
)


class MarketRegime(Enum):
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"


class DynamicRouter:
    """Dynamic Strategy Router (Method 3) for volatility-based agent switching.
    
    Monitors short-term rolling volatility of returns relative to long-term
    historical volatility distributions, and delegates control to DQN, PPO,
    DDPG, or A2C.
    """
    
    def __init__(
        self,
        vol_lookback: int = VOLATILITY_LOOKBACK,
        hist_window: int = HISTORICAL_VOL_WINDOW,
        hysteresis_bars: int = REGIME_HYSTERESIS_BARS
    ) -> None:
        self.vol_lookback = vol_lookback
        self.hist_window = hist_window
        self.hysteresis_bars = hysteresis_bars
        
        # State tracking for hysteresis
        self.current_regime: MarketRegime = MarketRegime.NORMAL_VOLATILITY
        self.pending_regime: MarketRegime = MarketRegime.NORMAL_VOLATILITY
        self.regime_counter: int = 0
        
        logger.info(
            "DynamicRouter initialised: vol_lookback={}, hist_window={}, hysteresis_bars={}",
            vol_lookback, hist_window, hysteresis_bars
        )
        
    def calculate_volatility_metrics(
        self,
        prices: np.ndarray | pd.Series
    ) -> Tuple[float, float, float]:
        """Compute current rolling volatility, long-term historical average, and std of historical vol.
        
        Parameters
        ----------
        prices : np.ndarray or pd.Series
            Historical price series (daily or hourly).
            
        Returns
        -------
        current_vol : float
            Standard deviation of returns over short rolling window.
        historical_mean_vol : float
            Average of rolling volatilities over the historical window.
        historical_std_vol : float
            Standard deviation of rolling volatilities over the historical window.
        """
        # 1. Compute log returns
        df_prices = pd.Series(prices)
        returns = np.log(df_prices / df_prices.shift(1)).dropna()
        
        if len(returns) < self.vol_lookback:
            # Fallback for insufficient data
            return 0.0, 0.0, 0.0
            
        # 2. Compute rolling volatility of returns (annualised/normalized)
        rolling_vols = returns.rolling(window=self.vol_lookback).std()
        
        # Current rolling volatility (latest non-nan)
        current_vol = float(rolling_vols.iloc[-1])
        
        # 3. Compute historical distributions over the lookback window
        hist_vols = rolling_vols.dropna().tail(self.hist_window)
        
        if len(hist_vols) < 2:
            return current_vol, current_vol, 0.0
            
        historical_mean_vol = float(hist_vols.mean())
        historical_std_vol = float(hist_vols.std())
        
        return current_vol, historical_mean_vol, historical_std_vol
        
    def classify_regime(
        self,
        current_vol: float,
        hist_mean: float,
        hist_std: float
    ) -> MarketRegime:
        """Classify current market state into volatility regimes.
        
        Rules:
        - EXTREME: current_vol > hist_mean + hist_std
        - HIGH:    hist_mean < current_vol <= hist_mean + hist_std
        - NORMAL:  hist_mean - hist_std < current_vol <= hist_mean
        - LOW:     current_vol <= hist_mean - hist_std
        """
        if current_vol <= 0.0 or hist_mean <= 0.0:
            return MarketRegime.NORMAL_VOLATILITY
            
        if current_vol > hist_mean + hist_std:
            return MarketRegime.EXTREME_VOLATILITY
        elif current_vol > hist_mean:
            return MarketRegime.HIGH_VOLATILITY
        elif current_vol > hist_mean - hist_std:
            return MarketRegime.NORMAL_VOLATILITY
        else:
            return MarketRegime.LOW_VOLATILITY
            
    def get_active_agent_name(
        self,
        regime: MarketRegime
    ) -> str:
        """Map volatility regime to the most suitable DRL agent.
        
        - EXTREME_VOLATILITY: DQN (Conservative discrete anchor, prevents drawdown)
        - HIGH_VOLATILITY:    A2C (Synchronous fast-adapting actor-critic)
        - NORMAL_VOLATILITY:  PPO (Robust, stable on-policy trend-following)
        - LOW_VOLATILITY:     DDPG (Continuous off-policy control for precise scaling)
        """
        mapping = {
            MarketRegime.EXTREME_VOLATILITY: "DQN",
            MarketRegime.HIGH_VOLATILITY: "A2C",
            MarketRegime.NORMAL_VOLATILITY: "PPO",
            MarketRegime.LOW_VOLATILITY: "DDPG"
        }
        return mapping[regime]
        
    def route(
        self,
        prices: np.ndarray | pd.Series
    ) -> Tuple[str, MarketRegime, Dict[str, float]]:
        """Determine which agent should execute trading operations given latest price data.
        
        Incorporates a hysteresis buffer to filter out temporary boundary crossings.
        
        Returns
        -------
        agent_name : str
            Name of the active agent ('DQN', 'PPO', 'DDPG', or 'A2C').
        regime : MarketRegime
            Classified volatility regime.
        metrics : dict
            Dictionary of calculated volatility parameters.
        """
        curr_vol, hist_mean, hist_std = self.calculate_volatility_metrics(prices)
        raw_regime = self.classify_regime(curr_vol, hist_mean, hist_std)
        
        # Apply Hysteresis Buffer
        if raw_regime == self.current_regime:
            # Reverted/Stable: reset counter
            self.pending_regime = raw_regime
            self.regime_counter = 0
        else:
            # Change detected, check if already pending
            if raw_regime == self.pending_regime:
                self.regime_counter += 1
                if self.regime_counter >= self.hysteresis_bars:
                    # Switch regime
                    logger.info(
                        "Regime transition confirmed: {} ➔ {} (Vol={:.5f}, HistMean={:.5f}, HistStd={:.5f})",
                        self.current_regime.value,
                        raw_regime.value,
                        curr_vol,
                        hist_mean,
                        hist_std
                    )
                    self.current_regime = raw_regime
                    self.regime_counter = 0
            else:
                self.pending_regime = raw_regime
                self.regime_counter = 1
                
        active_agent = self.get_active_agent_name(self.current_regime)
        
        metrics = {
            "current_vol": curr_vol,
            "historical_mean": hist_mean,
            "historical_std": hist_std,
            "raw_regime": raw_regime.value,
            "confirmed_regime": self.current_regime.value,
            "pending_regime": self.pending_regime.value,
            "counter": self.regime_counter
        }
        
        return active_agent, self.current_regime, metrics


if __name__ == "__main__":
    # Simple test of the switching router
    np.random.seed(42)
    router = DynamicRouter(vol_lookback=10, hist_window=50, hysteresis_bars=2)
    
    # Generate prices with low and high vol epochs
    prices = [100.0]
    for i in range(100):
        # Calm regime
        prices.append(prices[-1] * (1.0 + np.random.randn() * 0.005))
        
    for i in range(50):
        # Extreme volatility spike
        prices.append(prices[-1] * (1.0 + np.random.randn() * 0.08))
        
    for i in range(50):
        # Calm again
        prices.append(prices[-1] * (1.0 + np.random.randn() * 0.005))
        
    # Route step-by-step
    for idx in range(20, len(prices)):
        window = prices[:idx]
        agent, regime, met = router.route(window)
        if idx % 15 == 0:
            print(f"Step {idx:3d}: Vol={met['current_vol']:.4f} | Regime={regime.name:<18s} | Selected Agent={agent}")
