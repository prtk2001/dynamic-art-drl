"""
Dynamic ART-DRL — Options Trading Environments
===============================================
Custom Gymnasium environments for trading Nifty index options.
Implements the ATM Rolling Premium logic (Option A):
* At step t, buy ATM Call (CE) and/or Put (PE) options.
* At step t+1, mark-to-market using ce_close_next / pe_close_next,
  liquidate positions (paying transaction fees), and roll into new contracts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from config import settings
from config.assets import AssetConfig, NIFTY_OPT
from environment.reward import RewardCalculator, RewardType


# ═════════════════════════════════════════════════════════════════════════════
# Trade Record for Options
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionTradeRecord:
    """Log entry for an options trade execution."""
    step: int
    ce_action: float
    pe_action: float
    ce_qty: float
    pe_qty: float
    ce_premium: float
    pe_premium: float
    transaction_cost: float
    cash_after: float
    portfolio_value: float


# ═════════════════════════════════════════════════════════════════════════════
# Base Options Trading Environment
# ═════════════════════════════════════════════════════════════════════════════

class _BaseOptionsTradingEnv(gym.Env):
    """Abstract base for Nifty options trading environments.

    Parameters
    ----------
    features : np.ndarray
        Pre-computed market feature matrix (T × n_features).
    prices : np.ndarray
        Future close prices (T).
    ce_premiums : np.ndarray
        ATM Call close premiums (T).
    pe_premiums : np.ndarray
        ATM Put close premiums (T).
    ce_premiums_next : np.ndarray
        ATM Call close premiums at next step (T).
    pe_premiums_next : np.ndarray
        ATM Put close premiums at next step (T).
    asset_config : AssetConfig
        Options configuration settings.
    reward_type : RewardType
        Reward formula style.
    initial_capital : float
        Starting capital.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        ce_premiums: np.ndarray,
        pe_premiums: np.ndarray,
        ce_premiums_next: np.ndarray,
        pe_premiums_next: np.ndarray,
        asset_config: AssetConfig = NIFTY_OPT,
        reward_type: RewardType = RewardType.RISK_SENSITIVE,
        initial_capital: Optional[float] = None,
        max_position_pct: float = settings.MAX_POSITION_PCT,
        dates: Optional[Any] = None,
        expiries: Optional[Any] = None,
        timeframe: str = "1d",
    ) -> None:
        super().__init__()

        # Validate dimensions
        assert features.ndim == 2
        T = len(prices)
        assert len(features) == T
        assert len(ce_premiums) == T
        assert len(pe_premiums) == T
        assert len(ce_premiums_next) == T
        assert len(pe_premiums_next) == T

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.ce_premiums = ce_premiums.astype(np.float64)
        self.pe_premiums = pe_premiums.astype(np.float64)
        self.ce_premiums_next = ce_premiums_next.astype(np.float64)
        self.pe_premiums_next = pe_premiums_next.astype(np.float64)
        
        self.n_steps = T
        self.n_features = features.shape[1]

        # Asset and cost parameters
        self.asset_config = asset_config
        self.buy_cost_frac = asset_config.buy_cost_bps / 10_000.0
        self.sell_cost_frac = asset_config.sell_cost_bps / 10_000.0
        self.initial_capital = initial_capital or asset_config.initial_capital
        self.max_position_pct = max_position_pct

        # Setup timeframe-specific parameters
        self.timeframe = timeframe
        params = asset_config.timeframe_params.get(timeframe, {})
        self.profit_target = params.get("profit_target", 0.50)
        self.stop_loss = params.get("stop", params.get("stop_loss_pct", 0.15))
        self.time_decay_bars = params.get("time_decay_bars", 100000)
        self.cooldown_bars = params.get("cooldown", 0)
        self.leverage = params.get("leverage", asset_config.leverage_factor)
        
        # PCS & restricted compounding additions
        self.trailing_sl = params.get("trailing_sl", 0.0)
        self.decay_sl = params.get("decay_sl", False)
        self.max_lots = params.get("max_lots", 100)
        self.compounding = params.get("compounding", True)

        # Store dates and expiries
        import pandas as pd
        if dates is not None:
            self.dates = pd.to_datetime(dates)
        else:
            self.dates = pd.date_range("2020-01-01", periods=self.n_steps, freq="5min")
            
        if expiries is not None:
            self.expiries = expiries
        else:
            self.expiries = self.dates.strftime("%Y-%m-%d").values

        # Observation space: Market features + 4 portfolio-state variables
        # (cash_ratio, ce_ratio, pe_ratio, unrealised_pnl_ratio)
        obs_dim = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self.reward_calc = RewardCalculator(reward_type=reward_type)

        # Episode state variables
        self._current_step: int = 0
        self._cash: float = 0.0
        self._ce_held: float = 0.0
        self._pe_held: float = 0.0
        self._prev_ce_action: float = 0.0
        self._prev_pe_action: float = 0.0
        self._trade_log: List[OptionTradeRecord] = []
        self._portfolio_value_val: float = 0.0

        # Hold-until-Threshold state variables
        self._position_type: Optional[str] = None  # "CE", "PE", or None
        self._entry_premium: float = 0.0
        self._entry_strike: float = 0.0
        self._entry_step: int = 0
        self._cooldown_remaining: int = 0
        self._qty_held: float = 0.0
        self._previous_premium: float = 0.0

        logger.info(
            "OptionsTradingEnv created | asset={} T={} capital={:,.0f} timeframe={} profit_target={:.1%} stop_loss={:.1%}",
            asset_config.symbol,
            self.n_steps,
            self.initial_capital,
            self.timeframe,
            self.profit_target,
            self.stop_loss,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._current_step = 0
        self._cash = self.initial_capital
        self._ce_held = 0.0
        self._pe_held = 0.0
        self._prev_ce_action = 0.0
        self._prev_pe_action = 0.0
        self._portfolio_value_val = self.initial_capital
        self._trade_log = []

        # Reset Hold-until-Threshold states
        self._position_type = None
        self._entry_premium = 0.0
        self._entry_strike = 0.0
        self._entry_step = 0
        self._cooldown_remaining = 0
        self._qty_held = 0.0
        self._previous_premium = 0.0
        self._peak_premium = 0.0

        self.reward_calc.reset(self.initial_capital)

        return self._get_observation(), self._get_info()

    def step(
        self, action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        transaction_cost = 0.0
        
        # Map raw action to target Call (CE) and Put (PE) allocation fractions
        target_ce_frac, target_pe_frac = self._map_action(action)

        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Check if we are holding an active position
        if self._position_type is not None:
            # 1. Active position: calculate current premium
            steps_held = self._current_step - self._entry_step
            if steps_held == 1:
                if self._position_type == "CE":
                    current_premium = self.ce_premiums_next[self._entry_step]
                else:
                    current_premium = self.pe_premiums_next[self._entry_step]
            else:
                price_diff = self.prices[self._current_step] - self.prices[self._current_step - 1]
                if self._position_type == "CE":
                    current_premium = self._previous_premium + 0.5 * price_diff
                else:
                    current_premium = self._previous_premium - 0.5 * price_diff

            current_premium = max(0.05, current_premium)
            self._previous_premium = current_premium
            self._peak_premium = max(self._peak_premium, current_premium)

            # Update portfolio value
            self._portfolio_value_val = self._cash + self._qty_held * current_premium

            # Check exit conditions
            unrealized_pnl_pct = (current_premium - self._entry_premium) / self._entry_premium if self._entry_premium > 0 else 0.0
            time_held_bars = self._current_step - self._entry_step

            # Check if next step has a new expiry (expiry roll)
            is_expiry_roll = False
            if self._current_step < self.n_steps - 1:
                if self.expiries[self._current_step] != self.expiries[self._current_step + 1]:
                    is_expiry_roll = True

            # Check for EOD flat (Force close at 15:10 IST or later)
            current_time = self.dates[self._current_step]
            is_eod_flat = (current_time.hour == 15 and current_time.minute >= 10) or (current_time.hour > 15)

            # Check triggers
            should_exit = False
            exit_reason = ""

            # Academic PCS trailing stop-loss check
            is_tsl_hit = (self.trailing_sl > 0.0) and (current_premium < self._peak_premium * (1.0 - self.trailing_sl))
            
            # Academic PCS time-decay stop-loss check
            curr_sl = self.stop_loss
            if self.decay_sl:
                curr_sl = self.stop_loss * (1.0 - (time_held_bars / float(self.time_decay_bars)))
            is_sl_hit = (unrealized_pnl_pct <= -curr_sl)

            if unrealized_pnl_pct >= self.profit_target:
                should_exit = True
                exit_reason = "profit_target"
            elif is_sl_hit:
                should_exit = True
                exit_reason = "stop_loss"
            elif is_tsl_hit:
                should_exit = True
                exit_reason = "trailing_stop_loss"
            elif time_held_bars >= self.time_decay_bars:
                should_exit = True
                exit_reason = "time_decay"
            elif is_expiry_roll:
                should_exit = True
                exit_reason = "expiry_roll"
            elif is_eod_flat:
                should_exit = True
                exit_reason = "eod_flat"
            elif self._current_step >= self.n_steps - 1:
                should_exit = True
                exit_reason = "end_of_episode"

            if should_exit:
                # Close/liquidate position
                liquidated_val = self._qty_held * current_premium
                cost_sell = liquidated_val * self.sell_cost_frac
                transaction_cost += cost_sell
                self._cash += liquidated_val - cost_sell

                # Log trade
                if self._position_type == "CE":
                    self._log_trade(
                        self._prev_ce_action, 0.0, self._qty_held, 0.0,
                        current_premium, 0.0, cost_sell
                    )
                else:
                    self._log_trade(
                        0.0, self._prev_pe_action, 0.0, self._qty_held,
                        0.0, current_premium, cost_sell
                    )

                # Reset state variables
                self._position_type = None
                self._qty_held = 0.0
                self._ce_held = 0.0
                self._pe_held = 0.0
                self._cooldown_remaining = self.cooldown_bars
                self._portfolio_value_val = self._cash

                logger.debug("Closed options position at step {} due to {} | PnL: {:.2%}", 
                             self._current_step, exit_reason, unrealized_pnl_pct)

        else:
            # 2. No active position: look for entry signal
            # Check for EOD flat (prevent new entries at 15:10 IST or later)
            current_time = self.dates[self._current_step]
            is_eod_flat = (current_time.hour == 15 and current_time.minute >= 10) or (current_time.hour > 15)

            if self._cooldown_remaining == 0 and not is_eod_flat:
                has_signal = False
                is_call = False

                if target_ce_frac > 0.3 or target_pe_frac > 0.3:
                    has_signal = True
                    is_call = (target_ce_frac > target_pe_frac)

                if has_signal:
                    self._position_type = "CE" if is_call else "PE"
                    self._entry_premium = self.ce_premiums[self._current_step] if is_call else self.pe_premiums[self._current_step]
                    self._previous_premium = self._entry_premium
                    self._entry_strike = self.prices[self._current_step]
                    self._entry_step = self._current_step

                    # Allocate capital
                    max_investable = self._portfolio_value_val * self.max_position_pct * self.leverage
                    target_frac = target_ce_frac if is_call else target_pe_frac
                    target_val = target_frac * max_investable
                    cost_buy = target_val * self.buy_cost_frac
                    total_outflow = target_val + cost_buy

                    if total_outflow > self._cash:
                        scale = self._cash / max(total_outflow, 1e-8)
                        target_val *= scale

                    self._peak_premium = self._entry_premium
                    
                    # Calculate lot quantity
                    lot_size = self.asset_config.lot_size

                    if self.compounding:
                        lots = int(target_val / (self._entry_premium * lot_size)) if self._entry_premium > 0 else 0
                        lots = min(lots, self.max_lots)
                    else:
                        lots = self.max_lots
                        if lots * lot_size * self._entry_premium > self._cash:
                            lots = int(self._cash // (self._entry_premium * lot_size))

                    if lots > 0:
                        self._qty_held = lots * lot_size
                        if is_call:
                            self._ce_held = self._qty_held
                            self._pe_held = 0.0
                        else:
                            self._pe_held = self._qty_held
                            self._ce_held = 0.0

                        act_val = self._qty_held * self._entry_premium
                        cost_buy = act_val * self.buy_cost_frac
                        transaction_cost += cost_buy
                        self._cash -= (act_val + cost_buy)
                        self._portfolio_value_val = self._cash + act_val

                        # Update action tracking
                        self._prev_ce_action = target_ce_frac if is_call else 0.0
                        self._prev_pe_action = target_pe_frac if not is_call else 0.0

                        logger.debug("Opened {} position at step {} | strike: {} | entry premium: {}", 
                                     self._position_type, self._current_step, self._entry_strike, self._entry_premium)
                    else:
                        self._position_type = None

        # 3. Compute action delta and reward
        action_delta = abs(target_ce_frac - self._prev_ce_action) + abs(target_pe_frac - self._prev_pe_action)
        reward = self.reward_calc.compute(
            current_value=self._portfolio_value_val,
            transaction_cost=transaction_cost,
            action_delta=action_delta,
        )

        # Advance step
        self._current_step += 1

        terminated = self._current_step >= self.n_steps - 1
        truncated = False

        if self._portfolio_value_val < self.initial_capital * 0.01:
            terminated = True
            logger.warning("Options portfolio bankrupt at step {}!", self._current_step)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        idx = min(self._current_step, self.n_steps - 1)
        market_features = self.features[idx]
        
        pv_safe = max(self._portfolio_value_val, 1e-8)
        ce_premium = self.ce_premiums[idx]
        pe_premium = self.pe_premiums[idx]

        portfolio_state = np.array([
            self._cash / pv_safe,
            (self._ce_held * ce_premium) / pv_safe,
            (self._pe_held * pe_premium) / pv_safe,
            (self._portfolio_value_val - self.initial_capital) / self.initial_capital,
        ], dtype=np.float32)

        return np.concatenate([market_features, portfolio_state])

    def _get_info(self) -> Dict[str, Any]:
        idx = min(self._current_step, self.n_steps - 1)
        return {
            "step": self._current_step,
            "price": float(self.prices[idx]),
            "portfolio_value": self._portfolio_value_val,
            "cash": self._cash,
            "ce_held": self._ce_held,
            "pe_held": self._pe_held,
            "n_trades": len(self._trade_log),
        }

    def _log_trade(
        self, ce_action: float, pe_action: float, ce_qty: float, pe_qty: float,
        ce_prem: float, pe_prem: float, cost: float
    ) -> None:
        self._trade_log.append(OptionTradeRecord(
            step=self._current_step,
            ce_action=ce_action,
            pe_action=pe_action,
            ce_qty=ce_qty,
            pe_qty=pe_qty,
            ce_premium=ce_prem,
            pe_premium=pe_prem,
            transaction_cost=cost,
            cash_after=self._cash,
            portfolio_value=self._portfolio_value_val
        ))

    def _map_action(self, action: Any) -> Tuple[float, float]:
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════════
# Discrete Options Trading Environment (DQN)
# ═════════════════════════════════════════════════════════════════════════════

class DiscreteOptionsTradingEnv(_BaseOptionsTradingEnv):
    """Discrete option trading with 5 actions:
    0: 100% Call (CE)
    1: 50% Call (CE)
    2: 100% Cash / Flat
    3: 50% Put (PE)
    4: 100% Put (PE)
    """

    ACTION_MAP = {
        0: (1.0, 0.0),   # 100% Call
        1: (0.5, 0.0),   # 50% Call
        2: (0.0, 0.0),   # Cash
        3: (0.0, 0.5),   # 50% Put
        4: (0.0, 1.0),   # 100% Put
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(5)

    def _map_action(self, action: int) -> Tuple[float, float]:
        act = int(action)
        if act not in self.ACTION_MAP:
            raise ValueError(f"Invalid action {act}")
        return self.ACTION_MAP[act]


# ═════════════════════════════════════════════════════════════════════════════
# Continuous Options Trading Environment (PPO/DDPG/A2C)
# ═════════════════════════════════════════════════════════════════════════════

class ContinuousOptionsTradingEnv(_BaseOptionsTradingEnv):
    """Continuous option trading action space Box(-1, 1).
    * Action > 0: Allocates action fraction to Calls (CE).
    * Action < 0: Allocates abs(action) fraction to Puts (PE).
    * Action = 0: Cash / Flat.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

    def _map_action(self, action: np.ndarray) -> Tuple[float, float]:
        raw = float(np.clip(action, -1.0, 1.0).item() if hasattr(action, 'item') else np.clip(action, -1.0, 1.0).flat[0])
        if raw > 0:
            return raw, 0.0
        elif raw < 0:
            return 0.0, -raw
        else:
            return 0.0, 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Nifty Options Gymnasium Env.")
    parser.add_argument("--test", action="store_true", help="Run a test step simulation.")
    args = parser.parse_args()

    if args.test:
        logger.info("Running random step options environment test...")
        n = 100
        feat = np.random.randn(n, 12)
        prc = np.linspace(12000, 12500, n)
        
        ce_prem = np.linspace(200, 350, n)
        pe_prem = np.linspace(200, 100, n)
        
        # Settle premiums next day
        ce_prem_next = np.zeros(n)
        pe_prem_next = np.zeros(n)
        ce_prem_next[:-1] = ce_prem[1:]
        pe_prem_next[:-1] = pe_prem[1:]
        
        env = ContinuousOptionsTradingEnv(
            features=feat,
            prices=prc,
            ce_premiums=ce_prem,
            pe_premiums=pe_prem,
            ce_premiums_next=ce_prem_next,
            pe_premiums_next=pe_prem_next,
            initial_capital=50_00_000
        )
        
        obs, info = env.reset()
        print("Initial Observation shape:", obs.shape)
        print("Initial Info:", info)
        
        # Step
        obs, rew, done, trunc, info = env.step(np.array([0.8]))
        print("\nStep 1 (Action=0.8 - Buy Call):")
        print("Reward:", rew)
        print("Info:", info)
        
        obs, rew, done, trunc, info = env.step(np.array([-0.5]))
        print("\nStep 2 (Action=-0.5 - Buy Put):")
        print("Reward:", rew)
        print("Info:", info)
        
        logger.success("Environment test completed successfully!")
