"""
Dynamic ART-DRL — Backtesting Engine
====================================
Performs rolling walk-forward backtesting for the Dynamic ART-DRL switching model
and benchmarks it against individual static agents and baseline Buy-and-Hold.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from config.assets import AssetConfig
from router.dynamic_router import DynamicRouter
from environment.trading_env import ContinuousTradingEnv, DiscreteTradingEnv
from environment.reward import RewardCalculator
from backtest.metrics import calculate_performance_metrics


class BacktestEngine:
    """Walk-Forward Backtesting Engine for the Dynamic ART-DRL trading system."""
    
    def __init__(
        self,
        asset_config: AssetConfig,
        df_features: pd.DataFrame,
        timeframe: str = "1d",
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 63
    ) -> None:
        """
        Parameters
        ----------
        asset_config : AssetConfig
            Dataclass asset specifications.
        df_features : pd.DataFrame
            Preprocessed features DataFrame containing prices and engineered features.
        timeframe : str
            Active trading timeframe ('1d', '1h').
        train_window : int
            Rolling training window in days.
        test_window : int
            Rolling out-of-sample test window in days.
        step_size : int
            Step size to advance the rolling windows.
        """
        self.asset_config = asset_config
        self.df_features = df_features
        self.timeframe = timeframe
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
        # Load timeframe specific parameters
        tf_params = getattr(asset_config, "timeframe_params", {}).get(timeframe, {})
        self.leverage = tf_params.get("leverage", asset_config.leverage_factor)
        self.fast_window = tf_params.get("fast", asset_config.fast_window)
        self.slow_window = tf_params.get("slow", asset_config.slow_window)
        self.stop_loss_pct = tf_params.get("stop", asset_config.stop_loss_pct)
        self.dqn_exit = tf_params.get("dqn_exit", asset_config.dqn_exit)
        
        # Create standard switching router
        self.router = DynamicRouter()
        
        # Trailing stop loss state tracking
        self._in_position = False
        self._peak_price = 0.0
        
        logger.info(
            "BacktestEngine initialised for {} ({}): data_rows={}, train_window={}, test_window={}, step_size={}, leverage={}, fast_window={}, slow_window={}",
            asset_config.name, timeframe, len(df_features), train_window, test_window, step_size, self.leverage, self.fast_window, self.slow_window
        )
        
    def generate_walk_forward_splits(self) -> List[Tuple[int, int, int, int]]:
        """Generate rolling start/end index boundaries for train and test splits.
        
        Returns
        -------
        splits : list of tuples
            Each tuple is (train_start, train_end, test_start, test_end) index.
        """
        splits = []
        n_samples = len(self.df_features)
        
        train_start = 0
        while True:
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            if test_end > n_samples:
                # Last remaining slice, if large enough
                if n_samples - test_start >= 10:
                    splits.append((train_start, train_end, test_start, n_samples))
                break
                
            splits.append((train_start, train_end, test_start, test_end))
            train_start += self.step_size
            
        return splits
        
    def run_regime_backtest(self) -> Dict[str, Any]:
        """Run Method 3 Dynamic Switching Router over the complete out-of-sample period.
        
        Returns
        -------
        results : dict
            Contains out-of-sample portfolio equity curve, trades log, and regime tracking.
        """
        splits = self.generate_walk_forward_splits()
        if not splits:
            raise ValueError("Data insufficient for walk-forward splits.")
            
        # We start out-of-sample testing from the first split's test_start
        first_test_idx = splits[0][2]
        n_samples = len(self.df_features)
        prices = self.df_features["close"].values
        
        if "_OPT" in self.asset_config.symbol:
            # Walk-forward options backtesting using Gymnasium Environment (Option A)
            from environment.options_trading_env import ContinuousOptionsTradingEnv
            
            slice_df = self.df_features.iloc[first_test_idx:]
            feature_cols = ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_width', 'cci_30', 'dx_30', 'atr_14', 'return_simple', 'return_log', 'vol_rolling_30', 'vol_ewma']
            
            env = ContinuousOptionsTradingEnv(
                features=slice_df[feature_cols].values,
                prices=slice_df["close"].values,
                ce_premiums=slice_df["ce_close"].values,
                pe_premiums=slice_df["pe_close"].values,
                ce_premiums_next=slice_df["ce_close_next"].values,
                pe_premiums_next=slice_df["pe_close_next"].values,
                asset_config=self.asset_config,
                initial_capital=self.asset_config.initial_capital,
                max_position_pct=0.15,
                dates=slice_df.index,
                expiries=slice_df["expiry"].values,
                timeframe=self.timeframe
            )
            
            env.reset()
            
            portfolio_values = np.zeros(n_samples)
            portfolio_values[:first_test_idx] = self.asset_config.initial_capital
            
            active_agents = []
            vol_regimes = []
            trade_log = []
            
            logger.info("Executing walk-forward options backtest using Gymnasium Environment...")
            
            for idx in range(first_test_idx, n_samples):
                sub_prices = prices[:idx]
                agent_name, regime, vol_met = self.router.route(sub_prices)
                
                active_agents.append(agent_name)
                vol_regimes.append(regime.value)
                
                # Predict action
                action = self._mock_agent_prediction_options(agent_name, idx)
                
                # Step env (requires array input)
                obs, reward, done, truncated, info = env.step(np.array([action]))
                
                portfolio_values[idx] = env._portfolio_value_val
                
                if done:
                    # Fill the rest with last portfolio value
                    portfolio_values[idx:] = env._portfolio_value_val
                    remaining = n_samples - 1 - idx
                    active_agents.extend([agent_name] * remaining)
                    vol_regimes.extend([regime.value] * remaining)
                    break
                    
            # Map env trade log to backtest format
            for rec in env._trade_log:
                step_idx = first_test_idx + rec.step
                if step_idx >= n_samples:
                    continue
                action_type = "FLAT"
                if rec.ce_action > 0:
                    action_type = "BUY_CALL"
                elif rec.pe_action > 0:
                    action_type = "BUY_PUT"
                    
                trade_log.append({
                    "step": step_idx,
                    "date": self.df_features.index[step_idx] if isinstance(self.df_features.index, pd.DatetimeIndex) else step_idx,
                    "symbol": self.asset_config.symbol,
                    "action": action_type,
                    "qty": rec.ce_qty if rec.ce_action > 0 else (rec.pe_qty if rec.pe_action > 0 else 0.0),
                    "price": prices[step_idx],
                    "cost": rec.transaction_cost,
                    "avg_entry_price": rec.ce_premium if rec.ce_action > 0 else (rec.pe_premium if rec.pe_action > 0 else 0.0),
                    "realized_gross_pnl": 0.0,
                    "realized_net_pnl": 0.0,
                    "position_size": rec.ce_qty if rec.ce_action > 0 else (rec.pe_qty if rec.pe_action > 0 else 0.0),
                    "active_agent": active_agents[rec.step] if rec.step < len(active_agents) else "UNKNOWN",
                    "regime": vol_regimes[rec.step] if rec.step < len(vol_regimes) else "UNKNOWN"
                })
                
        else:
            # Initialize tracking arrays for standard assets
            portfolio_values = np.zeros(n_samples)
            portfolio_values[:first_test_idx] = self.asset_config.initial_capital
            
            # Hold state
            cash = self.asset_config.initial_capital
            shares = 0.0
            self._prev_action = 0.0
            
            # Realized PnL & Position average cost tracking
            avg_entry_price = 0.0
            current_shares = 0.0
            
            active_agents = []
            vol_regimes = []
            trade_log = []
            
            # Dynamic switching backtest
            logger.info("Executing walk-forward dynamic switching backtest...")
        
            for idx in range(first_test_idx, n_samples):
                # 1. Volatility Regime switching decision
                # Feed prices up to current tick
                sub_prices = prices[:idx]
                agent_name, regime, vol_met = self.router.route(sub_prices)
                
                active_agents.append(agent_name)
                vol_regimes.append(regime.value)
                
                # 2. Agent policy action prediction
                # In a full run, we load the trained checkpoint for the active agent
                # corresponding to the current rolling train window.
                # Here we mock the action based on the active agent's policy profile
                # to make the backtest runner completely self-contained and runnable:
                # - DQN: Discrete allocation anchor, acts cautiously during extreme spikes
                # - PPO/DDPG: Captures trends smoothly
                # - A2C: Adaptive fast transitions
                action = self._mock_agent_prediction(agent_name, idx)
                
                # 3. Step portfolio simulator with transaction costs
                curr_price = prices[idx]
                portfolio_value = cash + shares * curr_price
                
                # Avoid churning: if action has not changed, keep the exact same number of shares
                if action == self._prev_action:
                    target_shares = shares
                else:
                    leverage = self.leverage
                    target_alloc_value = portfolio_value * action * leverage
                    target_shares = target_alloc_value / curr_price
                    
                    # Handle lot size rounding for futures if lot_size > 1
                    if self.asset_config.lot_size > 1:
                        target_lots = int(target_alloc_value / (curr_price * self.asset_config.lot_size))
                        target_shares = target_lots * self.asset_config.lot_size
                        
                share_diff = target_shares - shares
                trade_value = abs(share_diff) * curr_price
                
                if share_diff != 0 and action != self._prev_action:
                    cost_bps = self.asset_config.buy_cost_bps if share_diff > 0 else self.asset_config.sell_cost_bps
                    cost = trade_value * (cost_bps / 10000.0)
                    
                    # Calculate Realized P&L
                    realized_gross = 0.0
                    realized_net = 0.0
                    
                    if share_diff > 0:
                        # BUY trade: opens or increases position
                        new_shares = current_shares + share_diff
                        avg_entry_price = (current_shares * avg_entry_price + share_diff * curr_price) / new_shares
                        current_shares = new_shares
                    else:
                        # SELL trade: closes or reduces position
                        qty_sold = abs(share_diff)
                        # Use the minimum of current shares or qty_sold to avoid underflow
                        shares_to_calc = min(current_shares, qty_sold)
                        realized_gross = shares_to_calc * (curr_price - avg_entry_price)
                        realized_net = realized_gross - cost
                        current_shares = max(current_shares - qty_sold, 0.0)
                        if current_shares < 1e-5:
                            avg_entry_price = 0.0
                            current_shares = 0.0
                    
                    # Update Cash (cash can go negative representing margin borrowing)
                    cash -= share_diff * curr_price + cost
                    shares = target_shares
                    self._prev_action = action
                    
                    trade_log.append({
                        "step": idx,
                        "date": self.df_features.index[idx] if isinstance(self.df_features.index, pd.DatetimeIndex) else idx,
                        "symbol": self.asset_config.symbol,
                        "action": "BUY" if share_diff > 0 else "SELL",
                        "qty": abs(share_diff),
                        "price": curr_price,
                        "cost": cost,
                        "avg_entry_price": avg_entry_price,
                        "realized_gross_pnl": realized_gross,
                        "realized_net_pnl": realized_net,
                        "position_size": current_shares,
                        "active_agent": agent_name,
                        "regime": regime.value
                    })
                else:
                    # Do not execute trade, keep current shares, cash remains unchanged
                    pass
                    
                # Compute portfolio value
                portfolio_values[idx] = cash + shares * curr_price
                
                # Enforce Bankruptcy Liquidation (Real-world scenario)
                if portfolio_values[idx] <= 0:
                    logger.error(
                        "BANKRUPTCY ALERT: Portfolio went bankrupt at step {} ({}) due to extreme leverage drawdowns! All assets liquidated to 0.",
                        idx,
                        self.df_features.index[idx] if isinstance(self.df_features.index, pd.DatetimeIndex) else idx
                    )
                    portfolio_values[idx:] = 0.0
                    cash = 0.0
                    shares = 0.0
                    # Pad active_agents and vol_regimes to match length
                    remaining_steps = (n_samples - first_test_idx) - len(active_agents)
                    active_agents.extend(["LIQUIDATED"] * remaining_steps)
                    vol_regimes.extend(["EXTREME_VOLATILITY"] * remaining_steps)
                    break
            
        # Compile results
        df_results = pd.DataFrame(index=self.df_features.index[first_test_idx:])
        df_results["portfolio_value"] = portfolio_values[first_test_idx:]
        df_results["active_agent"] = active_agents
        df_results["regime"] = vol_regimes
        df_results["price"] = prices[first_test_idx:]
        
        # Calculate daily returns
        df_results["returns"] = df_results["portfolio_value"].pct_change().fillna(0.0)
        
        # Compute baseline Buy-and-Hold
        initial_price = prices[first_test_idx]
        initial_qty = self.asset_config.initial_capital / initial_price
        if self.asset_config.exchange == "MCX":
            initial_lots = int(self.asset_config.initial_capital / (initial_price * self.asset_config.lot_size))
            initial_qty = initial_lots * self.asset_config.lot_size
            
        bh_cash = self.asset_config.initial_capital - (initial_qty * initial_price)
        df_results["buy_and_hold"] = bh_cash + initial_qty * df_results["price"]
        df_results["bh_returns"] = df_results["buy_and_hold"].pct_change().fillna(0.0)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(df_results["returns"])
        bh_metrics = calculate_performance_metrics(df_results["bh_returns"])
        
        logger.success("Dynamic switching backtest finished.")
        
        return {
            "df_results": df_results,
            "trade_log": trade_log,
            "metrics": metrics,
            "bh_metrics": bh_metrics
        }
        
    def _mock_agent_prediction(self, agent_name: str, idx: int) -> float:
        """Simulate agent policy allocation decision using double-smoothed Kalman-price SMA crossovers."""
        curr_price = self.df_features["close"].values[idx]
        
        fast_win = self.fast_window
        slow_win = self.slow_window
        
        if idx < slow_win:
            return 0.0
            
        # 1. Fetch preprocessed Kalman features
        kalman_price_series = self.df_features["kalman_price"].values
        sma_fast = np.mean(kalman_price_series[idx - fast_win : idx])
        sma_slow = np.mean(kalman_price_series[idx - slow_win : idx])
        
        # 2. Long-Only trend following crossover signal
        sig = 1.0 if sma_fast > sma_slow else 0.0
        
        # 3. Volatility regime defense: Stay 100% flat during extreme panic to preserve capital (if enabled)
        if agent_name == "DQN" and self.dqn_exit:
            sig = 0.0
            
        # 4. Trailing stop-loss: cut losses quickly under high leverage
        stop_loss_pct = self.stop_loss_pct
        
        if self._in_position:
            self._peak_price = max(self._peak_price, curr_price)
            drawdown = (self._peak_price - curr_price) / self._peak_price
            if drawdown > stop_loss_pct:
                sig = 0.0 # Trigger trailing stop-loss exit
                
        # Update tracking state
        if sig > 0.0:
            if not self._in_position:
                self._in_position = True
                self._peak_price = curr_price
        else:
            self._in_position = False
            self._peak_price = 0.0
            
        return sig

    def _mock_agent_prediction_options(self, agent_name: str, idx: int) -> float:
        """Simulate options agent policy allocation decision."""
        curr_price = self.df_features["close"].values[idx]
        
        fast_win = self.fast_window
        slow_win = self.slow_window
        
        if idx < slow_win:
            return 0.0
            
        # 1. Fetch preprocessed Kalman features
        kalman_price_series = self.df_features["kalman_price"].values
        sma_fast = np.mean(kalman_price_series[idx - fast_win : idx])
        sma_slow = np.mean(kalman_price_series[idx - slow_win : idx])
        
        # 2. Long signal -> +1.0 (Buy Call), Short signal -> -1.0 (Buy Put)
        sig = 1.0 if sma_fast > sma_slow else -1.0
        
        # 3. Volatility regime defense
        if agent_name == "DQN" and self.dqn_exit:
            sig = 0.0
            
        # 4. Trailing stop-loss
        stop_loss_pct = self.stop_loss_pct
        if self._in_position:
            self._peak_price = max(self._peak_price, curr_price)
            # Spot price drawdown is cleaner
            drawdown = abs(self._peak_price - curr_price) / self._peak_price
            if drawdown > stop_loss_pct:
                sig = 0.0
                
        # Update tracking state
        if sig != 0.0:
            if not self._in_position:
                self._in_position = True
                self._peak_price = curr_price
        else:
            self._in_position = False
            self._peak_price = 0.0
            
        return sig

