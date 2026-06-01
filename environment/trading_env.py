"""
Dynamic ART-DRL — Trading Environments
========================================
Custom Gymnasium environments for single-asset commodity trading.

Two concrete classes:
    * DiscreteTradingEnv  — Discrete(5) action space for DQN.
    * ContinuousTradingEnv — Box(-1, 1) action space for PPO/DDPG/A2C.

Both wrap a shared ``_BaseTradingEnv`` that handles portfolio accounting,
transaction costs, reward computation, and trade logging.

Usage
-----
>>> from environment.trading_env import DiscreteTradingEnv
>>> import numpy as np
>>> features = np.random.randn(500, 16)   # T×F pre-computed
>>> prices   = np.random.uniform(100, 200, size=500)
>>> env = DiscreteTradingEnv(features=features, prices=prices)
>>> obs, info = env.reset()
>>> obs, reward, done, truncated, info = env.step(2)  # hold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from config import settings
from config.assets import AssetConfig, GLD_ALPACA
from environment.reward import RewardCalculator, RewardType


# ═════════════════════════════════════════════════════════════════════════════
# Trade Record
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Immutable log entry for a single trade execution."""
    step: int
    action_raw: float          # Raw action value (int for discrete, float for continuous)
    direction: str             # "buy" | "sell" | "hold"
    shares_traded: float       # Absolute number of shares traded
    trade_value: float         # Dollar value of the trade (before cost)
    transaction_cost: float    # Dollar cost incurred
    price: float               # Execution price
    cash_after: float          # Cash balance after trade
    shares_after: float        # Position after trade
    portfolio_value: float     # Total portfolio value after trade


# ═════════════════════════════════════════════════════════════════════════════
# Base Trading Environment (shared logic)
# ═════════════════════════════════════════════════════════════════════════════

class _BaseTradingEnv(gym.Env):
    """Abstract base for discrete / continuous trading environments.

    This class is **not** meant to be instantiated directly.  Subclasses
    must define ``action_space`` and implement ``_map_action``.

    Parameters
    ----------
    features : np.ndarray
        Pre-computed feature matrix of shape ``(T, n_features)``.
        Each row is one time-step's observation vector.
    prices : np.ndarray
        Price series of length ``T`` aligned with *features*.
    asset_config : AssetConfig
        Asset-specific parameters (transaction costs, capital, etc.).
    reward_type : RewardType
        Which reward formula to use.
    initial_capital : float | None
        Override starting cash (defaults to ``asset_config.initial_capital``).
    max_position_pct : float
        Maximum fraction of portfolio value investable.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        asset_config: AssetConfig = GLD_ALPACA,
        reward_type: RewardType = RewardType.RISK_SENSITIVE,
        initial_capital: Optional[float] = None,
        max_position_pct: float = settings.MAX_POSITION_PCT,
    ) -> None:
        super().__init__()

        # ── Validate inputs ─────────────────────────────────────────────
        assert features.ndim == 2, "features must be 2-D (T × n_features)"
        assert len(prices) == len(features), "prices and features must align"
        assert len(prices) >= 2, "Need at least 2 time-steps"

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.n_steps = len(prices)
        self.n_features = features.shape[1]

        # ── Asset / cost parameters ─────────────────────────────────────
        self.asset_config = asset_config
        self.buy_cost_frac = asset_config.buy_cost_bps / 10_000.0
        self.sell_cost_frac = asset_config.sell_cost_bps / 10_000.0
        self.initial_capital = initial_capital or asset_config.initial_capital
        self.max_position_pct = max_position_pct

        # ── Observation space ───────────────────────────────────────────
        # Features + 3 portfolio-state values (cash_ratio, position_ratio, unrealised_pnl_ratio)
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # ── Reward calculator ───────────────────────────────────────────
        self.reward_calc = RewardCalculator(reward_type=reward_type)

        # ── Episode state (set in reset) ────────────────────────────────
        self._current_step: int = 0
        self._cash: float = 0.0
        self._shares_held: float = 0.0
        self._prev_action: float = 0.0   # For Δa penalty
        self._trade_log: List[TradeRecord] = []

        logger.info(
            "TradingEnv created | asset={} T={} features={} capital={:,.0f}",
            asset_config.symbol,
            self.n_steps,
            self.n_features,
            self.initial_capital,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Core Gym interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to the beginning of the episode.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        super().reset(seed=seed)

        self._current_step = 0
        self._cash = self.initial_capital
        self._shares_held = 0.0
        self._prev_action = 0.0
        self._trade_log = []

        # Reset reward tracker
        self.reward_calc.reset(self.initial_capital)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one trading step.

        Parameters
        ----------
        action
            Discrete int or continuous float depending on subclass.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        # Map raw action to a target-position fraction in [-1, 1]
        target_position_frac = self._map_action(action)

        # Execute trade
        transaction_cost = self._execute_trade(target_position_frac)

        # Compute action delta for reward
        action_delta = abs(target_position_frac - self._prev_action)
        self._prev_action = target_position_frac

        # Portfolio value after trade
        portfolio_value = self._portfolio_value()

        # Reward
        reward = self.reward_calc.compute(
            current_value=portfolio_value,
            transaction_cost=transaction_cost,
            action_delta=action_delta,
        )

        # Advance time
        self._current_step += 1

        # Check termination
        terminated = self._current_step >= self.n_steps - 1
        truncated = False

        # Check for bankruptcy (portfolio value < 1% of initial)
        if portfolio_value < self.initial_capital * 0.01:
            terminated = True
            logger.warning(
                "Portfolio bankrupt at step {} | V={:.2f}",
                self._current_step,
                portfolio_value,
            )

        obs = self._get_observation()
        info = self._get_info()
        info["transaction_cost"] = transaction_cost
        info["reward_diagnostics"] = self.reward_calc.get_diagnostics()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Abstract: subclasses must implement
    # ──────────────────────────────────────────────────────────────────────

    def _map_action(self, action: Any) -> float:
        """Convert raw action to target position fraction in [-1, 1].

        * -1 → fully short / sell all
        *  0 → flat / hold
        * +1 → fully long / buy all
        """
        raise NotImplementedError

    # ──────────────────────────────────────────────────────────────────────
    # Trade Execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute_trade(self, target_position_frac: float) -> float:
        """Adjust position toward *target_position_frac* and return cost.

        The target fraction is expressed as a fraction of *max investable*
        capital (``max_position_pct * portfolio_value``).

        Returns
        -------
        float
            Total dollar transaction cost paid.
        """
        current_price = self.prices[self._current_step]
        portfolio_value = self._portfolio_value()
        max_investable = self.max_position_pct * portfolio_value

        # Target number of shares (fractional allowed for ETFs)
        # target_frac ∈ [-1, 1]:  +1 → buy up to max, -1 → sell all
        if target_position_frac > 0:
            target_shares = (target_position_frac * max_investable) / current_price
        elif target_position_frac < 0:
            # Negative fraction: sell shares proportionally
            target_shares = self._shares_held * (1.0 + target_position_frac)
            target_shares = max(target_shares, 0.0)
        else:
            # Hold — no change
            target_shares = self._shares_held

        shares_delta = target_shares - self._shares_held
        trade_value = abs(shares_delta) * current_price
        transaction_cost = 0.0

        # Minimum trade filter
        if trade_value < settings.MIN_TRADE_VALUE:
            # Too small — skip
            self._log_trade(0.0, "hold", 0.0, 0.0, 0.0, current_price)
            return 0.0

        if shares_delta > 0:
            # ── BUY ──
            cost = trade_value * self.buy_cost_frac
            total_outflow = trade_value + cost
            if total_outflow > self._cash:
                # Reduce to affordable amount
                affordable_value = self._cash / (1.0 + self.buy_cost_frac)
                shares_delta = affordable_value / current_price
                trade_value = shares_delta * current_price
                cost = trade_value * self.buy_cost_frac
                total_outflow = trade_value + cost

            self._cash -= total_outflow
            self._shares_held += shares_delta
            transaction_cost = cost
            self._log_trade(
                target_position_frac, "buy", shares_delta,
                trade_value, cost, current_price,
            )

        elif shares_delta < 0:
            # ── SELL ──
            shares_to_sell = min(abs(shares_delta), self._shares_held)
            trade_value = shares_to_sell * current_price
            cost = trade_value * self.sell_cost_frac
            self._cash += trade_value - cost
            self._shares_held -= shares_to_sell
            transaction_cost = cost
            self._log_trade(
                target_position_frac, "sell", shares_to_sell,
                trade_value, cost, current_price,
            )
        else:
            self._log_trade(
                target_position_frac, "hold", 0.0, 0.0, 0.0, current_price,
            )

        return transaction_cost

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _portfolio_value(self) -> float:
        """Cash + mark-to-market value of holdings."""
        price = self.prices[min(self._current_step, self.n_steps - 1)]
        return self._cash + self._shares_held * price

    def _unrealised_pnl(self) -> float:
        """Unrealised P&L relative to initial capital."""
        return self._portfolio_value() - self.initial_capital

    def _get_observation(self) -> np.ndarray:
        """Concatenate market features with normalised portfolio state."""
        idx = min(self._current_step, self.n_steps - 1)
        market_features = self.features[idx]

        portfolio_value = self._portfolio_value()
        pv_safe = max(portfolio_value, 1e-8)

        portfolio_state = np.array([
            self._cash / pv_safe,                    # cash ratio
            (self._shares_held * self.prices[idx]) / pv_safe,  # position ratio
            self._unrealised_pnl() / self.initial_capital,     # normalised PnL
        ], dtype=np.float32)

        return np.concatenate([market_features, portfolio_state])

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information dict."""
        idx = min(self._current_step, self.n_steps - 1)
        return {
            "step": self._current_step,
            "price": float(self.prices[idx]),
            "portfolio_value": self._portfolio_value(),
            "cash": self._cash,
            "shares_held": self._shares_held,
            "unrealised_pnl": self._unrealised_pnl(),
            "n_trades": len(self._trade_log),
        }

    def _log_trade(
        self,
        action_raw: float,
        direction: str,
        shares_traded: float,
        trade_value: float,
        transaction_cost: float,
        price: float,
    ) -> None:
        """Append a trade record to the log."""
        record = TradeRecord(
            step=self._current_step,
            action_raw=action_raw,
            direction=direction,
            shares_traded=shares_traded,
            trade_value=trade_value,
            transaction_cost=transaction_cost,
            price=price,
            cash_after=self._cash,
            shares_after=self._shares_held,
            portfolio_value=self._portfolio_value(),
        )
        self._trade_log.append(record)

    @property
    def trade_log(self) -> List[TradeRecord]:
        """Return the full trade log for this episode."""
        return list(self._trade_log)

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value."""
        return self._portfolio_value()


# ═════════════════════════════════════════════════════════════════════════════
# Discrete Trading Environment (for DQN)
# ═════════════════════════════════════════════════════════════════════════════

class DiscreteTradingEnv(_BaseTradingEnv):
    """Gymnasium environment with ``Discrete(5)`` action space.

    Action Mapping
    --------------
    0 → sell_all    (target_frac = -1.0)
    1 → sell_half   (target_frac = -0.5)
    2 → hold        (target_frac =  0.0)
    3 → buy_half    (target_frac = +0.5)
    4 → buy_all     (target_frac = +1.0)
    """

    # Class-level action map: action_int → target_position_fraction
    ACTION_MAP: Dict[int, float] = {
        0: -1.0,   # sell_all
        1: -0.5,   # sell_half
        2:  0.0,   # hold
        3:  0.5,   # buy_half
        4:  1.0,   # buy_all
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "sell_all",
        1: "sell_half",
        2: "hold",
        3: "buy_half",
        4: "buy_all",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(settings.DQN_DISCRETE_ACTIONS)
        logger.debug("DiscreteTradingEnv | actions={}", self.ACTION_NAMES)

    def _map_action(self, action: int) -> float:
        """Map discrete action int to target position fraction."""
        action_int = int(action)
        if action_int not in self.ACTION_MAP:
            raise ValueError(
                f"Invalid action {action_int}. Expected one of {list(self.ACTION_MAP.keys())}"
            )
        return self.ACTION_MAP[action_int]


# ═════════════════════════════════════════════════════════════════════════════
# Continuous Trading Environment (for PPO / DDPG / A2C)
# ═════════════════════════════════════════════════════════════════════════════

class ContinuousTradingEnv(_BaseTradingEnv):
    """Gymnasium environment with ``Box(-1, 1, shape=(1,))`` action space.

    Action Interpretation
    ---------------------
    * -1.0 → full sell (liquidate all shares)
    *  0.0 → hold (no trade)
    * +1.0 → full buy (invest up to max_position_pct)

    Values in between are interpolated linearly.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )
        logger.debug("ContinuousTradingEnv | action_space={}", self.action_space)

    def _map_action(self, action: np.ndarray) -> float:
        """Map continuous action array to target position fraction."""
        # SB3 passes shape (1,) array — extract scalar
        raw = float(np.clip(action, -1.0, 1.0).item() if hasattr(action, 'item') else np.clip(action, -1.0, 1.0).flat[0])
        return raw
