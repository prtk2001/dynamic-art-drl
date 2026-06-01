"""
Dynamic ART-DRL — Reward Functions
====================================
Implements risk-sensitive and alternative reward functions for the
trading environment.

Reward Variants:
    1. Risk-Sensitive (default):
       r_t = ΔV_t - λ₁·DD_t - λ₂·|Δa_t|·c + λ₃·SR_rolling

    2. Simple PnL:
       r_t = V_t - V_{t-1}

    3. Sortino-Based:
       r_t = ΔV_t + λ₃ · Sortino_rolling

Parameters are loaded from config.settings.
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger

from config import settings


class RewardType(Enum):
    """Supported reward function variants."""
    RISK_SENSITIVE = "risk_sensitive"
    SIMPLE_PNL = "simple_pnl"
    SORTINO = "sortino"


class RewardCalculator:
    """Computes step-wise rewards for the trading environment.

    The calculator maintains internal state (peak portfolio value,
    recent return history) and must be reset between episodes.

    Parameters
    ----------
    reward_type : RewardType
        Which reward formula to use.
    lambda_drawdown : float
        λ₁ — penalty weight on current drawdown fraction.
    lambda_transaction : float
        λ₂ — penalty weight on transaction-cost component.
    lambda_sharpe : float
        λ₃ — bonus weight on rolling Sharpe / Sortino ratio.
    rolling_window : int
        Number of past steps used for rolling Sharpe / Sortino.
    risk_free_rate : float
        Annualised risk-free rate (converted to per-step for Sharpe).
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.RISK_SENSITIVE,
        lambda_drawdown: float = settings.REWARD_DRAWDOWN_PENALTY,
        lambda_transaction: float = settings.REWARD_TRANSACTION_PENALTY,
        lambda_sharpe: float = settings.REWARD_SHARPE_BONUS,
        rolling_window: int = settings.REWARD_ROLLING_WINDOW,
        risk_free_rate: float = settings.RISK_FREE_RATE,
    ) -> None:
        self.reward_type = reward_type
        self.lambda_dd = lambda_drawdown
        self.lambda_tx = lambda_transaction
        self.lambda_sr = lambda_sharpe
        self.rolling_window = rolling_window
        self.risk_free_rate = risk_free_rate

        # ── Internal state (reset per episode) ──
        self._peak_value: float = 0.0
        self._prev_value: float = 0.0
        self._return_history: deque[float] = deque(maxlen=rolling_window)
        self._step_count: int = 0

        logger.debug(
            "RewardCalculator initialised | type={} λ₁={} λ₂={} λ₃={} window={}",
            reward_type.value,
            lambda_drawdown,
            lambda_transaction,
            lambda_sharpe,
            rolling_window,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, initial_portfolio_value: float) -> None:
        """Reset internal state at the start of a new episode.

        Parameters
        ----------
        initial_portfolio_value : float
            Starting portfolio value (cash + holdings).
        """
        self._peak_value = initial_portfolio_value
        self._prev_value = initial_portfolio_value
        self._return_history.clear()
        self._step_count = 0
        logger.trace("RewardCalculator reset | V₀={:.2f}", initial_portfolio_value)

    def compute(
        self,
        current_value: float,
        transaction_cost: float,
        action_delta: float,
    ) -> float:
        """Compute the reward for the current step.

        Parameters
        ----------
        current_value : float
            Portfolio value *after* executing the trade.
        transaction_cost : float
            Dollar cost of the trade (>= 0).
        action_delta : float
            Absolute change in position expressed as a fraction [0, 1].
            For discrete envs this is mapped from the action;
            for continuous envs it is ``|a_t - a_{t-1}|``.

        Returns
        -------
        float
            Scalar reward.
        """
        self._step_count += 1

        # ΔV_t (dollar change in portfolio value)
        delta_v = current_value - self._prev_value

        # Track peak for drawdown
        self._peak_value = max(self._peak_value, current_value)

        # Store step return for rolling statistics
        step_return = delta_v / max(self._prev_value, 1e-8)
        self._return_history.append(step_return)

        # Dispatch to the appropriate formula
        if self.reward_type == RewardType.RISK_SENSITIVE:
            reward = self._risk_sensitive(
                delta_v, current_value, transaction_cost, action_delta,
            )
        elif self.reward_type == RewardType.SIMPLE_PNL:
            reward = self._simple_pnl(delta_v)
        elif self.reward_type == RewardType.SORTINO:
            reward = self._sortino_reward(delta_v)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        # Advance
        self._prev_value = current_value
        return reward

    # ──────────────────────────────────────────────────────────────────────
    # Reward Formulae (private)
    # ──────────────────────────────────────────────────────────────────────

    def _risk_sensitive(
        self,
        delta_v: float,
        current_value: float,
        transaction_cost: float,
        action_delta: float,
    ) -> float:
        """r_t = ΔV_t − λ₁·DD_t − λ₂·|Δa_t|·c + λ₃·SR_rolling

        Components:
            * ΔV_t        — raw P&L change (reward for profit).
            * DD_t         — current drawdown fraction (penalty for risk).
            * |Δa_t|·c     — position-change × cost (penalty for churning).
            * SR_rolling   — rolling Sharpe ratio (bonus for consistency).
        """
        dd = self._drawdown_fraction(current_value)
        sr = self._rolling_sharpe()

        reward = (
            delta_v
            - self.lambda_dd * dd
            - self.lambda_tx * action_delta * transaction_cost
            + self.lambda_sr * sr
        )
        return reward

    def _simple_pnl(self, delta_v: float) -> float:
        """r_t = V_t − V_{t-1} (raw dollar P&L)."""
        return delta_v

    def _sortino_reward(self, delta_v: float) -> float:
        """r_t = ΔV_t + λ₃ · Sortino_rolling

        Like risk-sensitive but replaces Sharpe with Sortino and
        drops the explicit drawdown / transaction penalties.
        """
        sortino = self._rolling_sortino()
        return delta_v + self.lambda_sr * sortino

    # ──────────────────────────────────────────────────────────────────────
    # Metrics helpers
    # ──────────────────────────────────────────────────────────────────────

    def _drawdown_fraction(self, current_value: float) -> float:
        """Compute the fractional drawdown from peak.

        DD_t = (V_peak − V_t) / V_peak   ∈ [0, 1]
        """
        if self._peak_value <= 0:
            return 0.0
        dd = (self._peak_value - current_value) / self._peak_value
        return max(dd, 0.0)

    def _rolling_sharpe(self) -> float:
        """Compute the rolling Sharpe ratio over the recent window.

        SR = (μ_r − r_f_step) / σ_r

        Returns 0.0 if fewer than 2 observations are available.
        """
        if len(self._return_history) < 2:
            return 0.0

        returns = np.array(self._return_history, dtype=np.float64)
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))

        if std_r < 1e-10:
            return 0.0

        # Convert annual risk-free to per-step (assuming ~252 trading days)
        rf_step = self.risk_free_rate / 252.0
        sharpe = (mean_r - rf_step) / std_r
        return sharpe

    def _rolling_sortino(self) -> float:
        """Compute the rolling Sortino ratio (penalises only downside vol).

        Sortino = (μ_r − r_f_step) / σ_downside

        Returns 0.0 if fewer than 2 observations or no downside deviation.
        """
        if len(self._return_history) < 2:
            return 0.0

        returns = np.array(self._return_history, dtype=np.float64)
        rf_step = self.risk_free_rate / 252.0
        mean_r = float(np.mean(returns))

        # Downside deviation: σ of returns below the risk-free rate
        downside = returns[returns < rf_step] - rf_step
        if len(downside) < 1:
            return 0.0

        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if downside_std < 1e-10:
            return 0.0

        sortino = (mean_r - rf_step) / downside_std
        return sortino

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    @property
    def current_drawdown(self) -> float:
        """Current drawdown fraction from peak."""
        return self._drawdown_fraction(self._prev_value)

    @property
    def peak_value(self) -> float:
        """Highest portfolio value observed so far in this episode."""
        return self._peak_value

    @property
    def step_count(self) -> int:
        """Number of steps computed since last reset."""
        return self._step_count

    def get_diagnostics(self) -> dict:
        """Return a dictionary of current diagnostic metrics.

        Useful for logging / TensorBoard callbacks.
        """
        return {
            "reward/peak_value": self._peak_value,
            "reward/drawdown": self.current_drawdown,
            "reward/rolling_sharpe": self._rolling_sharpe(),
            "reward/rolling_sortino": self._rolling_sortino(),
            "reward/step_count": self._step_count,
            "reward/return_buffer_len": len(self._return_history),
        }

    def __repr__(self) -> str:
        return (
            f"RewardCalculator(type={self.reward_type.value}, "
            f"λ_dd={self.lambda_dd}, λ_tx={self.lambda_tx}, "
            f"λ_sr={self.lambda_sr}, window={self.rolling_window})"
        )
