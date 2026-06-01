"""
Dynamic ART-DRL — Quantitative Performance Metrics
===================================================
Calculates standard risk-adjusted returns metrics (Sharpe, Sortino, Calmar, etc.)
for backtesting evaluation (PRUDEX-Compass specifications).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_performance_metrics(
    daily_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.05
) -> Dict[str, float]:
    """Calculate key portfolio performance and risk metrics.
    
    Parameters
    ----------
    daily_returns : pd.Series or np.ndarray
        Array/series of daily portfolio returns.
    risk_free_rate : float
        Annualized risk-free rate (default 5% / 0.05).
        
    Returns
    -------
    metrics : dict
        Calculated key metrics.
    """
    returns = pd.Series(daily_returns).fillna(0.0)
    
    if len(returns) == 0:
        return {
            "cumulative_return": 0.0,
            "annualised_return": 0.0,
            "annualised_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0
        }
        
    # 1. Cumulative Return
    cumulative = np.prod(1.0 + returns) - 1.0
    
    # 2. Annualised Return
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
        years = float((returns.index[-1] - returns.index[0]).total_seconds() / (365.25 * 86400.0))
        years = max(years, 1.0 / 252.0)
    else:
        n_days = len(returns)
        years = n_days / 252.0 if n_days > 0 else 1.0
        
    annualised_return = (cumulative + 1.0) ** (1.0 / years) - 1.0 if cumulative > -1.0 else -1.0
    
    # 3. Annualised Volatility
    vol = returns.std() * np.sqrt(252)
    
    # 4. Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    
    # 5. Downside deviation & Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (excess_returns.mean() / downside_returns.std() * np.sqrt(252)) if downside_returns.std() > 0 else 0.0
    
    # 6. Max Drawdown
    equity_curve = np.cumprod(1.0 + returns)
    cum_max = np.maximum.accumulate(equity_curve)
    drawdowns = (cum_max - equity_curve) / cum_max
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0
    
    # 7. Calmar Ratio
    calmar = (annualised_return / max_dd) if max_dd > 0 else 0.0
    
    return {
        "cumulative_return": float(cumulative),
        "annualised_return": float(annualised_return),
        "annualised_volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(-max_dd),  # negative to represent loss
        "calmar_ratio": float(calmar)
    }
