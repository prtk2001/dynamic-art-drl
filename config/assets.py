"""
Dynamic ART-DRL — Asset Configuration
======================================
Defines asset-specific parameters for each tradeable instrument.
Supports both Alpaca ETF proxies and Upstox MCX futures.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AssetConfig:
    """Configuration for a single tradeable asset."""
    name: str                         # Human-readable name
    symbol: str                       # Primary trading symbol
    exchange: str                     # Exchange (ALPACA / MCX)
    instrument_key: str               # Upstox instrument key (for MCX)
    asset_class: str                  # gold / crude_oil
    currency: str                     # USD / INR
    lot_size: int                     # Minimum lot size for futures
    tick_size: float                  # Minimum price movement
    initial_capital: float            # Starting capital for this asset
    buy_cost_bps: float = 5.0         # Transaction cost (buy) in bps
    sell_cost_bps: float = 15.0       # Transaction cost (sell) in bps
    kalman_process_noise: float = 0.5
    kalman_measurement_noise: float = 5.0
    leverage_factor: float = 1.0       # Position sizing leverage multiplier
    fast_window: int = 10              # SMA fast crossover window (daily basis)
    slow_window: int = 80              # SMA slow crossover window (daily basis)
    stop_loss_pct: float = 0.05        # Trailing stop-loss exit threshold
    dqn_exit: bool = True              # Flat out during extreme volatility
    timeframe_params: dict = field(default_factory=dict) # Timeframe specific parameters
    description: str = ""


# ── Alpaca ETF Proxies (US Market) ───────────────────────────────────────────

GLD_ALPACA = AssetConfig(
    name="Gold ETF (GLD)",
    symbol="GLD",
    exchange="ALPACA",
    instrument_key="GLD",
    asset_class="gold",
    currency="USD",
    lot_size=1,
    tick_size=0.01,
    initial_capital=100_000,
    buy_cost_bps=5.0,
    sell_cost_bps=15.0,
    kalman_process_noise=0.3,
    kalman_measurement_noise=3.0,
    leverage_factor=10.0,             # Default leverage
    fast_window=30,
    slow_window=50,
    stop_loss_pct=0.06,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 10.0, "fast": 30, "slow": 50, "stop": 0.06, "dqn_exit": False},
        "1h": {"leverage": 5.0, "fast": 420, "slow": 560, "stop": 0.08, "dqn_exit": False}
    },
    description="SPDR Gold Shares ETF — proxy for gold futures"
)

USO_ALPACA = AssetConfig(
    name="Crude Oil ETF (USO)",
    symbol="USO",
    exchange="ALPACA",
    instrument_key="USO",
    asset_class="crude_oil",
    currency="USD",
    lot_size=1,
    tick_size=0.01,
    initial_capital=100_000,
    buy_cost_bps=5.0,
    sell_cost_bps=15.0,
    kalman_process_noise=0.8,
    kalman_measurement_noise=8.0,
    leverage_factor=4.0,
    fast_window=20,
    slow_window=25,
    stop_loss_pct=0.02,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 4.0, "fast": 20, "slow": 25, "stop": 0.02, "dqn_exit": False},
        "1h": {"leverage": 2.0, "fast": 140, "slow": 175, "stop": 0.03, "dqn_exit": False}
    },
    description="United States Oil Fund ETF — proxy for crude oil futures"
)

# ── Upstox MCX Futures (Indian Market) — PRIMARY ────────────────────────────

GOLD_MCX = AssetConfig(
    name="MCX Gold Futures",
    symbol="GOLD",
    exchange="MCX",
    instrument_key="MCX_FO|GOLD",    # Resolved at runtime
    asset_class="gold",
    currency="INR",
    lot_size=1,                       # 1 lot = 100 grams
    tick_size=1.0,                    # ₹1 per gram
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=3.0,                 # MCX brokerage is typically lower
    sell_cost_bps=3.0,
    kalman_process_noise=0.3,
    kalman_measurement_noise=3.0,
    leverage_factor=10.0,
    fast_window=30,
    slow_window=50,
    stop_loss_pct=0.06,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 10.0, "fast": 30, "slow": 50, "stop": 0.06, "dqn_exit": False},
        "1h": {"leverage": 4.0, "fast": 420, "slow": 560, "stop": 0.08, "dqn_exit": False}
    },
    description="MCX Gold futures — near-month active contract"
)

CRUDE_MCX = AssetConfig(
    name="MCX Crude Oil Futures",
    symbol="CRUDEOIL",
    exchange="MCX",
    instrument_key="MCX_FO|CRUDEOIL",  # Resolved at runtime
    asset_class="crude_oil",
    currency="INR",
    lot_size=1,                         # 1 lot = 100 barrels
    tick_size=1.0,                      # ₹1 per barrel
    initial_capital=50_00_000,          # ₹50 Lakhs
    buy_cost_bps=3.0,
    sell_cost_bps=3.0,
    kalman_process_noise=0.8,
    kalman_measurement_noise=8.0,
    leverage_factor=3.0,
    fast_window=20,
    slow_window=25,
    stop_loss_pct=0.02,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 3.0, "fast": 20, "slow": 25, "stop": 0.02, "dqn_exit": False},
        "1h": {"leverage": 1.5, "fast": 140, "slow": 175, "stop": 0.03, "dqn_exit": False}
    },
    description="MCX Crude Oil futures — near-month active contract"
)



# ── Asset Registry ──────────────────────────────────────────────────────────

ASSET_REGISTRY = {
    # Alpaca (US)
    "GLD": GLD_ALPACA,
    "USO": USO_ALPACA,
    # Upstox (MCX India) — PRIMARY
    "GOLD_MCX": GOLD_MCX,
    "CRUDE_MCX": CRUDE_MCX,
}

# Default assets for backtesting
BACKTEST_ASSETS_ALPACA = ["GLD", "USO"]
BACKTEST_ASSETS_MCX = ["GOLD_MCX", "CRUDE_MCX"]

# All timeframe + asset combinations to test
BACKTEST_COMBOS = [
    ("GLD", "1d"), ("GLD", "1h"),
    ("USO", "1d"), ("USO", "1h"),
    ("GOLD_MCX", "1d"), ("GOLD_MCX", "1h"),
    ("CRUDE_MCX", "1d"), ("CRUDE_MCX", "1h"),
]


def get_asset(name: str) -> AssetConfig:
    """Retrieve asset config by name."""
    if name not in ASSET_REGISTRY:
        raise ValueError(
            f"Unknown asset '{name}'. Available: {list(ASSET_REGISTRY.keys())}"
        )
    return ASSET_REGISTRY[name]
