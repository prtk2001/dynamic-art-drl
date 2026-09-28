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

# ── Nifty 50 Index Derivatives (Indian Market) — Version 2 ────────────────────

NIFTY_FUT = AssetConfig(
    name="Nifty 50 Futures",
    symbol="NIFTY",
    exchange="NSE_FO",
    instrument_key="NSE_FO|NIFTY",    # Resolved at runtime
    asset_class="equity_index",
    currency="INR",
    lot_size=65,                       # Nifty lot size (updated 2026)
    tick_size=0.05,                    # Tick size is 0.05
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=2.0,                  # NSE transaction charges are low
    sell_cost_bps=2.0,
    kalman_process_noise=0.5,
    kalman_measurement_noise=5.0,
    leverage_factor=5.0,
    fast_window=10,
    slow_window=80,
    stop_loss_pct=0.03,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 5.0, "fast": 10, "slow": 80, "stop": 0.03, "dqn_exit": False},
        "1h": {"leverage": 2.5, "fast": 70, "slow": 560, "stop": 0.04, "dqn_exit": False},
        "5min": {"leverage": 2.0, "fast": 60, "slow": 480, "stop": 0.005, "dqn_exit": False},
        "15min": {"leverage": 2.5, "fast": 20, "slow": 160, "stop": 0.008, "dqn_exit": False}
    },
    description="Nifty 50 Index Futures — continuous near-month contract"
)

NIFTY_OPT = AssetConfig(
    name="Nifty 50 ATM Options",
    symbol="NIFTY_OPT",
    exchange="NSE_FO",
    instrument_key="NSE_FO|NIFTY_OPT", # Resolved at runtime
    asset_class="equity_index_option",
    currency="INR",
    lot_size=65,                       # Lot size (updated 2026)
    tick_size=0.05,
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    kalman_process_noise=0.5,
    kalman_measurement_noise=5.0,
    leverage_factor=1.0,               # Options have high inherent leverage
    fast_window=10,
    slow_window=80,
    stop_loss_pct=0.15,                # Options have higher volatility
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 1.0, "fast": 10, "slow": 80, "stop": 0.15, "dqn_exit": False},
        "1h": {"leverage": 1.0, "fast": 70, "slow": 560, "stop": 0.20, "dqn_exit": False},
        "5min": {"leverage": 1.0, "fast": 60, "slow": 480, "stop": 0.03, "dqn_exit": False,
                 "profit_target": 0.05, "time_decay_bars": 8, "cooldown": 8,
                 "trailing_sl": 0.08, "decay_sl": False, "max_lots": 10, "compounding": True},
        "15min": {"leverage": 1.0, "fast": 20, "slow": 160, "stop": 0.25, "dqn_exit": False,
                   "profit_target": 0.40, "time_decay_bars": 8, "cooldown": 2}
    },
    description="Nifty 50 Index Options — ATM CE/PE premium trading"
)

# ── Bank Nifty Index Derivatives (Indian Market) — Added 2026 ─────────────────

BANKNIFTY_FUT = AssetConfig(
    name="Bank Nifty Futures",
    symbol="BANKNIFTY",
    exchange="NSE_FO",
    instrument_key="NSE_FO|BANKNIFTY",
    asset_class="equity_index",
    currency="INR",
    lot_size=30,                       # Bank Nifty lot size (updated 2026)
    tick_size=0.05,
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    kalman_process_noise=0.5,
    kalman_measurement_noise=5.0,
    leverage_factor=5.0,
    fast_window=10,
    slow_window=80,
    stop_loss_pct=0.03,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 5.0, "fast": 10, "slow": 80, "stop": 0.03, "dqn_exit": False},
        "1h": {"leverage": 2.5, "fast": 70, "slow": 560, "stop": 0.04, "dqn_exit": False},
        "5min": {"leverage": 2.0, "fast": 60, "slow": 480, "stop": 0.005, "dqn_exit": False},
        "15min": {"leverage": 2.5, "fast": 20, "slow": 160, "stop": 0.008, "dqn_exit": False}
    },
    description="Bank Nifty Index Futures — continuous near-month contract"
)

BANKNIFTY_OPT = AssetConfig(
    name="Bank Nifty ATM Options",
    symbol="BANKNIFTY_OPT",
    exchange="NSE_FO",
    instrument_key="NSE_FO|BANKNIFTY_OPT",
    asset_class="equity_index_option",
    currency="INR",
    lot_size=30,                       # Bank Nifty lot size (updated 2026)
    tick_size=0.05,
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    kalman_process_noise=0.5,
    kalman_measurement_noise=5.0,
    leverage_factor=1.0,
    fast_window=10,
    slow_window=80,
    stop_loss_pct=0.15,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 1.0, "fast": 10, "slow": 80, "stop": 0.15, "dqn_exit": False},
        "1h": {"leverage": 1.0, "fast": 70, "slow": 560, "stop": 0.20, "dqn_exit": False},
        "5min": {"leverage": 1.0, "fast": 60, "slow": 480, "stop": 0.03, "dqn_exit": False,
                 "profit_target": 0.05, "time_decay_bars": 8, "cooldown": 8,
                 "trailing_sl": 0.08, "decay_sl": False, "max_lots": 10, "compounding": True},
        "15min": {"leverage": 1.0, "fast": 20, "slow": 160, "stop": 0.25, "dqn_exit": False,
                   "profit_target": 0.40, "time_decay_bars": 8, "cooldown": 2}
    },
    description="Bank Nifty Index Options — ATM CE/PE premium trading"
)

# ── BSE Sensex Index Derivatives (Indian Market) — Added 2026 ─────────────────

SENSEX_FUT = AssetConfig(
    name="BSE Sensex Futures",
    symbol="SENSEX",
    exchange="BSE",
    instrument_key="BSE|SENSEX",
    asset_class="equity_index",
    currency="INR",
    lot_size=20,                       # Sensex lot size (updated 2026)
    tick_size=0.05,
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    kalman_process_noise=0.5,
    kalman_measurement_noise=5.0,
    leverage_factor=5.0,
    fast_window=10,
    slow_window=80,
    stop_loss_pct=0.03,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 5.0, "fast": 10, "slow": 80, "stop": 0.03, "dqn_exit": False},
        "1h": {"leverage": 2.5, "fast": 70, "slow": 560, "stop": 0.04, "dqn_exit": False},
        "5min": {"leverage": 2.0, "fast": 60, "slow": 480, "stop": 0.005, "dqn_exit": False},
        "15min": {"leverage": 2.5, "fast": 20, "slow": 160, "stop": 0.008, "dqn_exit": False}
    },
    description="BSE Sensex Index Futures — continuous near-month contract"
)

SENSEX_OPT = AssetConfig(
    name="BSE Sensex ATM Options",
    symbol="SENSEX_OPT",
    exchange="BSE",
    instrument_key="BSE|SENSEX_OPT",
    asset_class="equity_index_option",
    currency="INR",
    lot_size=20,                       # Sensex lot size (updated 2026)
    tick_size=0.05,
    initial_capital=50_00_000,        # ₹50 Lakhs
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    kalman_process_noise=0.5,
    kalman_measurement_noise=5.0,
    leverage_factor=1.0,
    fast_window=10,
    slow_window=80,
    stop_loss_pct=0.15,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 1.0, "fast": 10, "slow": 80, "stop": 0.15, "dqn_exit": False},
        "1h": {"leverage": 1.0, "fast": 70, "slow": 560, "stop": 0.20, "dqn_exit": False},
        "5min": {"leverage": 1.0, "fast": 60, "slow": 480, "stop": 0.03, "dqn_exit": False,
                 "profit_target": 0.05, "time_decay_bars": 8, "cooldown": 8,
                 "trailing_sl": 0.08, "decay_sl": False, "max_lots": 10, "compounding": True},
        "15min": {"leverage": 1.0, "fast": 20, "slow": 160, "stop": 0.25, "dqn_exit": False,
                   "profit_target": 0.40, "time_decay_bars": 8, "cooldown": 2}
    },
    description="BSE Sensex Index Options — ATM CE/PE premium trading"
)

# ── Forex USD/EUR (Global Market) — Version 2 ────────────────────────────────

USDEUR_FOREX = AssetConfig(
    name="USD/EUR Forex Pair",
    symbol="USDEUR",
    exchange="FOREX",
    instrument_key="USDEUR=X",        # Yahoo Finance ticker
    asset_class="forex",
    currency="EUR",
    lot_size=1,
    tick_size=0.0001,
    initial_capital=100_000,          # $100,000
    buy_cost_bps=1.0,                  # Forex spreads are tight
    sell_cost_bps=1.0,
    kalman_process_noise=0.2,
    kalman_measurement_noise=2.0,
    leverage_factor=20.0,              # Forex typically has high leverage
    fast_window=20,
    slow_window=50,
    stop_loss_pct=0.015,
    dqn_exit=False,
    timeframe_params={
        "1d": {"leverage": 20.0, "fast": 20, "slow": 50, "stop": 0.015, "dqn_exit": False},
        "1h": {"leverage": 10.0, "fast": 140, "slow": 350, "stop": 0.02, "dqn_exit": False},
        "5min": {"leverage": 5.0, "fast": 60, "slow": 480, "stop": 0.003, "dqn_exit": False},
        "15min": {"leverage": 8.0, "fast": 20, "slow": 160, "stop": 0.005, "dqn_exit": False}
    },
    description="USD/EUR exchange rate — fetched from Yahoo Finance"
)

# ── Asset Registry ──────────────────────────────────────────────────────────

ASSET_REGISTRY = {
    # Alpaca (US)
    "GLD": GLD_ALPACA,
    "USO": USO_ALPACA,
    # Upstox (MCX India)
    "GOLD_MCX": GOLD_MCX,
    "CRUDE_MCX": CRUDE_MCX,
    # Nifty, Bank Nifty, Sensex & Forex (Version 2)
    "NIFTY_FUT": NIFTY_FUT,
    "NIFTY_OPT": NIFTY_OPT,
    "BANKNIFTY_FUT": BANKNIFTY_FUT,
    "BANKNIFTY_OPT": BANKNIFTY_OPT,
    "SENSEX_FUT": SENSEX_FUT,
    "SENSEX_OPT": SENSEX_OPT,
    "USDEUR": USDEUR_FOREX,
}

# Default assets for backtesting
BACKTEST_ASSETS_ALPACA = ["GLD", "USO"]
BACKTEST_ASSETS_MCX = ["GOLD_MCX", "CRUDE_MCX"]
BACKTEST_ASSETS_V2 = ["NIFTY_FUT", "NIFTY_OPT", "BANKNIFTY_FUT", "BANKNIFTY_OPT", "SENSEX_FUT", "SENSEX_OPT", "USDEUR"]

# All timeframe + asset combinations to test
BACKTEST_COMBOS = [
    ("GLD", "1d"), ("GLD", "1h"),
    ("USO", "1d"), ("USO", "1h"),
    ("GOLD_MCX", "1d"), ("GOLD_MCX", "1h"),
    ("CRUDE_MCX", "1d"), ("CRUDE_MCX", "1h"),
    # Version 2
    ("NIFTY_FUT", "1d"), ("NIFTY_FUT", "1h"), ("NIFTY_FUT", "5min"), ("NIFTY_FUT", "15min"),
    ("NIFTY_OPT", "1d"), ("NIFTY_OPT", "1h"), ("NIFTY_OPT", "5min"), ("NIFTY_OPT", "15min"),
    ("BANKNIFTY_FUT", "1d"), ("BANKNIFTY_FUT", "1h"), ("BANKNIFTY_FUT", "5min"), ("BANKNIFTY_FUT", "15min"),
    ("BANKNIFTY_OPT", "1d"), ("BANKNIFTY_OPT", "1h"), ("BANKNIFTY_OPT", "5min"), ("BANKNIFTY_OPT", "15min"),
    ("SENSEX_FUT", "1d"), ("SENSEX_FUT", "1h"), ("SENSEX_FUT", "5min"), ("SENSEX_FUT", "15min"),
    ("SENSEX_OPT", "1d"), ("SENSEX_OPT", "1h"), ("SENSEX_OPT", "5min"), ("SENSEX_OPT", "15min"),
    ("USDEUR", "1d"), ("USDEUR", "1h"), ("USDEUR", "5min"), ("USDEUR", "15min"),
]


def get_asset(name: str) -> AssetConfig:
    """Retrieve asset config by name."""
    if name not in ASSET_REGISTRY:
        raise ValueError(
            f"Unknown asset '{name}'. Available: {list(ASSET_REGISTRY.keys())}"
        )
    return ASSET_REGISTRY[name]

