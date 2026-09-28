"""
Dynamic ART-DRL — Centralized Configuration
============================================
Loads API credentials from .env and defines all hyperparameters,
training constants, and system defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── API Credentials ──────────────────────────────────────────────────────────

# Alpaca (Paper Trading — US ETFs: GLD, USO)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_ENDPOINT = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")

# Upstox (MCX Futures — Gold, Crude Oil) — PRIMARY
UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# Groq LLM (Meta-Analyst)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── Data Configuration ───────────────────────────────────────────────────────
DATA_CACHE_DIR = ROOT_DIR / "data_cache"
DATA_CACHE_DIR.mkdir(exist_ok=True)

# Historical data range (8 years as per Kalman paper)
DATA_START_DATE = "2017-01-01"
DATA_END_DATE = "2026-05-30"


# Timeframes to test (all combos as requested)
TIMEFRAMES = ["1d", "1h", "5min", "15min"]  # Daily, Hourly, 5-min, 15-min

# ── Trading Environment Parameters ──────────────────────────────────────────

# Initial capital
INITIAL_CAPITAL_USD = 100_000       # Alpaca paper trading
INITIAL_CAPITAL_INR = 50_00_000     # Upstox (₹50 Lakhs)

# Transaction costs (in basis points)
BUY_COST_BPS = 5       # 0.05%
SELL_COST_BPS = 15      # 0.15%

# Position constraints
MAX_POSITION_PCT = 0.95   # Max 95% of capital in a single position
MIN_TRADE_VALUE = 100     # Minimum trade value

# ── Kalman Filter Parameters ────────────────────────────────────────────────

KALMAN_PROCESS_NOISE = 0.5        # Q — model uncertainty
KALMAN_MEASUREMENT_NOISE = 5.0    # R — market microstructure noise
KALMAN_WARMUP_BARS = 20           # Bars to skip before using filtered signal

# ── VSN + Transformer Parameters ─────────────────────────────────────────────

# Variable Selection Network
VSN_HIDDEN_SIZE = 64
VSN_DROPOUT = 0.1

# Transformer Encoder
TRANSFORMER_HIDDEN_SIZE = 64
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1
LOOKBACK_WINDOW = 60    # Bars of history fed to the encoder

# ── DRL Agent Hyperparameters ────────────────────────────────────────────────

# Shared
GAMMA = 0.99                  # Discount factor
TRAINING_TIMESTEPS = 500_000  # Default total timesteps per agent (scaled for V2)

# DQN
DQN_LEARNING_RATE = 1e-4
DQN_BUFFER_SIZE = 100_000
DQN_BATCH_SIZE = 64
DQN_TAU = 0.005               # Soft update coefficient
DQN_EXPLORATION_FRACTION = 0.2
DQN_EXPLORATION_FINAL_EPS = 0.05
DQN_DISCRETE_ACTIONS = 5      # {sell_all, sell_half, hold, buy_half, buy_all}

# PPO
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_CLIP_RANGE = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_ENT_COEF = 0.01

# DDPG
DDPG_LEARNING_RATE = 1e-3
DDPG_BUFFER_SIZE = 100_000
DDPG_BATCH_SIZE = 256
DDPG_TAU = 0.005
DDPG_NOISE_TYPE = "ornstein-uhlenbeck"
DDPG_NOISE_SIGMA = 0.1

# A2C
A2C_LEARNING_RATE = 7e-4
A2C_N_STEPS = 5
A2C_GAE_LAMBDA = 0.95
A2C_ENT_COEF = 0.01
A2C_VF_COEF = 0.5

# ── Dynamic Router Parameters ───────────────────────────────────────────────

VOLATILITY_LOOKBACK = 30         # Rolling window for σ_rolling (days)
HISTORICAL_VOL_WINDOW = 252      # Window for σ̄_historical (1 year)
REGIME_HYSTERESIS_BARS = 3       # Bars regime must persist before switching

# ── Reward Function Parameters ──────────────────────────────────────────────

REWARD_DRAWDOWN_PENALTY = 0.5    # λ₁ — drawdown penalty weight
REWARD_TRANSACTION_PENALTY = 1.0 # λ₂ — transaction cost weight
REWARD_SHARPE_BONUS = 0.1        # λ₃ — rolling Sharpe bonus weight
REWARD_ROLLING_WINDOW = 30       # Window for rolling Sharpe in reward

# ── Backtesting Parameters ──────────────────────────────────────────────────

BACKTEST_TRAIN_WINDOW = 252      # Training window (trading days)
BACKTEST_TEST_WINDOW = 63        # Testing window (1 quarter)
BACKTEST_STEP_SIZE = 63          # Walk-forward step size
RISK_FREE_RATE = 0.05            # Annual risk-free rate (US 10Y Treasury approx)

# ── Paths ────────────────────────────────────────────────────────────────────

NIFTY_LOCAL_DATA_DIR = Path("/Users/prateek/Documents/nifty_options/data/nifty_bank_nifty")

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

TENSORBOARD_DIR = ROOT_DIR / "tensorboard_logs"
TENSORBOARD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
