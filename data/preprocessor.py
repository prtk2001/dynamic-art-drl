"""
Dynamic ART-DRL — Data Preprocessor & Feature Engineering
==========================================================
Transforms raw OHLCV DataFrames into fully-normalised feature tensors
ready for the Transformer encoder and RL environment.

Pipeline
--------
1. **Technical indicators** via the ``ta`` library: RSI(14), MACD(12,26,9),
   Bollinger Band Width, CCI(30), DX(30), ATR(14).
2. **Volatility metrics**: 30-day rolling σ, EWMA volatility.
3. **Returns**: simple and log returns.
4. **Z-score normalisation** with a configurable rolling window.
5. **State tensor construction**: sliding-window 3-D array
   ``(samples, lookback, features)`` for the Transformer encoder.

All operations are in-place-safe (return new DataFrames / arrays).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    LOOKBACK_WINDOW,
    VOLATILITY_LOOKBACK,
)

# ---------------------------------------------------------------------------
# Lazy-import the ``ta`` library
# ---------------------------------------------------------------------------

_TA_AVAILABLE: bool = True
try:
    import ta
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, CCIIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("'ta' library not installed — indicator computation will fail.")


# ---------------------------------------------------------------------------
# Feature column registries
# ---------------------------------------------------------------------------

# Columns added by compute_technical_indicators()
_INDICATOR_COLS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_width",
    "cci_30",
    "dx_30",
    "atr_14",
]

# Columns added by compute_volatility_metrics()
_VOLATILITY_COLS = [
    "return_simple",
    "return_log",
    "vol_rolling_30",
    "vol_ewma",
]

# All feature columns (excluding OHLCV)
ALL_FEATURE_COLS: list[str] = _INDICATOR_COLS + _VOLATILITY_COLS


class Preprocessor:
    """Feature engineering and normalisation pipeline.

    Parameters
    ----------
    lookback : int
        Number of historical bars per sample for the state tensor.
    vol_window : int
        Rolling window for volatility metrics (days).
    zscore_window : int
        Rolling window for z-score normalisation.
    """

    def __init__(
        self,
        lookback: int = LOOKBACK_WINDOW,
        vol_window: int = VOLATILITY_LOOKBACK,
        zscore_window: int = 252,
    ) -> None:
        self.lookback = lookback
        self.vol_window = vol_window
        self.zscore_window = zscore_window

        if not _TA_AVAILABLE:
            logger.error(
                "The 'ta' library is required for Preprocessor. "
                "Install with: pip install ta"
            )

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical-indicator columns to *df*.

        Requires columns: ``high, low, close`` (and ``volume`` if
        available, though currently unused).

        Added columns
        -------------
        rsi_14, macd, macd_signal, macd_diff, bb_width, cci_30, dx_30, atr_14
        """
        if not _TA_AVAILABLE:
            raise ImportError("Install 'ta' library: pip install ta")

        required = {"high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        out = df.copy()
        close = out["close"]
        high = out["high"]
        low = out["low"]

        # RSI(14)
        out["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

        # MACD(12, 26, 9)
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_diff"] = macd.macd_diff()

        # Bollinger Band Width (20-period default)
        bb = BollingerBands(close=close, window=20, window_dev=2)
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        mid = bb.bollinger_mavg()
        # Width = (upper - lower) / mid
        out["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

        # CCI(30)
        out["cci_30"] = CCIIndicator(
            high=high, low=low, close=close, window=30
        ).cci()

        # DX(30)  — Directional Index (using ADX as proxy)
        out["dx_30"] = ADXIndicator(
            high=high, low=low, close=close, window=30
        ).adx()

        # ATR(14)
        out["atr_14"] = AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        logger.debug("Computed {} technical indicators.", len(_INDICATOR_COLS))
        return out

    # ------------------------------------------------------------------
    # Volatility metrics
    # ------------------------------------------------------------------

    def compute_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return and volatility columns.

        Added columns
        -------------
        return_simple, return_log, vol_rolling_30, vol_ewma
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must have a 'close' column.")

        out = df.copy()
        close = out["close"]

        # Simple return
        out["return_simple"] = close.pct_change()

        # Log return
        out["return_log"] = np.log(close / close.shift(1))

        # Rolling standard deviation of log returns (annualised is optional)
        out["vol_rolling_30"] = out["return_log"].rolling(
            window=self.vol_window, min_periods=1
        ).std()

        # EWMA volatility (span = vol_window)
        out["vol_ewma"] = out["return_log"].ewm(
            span=self.vol_window, min_periods=1
        ).std()

        logger.debug(
            "Computed volatility metrics (window={}).",
            self.vol_window,
        )
        return out

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def zscore_normalise(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """Apply rolling z-score normalisation to selected columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : sequence of str, optional
            Columns to normalise.  Defaults to ``ALL_FEATURE_COLS``.
        window : int, optional
            Rolling window.  Defaults to ``self.zscore_window``.

        Returns
        -------
        pd.DataFrame
            Copy with normalised values in the specified columns.
        """
        out = df.copy()
        cols = list(columns) if columns else ALL_FEATURE_COLS
        win = window or self.zscore_window

        # Only normalise columns that actually exist
        cols_present = [c for c in cols if c in out.columns]
        cols_missing = set(cols) - set(cols_present)
        if cols_missing:
            logger.warning(
                "zscore_normalise: columns not found, skipping: {}", cols_missing
            )

        for col in cols_present:
            roll_mean = out[col].rolling(window=win, min_periods=1).mean()
            roll_std = out[col].rolling(window=win, min_periods=1).std()
            # Avoid division by zero
            roll_std = roll_std.replace(0, np.nan)
            out[col] = (out[col] - roll_mean) / roll_std

        logger.debug(
            "Z-score normalised {} columns (window={}).",
            len(cols_present),
            win,
        )
        return out

    # ------------------------------------------------------------------
    # Full preprocess pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        raw_df: pd.DataFrame,
        normalise: bool = True,
    ) -> pd.DataFrame:
        """Run the full preprocessing pipeline.

        Steps
        -----
        1. Compute technical indicators (RSI, MACD, BB, CCI, DX, ATR).
        2. Compute volatility metrics (returns, rolling σ, EWMA σ).
        3. (Optional) Z-score normalise feature columns.
        4. Forward-fill then drop remaining NaN rows.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Raw OHLCV DataFrame with columns ``open, high, low, close, volume``.
        normalise : bool
            Whether to apply z-score normalisation.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame ready for tensor construction.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(raw_df.columns)
        if missing:
            raise ValueError(
                f"Raw DataFrame missing required columns: {missing}. "
                f"Got: {list(raw_df.columns)}"
            )

        logger.info(
            "Preprocessing {} rows × {} cols …",
            len(raw_df),
            len(raw_df.columns),
        )

        # Step 1: Technical indicators
        df = self.compute_technical_indicators(raw_df)

        # Step 2: Volatility metrics
        df = self.compute_volatility_metrics(df)

        # Step 3: Normalise
        if normalise:
            df = self.zscore_normalise(df)

        # Step 4: Clean up NaNs
        n_before = len(df)
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        n_after = len(df)
        if n_before != n_after:
            logger.info(
                "Dropped {} NaN rows ({} → {}).",
                n_before - n_after,
                n_before,
                n_after,
            )

        logger.info(
            "Preprocessing complete: {} rows × {} cols.  "
            "Feature cols: {}",
            len(df),
            len(df.columns),
            ALL_FEATURE_COLS,
        )
        return df

    # ------------------------------------------------------------------
    # Tensor construction
    # ------------------------------------------------------------------

    def build_state_tensor(
        self,
        feature_df: pd.DataFrame,
        lookback: Optional[int] = None,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Build a 3-D sliding-window tensor for the RL environment.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Output of :meth:`preprocess`.
        lookback : int, optional
            Window length.  Defaults to ``self.lookback``.
        feature_cols : sequence of str, optional
            Columns to include.  Defaults to ``ALL_FEATURE_COLS``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, lookback, n_features)`` where
            ``n_samples = len(feature_df) - lookback + 1``.

        Raises
        ------
        ValueError
            If the DataFrame is too short for the requested lookback.
        """
        lb = lookback or self.lookback
        cols = list(feature_cols) if feature_cols else ALL_FEATURE_COLS

        # Keep only available feature columns
        available = [c for c in cols if c in feature_df.columns]
        if not available:
            raise ValueError(
                "No feature columns found in DataFrame. "
                f"Expected some of: {cols}. Got: {list(feature_df.columns)}"
            )

        data = feature_df[available].values.astype(np.float32)
        n_rows, n_feats = data.shape

        if n_rows < lb:
            raise ValueError(
                f"DataFrame has {n_rows} rows but lookback={lb}. "
                f"Need at least {lb} rows."
            )

        # Replace any remaining NaN / inf with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Sliding window via stride tricks (memory-efficient)
        n_samples = n_rows - lb + 1
        strides = (data.strides[0], data.strides[0], data.strides[1])
        tensor = np.lib.stride_tricks.as_strided(
            data,
            shape=(n_samples, lb, n_feats),
            strides=strides,
        )
        # Return a contiguous copy so downstream code doesn't depend on
        # the stride-trick memory layout
        tensor = np.ascontiguousarray(tensor)

        logger.info(
            "Built state tensor: shape={} (samples={}, lookback={}, features={}).",
            tensor.shape,
            n_samples,
            lb,
            n_feats,
        )
        return tensor

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_names() -> list[str]:
        """Return the canonical list of feature column names."""
        return list(ALL_FEATURE_COLS)

    @staticmethod
    def describe_features(df: pd.DataFrame) -> pd.DataFrame:
        """Return descriptive statistics for all feature columns."""
        cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        return df[cols].describe()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_preprocessor(**kwargs) -> Preprocessor:
    """Factory for the default Preprocessor with project config."""
    return Preprocessor(**kwargs)


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    opn = close + np.random.randn(n) * 0.3
    vol = np.random.randint(100_000, 1_000_000, size=n)

    raw = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )

    pp = get_preprocessor()
    features = pp.preprocess(raw)
    print("Feature columns:", list(features.columns))
    print(features.tail(3))

    tensor = pp.build_state_tensor(features)
    print(f"\nState tensor shape: {tensor.shape}")
    print(f"Feature names: {pp.get_feature_names()}")
