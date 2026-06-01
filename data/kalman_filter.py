"""
Dynamic ART-DRL — Kalman Filter for Financial Time-Series
==========================================================
Implements a recursive Kalman Filter with a 2D state vector ``[price, velocity]``
for denoising financial price series and extracting trend signals.

Three factory functions provide ready-to-use configurations:

* :func:`create_price_filter` — tight tracking of the close price.
* :func:`create_trend_filter` — smoother output emphasising trend direction.
* :func:`create_adaptive_filter` — dynamically adjusts ``Q`` / ``R`` via ATR.

Theory
------
**Prediction step**::

    x̂ₖ|ₖ₋₁ = F · x̂ₖ₋₁
    Pₖ|ₖ₋₁ = F · Pₖ₋₁ · Fᵀ + Q

**Update step**::

    Kₖ = Pₖ|ₖ₋₁ · Hᵀ · (H · Pₖ|ₖ₋₁ · Hᵀ + R)⁻¹
    x̂ₖ = x̂ₖ|ₖ₋₁ + Kₖ · (zₖ − H · x̂ₖ|ₖ₋₁)
    Pₖ = (I − Kₖ · H) · Pₖ|ₖ₋₁
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_PROCESS_NOISE,
    KALMAN_WARMUP_BARS,
)


@dataclass
class KalmanState:
    """Mutable internal state of a 2-D Kalman filter."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(2))
    """State estimate ``[price, velocity]``."""

    P: np.ndarray = field(
        default_factory=lambda: np.eye(2) * 1000.0
    )
    """Error covariance matrix."""

    initialised: bool = False
    """Whether the filter has been seeded with the first observation."""


class KalmanFilter:
    """Recursive 2-D Kalman filter for financial price denoising.

    Parameters
    ----------
    process_noise : float
        Scalar variance for the process noise matrix *Q*.
    measurement_noise : float
        Scalar variance for the measurement noise *R*.
    warmup_bars : int
        Number of initial bars to skip when reporting filtered output
        (the filter needs a ramp-up period).
    adaptive : bool
        If ``True``, ``Q`` and ``R`` are re-estimated at each step using
        the rolling ATR.
    atr_period : int
        Period for ATR calculation when ``adaptive=True``.
    dt : float
        Time-step spacing (normalised; default 1.0).
    """

    def __init__(
        self,
        process_noise: float = KALMAN_PROCESS_NOISE,
        measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
        warmup_bars: int = KALMAN_WARMUP_BARS,
        adaptive: bool = False,
        atr_period: int = 14,
        dt: float = 1.0,
    ) -> None:
        self.dt = dt
        self.warmup_bars = warmup_bars
        self.adaptive = adaptive
        self.atr_period = atr_period

        # State-transition matrix  F = [[1, dt], [0, 1]]
        self.F: np.ndarray = np.array(
            [[1.0, dt], [0.0, 1.0]], dtype=np.float64
        )

        # Observation matrix  H = [1, 0]  (we observe price only)
        self.H: np.ndarray = np.array([[1.0, 0.0]], dtype=np.float64)

        # Process noise covariance
        self._base_q: float = process_noise
        self.Q: np.ndarray = np.array(
            [
                [0.25 * dt**4, 0.5 * dt**3],
                [0.5 * dt**3, dt**2],
            ],
            dtype=np.float64,
        ) * process_noise

        # Measurement noise covariance (scalar → 1×1 matrix)
        self._base_r: float = measurement_noise
        self.R: np.ndarray = np.array(
            [[measurement_noise]], dtype=np.float64
        )

        # Identity matrix for the update step
        self._I: np.ndarray = np.eye(2, dtype=np.float64)

        # Internal state
        self._state: KalmanState = KalmanState()
        self._step: int = 0

        logger.debug(
            "KalmanFilter created: Q_base={}, R_base={}, adaptive={}, warmup={}",
            process_noise,
            measurement_noise,
            adaptive,
            warmup_bars,
        )

    # ------------------------------------------------------------------
    # Core recursive step
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Prediction step: propagate state and covariance forward."""
        s = self._state
        s.x = self.F @ s.x
        s.P = self.F @ s.P @ self.F.T + self.Q

    def update(self, z: float) -> np.ndarray:
        """Update step: assimilate a new observation.

        Parameters
        ----------
        z : float
            New observed price.

        Returns
        -------
        np.ndarray
            Updated state ``[price, velocity]``.
        """
        s = self._state
        H, R, I = self.H, self.R, self._I

        # Innovation
        y = np.array([[z]]) - H @ s.x

        # Innovation covariance
        S = H @ s.P @ H.T + R

        # Kalman gain
        K = s.P @ H.T @ np.linalg.inv(S)

        # State update
        s.x = s.x + (K @ y).flatten()

        # Covariance update (Joseph form for numerical stability)
        IKH = I - K @ H
        s.P = IKH @ s.P @ IKH.T + K @ R @ K.T

        return s.x.copy()

    def step(self, z: float) -> np.ndarray:
        """Run one full predict → update cycle.

        Parameters
        ----------
        z : float
            New price observation.

        Returns
        -------
        np.ndarray
            Updated state ``[filtered_price, velocity]``.
        """
        s = self._state
        self._step += 1

        if not s.initialised:
            s.x = np.array([z, 0.0], dtype=np.float64)
            s.P = np.eye(2, dtype=np.float64) * 1000.0
            s.initialised = True
            return s.x.copy()

        self.predict()
        return self.update(z)

    def reset(self) -> None:
        """Reset the filter to its initial un-seeded state."""
        self._state = KalmanState()
        self._step = 0
        logger.debug("KalmanFilter reset.")

    # ------------------------------------------------------------------
    # Adaptive noise estimation
    # ------------------------------------------------------------------

    def _adapt_noise(self, atr_value: float) -> None:
        """Adjust ``Q`` and ``R`` based on the current ATR.

        When ATR is high (volatile market), increase both Q and R so
        the filter tracks faster and trusts observations less.
        When ATR is low, tighten the filter for smoother output.
        """
        if atr_value <= 0:
            return

        # Scale factor relative to base noise
        scale = atr_value / max(self._base_r, 1e-8)
        scale = np.clip(scale, 0.1, 10.0)

        self.Q = (
            np.array(
                [
                    [0.25 * self.dt**4, 0.5 * self.dt**3],
                    [0.5 * self.dt**3, self.dt**2],
                ],
                dtype=np.float64,
            )
            * self._base_q
            * scale
        )
        self.R = np.array([[self._base_r * scale]], dtype=np.float64)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Compute Average True Range (ATR) for noise adaptation.

        Returns an array the same length as the inputs.  The first
        ``period`` values are forward-filled with the first valid ATR.
        """
        n = len(close)
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.full(n, np.nan, dtype=np.float64)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Forward-fill NaNs at the start
        first_valid = period - 1 if n >= period else n - 1
        atr[:first_valid] = atr[first_valid] if not np.isnan(atr[first_valid]) else tr[0]
        return atr

    def filter_series(
        self,
        prices: np.ndarray | pd.Series,
        high: Optional[np.ndarray | pd.Series] = None,
        low: Optional[np.ndarray | pd.Series] = None,
        close: Optional[np.ndarray | pd.Series] = None,
    ) -> np.ndarray:
        """Filter a 1-D price series and return filtered values.

        Parameters
        ----------
        prices : array-like
            Raw price series to filter (typically close prices).
        high, low, close : array-like, optional
            Required only when ``adaptive=True`` for ATR computation.
            If not given and adaptive is on, ``prices`` is used for all three.

        Returns
        -------
        np.ndarray
            Filtered prices, same length as *prices*.  The first
            ``warmup_bars`` values are taken from the raw input.
        """
        self.reset()
        prices_arr = np.asarray(prices, dtype=np.float64)
        n = len(prices_arr)
        filtered = np.empty(n, dtype=np.float64)

        # ATR for adaptive mode
        atr_arr: Optional[np.ndarray] = None
        if self.adaptive:
            h = np.asarray(high if high is not None else prices_arr, dtype=np.float64)
            l = np.asarray(low if low is not None else prices_arr, dtype=np.float64)
            c = np.asarray(close if close is not None else prices_arr, dtype=np.float64)
            atr_arr = self._compute_atr(h, l, c, self.atr_period)

        for i in range(n):
            if self.adaptive and atr_arr is not None:
                self._adapt_noise(atr_arr[i])

            state = self.step(prices_arr[i])
            if i < self.warmup_bars:
                filtered[i] = prices_arr[i]  # raw value during warmup
            else:
                filtered[i] = state[0]  # filtered price

        logger.debug("Filtered {} prices (warmup={}).", n, self.warmup_bars)
        return filtered

    def filter_ohlcv(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        prefix: str = "kalman_",
    ) -> pd.DataFrame:
        """Apply the Kalman filter to an OHLCV DataFrame.

        New columns added:
        * ``{prefix}price`` — filtered close price
        * ``{prefix}velocity`` — estimated price velocity (trend)
        * ``{prefix}residual`` — raw − filtered (innovation proxy)

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least ``close``; ``high`` and ``low`` are
            used when ``adaptive=True``.
        price_col : str
            Column to filter.  Default ``"close"``.
        prefix : str
            Prefix for new column names.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with additional Kalman columns.
        """
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame.")

        out = df.copy()
        self.reset()

        prices = df[price_col].values.astype(np.float64)
        n = len(prices)
        filtered_price = np.empty(n, dtype=np.float64)
        velocity = np.empty(n, dtype=np.float64)

        # ATR for adaptive mode
        atr_arr: Optional[np.ndarray] = None
        if self.adaptive and {"high", "low", "close"}.issubset(df.columns):
            atr_arr = self._compute_atr(
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                df["close"].values.astype(np.float64),
                self.atr_period,
            )

        for i in range(n):
            if self.adaptive and atr_arr is not None:
                self._adapt_noise(atr_arr[i])

            state = self.step(prices[i])

            if i < self.warmup_bars:
                filtered_price[i] = prices[i]
                velocity[i] = 0.0
            else:
                filtered_price[i] = state[0]
                velocity[i] = state[1]

        out[f"{prefix}price"] = filtered_price
        out[f"{prefix}velocity"] = velocity
        out[f"{prefix}residual"] = prices - filtered_price

        logger.info(
            "Kalman-filtered {} ({} rows, warmup={}). "
            "New cols: {}price, {}velocity, {}residual.",
            price_col,
            n,
            self.warmup_bars,
            prefix,
            prefix,
            prefix,
        )
        return out

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Current state estimate ``[price, velocity]``."""
        return self._state.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current error covariance ``P``."""
        return self._state.P.copy()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_price_filter(
    process_noise: float = 0.3,
    measurement_noise: float = 3.0,
    warmup: int = KALMAN_WARMUP_BARS,
) -> KalmanFilter:
    """Create a Kalman filter tuned for tight price tracking.

    Lower ``Q`` (process noise) makes the filter track the actual price
    more closely; lower ``R`` (measurement noise) means we trust
    observations more.
    """
    return KalmanFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        warmup_bars=warmup,
        adaptive=False,
    )


def create_trend_filter(
    process_noise: float = 0.1,
    measurement_noise: float = 10.0,
    warmup: int = KALMAN_WARMUP_BARS,
) -> KalmanFilter:
    """Create a Kalman filter emphasising trend extraction.

    Higher ``R`` means we trust each individual observation less, producing
    a smoother, more trend-following output.
    """
    return KalmanFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        warmup_bars=warmup,
        adaptive=False,
    )


def create_adaptive_filter(
    process_noise: float = KALMAN_PROCESS_NOISE,
    measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
    warmup: int = KALMAN_WARMUP_BARS,
    atr_period: int = 14,
) -> KalmanFilter:
    """Create an adaptive Kalman filter that adjusts Q/R via ATR.

    During high-volatility regimes (large ATR), the filter loosens to
    track faster; during calm markets it tightens for smoother output.
    """
    return KalmanFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        warmup_bars=warmup,
        adaptive=True,
        atr_period=atr_period,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    # Synthetic price: random walk + trend
    raw = 100.0 + np.cumsum(np.random.randn(n) * 0.5) + np.linspace(0, 10, n)

    for name, factory in [
        ("price", create_price_filter),
        ("trend", create_trend_filter),
        ("adaptive", create_adaptive_filter),
    ]:
        kf = factory()
        filtered = kf.filter_series(raw)
        rmse = np.sqrt(np.mean((raw[kf.warmup_bars:] - filtered[kf.warmup_bars:]) ** 2))
        print(f"{name:>10s} filter — RMSE vs raw: {rmse:.4f}")
