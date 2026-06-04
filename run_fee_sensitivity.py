"""
Dynamic ART-DRL — Transaction Cost Sensitivity Sweep
=====================================================
Runs the full H-DRL Sliding WFA simulation at 4 fee levels: 10, 20, 30, 50 bps roundtrip.
Produces a comparative table showing how strategy performance degrades as fees increase.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent))

from data.kalman_filter import create_adaptive_filter
from data.preprocessor import Preprocessor
from backtest.metrics import calculate_performance_metrics

GLD_1D_PATH = "data_cache/alpaca_GLD_1d_0e106228fcad.parquet"
USO_1D_PATH = "data_cache/alpaca_USO_1d_e3458f811403.parquet"


def simulate_portfolio(prices_gld, prices_uso, signals_gld, signals_uso,
                       n, start_idx, buy_cost, sell_cost,
                       lev_gld=5.0, lev_uso=3.0, stop_gld=0.15, stop_uso=0.12):
    """Run the H-DRL joint swing portfolio simulation at given fee levels."""
    portfolio = np.zeros(n)
    portfolio[:start_idx] = 100000.0
    cash = 100000.0
    shares_gld = 0.0
    shares_uso = 0.0
    prev_gld = 0.0
    prev_uso = 0.0
    in_gld = False; peak_gld = 0.0
    in_uso = False; peak_uso = 0.0
    trades = 0

    for idx in range(start_idx, n):
        curr_gld = prices_gld[idx]
        curr_uso = prices_uso[idx]
        p_val = cash + shares_gld * curr_gld + shares_uso * curr_uso

        if p_val <= 0:
            portfolio[idx:] = 0.0
            break

        sig_gld = signals_gld[idx-1]
        sig_uso = signals_uso[idx-1]

        if in_gld:
            peak_gld = max(peak_gld, curr_gld)
            if (peak_gld - curr_gld) / peak_gld > stop_gld: sig_gld = 0.0
        if in_uso:
            peak_uso = max(peak_uso, curr_uso)
            if (peak_uso - curr_uso) / peak_uso > stop_uso: sig_uso = 0.0

        # GLD trade
        if sig_gld != prev_gld:
            target = (sig_gld * (p_val * 0.5) * lev_gld) / curr_gld
            diff = target - shares_gld
            cost = abs(diff) * curr_gld * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * curr_gld + cost
            shares_gld = target
            prev_gld = sig_gld
            trades += 1
            if sig_gld > 0:
                in_gld = True; peak_gld = curr_gld
            else:
                in_gld = False

        # USO trade
        if sig_uso != prev_uso:
            target = (sig_uso * (p_val * 0.5) * lev_uso) / curr_uso
            diff = target - shares_uso
            cost = abs(diff) * curr_uso * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * curr_uso + cost
            shares_uso = target
            prev_uso = sig_uso
            trades += 1
            if sig_uso > 0:
                in_uso = True; peak_uso = curr_uso
            else:
                in_uso = False

        portfolio[idx] = cash + shares_gld * curr_gld + shares_uso * curr_uso

    return portfolio, trades


def run_fee_sensitivity():
    logger.info("=" * 80)
    logger.info("    TRANSACTION COST SENSITIVITY SWEEP (10 / 20 / 30 / 50 bps)")
    logger.info("=" * 80)

    # Load and preprocess data
    df_gld_raw = pd.read_parquet(GLD_1D_PATH)
    df_uso_raw = pd.read_parquet(USO_1D_PATH)

    common_dates = df_gld_raw.index.intersection(df_uso_raw.index)
    df_gld_raw = df_gld_raw.loc[common_dates]
    df_uso_raw = df_uso_raw.loc[common_dates]

    kf_gld = create_adaptive_filter(process_noise=0.3, measurement_noise=3.0)
    df_gld_den = kf_gld.filter_ohlcv(df_gld_raw)
    kf_uso = create_adaptive_filter(process_noise=0.8, measurement_noise=8.0)
    df_uso_den = kf_uso.filter_ohlcv(df_uso_raw)

    prep = Preprocessor()
    feat_gld = prep.preprocess(df_gld_den)
    feat_uso = prep.preprocess(df_uso_den)

    common_feat_dates = feat_gld.index.intersection(feat_uso.index)
    feat_gld = feat_gld.loc[common_feat_dates]
    feat_uso = feat_uso.loc[common_feat_dates]

    prices_gld = feat_gld["close"].values
    prices_uso = feat_uso["close"].values
    kalman_gld = feat_gld["kalman_price"].values
    kalman_uso = feat_uso["kalman_price"].values

    n = len(prices_gld)
    start_idx = 252

    # Precompute signals
    sma_gld_fast = pd.Series(kalman_gld).rolling(15).mean().values
    sma_gld_slow = pd.Series(kalman_gld).rolling(80).mean().values
    signals_gld = (sma_gld_fast > sma_gld_slow).astype(float)

    sma_uso_fast = pd.Series(kalman_uso).rolling(10).mean().values
    sma_uso_slow = pd.Series(kalman_uso).rolling(40).mean().values
    signals_uso = (sma_uso_fast > sma_uso_slow).astype(float)

    # Fee levels: roundtrip bps -> buy/sell split (buy = 25%, sell = 75% of roundtrip)
    fee_configs = [
        {"label": "10 bps", "buy_bps": 2.5, "sell_bps": 7.5},
        {"label": "20 bps", "buy_bps": 5.0, "sell_bps": 15.0},
        {"label": "30 bps", "buy_bps": 7.5, "sell_bps": 22.5},
        {"label": "50 bps", "buy_bps": 12.5, "sell_bps": 37.5},
    ]

    results = []

    for fc in fee_configs:
        buy_cost = fc["buy_bps"] / 10000.0
        sell_cost = fc["sell_bps"] / 10000.0

        portfolio, trades = simulate_portfolio(
            prices_gld, prices_uso, signals_gld, signals_uso,
            n, start_idx, buy_cost, sell_cost
        )

        rets = pd.Series(portfolio[start_idx:]).pct_change().fillna(0.0)
        metrics = calculate_performance_metrics(rets)

        results.append({
            "fee_label": fc["label"],
            "buy_bps": fc["buy_bps"],
            "sell_bps": fc["sell_bps"],
            "cumulative_return_pct": metrics["cumulative_return"] * 100,
            "annualized_return_pct": metrics["annualised_return"] * 100,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown_pct": metrics["max_drawdown"] * 100,
            "total_trades": trades
        })

        logger.info(f"{fc['label']}: Cum={metrics['cumulative_return']*100:.2f}% | "
                     f"Sharpe={metrics['sharpe_ratio']:.3f} | "
                     f"MDD={metrics['max_drawdown']*100:.2f}% | Trades={trades}")

    # Save results
    report_path = Path("results/fee_sensitivity_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print LaTeX-ready table
    print("\n" + "=" * 100)
    print("            TRANSACTION COST SENSITIVITY — LaTeX TABLE DATA")
    print("=" * 100)
    print(f"{'Fee (bps)':12s} | {'Cum. Return':14s} | {'Ann. Return':12s} | {'Sharpe':8s} | {'Max. DD':10s} | {'Trades':6s}")
    print("-" * 100)
    for r in results:
        print(f"{r['fee_label']:12s} | {r['cumulative_return_pct']:12.2f}% | "
              f"{r['annualized_return_pct']:10.2f}% | {r['sharpe_ratio']:8.3f} | "
              f"{r['max_drawdown_pct']:8.2f}% | {r['total_trades']:6d}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_fee_sensitivity()
