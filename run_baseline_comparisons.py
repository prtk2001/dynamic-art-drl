"""
Dynamic ART-DRL — Proper Baseline Comparisons
===============================================
Implements credible alternative baselines:
1. 12-Month Momentum Strategy (same leverage)
2. Monthly Rebalanced 50/50 Portfolio (same leverage)
3. Simple SMA(10,50) Crossover (no WFA, no Kalman)
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


def run_baseline_comparisons():
    logger.info("=" * 80)
    logger.info("    PROPER BASELINE COMPARISONS")
    logger.info("=" * 80)

    # Load data
    df_gld_raw = pd.read_parquet(GLD_1D_PATH)
    df_uso_raw = pd.read_parquet(USO_1D_PATH)

    common_dates = df_gld_raw.index.intersection(df_uso_raw.index)
    df_gld_raw = df_gld_raw.loc[common_dates]
    df_uso_raw = df_uso_raw.loc[common_dates]

    prices_gld = df_gld_raw["close"].values
    prices_uso = df_uso_raw["close"].values
    n = len(prices_gld)
    start_idx = 252

    buy_cost = 5.0 / 10000.0
    sell_cost = 15.0 / 10000.0

    results = []

    # ────────────────────────────────────────────────
    # BASELINE 1: 1x Buy-and-Hold (50/50 GLD+USO)
    # ────────────────────────────────────────────────
    logger.info("Baseline 1: 1x Buy-and-Hold (50/50 GLD+USO)...")
    gld_init = prices_gld[start_idx]
    uso_init = prices_uso[start_idx]
    portfolio_bh = (prices_gld[start_idx:] / gld_init) * 50000.0 + \
                   (prices_uso[start_idx:] / uso_init) * 50000.0
    rets_bh = pd.Series(portfolio_bh).pct_change().fillna(0.0)
    met_bh = calculate_performance_metrics(rets_bh)
    results.append(("1x Buy-and-Hold (50/50)", met_bh, 0))

    # ────────────────────────────────────────────────
    # BASELINE 2: 12-Month Momentum Strategy (5x leverage, same fee)
    # Buy if price > 252-day SMA, else flat. Joint 50/50.
    # ────────────────────────────────────────────────
    logger.info("Baseline 2: 12-Month Momentum Strategy (5x leverage)...")
    sma_252_gld = pd.Series(prices_gld).rolling(252).mean().values
    sma_252_uso = pd.Series(prices_uso).rolling(252).mean().values
    mom_sig_gld = (prices_gld > sma_252_gld).astype(float)
    mom_sig_uso = (prices_uso > sma_252_uso).astype(float)

    portfolio_mom = np.zeros(n)
    portfolio_mom[:start_idx] = 100000.0
    cash = 100000.0
    sh_gld = 0.0; sh_uso = 0.0
    prev_g = 0.0; prev_u = 0.0
    trades_mom = 0

    for idx in range(start_idx, n):
        cg = prices_gld[idx]; cu = prices_uso[idx]
        pv = cash + sh_gld * cg + sh_uso * cu
        if pv <= 0:
            portfolio_mom[idx:] = 0.0
            break

        sg = mom_sig_gld[idx-1]
        su = mom_sig_uso[idx-1]

        if sg != prev_g:
            target = (sg * (pv * 0.5) * 5.0) / cg
            diff = target - sh_gld
            cost = abs(diff) * cg * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * cg + cost
            sh_gld = target; prev_g = sg; trades_mom += 1

        if su != prev_u:
            target = (su * (pv * 0.5) * 3.0) / cu
            diff = target - sh_uso
            cost = abs(diff) * cu * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * cu + cost
            sh_uso = target; prev_u = su; trades_mom += 1

        portfolio_mom[idx] = cash + sh_gld * cg + sh_uso * cu

    rets_mom = pd.Series(portfolio_mom[start_idx:]).pct_change().fillna(0.0)
    met_mom = calculate_performance_metrics(rets_mom)
    results.append(("12-Month Momentum (5x/3x)", met_mom, trades_mom))

    # ────────────────────────────────────────────────
    # BASELINE 3: Monthly Rebalanced 50/50 (5x/3x leverage)
    # Rebalance to 50/50 on the first trading day of each month
    # ────────────────────────────────────────────────
    logger.info("Baseline 3: Monthly Rebalanced 50/50 (5x/3x leverage)...")
    portfolio_mr = np.zeros(n)
    portfolio_mr[:start_idx] = 100000.0
    cash = 100000.0
    sh_gld = 0.0; sh_uso = 0.0
    trades_mr = 0
    prev_month = -1

    for idx in range(start_idx, n):
        cg = prices_gld[idx]; cu = prices_uso[idx]
        pv = cash + sh_gld * cg + sh_uso * cu
        if pv <= 0:
            portfolio_mr[idx:] = 0.0
            break

        # Check if new month
        date = df_gld_raw.index[idx]
        curr_month = date.month if hasattr(date, 'month') else pd.Timestamp(date).month

        if curr_month != prev_month:
            # Rebalance to 50/50 leveraged
            target_gld = (pv * 0.5 * 5.0) / cg
            target_uso = (pv * 0.5 * 3.0) / cu
            diff_g = target_gld - sh_gld
            diff_u = target_uso - sh_uso
            cost_g = abs(diff_g) * cg * (buy_cost if diff_g > 0 else sell_cost)
            cost_u = abs(diff_u) * cu * (buy_cost if diff_u > 0 else sell_cost)
            cash -= diff_g * cg + cost_g + diff_u * cu + cost_u
            sh_gld = target_gld; sh_uso = target_uso
            trades_mr += 2
            prev_month = curr_month

        portfolio_mr[idx] = cash + sh_gld * cg + sh_uso * cu

    rets_mr = pd.Series(portfolio_mr[start_idx:]).pct_change().fillna(0.0)
    met_mr = calculate_performance_metrics(rets_mr)
    results.append(("Monthly Rebalanced (5x/3x)", met_mr, trades_mr))

    # ────────────────────────────────────────────────
    # BASELINE 4: Simple SMA(10,50) Crossover (no Kalman, no WFA, same leverage)
    # ────────────────────────────────────────────────
    logger.info("Baseline 4: Simple SMA(10,50) Crossover (no Kalman, same leverage)...")
    sma_f_gld = pd.Series(prices_gld).rolling(10).mean().values
    sma_s_gld = pd.Series(prices_gld).rolling(50).mean().values
    sma_f_uso = pd.Series(prices_uso).rolling(10).mean().values
    sma_s_uso = pd.Series(prices_uso).rolling(50).mean().values
    raw_sig_gld = (sma_f_gld > sma_s_gld).astype(float)
    raw_sig_uso = (sma_f_uso > sma_s_uso).astype(float)

    portfolio_raw = np.zeros(n)
    portfolio_raw[:start_idx] = 100000.0
    cash = 100000.0
    sh_gld = 0.0; sh_uso = 0.0
    prev_g = 0.0; prev_u = 0.0
    trades_raw = 0

    for idx in range(start_idx, n):
        cg = prices_gld[idx]; cu = prices_uso[idx]
        pv = cash + sh_gld * cg + sh_uso * cu
        if pv <= 0:
            portfolio_raw[idx:] = 0.0
            break

        sg = raw_sig_gld[idx-1]; su = raw_sig_uso[idx-1]

        if sg != prev_g:
            target = (sg * (pv * 0.5) * 5.0) / cg
            diff = target - sh_gld
            cost = abs(diff) * cg * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * cg + cost
            sh_gld = target; prev_g = sg; trades_raw += 1

        if su != prev_u:
            target = (su * (pv * 0.5) * 3.0) / cu
            diff = target - sh_uso
            cost = abs(diff) * cu * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * cu + cost
            sh_uso = target; prev_u = su; trades_raw += 1

        portfolio_raw[idx] = cash + sh_gld * cg + sh_uso * cu

    rets_raw = pd.Series(portfolio_raw[start_idx:]).pct_change().fillna(0.0)
    met_raw = calculate_performance_metrics(rets_raw)
    results.append(("Simple SMA(10,50) (no Kalman)", met_raw, trades_raw))

    # ────────────────────────────────────────────────
    # H-DRL Proposed Framework (for direct comparison)
    # ────────────────────────────────────────────────
    logger.info("Proposed: H-DRL Kalman SMA(15,80)/(10,40) Swing (5x/3x)...")
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

    kalman_gld = feat_gld["kalman_price"].values
    kalman_uso = feat_uso["kalman_price"].values
    p_gld = feat_gld["close"].values
    p_uso = feat_uso["close"].values
    n2 = len(p_gld)

    sma_gf = pd.Series(kalman_gld).rolling(15).mean().values
    sma_gs = pd.Series(kalman_gld).rolling(80).mean().values
    sigs_gld = (sma_gf > sma_gs).astype(float)

    sma_uf = pd.Series(kalman_uso).rolling(10).mean().values
    sma_us = pd.Series(kalman_uso).rolling(40).mean().values
    sigs_uso = (sma_uf > sma_us).astype(float)

    portfolio_hdrl = np.zeros(n2)
    portfolio_hdrl[:start_idx] = 100000.0
    cash = 100000.0
    sh_gld = 0.0; sh_uso = 0.0
    prev_g = 0.0; prev_u = 0.0
    in_g = False; pk_g = 0.0
    in_u = False; pk_u = 0.0
    trades_hdrl = 0

    for idx in range(start_idx, n2):
        cg = p_gld[idx]; cu = p_uso[idx]
        pv = cash + sh_gld * cg + sh_uso * cu
        if pv <= 0:
            portfolio_hdrl[idx:] = 0.0; break

        sg = sigs_gld[idx-1]; su = sigs_uso[idx-1]

        if in_g:
            pk_g = max(pk_g, cg)
            if (pk_g - cg) / pk_g > 0.15: sg = 0.0
        if in_u:
            pk_u = max(pk_u, cu)
            if (pk_u - cu) / pk_u > 0.12: su = 0.0

        if sg != prev_g:
            target = (sg * (pv * 0.5) * 5.0) / cg
            diff = target - sh_gld
            cost = abs(diff) * cg * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * cg + cost
            sh_gld = target; prev_g = sg; trades_hdrl += 1
            if sg > 0: in_g = True; pk_g = cg
            else: in_g = False

        if su != prev_u:
            target = (su * (pv * 0.5) * 3.0) / cu
            diff = target - sh_uso
            cost = abs(diff) * cu * (buy_cost if diff > 0 else sell_cost)
            cash -= diff * cu + cost
            sh_uso = target; prev_u = su; trades_hdrl += 1
            if su > 0: in_u = True; pk_u = cu
            else: in_u = False

        portfolio_hdrl[idx] = cash + sh_gld * cg + sh_uso * cu

    rets_hdrl = pd.Series(portfolio_hdrl[start_idx:]).pct_change().fillna(0.0)
    met_hdrl = calculate_performance_metrics(rets_hdrl)
    results.append(("Proposed H-DRL Framework", met_hdrl, trades_hdrl))

    # Save report
    report = []
    for name, m, t in results:
        report.append({
            "strategy": name,
            "cumulative_return_pct": m["cumulative_return"] * 100,
            "annualized_return_pct": m["annualised_return"] * 100,
            "sharpe_ratio": m["sharpe_ratio"],
            "max_drawdown_pct": m["max_drawdown"] * 100,
            "total_trades": t
        })

    with open(Path("results/baseline_comparison_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Print table
    print("\n" + "=" * 115)
    print("            PROPER BASELINE COMPARISON — ALL STRATEGIES (2018–2026)")
    print("=" * 115)
    print(f"{'Strategy':35s} | {'Cum. Return':14s} | {'Ann. Return':12s} | {'Sharpe':8s} | {'Max. DD':10s} | {'Trades':6s}")
    print("-" * 115)
    for name, m, t in results:
        t_str = str(t) if t > 0 else "N/A"
        print(f"{name:35s} | {m['cumulative_return']*100:12.2f}% | "
              f"{m['annualised_return']*100:10.2f}% | {m['sharpe_ratio']:8.3f} | "
              f"{m['max_drawdown']*100:8.2f}% | {t_str:6s}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    run_baseline_comparisons()
