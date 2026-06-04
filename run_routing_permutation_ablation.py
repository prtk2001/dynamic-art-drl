"""
Dynamic ART-DRL — Full 24-Permutation Routing Ablation
=======================================================
Tests all 4! = 24 possible assignments of {DQN, PPO, DDPG, A2C} to
{LOW, NORMAL, HIGH, EXTREME} volatility regimes. Reports the rank of the
proposed mapping and the full performance comparison.
"""

import sys
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent))

from data.kalman_filter import create_adaptive_filter
from data.preprocessor import Preprocessor
from router.dynamic_router import DynamicRouter, MarketRegime
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.ddpg_agent import DDPGAgent
from agents.a2c_agent import A2CAgent
from backtest.metrics import calculate_performance_metrics

GLD_1D_PATH = "data_cache/alpaca_GLD_1d_0e106228fcad.parquet"
USO_1D_PATH = "data_cache/alpaca_USO_1d_e3458f811403.parquet"


def get_signal(agent_name, obs, agents):
    """Get trading signal from the named agent."""
    if agent_name == "DQN":
        raw_act, _ = agents["DQN"].predict(obs, deterministic=True)
        action_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
        return action_map.get(int(raw_act), 0.0)
    else:
        raw_act, _ = agents[agent_name].predict(obs, deterministic=True)
        return float(raw_act[0])


def run_permutation_ablation():
    logger.info("=" * 80)
    logger.info("    24-PERMUTATION ROUTING ABLATION (All {DQN,PPO,DDPG,A2C} → {LOW,NORM,HIGH,EXT})")
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
    n = len(prices_gld)
    start_idx = 252

    feature_cols = ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_width',
                    'cci_30', 'dx_30', 'atr_14', 'return_simple', 'return_log',
                    'vol_rolling_30', 'vol_ewma']
    features_gld = feat_gld[feature_cols].values
    features_uso = feat_uso[feature_cols].values

    # Load pre-trained agents
    logger.info("Loading pre-trained DRL agent checkpoints...")
    checkpoint_dir = Path("checkpoints")

    agents_gld = {
        "DQN": DQNAgent(), "PPO": PPOAgent(),
        "DDPG": DDPGAgent(), "A2C": A2CAgent()
    }
    agents_gld["DQN"].load(checkpoint_dir / "dqn_GLD_1d")
    agents_gld["PPO"].load(checkpoint_dir / "ppo_GLD_1d")
    agents_gld["DDPG"].load(checkpoint_dir / "ddpg_GLD_1d")
    agents_gld["A2C"].load(checkpoint_dir / "a2c_GLD_1d")

    agents_uso = {
        "DQN": DQNAgent(), "PPO": PPOAgent(),
        "DDPG": DDPGAgent(), "A2C": A2CAgent()
    }
    agents_uso["DQN"].load(checkpoint_dir / "dqn_USO_1d")
    agents_uso["PPO"].load(checkpoint_dir / "ppo_USO_1d")
    agents_uso["DDPG"].load(checkpoint_dir / "ddpg_USO_1d")
    agents_uso["A2C"].load(checkpoint_dir / "a2c_USO_1d")

    buy_cost = 5.0 / 10000.0
    sell_cost = 15.0 / 10000.0

    # Generate all 24 permutations
    agent_names = ["DQN", "PPO", "DDPG", "A2C"]
    regime_order = [MarketRegime.LOW_VOLATILITY, MarketRegime.NORMAL_VOLATILITY,
                    MarketRegime.HIGH_VOLATILITY, MarketRegime.EXTREME_VOLATILITY]
    regime_labels = ["LOW", "NORMAL", "HIGH", "EXTREME"]

    all_permutations = list(itertools.permutations(agent_names))
    results = []

    for perm_idx, perm in enumerate(all_permutations):
        mapping = dict(zip(regime_order, perm))
        mapping_str = " | ".join([f"{regime_labels[i]}→{perm[i]}" for i in range(4)])
        logger.info(f"Permutation {perm_idx+1}/24: {mapping_str}")

        # Fresh router instances for each permutation
        router_gld = DynamicRouter()
        router_uso = DynamicRouter()

        portfolio = np.zeros(n)
        portfolio[:start_idx] = 100000.0
        cash = 100000.0
        shares_gld = 0.0
        shares_uso = 0.0
        prev_gld = 0.0
        prev_uso = 0.0
        in_pos_gld = False; peak_gld = 0.0
        in_pos_uso = False; peak_uso = 0.0
        trades = 0

        for idx in range(start_idx, n):
            curr_gld = prices_gld[idx]
            curr_uso = prices_uso[idx]
            p_val = cash + shares_gld * curr_gld + shares_uso * curr_uso
            if p_val <= 0:
                portfolio[idx:] = 0.0
                break

            # Route with custom mapping
            _, regime_gld, vol_gld = router_gld.route(prices_gld[:idx])
            _, regime_uso, vol_uso = router_uso.route(prices_uso[:idx])

            agent_gld_name = mapping[regime_gld]
            agent_uso_name = mapping[regime_uso]

            # Inverse vol allocation (Op C)
            std_gld = vol_gld["current_vol"]
            std_uso = vol_uso["current_vol"]
            if std_gld > 0 and std_uso > 0:
                w_gld = std_uso / (std_gld + std_uso)
                w_uso = std_gld / (std_gld + std_uso)
            else:
                w_gld, w_uso = 0.5, 0.5

            # Dynamic leverage (Op A)
            lev_gld = min(5.0, max(1.0, 5.0 * (0.006 / std_gld))) if std_gld > 0 else 5.0
            lev_uso = min(5.0, max(1.0, 5.0 * (0.015 / std_uso))) if std_uso > 0 else 5.0

            # Regime stops (Op B)
            stop_map = {"DQN": 0.015, "A2C": 0.02, "DDPG": 0.03, "PPO": 0.05}
            stop_gld = stop_map.get(agent_gld_name, 0.05)
            stop_uso_map = {"DQN": 0.008, "A2C": 0.01, "DDPG": 0.015, "PPO": 0.02}
            stop_uso = stop_uso_map.get(agent_uso_name, 0.02)

            pv_safe = max(p_val, 1e-8)
            state_gld = np.concatenate([features_gld[idx],
                [cash/pv_safe, (shares_gld*curr_gld)/pv_safe, (p_val-100000.0)/100000.0]])
            state_uso = np.concatenate([features_uso[idx],
                [cash/pv_safe, (shares_uso*curr_uso)/pv_safe, (p_val-100000.0)/100000.0]])

            sig_gld = get_signal(agent_gld_name, state_gld, agents_gld)
            sig_uso = get_signal(agent_uso_name, state_uso, agents_uso)

            if in_pos_gld:
                peak_gld = max(peak_gld, curr_gld)
                if (peak_gld - curr_gld) / peak_gld > stop_gld: sig_gld = -1.0
            if in_pos_uso:
                peak_uso = max(peak_uso, curr_uso)
                if (peak_uso - curr_uso) / peak_uso > stop_uso: sig_uso = -1.0

            # Execute GLD
            if sig_gld != prev_gld:
                if sig_gld > 0.0:
                    target_gld = (sig_gld * (p_val * w_gld) * lev_gld) / curr_gld
                elif sig_gld < 0.0:
                    target_gld = shares_gld * (1.0 + sig_gld)
                else:
                    target_gld = shares_gld
                diff = target_gld - shares_gld
                cost = abs(diff) * curr_gld * (buy_cost if diff > 0 else sell_cost)
                cash -= diff * curr_gld + cost
                shares_gld = target_gld
                prev_gld = sig_gld
                trades += 1
                if shares_gld > 1e-5:
                    in_pos_gld = True
                    if diff > 0: peak_gld = max(peak_gld, curr_gld)
                else:
                    in_pos_gld = False

            # Execute USO
            if sig_uso != prev_uso:
                if sig_uso > 0.0:
                    target_uso = (sig_uso * (p_val * w_uso) * lev_uso) / curr_uso
                elif sig_uso < 0.0:
                    target_uso = shares_uso * (1.0 + sig_uso)
                else:
                    target_uso = shares_uso
                diff = target_uso - shares_uso
                cost = abs(diff) * curr_uso * (buy_cost if diff > 0 else sell_cost)
                cash -= diff * curr_uso + cost
                shares_uso = target_uso
                prev_uso = sig_uso
                trades += 1
                if shares_uso > 1e-5:
                    in_pos_uso = True
                    if diff > 0: peak_uso = max(peak_uso, curr_uso)
                else:
                    in_pos_uso = False

            portfolio[idx] = cash + shares_gld * curr_gld + shares_uso * curr_uso

        rets = pd.Series(portfolio[start_idx:]).pct_change().fillna(0.0)
        metrics = calculate_performance_metrics(rets)

        is_proposed = (perm == ("DQN", "PPO", "DDPG", "A2C"))

        results.append({
            "permutation": list(perm),
            "mapping": mapping_str,
            "is_proposed": is_proposed,
            "cumulative_return_pct": metrics["cumulative_return"] * 100,
            "annualized_return_pct": metrics["annualised_return"] * 100,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown_pct": metrics["max_drawdown"] * 100,
            "total_trades": trades
        })

    # Sort by Sharpe descending
    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    # Determine rank of proposed mapping
    proposed_rank = next(i+1 for i, r in enumerate(results) if r["is_proposed"])

    # Save report
    report = {
        "proposed_mapping_rank": proposed_rank,
        "total_permutations": 24,
        "results": results
    }
    report_path = Path("results/routing_permutation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Print table
    print("\n" + "=" * 120)
    print("          24-PERMUTATION ROUTING ABLATION — RANKED BY SHARPE RATIO")
    print("=" * 120)
    print(f"{'Rank':5s} | {'Mapping':55s} | {'Cum. Return':14s} | {'Sharpe':8s} | {'Max. DD':10s} | {'Note':12s}")
    print("-" * 120)
    for i, r in enumerate(results):
        note = "★ PROPOSED" if r["is_proposed"] else ""
        print(f"{i+1:5d} | {r['mapping']:55s} | {r['cumulative_return_pct']:12.2f}% | "
              f"{r['sharpe_ratio']:8.3f} | {r['max_drawdown_pct']:8.2f}% | {note}")
    print("-" * 120)
    print(f"Proposed mapping rank: {proposed_rank}/24")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    run_permutation_ablation()
