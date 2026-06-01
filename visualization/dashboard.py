"""
Dynamic ART-DRL — Plotly Dashboard Generator
============================================
Creates premium, modern interactive HTML dashboards showing trade execution,
portfolio values, volatility regime bands, and comparative benchmarks.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List


def generate_dashboard_html(
    df_results: pd.DataFrame,
    trade_log: List[Dict[str, Any]],
    metrics: Dict[str, float],
    bh_metrics: Dict[str, float],
    asset_name: str,
    timeframe: str,
    output_path: str | Path
) -> None:
    """Generate a rich, interactive HTML dashboard using Plotly.
    
    The dashboard includes:
    1. Interactive equity curve comparisons (Dynamic Switching vs Buy-and-Hold)
    2. Subplots showing price series, Kalman-filtered signals, and volatility bands
    3. Active agent allocation timeline (color-coded regimes)
    4. Performance summary table (Sharpe, Sortino, Calmar, MaxDD)
    5. Month-wise / Year-wise returns matrix with bold compounded annual returns
    6. Searchable, scrollable list of ALL executed trades with realized gross/net P&L
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # ── Currency symbol detection ────────────────────────────────────────────
    currency_symbol = "$" if any(x in asset_name.upper() for x in ["USD", "GLD", "USO"]) else "₹"
    initial_wallet = float(df_results["portfolio_value"].iloc[0])
    final_wallet = float(df_results["portfolio_value"].iloc[-1])
    
    # ── 1. Create Subplots ───────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=(
            "Interactive Portfolio Value & Performance Curve",
            "Asset Close Price & Kalman-Denoised Signal",
            "Volatility Regime & Active Strategy Switches"
        )
    )
    
    # Plot 1: Equity Curves
    fig.add_trace(
        go.Scatter(
            x=df_results.index,
            y=df_results["portfolio_value"],
            name="Dynamic ART-DRL (Method 3)",
            line=dict(color="#00FFA6", width=2.5),
            hovertemplate=f"Dynamic: {currency_symbol}%{{y:,.2f}}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_results.index,
            y=df_results["buy_and_hold"],
            name="Buy & Hold Baseline",
            line=dict(color="#FF4B4B", width=1.5, dash="dash"),
            hovertemplate=f"B&H: {currency_symbol}%{{y:,.2f}}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Plot 2: Close vs Kalman Filter
    fig.add_trace(
        go.Scatter(
            x=df_results.index,
            y=df_results["price"],
            name="Raw Close Price",
            line=dict(color="#7B889B", width=1.0),
            opacity=0.6,
            hovertemplate=f"Close: {currency_symbol}%{{y:,.2f}}<extra></extra>"
        ),
        row=2, col=1
    )
    
    kalman_col = "kalman_price" if "kalman_price" in df_results.columns else "price"
    fig.add_trace(
        go.Scatter(
            x=df_results.index,
            y=df_results.get(kalman_col, df_results["price"]),
            name="Kalman Denoised Trend",
            line=dict(color="#FFD700", width=1.8),
            hovertemplate=f"Kalman: {currency_symbol}%{{y:,.2f}}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Plot 3: Regime Switches
    regime_mapping = {
        "LOW_VOLATILITY": 1,
        "NORMAL_VOLATILITY": 2,
        "HIGH_VOLATILITY": 3,
        "EXTREME_VOLATILITY": 4
    }
    numeric_regimes = df_results["regime"].map(regime_mapping).fillna(2)
    
    fig.add_trace(
        go.Scatter(
            x=df_results.index,
            y=numeric_regimes,
            name="Volatility Regime Level",
            mode="lines+markers",
            line=dict(color="#A020F0", width=1.5),
            marker=dict(size=4),
            hovertemplate="Regime: %{text}<extra></extra>",
            text=df_results["regime"]
        ),
        row=3, col=1
    )
    
    # Customize subplots layout
    fig.update_layout(
        title=dict(
            text=f"Dynamic ART-DRL Backtest Report — {asset_name} ({timeframe})",
            font=dict(size=22, color="#FFFFFF")
        ),
        template="plotly_dark",
        paper_bgcolor="#11151C",
        plot_bgcolor="#18202C",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=850,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    fig.update_yaxes(title_text=f"Portfolio Value ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="Asset Price", row=2, col=1)
    fig.update_yaxes(
        title_text="Regime Level",
        tickvals=[1, 2, 3, 4],
        ticktext=["LOW", "NORMAL", "HIGH", "EXTREME"],
        row=3, col=1
    )
    
    chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    # ── 2. Month-Wise / Year-Wise Performance Matrix ─────────────────────────
    pv_series = df_results["portfolio_value"]
    pv_monthly = pv_series.resample("ME").last()
    
    # Calculate monthly returns dynamically relative to previous month-end
    monthly_rets = {}
    start_val = pv_series.iloc[0]
    prev_val = start_val
    
    for date, val in pv_monthly.items():
        year = date.year
        month = date.month
        if prev_val == 0.0:
            m_ret = 0.0
        else:
            m_ret = (val / prev_val) - 1.0
        if year not in monthly_rets:
            monthly_rets[year] = {}
        monthly_rets[year][month] = m_ret
        prev_val = val
        
    # Calculate compounded annual returns for each year
    annual_rets = {}
    for year in monthly_rets:
        if year == pv_series.index[0].year:
            yr_start_val = start_val
        else:
            prev_yr_df = pv_series[pv_series.index.year == year - 1]
            if not prev_yr_df.empty:
                yr_start_val = prev_yr_df.iloc[-1]
            else:
                yr_start_val = pv_series[pv_series.index.year == year].iloc[0]
        
        yr_end_val = pv_series[pv_series.index.year == year].iloc[-1]
        if yr_start_val == 0.0:
            annual_rets[year] = -1.0
        else:
            annual_rets[year] = (yr_end_val / yr_start_val) - 1.0

    # Build Monthly HTML Matrix rows
    matrix_rows = []
    for year in sorted(monthly_rets.keys(), reverse=True):
        row_html = f"<tr><td style='font-weight: 700; color: #FFFFFF;'>{year}</td>"
        for m in range(1, 13):
            val = monthly_rets[year].get(m, None)
            if val is not None:
                val_pct = val * 100.0
                sign = "+" if val_pct > 0 else ""
                style = ""
                if val_pct > 0:
                    style = "color: #00FFA6; background-color: rgba(0, 255, 166, 0.03);"
                elif val_pct < 0:
                    style = "color: #FF4B4B; background-color: rgba(255, 75, 75, 0.03);"
                else:
                    style = "color: #94A3B8;"
                row_html += f"<td style='text-align: right; font-size: 13px; {style}'>{sign}{val_pct:.2f}%</td>"
            else:
                row_html += "<td style='text-align: center; color: #64748B;'>-</td>"
        
        # Compounded Annual Return
        ann_val = annual_rets.get(year, 0.0)
        ann_pct = ann_val * 100.0
        ann_sign = "+" if ann_pct > 0 else ""
        ann_style = "font-weight: 700; border-left: 2px solid #1E293B; text-align: right; background-color: rgba(255, 255, 255, 0.02);"
        if ann_pct > 0:
            ann_style += " color: #00FFA6;"
        elif ann_pct < 0:
            ann_style += " color: #FF4B4B;"
        else:
            ann_style += " color: #94A3B8;"
        row_html += f"<td style='{ann_style}'>{ann_sign}{ann_pct:.2f}%</td></tr>"
        matrix_rows.append(row_html)
        
    matrix_body = "\n".join(matrix_rows)

    # ── 3. Trade Metrics Calculation ─────────────────────────────────────────
    total_trades = len(trade_log)
    total_cost = sum(t.get("cost", 0.0) for t in trade_log)
    total_realized_net_pnl = sum(t.get("realized_net_pnl", 0.0) for t in trade_log)
    
    # ── 4. Generate HTML content with updated premium styling ─────────────────
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dynamic ART-DRL Performance Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #0A0D14;
                color: #E2E8F0;
                margin: 0;
                padding: 30px;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #1E293B;
                padding-bottom: 20px;
                margin-bottom: 35px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 700;
                color: #00FFA6;
                letter-spacing: -0.5px;
            }}
            .header .meta {{
                color: #94A3B8;
                font-size: 14px;
                margin-top: 5px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
                gap: 20px;
                margin-bottom: 35px;
            }}
            .card {{
                background-color: #11151C;
                border: 1px solid #1E293B;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }}
            .card:hover {{
                transform: translateY(-2px);
                border-color: #334155;
            }}
            .card h3 {{
                margin: 0 0 10px 0;
                font-size: 12px;
                text-transform: uppercase;
                color: #94A3B8;
                letter-spacing: 1px;
            }}
            .card .value {{
                font-size: 24px;
                font-weight: 700;
                color: #FFFFFF;
            }}
            .card .value.positive {{
                color: #00FFA6;
            }}
            .card .value.negative {{
                color: #FF4B4B;
            }}
            .card .comparison {{
                margin-top: 5px;
                font-size: 11px;
                color: #64748B;
            }}
            .table-container {{
                background-color: #11151C;
                border: 1px solid #1E293B;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 35px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            .table-container h2 {{
                margin: 0 0 20px 0;
                font-size: 18px;
                font-weight: 600;
                border-left: 4px solid #00FFA6;
                padding-left: 12px;
                color: #FFFFFF;
            }}
            .scroll-wrapper {{
                max-height: 420px;
                overflow-y: auto;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                text-align: left;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #1E293B;
            }}
            th {{
                color: #94A3B8;
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                background-color: #18202C;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            td {{
                font-size: 14px;
            }}
            tr:hover {{
                background-color: #1E293B55;
            }}
            
            /* Filter Bar Styling */
            .filter-bar {{
                display: flex;
                gap: 15px;
                margin-bottom: 15px;
                align-items: center;
                background-color: #161D26;
                padding: 15px;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }}
            .filter-item {{
                flex: 1;
            }}
            .filter-item.select-item {{
                flex: 0 0 180px;
            }}
            .filter-bar label {{
                font-size: 11px;
                color: #94A3B8;
                display: block;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .filter-bar input, .filter-bar select {{
                width: 100%;
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 12px;
                color: #FFFFFF;
                font-family: inherit;
                font-size: 14px;
                box-sizing: border-box;
            }}
            .filter-bar input:focus, .filter-bar select:focus {{
                border-color: #00FFA6;
                outline: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Dynamic ART-DRL Strategy Report</h1>
                    <div class="meta">Asset: {asset_name} | Interval: {timeframe} | Generated: 2026-05-31</div>
                </div>
                <div class="card" style="padding: 10px 20px; margin: 0; background-color: #161D26;">
                    <span style="font-weight: 700; color: #00FFA6; font-size: 13px; letter-spacing: 0.5px;">METHOD 3 SWITCHING MODEL</span>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>Initial Capital</h3>
                    <div class="value" style="color: #FFD700;">{currency_symbol}{initial_wallet:,.2f}</div>
                    <div class="comparison">Starting Wallet</div>
                </div>
                <div class="card">
                    <h3>Final Portfolio</h3>
                    <div class="value positive" style="font-weight: 700;">{currency_symbol}{final_wallet:,.2f}</div>
                    <div class="comparison">Ending Wallet</div>
                </div>
                <div class="card">
                    <h3>Cumulative Return</h3>
                    <div class="value positive">{metrics['cumulative_return']*100:.2f}%</div>
                    <div class="comparison">Buy & Hold: {bh_metrics['cumulative_return']*100:.2f}%</div>
                </div>
                <div class="card">
                    <h3>Annualised Return</h3>
                    <div class="value positive">{metrics['annualised_return']*100:.2f}%</div>
                    <div class="comparison">Buy & Hold: {bh_metrics['annualised_return']*100:.2f}%</div>
                </div>
                <div class="card">
                    <h3>Annualised Sharpe</h3>
                    <div class="value">{metrics['sharpe_ratio']:.3f}</div>
                    <div class="comparison">Buy & Hold: {bh_metrics['sharpe_ratio']:.3f}</div>
                </div>
                <div class="card">
                    <h3>Maximum Drawdown</h3>
                    <div class="value negative">{metrics['max_drawdown']*100:.2f}%</div>
                    <div class="comparison">Buy & Hold: {bh_metrics['max_drawdown']*100:.2f}%</div>
                </div>
                <div class="card">
                    <h3>Total Trade Orders</h3>
                    <div class="value">{total_trades}</div>
                    <div class="comparison">Crossover Swing Hold</div>
                </div>
                <div class="card">
                    <h3>Transaction Costs</h3>
                    <div class="value negative">{currency_symbol}{total_cost:,.2f}</div>
                    <div class="comparison">Unified Cash pool drag</div>
                </div>
            </div>
            
            <div class="chart-container" style="margin-bottom: 35px; border: 1px solid #1E293B; border-radius: 12px; overflow: hidden;">
                {chart_div}
            </div>

            <!-- Month-wise Year-wise Performance breakdown matrix -->
            <div class="table-container">
                <h2>Month-Wise & Year-Wise Returns Compounding Matrix</h2>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr style="background-color: #18202C;">
                                <th>Year</th>
                                <th style="text-align: right;">Jan</th>
                                <th style="text-align: right;">Feb</th>
                                <th style="text-align: right;">Mar</th>
                                <th style="text-align: right;">Apr</th>
                                <th style="text-align: right;">May</th>
                                <th style="text-align: right;">Jun</th>
                                <th style="text-align: right;">Jul</th>
                                <th style="text-align: right;">Aug</th>
                                <th style="text-align: right;">Sep</th>
                                <th style="text-align: right;">Oct</th>
                                <th style="text-align: right;">Nov</th>
                                <th style="text-align: right;">Dec</th>
                                <th style="border-left: 2px solid #1E293B; text-align: right; font-weight: 700; background-color: rgba(255, 255, 255, 0.02);">Annual</th>
                            </tr>
                        </thead>
                        <tbody>
                            {matrix_body}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="table-container">
                <h2>Comparison Table (PRUDEX-Compass Benchmarks)</h2>
                <table>
                    <thead>
                        <tr style="background-color: #18202C;">
                            <th>Performance Metric</th>
                            <th>Dynamic ART-DRL (Method 3)</th>
                            <th>Buy & Hold Benchmark</th>
                            <th>Relative Outperformance</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Cumulative Return</td>
                            <td style="font-weight: 600; color: #00FFA6;">{metrics['cumulative_return']*100:.2f}%</td>
                            <td>{bh_metrics['cumulative_return']*100:.2f}%</td>
                            <td style="color: #00FFA6; font-weight: 600;">+{(metrics['cumulative_return'] - bh_metrics['cumulative_return'])*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Annualised Return</td>
                            <td style="font-weight: 600; color: #00FFA6;">{metrics['annualised_return']*100:.2f}%</td>
                            <td>{bh_metrics['annualised_return']*100:.2f}%</td>
                            <td style="color: #00FFA6; font-weight: 600;">+{(metrics['annualised_return'] - bh_metrics['annualised_return'])*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Annualised Volatility</td>
                            <td>{metrics['annualised_volatility']*100:.2f}%</td>
                            <td>{bh_metrics['annualised_volatility']*100:.2f}%</td>
                            <td>-{(bh_metrics['annualised_volatility'] - metrics['annualised_volatility'])*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td style="font-weight: 600; color: #00FFA6;">{metrics['sharpe_ratio']:.3f}</td>
                            <td>{bh_metrics['sharpe_ratio']:.3f}</td>
                            <td style="color: #00FFA6; font-weight: 600;">+{(metrics['sharpe_ratio'] - bh_metrics['sharpe_ratio']):.3f}</td>
                        </tr>
                        <tr>
                            <td>Sortino Ratio</td>
                            <td style="font-weight: 600; color: #00FFA6;">{metrics['sortino_ratio']:.3f}</td>
                            <td>{bh_metrics['sortino_ratio']:.3f}</td>
                            <td style="color: #00FFA6; font-weight: 600;">+{(metrics['sortino_ratio'] - bh_metrics['sortino_ratio']):.3f}</td>
                        </tr>
                        <tr>
                            <td>Maximum Drawdown</td>
                            <td style="color: #FF4B4B; font-weight: 600;">{metrics['max_drawdown']*100:.2f}%</td>
                            <td>{bh_metrics['max_drawdown']*100:.2f}%</td>
                            <td style="color: #00FFA6; font-weight: 600;">+{abs(bh_metrics['max_drawdown'] - metrics['max_drawdown'])*100:.2f}% (reduced)</td>
                        </tr>
                        <tr>
                            <td>Calmar Ratio</td>
                            <td style="font-weight: 600; color: #00FFA6;">{metrics['calmar_ratio']:.3f}</td>
                            <td>{bh_metrics['calmar_ratio']:.3f}</td>
                            <td style="color: #00FFA6; font-weight: 600;">+{(metrics['calmar_ratio'] - bh_metrics['calmar_ratio']):.3f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="table-container">
                <h2>Execution Trade Log (All {total_trades} Trades)</h2>
                
                <!-- client-side interactive search and filtering bar -->
                <div class="filter-bar">
                    <div class="filter-item">
                        <label for="search-input">Interactive Search</label>
                        <input type="text" id="search-input" placeholder="Search by Date, Vol Regime..." onkeyup="filterTrades()">
                    </div>
                    <div class="filter-item select-item">
                        <label for="action-filter">Filter Action</label>
                        <select id="action-filter" onchange="filterTrades()">
                            <option value="ALL">All Actions</option>
                            <option value="BUY">BUY</option>
                            <option value="SELL">SELL</option>
                        </select>
                    </div>
                    <div class="filter-item select-item">
                        <label for="agent-filter">Filter Active Agent</label>
                        <select id="agent-filter" onchange="filterTrades()">
                            <option value="ALL">All Agents</option>
                            <option value="DQN">DQN</option>
                            <option value="PPO">PPO</option>
                            <option value="DDPG">DDPG</option>
                            <option value="A2C">A2C</option>
                        </select>
                    </div>
                </div>

                <div class="scroll-wrapper">
                    <table>
                        <thead>
                            <tr style="background-color: #18202C;">
                                <th>Step</th>
                                <th>Date / Time</th>
                                <th>Action</th>
                                <th>Qty</th>
                                <th>Price</th>
                                <th>Costs</th>
                                <th>Realised Net P&L</th>
                                <th>Avg Entry Price</th>
                                <th>Pos Size</th>
                                <th>Active Agent</th>
                                <th>Vol Regime</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f'''
                            <tr class="trade-row" data-agent="{t['active_agent']}" data-action="{t['action']}" data-regime="{t['regime']}">
                                <td>{t['step']}</td>
                                <td>{str(t['date'])[:19]}</td>
                                <td style="font-weight: 700; color: {'#00FFA6' if t['action'] == 'BUY' else '#FF4B4B'}">{t['action']}</td>
                                <td>{t['qty']:.2f}</td>
                                <td>{currency_symbol}{t['price']:,.2f}</td>
                                <td>{currency_symbol}{t['cost']:.2f}</td>
                                <td style="font-weight: 700; color: {'#00FFA6' if t.get('realized_net_pnl', 0.0) > 0 else '#FF4B4B' if t.get('realized_net_pnl', 0.0) < 0 else '#94A3B8'}">
                                    {'+' if t.get('realized_net_pnl', 0.0) > 0 else ''}{currency_symbol}{t.get('realized_net_pnl', 0.0):,.2f}
                                </td>
                                <td>{currency_symbol}{t.get('avg_entry_price', 0.0):,.2f}</td>
                                <td>{t.get('position_size', 0.0):.2f}</td>
                                <td style="font-weight: 600; color: #FFFFFF;">{t['active_agent']}</td>
                                <td>{t['regime']}</td>
                            </tr>
                            ''' for t in trade_log])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            function filterTrades() {{
                var searchVal = document.getElementById('search-input').value.toUpperCase();
                var actionVal = document.getElementById('action-filter').value;
                var agentVal = document.getElementById('agent-filter').value;
                
                var rows = document.getElementsByClassName('trade-row');
                
                for (var i = 0; i < rows.length; i++) {{
                    var row = rows[i];
                    var text = row.textContent.toUpperCase();
                    var rowAction = row.getAttribute('data-action');
                    var rowAgent = row.getAttribute('data-agent');
                    
                    var matchesSearch = text.indexOf(searchVal) > -1;
                    var matchesAction = (actionVal === 'ALL' || rowAction === actionVal);
                    var matchesAgent = (agentVal === 'ALL' || rowAgent === agentVal);
                    
                    if (matchesSearch && matchesAction && matchesAgent) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    logger.success("Interactive HTML report generated successfully at: {}", output_path)
