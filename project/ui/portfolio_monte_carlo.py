"""
Portfolio analysis page with a market-dashboard layout.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import kurtosis, skew
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def _interpolate_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    rgb = (
        int(round(r1 + (r2 - r1) * t)),
        int(round(g1 + (g2 - g1) * t)),
        int(round(b1 + (b2 - b1) * t)),
    )
    return _rgb_to_hex(rgb)

def ranked_blue_map(values: pd.Series, darkest: str = "#104861", lightest: str = "#d9f0ff") -> dict:
    """Map values to ranked shades of blue (highest = darkest)."""
    vals = values.fillna(0.0).astype(float)
    if vals.empty:
        return {}
    sorted_idx = vals.sort_values(ascending=True).index.tolist()
    n = len(sorted_idx)
    if n == 1:
        return {sorted_idx[0]: darkest}

    color_map = {}
    for rank, key in enumerate(sorted_idx):
        t = rank / (n - 1)
        color_map[key] = _interpolate_hex(lightest, darkest, t)
    return color_map

def inject_custom_css():
    """Inject custom CSS for portfolio page styling."""
    st.markdown(
        """
        <style>
        .portfolio-hero {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(67, 89, 119, 0.28);
            background:
                radial-gradient(circle at 8% 12%, rgba(31, 119, 180, 0.18), transparent 42%),
                radial-gradient(circle at 92% 8%, rgba(17, 94, 89, 0.14), transparent 38%),
                linear-gradient(160deg, rgba(18, 30, 49, 0.06), rgba(12, 19, 31, 0.03));
            margin-bottom: 1rem;
        }
        .portfolio-hero-title {
            margin: 0;
            font-size: 1.45rem;
            letter-spacing: 0.2px;
        }
        .portfolio-hero-subtitle {
            margin-top: 0.3rem;
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .portfolio-badge-row {
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .portfolio-badge {
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.2px;
            border: 1px solid rgba(31, 119, 180, 0.35);
            background: rgba(31, 119, 180, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def plot_efficient_frontier(mc_risk, mc_ret, curve_risk, curve_ret, port_vol, port_return, eq_grid):
    """Create efficient frontier visualization."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.array(mc_risk) * 100,
            y=np.array(mc_ret) * 100,
            mode="markers",
            marker=dict(
                size=2.5,
                color=np.array(mc_risk) * 100,
                colorscale="RdYlBu_r",
                opacity=0.5,
                showscale=False,
            ),
            name="Simulations",
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(curve_risk) * 100,
            y=np.array(curve_ret) * 100,
            mode="lines",
            line=dict(color="#1f77b4", width=3.4),
            name="Efficient Frontier",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[port_vol * 100],
            y=[port_return * 100],
            mode="markers",
            marker=dict(size=18, color="gold", symbol="star", line=dict(color="black", width=2)),
            name="Your Portfolio",
        )
    )

    label_points = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for lp in label_points:
        i = np.argmin(np.abs(eq_grid - lp))
        fig.add_annotation(
            x=curve_risk[i] * 100,
            y=curve_ret[i] * 100,
            text=f"{int(lp * 100)}%",
            showarrow=False,
            font=dict(size=9, color="gray"),
            opacity=0.8,
            yshift=-12,
        )

    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=495,
        hovermode="closest",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=55, r=95, t=30, b=55),
    )
    return fig

def plot_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    color_map: dict = None,
) -> go.Figure:
    """Plot risk contribution by asset."""
    portfolio_variance = weights @ cov_matrix @ weights
    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / portfolio_variance
    risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

    risk_color_map = color_map if isinstance(color_map, dict) else ranked_blue_map(risk_contrib_pct)
    colors = [risk_color_map.get(asset, "#1f77b4") for asset in risk_contrib_pct.index]
    fig = go.Figure(
        go.Bar(
            x=risk_contrib_pct.index,
            y=risk_contrib_pct.values,
            marker=dict(color=colors[: len(risk_contrib_pct)], line=dict(color="rgba(0,0,0,0.1)", width=0.5)),
            text=[f"{v:.1f}%" for v in risk_contrib_pct.values],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis=dict(title="Asset", showgrid=False),
        yaxis=dict(
            title="Risk Contribution (%)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
            ticksuffix="%",
            tickfont=dict(color="#ffffff"),
        ),
        xaxis_title="Asset",
        yaxis_title="Risk Contribution (%)",
        height=360,
        template="plotly_white",
        margin=dict(l=20, r=20, t=10, b=20),
    )
    return fig

def plot_rolling_volatility(returns: pd.Series, window: int = 252) -> go.Figure:
    """Plot rolling volatility."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values * 100,
            name="Rolling Volatility (1Y)",
            line=dict(color="#EF553B", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(239, 85, 59, 0.2)",
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=495,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=30, t=10, b=50),
    )
    return fig

def plot_cumulative_returns(returns: pd.Series, benchmark_returns: pd.Series = None) -> go.Figure:
    """Plot cumulative returns vs benchmark."""
    cum_returns = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=(cum_returns - 1) * 100,
            name="Portfolio",
            line=dict(color="#1f77b4", width=3),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.1)",
        )
    )
    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum_bench.index,
                y=(cum_bench - 1) * 100,
                name="Benchmark (SPY)",
                line=dict(color="orange", width=2.5, dash="dash"),
            )
        )

    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=360,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="left", x=0),
        margin=dict(l=50, r=30, t=10, b=50),
    )
    return fig

def plot_allocation_donut(weights: pd.Series, color_map: dict = None) -> go.Figure:
    """Portfolio weights donut chart."""
    labels = []
    values = []
    colors = []

    weight_pct = weights * 100
    asset_color_map = color_map if isinstance(color_map, dict) else ranked_blue_map(weight_pct)

    for i, asset in enumerate(weights.index):
        val = float(weights.loc[asset]) * 100
        if val <= 0:
            continue
        labels.append(asset)
        values.append(val)
        colors.append(asset_color_map.get(asset, "#1f77b4"))

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(width=0)),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig

def plot_correlation_heatmap(asset_returns: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for portfolio assets."""
    corr = asset_returns.corr().fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr"),
            hovertemplate="%{y} vs %{x}<br>Corr: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        height=430,
        template="plotly_white",
        margin=dict(l=35, r=25, t=10, b=30),
    )
    return fig

def plot_drawdown(returns: pd.Series) -> go.Figure:
    """Portfolio drawdown chart."""
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            name="Drawdown",
            line=dict(color="#dc2626", width=2.3),
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.2)",
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=360,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=30, t=10, b=50),
    )
    return fig

def plot_return_distribution(returns: pd.Series) -> go.Figure:
    """Daily returns distribution."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns.values * 100,
            nbinsx=70,
            marker=dict(color="#2563eb", line=dict(color="white", width=0.3)),
            opacity=0.9,
            name="Daily Returns",
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=360,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=50, r=30, t=10, b=45),
    )
    return fig

def render_portfolio_monte_carlo():
    """Render the Portfolio page."""
    inject_custom_css()
    st.markdown(
        '<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:4px;">Portfolio</h1>',
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        if "portfolio_stocks_input" not in st.session_state:
            st.session_state["portfolio_stocks_input"] = "ASML\nCVX\nGOOGL\nMSFT\nSTRL\nTSM"
        # Auto-size the textarea so the ticker list stays visible without internal scrolling.
        current_ticker_text = st.session_state.get("portfolio_stocks_input", "")
        visible_lines = max(8, min(28, len([ln for ln in current_ticker_text.splitlines() if ln.strip()]) + 3))
        stocks_input = st.text_area(
            "Stock Tickers",
            key="portfolio_stocks_input",
            height=visible_lines * 24,
        )
        bonds_input = st.text_input("Bond Ticker", value="VGIT")

        # As requested: MC + Years above Risk Appetite
        col_a, col_b = st.columns(2)
        with col_a:
            mc_sims = st.number_input("MC Simulations", 1000, 10000, 3000, 500)
        with col_b:
            years_back = st.number_input("Years of Data", 5, 30, 25, 1)

        risk_appetite = st.slider(
            "Risk Appetite",
            0.0,
            1.0,
            0.75,
            0.05,
            help="0 = Conservative (bonds only), 1 = Aggressive (equity only)",
        )

        stocks = [s.strip().upper() for s in stocks_input.split("\n") if s.strip()]
        bonds = [bonds_input.strip().upper()] if bonds_input.strip() else []
        assets = stocks + bonds

        st.markdown("#### Expected Returns")
        n_cols = min(len(assets), 3) if assets else 1
        dcf_returns = {}
        for i in range(0, len(assets), n_cols):
            cols = st.columns(n_cols)
            for j, asset in enumerate(assets[i : i + n_cols]):
                with cols[j]:
                    default_val = 0.12 if asset in stocks else 0.03
                    dcf_returns[asset] = st.number_input(
                        asset,
                        0.0,
                        1.0,
                        default_val,
                        0.01,
                        format="%.2f",
                        key=f"dcf_{asset}",
                    )

        run_button = st.button("Run Optimization", type="primary", width="stretch")

    with right_col:
        st.markdown("#### Holdings Mix")
        holdings_mix_slot = st.empty()
        cached_weights = st.session_state.get("portfolio_last_weights")
        if isinstance(cached_weights, dict) and cached_weights:
            w_cached = pd.Series(cached_weights, dtype=float)
            w_cached = w_cached[w_cached > 0]
            if not w_cached.empty:
                shared_color_map = ranked_blue_map(w_cached * 100)
                with holdings_mix_slot.container():
                    st.plotly_chart(plot_allocation_donut(w_cached, color_map=shared_color_map), width="stretch")
            else:
                holdings_mix_slot.info("Run optimization to generate contribution chart.")
        else:
            holdings_mix_slot.info("Run optimization to generate contribution chart.")

        st.markdown("#### Contribution to Portfolio Risk")
        risk_contrib_slot = st.empty()
        risk_contrib_slot.info("Run optimization to generate contribution chart.")

    if not assets:
        st.warning("Enter at least one ticker.")
        return

    if not run_button:
        return

    with st.spinner("Fetching market data and optimizing portfolio..."):
        try:
            risk_target = risk_appetite
            cdar_alpha = 0.05
            trading_days = 252
            mc_simulations = mc_sims
            eps = 1e-9

            start = (datetime.today() - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d")
            end = datetime.today().strftime("%Y-%m-%d")
            os.environ["YF_DISABLE_PROTOBUF"] = "1"

            def get_prices_safe(tickers, start_date, end_date):
                df_list = []
                for ticker in tickers:
                    try:
                        df = yf.Ticker(ticker).history(start=start_date, end=end_date)["Close"]
                        df_list.append(df.rename(ticker))
                    except Exception as exc:
                        st.warning(f"Could not fetch {ticker}: {exc}")
                if not df_list:
                    raise ValueError("No tickers could be fetched.")
                return pd.concat(df_list, axis=1)

            prices = get_prices_safe(assets, start, end)
            returns = prices.pct_change().dropna()

            stocks = [s for s in stocks if s in returns.columns]
            bonds = [b for b in bonds if b in returns.columns]
            assets = stocks + bonds
            if not assets:
                raise ValueError("No valid assets found in fetched data.")

            n_obs = len(returns)
            hist_ann_returns = (1 + returns).prod() ** (trading_days / n_obs) - 1
            if all(a in dcf_returns for a in assets):
                exp_returns = pd.Series(dcf_returns)
            else:
                exp_returns = 0.3 * hist_ann_returns + 0.7 * hist_ann_returns.mean()

            cov = LedoitWolf().fit(returns[assets]).covariance_ * trading_days
            cov = pd.DataFrame(cov, index=assets, columns=assets)
            market_ret = returns[stocks].mean(axis=1) if stocks else returns[assets].mean(axis=1)

            def downside_beta(asset_ret, market_series):
                down = market_series < 0
                if down.sum() < 2:
                    return 0.0
                return np.cov(asset_ret[down], market_series[down])[0, 1] / np.var(market_series[down])

            def cdar(x, alpha=cdar_alpha):
                equity = (1 + x).cumprod()
                dd = equity / equity.cummax() - 1
                tail = dd[dd <= dd.quantile(alpha)]
                return abs(tail.mean()) if len(tail) else 0.0

            def cvar(x, alpha=cdar_alpha):
                x_sorted = np.sort(x)
                n = max(int(alpha * len(x_sorted)), 1)
                return -x_sorted[:n].mean()

            asset_downbeta = returns[assets].apply(lambda x: downside_beta(x, market_ret))
            asset_cdar = returns[assets].apply(cdar)
            asset_cvar = returns[assets].apply(cvar)
            asset_vol = returns[assets].std() * np.sqrt(trading_days)
            asset_risk = 0.4 * asset_cdar + 0.4 * asset_cvar + 0.2 * asset_vol + 0.2 * asset_downbeta

            if stocks:
                stock_mix = (exp_returns[stocks] / (asset_risk[stocks] + eps)).clip(lower=0)
                stock_mix = (
                    pd.Series(1 / len(stocks), index=stocks)
                    if stock_mix.sum() <= 0
                    else stock_mix / stock_mix.sum()
                )
            else:
                stock_mix = pd.Series(dtype=float)

            eq_grid = np.linspace(0, 1, 60)
            curve_weights = []
            curve_ret, curve_risk = [], []
            for ew in eq_grid:
                w = pd.Series(0.0, index=assets)
                if stocks:
                    w[stocks] = stock_mix * ew
                if bonds:
                    w[bonds] = (1 - ew) / len(bonds)
                elif stocks:
                    w[stocks] = w[stocks] + (1 - ew) * stock_mix
                curve_weights.append(w)
                curve_ret.append(w @ exp_returns[assets])
                curve_risk.append(np.sqrt(w @ cov @ w))

            idx = int(risk_target * (len(eq_grid) - 1))
            weights = curve_weights[idx]
            weights_nonzero = weights[weights > 0]
            st.session_state["portfolio_last_weights"] = weights_nonzero.to_dict()
            shared_color_map = ranked_blue_map(weights_nonzero * 100)

            with holdings_mix_slot.container():
                st.plotly_chart(
                    plot_allocation_donut(weights_nonzero, color_map=shared_color_map),
                    width="stretch",
                )

            with risk_contrib_slot.container():
                st.plotly_chart(
                    plot_risk_contribution(weights, cov, color_map=shared_color_map),
                    width="stretch",
                )

            equity_weight = weights[stocks].sum() if stocks else 0.0
            bond_weight = weights[bonds].sum() if bonds else 0.0

            port_return = weights @ exp_returns[assets]
            port_vol = np.sqrt(weights @ cov @ weights)
            port_series = returns[assets] @ weights
            port_cdar = cdar(port_series)
            port_cvar = cvar(port_series)
            annual_ret = port_series.mean() * 252
            sharpe_ratio = (port_return - 0.02) / port_vol

            mc_risk, mc_ret = [], []
            for _ in range(mc_simulations):
                ew = np.random.rand()
                w = pd.Series(0.0, index=assets)
                if stocks:
                    sw = np.random.dirichlet(np.ones(len(stocks)))
                    w[stocks] = sw * ew
                if bonds:
                    w[bonds] = (1 - ew) / len(bonds)
                elif stocks:
                    w[stocks] = w[stocks] + (1 - ew) * sw
                mc_ret.append(w @ exp_returns[assets])
                mc_risk.append(np.sqrt(w @ cov @ w))

            st.markdown("---")
            metric_cols = st.columns(6)
            with metric_cols[0]:
                st.metric("Expected Return", f"{port_return * 100:.2f}%")
            with metric_cols[1]:
                st.metric("Historical Return", f"{annual_ret * 100:.2f}%")
            with metric_cols[2]:
                st.metric("Volatility", f"{port_vol * 100:.2f}%")
            with metric_cols[3]:
                st.metric("Sharpe", f"{sharpe_ratio:.2f}")
            with metric_cols[4]:
                st.metric("CDaR (5%)", f"{port_cdar * 100:.2f}%")
            with metric_cols[5]:
                st.metric("CVaR (5%)", f"{port_cvar * 100:.2f}%")

            st.markdown("")
            chart_left, chart_right = st.columns([1, 1], gap="medium")
            with chart_left:
                st.markdown("#### Efficient Frontier")
                st.plotly_chart(
                    plot_efficient_frontier(mc_risk, mc_ret, curve_risk, curve_ret, port_vol, port_return, eq_grid),
                    width="stretch",
                )
            with chart_right:
                st.markdown("#### Rolling Volatility")
                st.plotly_chart(plot_rolling_volatility(port_series), width="stretch")

            st.markdown("")
            st.markdown("#### Cumulative Performance vs Benchmark")
            try:
                spy = yf.Ticker("SPY").history(start=start, end=end)["Close"]
                spy_returns = spy.pct_change().dropna().reindex(port_series.index, fill_value=0)
                fig_cum = plot_cumulative_returns(port_series, spy_returns)
            except Exception:
                fig_cum = plot_cumulative_returns(port_series)
            st.plotly_chart(fig_cum, width="stretch")

        except Exception as exc:
            st.error(f"Error: {str(exc)}")
            with st.expander("Show Details"):
                import traceback

                st.code(traceback.format_exc())
