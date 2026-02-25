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


# ── Performance-page style constants ─────────────────────────────────────────
UP      = "#00C805"
DOWN    = "#FF3B30"
BLUE    = "#0A7CFF"
ORANGE  = "#FF9F0A"
GREY    = "#6E6E73"
BORDER  = "#E5E5EA"

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff", family="'SF Pro Display','Segoe UI',sans-serif"),
)

def _inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {BORDER}; background: transparent !important; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 14px; padding: 10px 18px; color: #ffffff; border-radius: 8px 8px 0 0; background: transparent !important; }}
    .stTabs [aria-selected="true"] {{ color: {BLUE} !important; border-bottom: 2px solid {BLUE} !important; background: transparent !important; }}
    div[data-testid="metric-container"] {{ background: transparent !important; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px; }}
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label {{ color: #ffffff !important; }}
    div[data-testid="stMetricValue"] > div {{ color: #ffffff !important; }}
    div[data-testid="stExpander"] {{ background: transparent !important; border: 1px solid {BORDER} !important; border-radius: 10px; }}
    div[data-testid="stExpander"] > details > summary {{ color: #ffffff !important; font-weight: 600; }}
    div[data-testid="stExpander"] > details > summary svg {{ fill: #ffffff !important; }}
    div[data-testid="stAlert"] {{ background: transparent !important; }}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] div {{ color: #ffffff !important; }}
    .stRadio > label, .stMultiSelect > label {{ color: #ffffff !important; }}
    .stCaption p, small {{ color: #ffffff !important; opacity: 0.7; }}
    .stDownloadButton button {{ background: transparent !important; border: 1px solid {BORDER} !important; color: #ffffff !important; }}
    .block-container {{ background: transparent !important; }}
    p, span, label, div {{ color: #ffffff; }}
    .stSelectbox label {{ color: #ffffff !important; }}
    </style>""", unsafe_allow_html=True)


def _metric_card(label: str, value_str: str, color: str = "#ffffff", tooltip: str = "") -> str:
    return f"""
<div title="{tooltip}" style="background:transparent;border-radius:12px;padding:16px 18px;
     cursor:{'help' if tooltip else 'default'};border:1px solid {BORDER};height:100%;">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.05em;color:#ffffff;
              text-transform:uppercase;margin-bottom:6px;">{label}</div>
  <div style="font-size:26px;font-weight:700;line-height:1.1;color:{color};">{value_str}</div>
  {f'<div style="font-size:11px;color:#ffffff;margin-top:6px;line-height:1.4;opacity:0.7;">{tooltip}</div>' if tooltip else ''}
</div>"""


def _verdict_card(title: str, verdict: str, verdict_color: str, body: str) -> str:
    return f"""
<div style="background:transparent;border-radius:12px;padding:18px 20px;border:1px solid {BORDER};
     border-left:4px solid {verdict_color};margin-bottom:12px;">
  <div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:4px;">{title}</div>
  <div style="font-size:15px;font-weight:700;color:{verdict_color};margin-bottom:6px;">{verdict}</div>
  <div style="font-size:13px;color:#ffffff;opacity:0.85;line-height:1.6;">{body}</div>
</div>"""


def _section_header(title: str, subtitle: str = "") -> None:
    sub = f'<p style="color:#ffffff;opacity:0.7;font-size:14px;margin:2px 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div style="margin:24px 0 12px 0;"><h3 style="font-size:20px;font-weight:700;'
        f'color:#ffffff;margin:0;">{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def _divider() -> None:
    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:20px 0;">',
                unsafe_allow_html=True)


def _sharpe_verdict(s: float):
    if np.isnan(s): return "Not enough data", GREY
    if s >= 2.0:    return "Excellent", UP
    if s >= 1.0:    return "Good", UP
    if s >= 0.5:    return "Acceptable", ORANGE
    if s >= 0.0:    return "Below average", ORANGE
    return "Poor", DOWN


def _vol_verdict(v: float):
    if np.isnan(v): return "Not enough data", GREY
    if v < 0.10:    return "Very calm", UP
    if v < 0.18:    return "Moderate", UP
    if v < 0.28:    return "Quite volatile", ORANGE
    return "Very volatile", DOWN


def render_portfolio_monte_carlo(holdings: dict = None):
    """
    Render the Portfolio Optimizer.

    When called from inside the Performance page's Portfolio tab,
    pass the current holdings dict (from _current_holdings()) so that
    the stocks text area is pre-populated automatically.

    When called standalone (direct page route), holdings=None and
    the user fills in tickers manually.
    """
    # ── CSS (only injected when called standalone; Performance page already injects it) ──
    _inject_css()

    # ── Compact config styles ─────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    .cfg-label {{
        font-size: 11px; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; color: rgba(255,255,255,0.4);
        margin: 0 0 4px 0; line-height: 1;
    }}
    .cfg-hint {{
        font-size: 11px; color: rgba(255,255,255,0.32);
        margin: 2px 0 0 0; line-height: 1.35;
    }}
    .holdings-source-badge {{
        display: inline-block;
        font-size: 10px; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; padding: 2px 8px; border-radius: 20px;
        background: rgba(10,124,255,0.18); color: {BLUE};
        border: 1px solid rgba(10,124,255,0.35); margin-left: 8px;
        vertical-align: middle;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Auto-populate stocks from current holdings ────────────────────────────
    # If holdings are passed in (from the Performance page), seed the text area
    # with the tickers the user actually owns — but only on the first render or
    # when the holdings set has changed (so manual edits are not overwritten).
    if holdings:
        held_tickers = sorted(holdings.keys())
        held_key     = ",".join(held_tickers)
        prev_key     = st.session_state.get("_portfolio_holdings_key", "")

        if held_key != prev_key:
            # Holdings changed (or first render) — refresh the text area seed.
            st.session_state["portfolio_stocks_input"]  = "\n".join(held_tickers)
            st.session_state["_portfolio_holdings_key"] = held_key
        auto_sourced = True
    else:
        auto_sourced = False
        if "portfolio_stocks_input" not in st.session_state:
            st.session_state["portfolio_stocks_input"] = "ASML\nCVX\nGOOGL\nMSFT\nSTRL\nTSM"

    # ── Configuration panel (horizontal, compact) ─────────────────────────────
    with st.expander("⚙️  Configure optimizer", expanded=st.session_state.get("portfolio_last_results") is None):

        col_stocks, col_bond, col_risk, col_years = st.columns([2, 1, 2, 1], gap="medium")

        with col_stocks:
            badge = '<span class="holdings-source-badge">auto</span>' if auto_sourced else ""
            st.markdown(f'<p class="cfg-label">Stocks {badge}</p>', unsafe_allow_html=True)
            stocks_input = st.text_area(
                "stocks",
                key="portfolio_stocks_input",
                height=110,
                label_visibility="collapsed",
                placeholder="AAPL\nMSFT\nNVDA\n...",
            )
            if auto_sourced:
                st.markdown('<p class="cfg-hint">Pre-filled from your current holdings — edit freely</p>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<p class="cfg-hint">One ticker per line</p>', unsafe_allow_html=True)

        with col_bond:
            st.markdown('<p class="cfg-label">Bond hedge</p>', unsafe_allow_html=True)
            bonds_input = st.text_input(
                "bond", value="VGIT",
                label_visibility="collapsed",
                placeholder="VGIT",
            )
            st.markdown('<p class="cfg-hint">Leave blank for equity-only</p>', unsafe_allow_html=True)

        with col_risk:
            st.markdown('<p class="cfg-label">Risk appetite</p>', unsafe_allow_html=True)
            risk_appetite = st.slider(
                "risk", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                label_visibility="collapsed",
            )
            risk_pct   = int(risk_appetite * 100)
            risk_label = (
                "Very conservative" if risk_pct <= 20 else
                "Conservative"      if risk_pct <= 40 else
                "Balanced"          if risk_pct <= 60 else
                "Growth"            if risk_pct <= 80 else
                "Aggressive"
            )
            st.markdown(
                f'<p class="cfg-hint">{risk_label} &nbsp;·&nbsp; {risk_pct}% equity / {100-risk_pct}% bonds</p>',
                unsafe_allow_html=True,
            )

        with col_years:
            st.markdown('<p class="cfg-label">History</p>', unsafe_allow_html=True)
            years_back = st.slider(
                "years", min_value=5, max_value=30, value=25, step=1,
                label_visibility="collapsed",
            )
            st.markdown(f'<p class="cfg-hint">{years_back} yrs</p>', unsafe_allow_html=True)

        # ── Expected returns (tucked away — advanced) ─────────────────────────
        stocks = [s.strip().upper() for s in stocks_input.split("\n") if s.strip()]
        bonds  = [bonds_input.strip().upper()] if bonds_input.strip() else []
        assets = stocks + bonds
        mc_sims = 3000

        dcf_returns = {}
        if assets:
            st.markdown(
                '<p class="cfg-label" style="margin-top:12px;">Expected annual returns</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="cfg-hint" style="margin-bottom:8px;">Adjust if you have a thesis; defaults are 12% equities / 3% bonds</p>',
                unsafe_allow_html=True,
            )
            ret_cols = st.columns(min(len(assets), 6))
            for i, asset in enumerate(assets):
                with ret_cols[i % len(ret_cols)]:
                    default_val = 0.12 if asset in stocks else 0.03
                    dcf_returns[asset] = st.number_input(
                        asset,
                        min_value=0.0, max_value=1.0,
                        value=default_val, step=0.01,
                        format="%.2f",
                        key=f"dcf_{asset}",
                    )

        # ── Run button ────────────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        run_button = st.button("▶  Run Optimization", type="primary", use_container_width=False)

    if not assets:
        st.warning("Add at least one stock ticker on the left to get started.")
        return

    if not run_button and "portfolio_last_results" not in st.session_state:
        st.info("Set your holdings and preferences above, then click **Run Optimization**.")
        return

    # ── Run or restore from cache ─────────────────────────────────────────────
    results = st.session_state.get("portfolio_last_results")

    if run_button:
        with st.spinner("Fetching market data and optimizing…"):
            try:
                cdar_alpha   = 0.05
                trading_days = 252
                eps          = 1e-9

                start = (datetime.today() - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d")
                end   = datetime.today().strftime("%Y-%m-%d")
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

                prices  = get_prices_safe(assets, start, end)
                returns = prices.pct_change().dropna()

                stocks_clean = [s for s in stocks if s in returns.columns]
                bonds_clean  = [b for b in bonds  if b in returns.columns]
                assets_clean = stocks_clean + bonds_clean
                if not assets_clean:
                    raise ValueError("No valid assets found in fetched data.")

                n_obs           = len(returns)
                hist_ann_returns = (1 + returns).prod() ** (trading_days / n_obs) - 1
                exp_returns = (
                    pd.Series({a: dcf_returns[a] for a in assets_clean})
                    if all(a in dcf_returns for a in assets_clean)
                    else 0.3 * hist_ann_returns + 0.7 * hist_ann_returns.mean()
                )

                cov        = LedoitWolf().fit(returns[assets_clean]).covariance_ * trading_days
                cov        = pd.DataFrame(cov, index=assets_clean, columns=assets_clean)
                market_ret = returns[stocks_clean].mean(axis=1) if stocks_clean else returns[assets_clean].mean(axis=1)

                def downside_beta(asset_ret, market_series):
                    down = market_series < 0
                    if down.sum() < 2: return 0.0
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

                asset_downbeta = returns[assets_clean].apply(lambda x: downside_beta(x, market_ret))
                asset_cdar     = returns[assets_clean].apply(cdar)
                asset_cvar     = returns[assets_clean].apply(cvar)
                asset_vol      = returns[assets_clean].std() * np.sqrt(trading_days)
                asset_risk     = 0.4*asset_cdar + 0.4*asset_cvar + 0.2*asset_vol + 0.2*asset_downbeta

                if stocks_clean:
                    stock_mix = (exp_returns[stocks_clean] / (asset_risk[stocks_clean] + eps)).clip(lower=0)
                    stock_mix = (pd.Series(1/len(stocks_clean), index=stocks_clean)
                                 if stock_mix.sum() <= 0 else stock_mix / stock_mix.sum())
                else:
                    stock_mix = pd.Series(dtype=float)

                eq_grid       = np.linspace(0, 1, 60)
                curve_weights, curve_ret, curve_risk = [], [], []
                for ew in eq_grid:
                    w = pd.Series(0.0, index=assets_clean)
                    if stocks_clean: w[stocks_clean] = stock_mix * ew
                    if bonds_clean:  w[bonds_clean]  = (1 - ew) / len(bonds_clean)
                    elif stocks_clean: w[stocks_clean] = w[stocks_clean] + (1 - ew) * stock_mix
                    curve_weights.append(w)
                    curve_ret.append(w @ exp_returns[assets_clean])
                    curve_risk.append(np.sqrt(w @ cov @ w))

                idx             = int(risk_appetite * (len(eq_grid) - 1))
                weights         = curve_weights[idx]
                weights_nonzero = weights[weights > 0]

                port_return  = weights @ exp_returns[assets_clean]
                port_vol_val = np.sqrt(weights @ cov @ weights)
                port_series  = returns[assets_clean] @ weights
                port_cdar    = cdar(port_series)
                port_cvar    = cvar(port_series)
                annual_ret   = port_series.mean() * 252
                sharpe       = (port_return - 0.02) / port_vol_val

                mc_risk, mc_ret = [], []
                for _ in range(mc_sims):
                    ew = np.random.rand()
                    w  = pd.Series(0.0, index=assets_clean)
                    if stocks_clean:
                        sw = np.random.dirichlet(np.ones(len(stocks_clean)))
                        w[stocks_clean] = sw * ew
                    if bonds_clean:  w[bonds_clean]  = (1 - ew) / len(bonds_clean)
                    elif stocks_clean: w[stocks_clean] = w[stocks_clean] + (1 - ew) * sw
                    mc_ret.append(w @ exp_returns[assets_clean])
                    mc_risk.append(np.sqrt(w @ cov @ w))

                try:
                    spy         = yf.Ticker("SPY").history(start=start, end=end)["Close"]
                    spy_returns = spy.pct_change().dropna().reindex(port_series.index, fill_value=0)
                except Exception:
                    spy_returns = None

                results = dict(
                    weights=weights, weights_nonzero=weights_nonzero,
                    cov=cov, port_series=port_series,
                    port_return=port_return, port_vol=port_vol_val,
                    annual_ret=annual_ret, sharpe=sharpe,
                    port_cdar=port_cdar, port_cvar=port_cvar,
                    mc_risk=mc_risk, mc_ret=mc_ret,
                    curve_risk=curve_risk, curve_ret=curve_ret,
                    eq_grid=eq_grid, spy_returns=spy_returns,
                    stocks=stocks_clean, bonds=bonds_clean,
                    start=start, end=end,
                )
                st.session_state["portfolio_last_weights"]  = weights_nonzero.to_dict()
                st.session_state["portfolio_last_results"]  = results

            except Exception as exc:
                import traceback as _tb
                st.error(f"Optimization failed: {exc}")
                with st.expander("Show details"):
                    st.code(_tb.format_exc())
                return

    if not results:
        return

    # ── Unpack results ────────────────────────────────────────────────────────
    weights         = results["weights"]
    weights_nonzero = results["weights_nonzero"]
    cov             = results["cov"]
    port_series     = results["port_series"]
    port_return     = results["port_return"]
    port_vol_val    = results["port_vol"]
    annual_ret      = results["annual_ret"]
    sharpe          = results["sharpe"]
    port_cdar       = results["port_cdar"]
    port_cvar       = results["port_cvar"]
    mc_risk         = results["mc_risk"]
    mc_ret          = results["mc_ret"]
    curve_risk      = results["curve_risk"]
    curve_ret       = results["curve_ret"]
    eq_grid         = results["eq_grid"]
    spy_returns     = results["spy_returns"]
    stocks_clean    = results["stocks"]
    bonds_clean     = results["bonds"]
    start           = results["start"]
    end             = results["end"]

    shared_color_map = ranked_blue_map(weights_nonzero * 100)

    # ── KPI metric cards (top row) ────────────────────────────────────────────
    ret_color = UP if port_return >= 0 else DOWN
    ret_sign  = "+" if port_return >= 0 else ""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(_metric_card("Expected Return", f"{ret_sign}{port_return*100:.1f}%",
            ret_color, tooltip="Model-blended annualized expected return."), unsafe_allow_html=True)
    with c2:
        hist_color = UP if annual_ret >= 0 else DOWN
        st.markdown(_metric_card("Historical Return", f"{'+' if annual_ret>=0 else ''}{annual_ret*100:.1f}%",
            hist_color, tooltip="Actual annualized return earned over the look-back window."), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_card("Volatility", f"{port_vol_val*100:.1f}%",
            ORANGE if port_vol_val < 0.25 else DOWN, tooltip="Annualized portfolio standard deviation."), unsafe_allow_html=True)
    with c4:
        sh_color = UP if sharpe >= 1 else (ORANGE if sharpe >= 0.5 else DOWN)
        st.markdown(_metric_card("Sharpe Ratio", f"{sharpe:.2f}",
            sh_color, tooltip="Return per unit of risk above a 2% risk-free rate. Above 1 is good."), unsafe_allow_html=True)
    with c5:
        st.markdown(_metric_card("CDaR (5%)", f"{port_cdar*100:.1f}%",
            DOWN if port_cdar > 0.15 else ORANGE, tooltip="Average of the 5% worst drawdowns."), unsafe_allow_html=True)
    with c6:
        st.markdown(_metric_card("CVaR (5%)", f"{port_cvar*100:.1f}%",
            DOWN if port_cvar > 0.03 else ORANGE, tooltip="Expected daily loss in the worst 5% of days."), unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Allocation", "Frontier & Risk", "Performance", "Volatility"])

    # ════════ TAB 0 — ALLOCATION ═════════════════════════════════════════════
    with tabs[0]:
        col_donut, col_risk = st.columns([1, 1], gap="medium")

        with col_donut:
            _section_header("Holdings Mix")
            st.plotly_chart(
                plot_allocation_donut(weights_nonzero, color_map=shared_color_map),
                width="stretch",
            )

        with col_risk:
            _section_header("Contribution to Portfolio Risk")
            st.plotly_chart(
                plot_risk_contribution(weights, cov, color_map=shared_color_map),
                width="stretch",
            )

        s_label, s_color = _sharpe_verdict(sharpe)
        v_label, v_color = _vol_verdict(port_vol_val)
        ca, cb = st.columns(2)
        with ca:
            st.markdown(_verdict_card(
                "Quality of Return (Sharpe)", f"{s_label} ({sharpe:.2f})", s_color,
                f"For every unit of total risk, this portfolio earns a Sharpe of {sharpe:.2f}. "
                f"Above 1.0 is generally considered good; above 2.0 is excellent."
            ), unsafe_allow_html=True)
        with cb:
            st.markdown(_verdict_card(
                "Volatility Profile", f"{v_label} ({port_vol_val*100:.1f}%/yr)", v_color,
                f"The portfolio swings {port_vol_val*100:.1f}% per year. "
                f"{'Lower volatility makes it easier to stay the course in down markets.' if port_vol_val < 0.20 else 'Higher volatility means larger swings in both directions.'}"
            ), unsafe_allow_html=True)

    # ════════ TAB 1 — FRONTIER & RISK ════════════════════════════════════════
    with tabs[1]:
        _section_header("Efficient Frontier")
        st.plotly_chart(
            plot_efficient_frontier(mc_risk, mc_ret, curve_risk, curve_ret, port_vol_val, port_return, eq_grid),
            width="stretch",
        )

        col_cdar, col_cvar = st.columns(2)
        with col_cdar:
            cdar_color = DOWN if port_cdar > 0.15 else ORANGE
            st.markdown(_verdict_card(
                "Conditional Drawdown at Risk (CDaR 5%)",
                f"{port_cdar*100:.1f}% average worst-case drawdown", cdar_color,
                "In the 5% worst drawdown scenarios, the portfolio fell an average of "
                f"{port_cdar*100:.1f}% from its peak. Lower is better."
            ), unsafe_allow_html=True)
        with col_cvar:
            cvar_color = DOWN if port_cvar > 0.03 else ORANGE
            st.markdown(_verdict_card(
                "Conditional Value at Risk (CVaR 5%)",
                f"{port_cvar*100:.2f}% expected daily loss (tail)", cvar_color,
                "On the worst 5% of trading days, the portfolio lost an average of "
                f"{port_cvar*100:.2f}%. This is a tail-risk measure — lower is better."
            ), unsafe_allow_html=True)

    # ════════ TAB 2 — PERFORMANCE ════════════════════════════════════════════
    with tabs[2]:
        _section_header("Cumulative Return vs S&P 500")
        try:
            fig_cum = plot_cumulative_returns(port_series, spy_returns)
        except Exception:
            fig_cum = plot_cumulative_returns(port_series)
        st.plotly_chart(fig_cum, width="stretch")

        if spy_returns is not None:
            excess    = port_series.mean() * 252 - spy_returns.mean() * 252
            exc_color = UP if excess >= 0 else DOWN
            st.markdown(_verdict_card(
                "Extra return vs S&P 500 (annualized)",
                f"{'+' if excess>=0 else ''}{excess*100:.1f}% per year", exc_color,
                f"Over the look-back period, this portfolio {'outperformed' if excess>=0 else 'underperformed'} "
                f"the S&P 500 by {abs(excess)*100:.1f} percentage points per year on a historical basis."
            ), unsafe_allow_html=True)

    # ════════ TAB 3 — VOLATILITY ═════════════════════════════════════════════
    with tabs[3]:
        _section_header("Rolling 1-Year Volatility")
        st.plotly_chart(plot_rolling_volatility(port_series), width="stretch")
