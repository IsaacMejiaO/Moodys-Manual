# ============================
# TEARSHEET.PY (OPTIMIZED LAYOUT)
# ============================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np

from sec_engine.cik_loader import load_full_cik_map
from sec_engine.sec_fetch import fetch_company_facts
from sec_engine.normalize import GAAP_MAP
from sec_engine.ltm import extract_annual_series, extract_quarterly_series
from sec_engine.peer_finder import find_peers_by_sic
from sec_engine.capital_iq_style_peer_finder import find_best_peers_automated

# ── Design tokens (matches performance.py, ratios.py, multiples.py) ───────────
UP      = "#00C805"
DOWN    = "#FF3B30"
BLUE    = "#0A7CFF"
ORANGE  = "#FF9F0A"
GREY    = "#6E6E73"
BORDER  = "#E5E5EA"
CARD_BG = "transparent"

# ── Chart base layout ─────────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'SF Pro Display', 'Segoe UI', sans-serif", size=12, color="#ffffff"),
)


def _inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {BORDER}; background: transparent !important; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 14px; padding: 10px 18px; color: #ffffff;
        border-radius: 8px 8px 0 0; background: transparent !important; }}
    .stTabs [aria-selected="true"] {{ color: {BLUE} !important; border-bottom: 2px solid {BLUE} !important;
        background: transparent !important; }}
    </style>""", unsafe_allow_html=True)

# ----------------------------
# Formatting helpers (unchanged)
# ----------------------------
def format_market_cap(value):
    if not isinstance(value, (int, float)):
        return value
    if value >= 1_000_000_000_000:
        return f"{value/1_000_000_000_000:.0f}T"
    elif value >= 1_000_000_000:
        return f"{value/1_000_000_000:.0f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.0f}M"
    else:
        return f"{value:,}"

def fmt_int(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return str(int(round(float(value))))
    except Exception:
        return "N/A"

def fmt_one_decimal(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}"
    except Exception:
        return "N/A"

def fmt_two_decimal(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"

def fmt_pct(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.0f}%"
    except Exception:
        return "N/A"

def is_valid_number(value):
    return value is not None and not pd.isna(value)

def safe_divide(a, b, min_abs=1e-12):
    try:
        if a is None or b is None:
            return None
        if pd.isna(a) or pd.isna(b):
            return None
        if abs(b) < min_abs:
            return None
        return a / b
    except Exception:
        return None

# ----------------------------
# _metric_card — shared with performance.py / ratios.py / multiples.py
# ----------------------------
def _metric_card(label: str, value_str: str, color: str = "#ffffff",
                 emoji: str = "", tooltip: str = "") -> str:
    return f"""
<div title="{tooltip}" style="background:{CARD_BG};border-radius:12px;padding:16px 18px;
     cursor:{'help' if tooltip else 'default'};border:1px solid {BORDER};height:100%;">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.05em;color:#ffffff;
              text-transform:uppercase;margin-bottom:6px;">{(emoji + ' ') if emoji else ''}{label}</div>
  <div style="font-size:26px;font-weight:700;line-height:1.1;color:{color};">{value_str}</div>
</div>"""

# ----------------------------
# Cached YF helpers (unchanged)
# ----------------------------
@st.cache_data(ttl=3600)
def get_yf_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_yf_financials(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.financials if hasattr(t, "financials") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf_balance_sheet(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf_cashflow(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.cashflow if hasattr(t, "cashflow") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf_quarterly_financials(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.quarterly_income_stmt if hasattr(t, "quarterly_income_stmt") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf_quarterly_balance(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf_quarterly_cashflow(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yf_history(ticker, period="5y"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_benchmark_history(symbol="^GSPC", period="5y"):
    try:
        return yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_cik_map():
    return load_full_cik_map()

# ----------------------------
# Y-axis formatting helpers (unchanged)
# ----------------------------
def format_yaxis_currency(values):
    values = [v for v in values if not pd.isna(v) and v != 0]
    if not values:
        return "USD", 1
    max_val = max(abs(v) for v in values)
    if max_val >= 1_000_000_000_000:
        return "USD (Trillions)", 1_000_000_000_000
    elif max_val >= 1_000_000_000:
        return "USD (Billions)", 1_000_000_000
    elif max_val >= 1_000_000:
        return "USD (Millions)", 1_000_000
    elif max_val >= 1_000:
        return "USD (Thousands)", 1_000
    else:
        return "USD", 1

# ----------------------------
# Helper functions (unchanged)
# ----------------------------
def get_value_by_year(df, label, year):
    if df is None or df.empty or label not in df.index:
        return np.nan
    for col in df.columns:
        try:
            if pd.to_datetime(col).year == year:
                return pd.to_numeric(df.loc[label, col], errors="coerce")
        except Exception:
            continue
    return np.nan

def get_latest_value(df, label):
    if df is None or df.empty or label not in df.index:
        return np.nan
    series = df.loc[label]
    if isinstance(series, pd.Series):
        try:
            idx = pd.to_datetime(series.index)
            series = series.copy()
            series.index = idx
            series = series.sort_index()
            return pd.to_numeric(series.iloc[-1], errors="coerce")
        except Exception:
            return pd.to_numeric(series.dropna().iloc[0], errors="coerce") if not series.dropna().empty else np.nan
    return np.nan

def get_all_years_from_dfs(dfs):
    years = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        try:
            years.update(pd.to_datetime(df.columns).year.tolist())
        except Exception:
            continue
    if not years:
        return pd.Index([])
    return pd.Index(sorted(years))

def median_ignore_nan(values):
    clean = [v for v in values if is_valid_number(v)]
    if not clean:
        return np.nan
    return float(np.median(clean))

def compute_sector_median(peer_labels, peer_values, target_sector):
    if not target_sector:
        return np.nan
    sector_vals = []
    for label, val in zip(peer_labels, peer_values):
        info = get_yf_info(label)
        if info.get("sector") == target_sector and is_valid_number(val):
            sector_vals.append(val)
    return median_ignore_nan(sector_vals)

def append_median_bars(x_labels, values, colors, industry_median, sector_median):
    median_color = "#f2b632"
    sector_color = "#3fb41f"
    if is_valid_number(industry_median):
        x_labels.append("Industry Median")
        values.append(industry_median)
        colors.append(median_color)
    if is_valid_number(sector_median):
        x_labels.append("Sector Median")
        values.append(sector_median)
        colors.append(sector_color)
    return x_labels, values, colors


def _render_public_comps_style_bar(labels, values, colors, x_title, ticker, tick_suffix="", height=300):
    pairs = [(lbl, val) for lbl, val, _ in zip(labels, values, colors) if not pd.isna(val)]
    if not pairs:
        st.info("No peer data available for chart.")
        return

    # Match multiples.py style: sort company bars by value, keep benchmark rows at end.
    special_labels = {"Industry Median", "Industry Avg", "Sector Median"}
    main_pairs = [(lbl, val) for lbl, val in pairs if lbl not in special_labels]
    special_pairs = [(lbl, val) for lbl, val in pairs if lbl in special_labels]
    main_pairs.sort(key=lambda x: x[1])
    ordered = main_pairs + special_pairs

    labels_sorted = [p[0] for p in ordered]
    values_sorted = [p[1] for p in ordered]
    colors_sorted = []
    for lbl in labels_sorted:
        if lbl == ticker:
            colors_sorted.append(BLUE)
        elif lbl in special_labels:
            colors_sorted.append(ORANGE)
        else:
            colors_sorted.append("rgba(10,124,255,0.35)")

    fig_p = go.Figure(go.Bar(
        y=labels_sorted,
        x=values_sorted,
        orientation="h",
        marker=dict(color=colors_sorted, line=dict(color="rgba(0,0,0,0.1)", width=0.5)),
        text=[f"{v:.1f}{tick_suffix}" for v in values_sorted],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        hovertemplate=f"<b>%{{y}}</b><br>%{{x:.1f}}{tick_suffix}<extra></extra>",
        showlegend=False,
    ))
    fig_p.update_layout(
        **_CHART_LAYOUT,
        height=height,
        margin=dict(l=120, r=20, t=10, b=20),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   zeroline=False, tickfont=dict(color="#ffffff"), ticksuffix=tick_suffix),
        yaxis=dict(showgrid=False, tickfont=dict(color="#ffffff")),
    )
    st.plotly_chart(fig_p, width="stretch")

# ----------------------------
# Stock Chart — restyled with shared design system
# ----------------------------
def render_stock_chart(ticker, selected_period):
    hist       = get_yf_history(ticker, period="5y")
    sp500_hist = get_benchmark_history("^GSPC", period="5y")
    perf_return = None

    if hist.empty or "Close" not in hist.columns:
        st.info("Price history unavailable for the selected period.")
        return perf_return

    hist = hist.copy()
    hist["Pct_Change"] = hist["Close"].pct_change().fillna(0).add(1).cumprod().sub(1) * 100

    days_map = {"1M": 21, "3M": 63, "6M": 126, "YTD": None, "1Y": 252, "3Y": 252*3, "5Y": 252*5}
    if selected_period == "YTD":
        year_mask = hist.index.year == hist.index[-1].year
        year_idx  = hist.loc[year_mask].index
        start_idx = year_idx[0] if len(year_idx) > 0 else hist.index[0]
    else:
        days      = days_map[selected_period]
        start_idx = hist.index[-days] if days is not None and len(hist) > days else hist.index[0]

    hist_period  = hist.loc[start_idx:]
    perf_return  = hist_period["Pct_Change"].iloc[-1] if not hist_period.empty else None
    sp500_period = sp500_hist.loc[start_idx:] if not sp500_hist.empty else pd.DataFrame()

    # Indexed to 0 for both series
    stock_indexed = (((hist_period["Close"] / hist_period["Close"].iloc[0]) - 1) * 100
                     if not hist_period.empty else pd.Series())
    sp500_indexed = (((sp500_period["Close"] / sp500_period["Close"].iloc[0]) - 1) * 100
                     if not sp500_period.empty and "Close" in sp500_period.columns else pd.Series())

    ret_color = UP if (perf_return or 0) >= 0 else DOWN

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price — primary y-axis (left) with BLUE fill
    if not hist_period.empty:
        fig.add_trace(go.Scatter(
            x=hist_period.index, y=hist_period["Close"],
            mode="lines", name=f"{ticker} Price",
            line=dict(color=BLUE, width=3),
            fill="tozeroy", fillcolor="rgba(10,124,255,0.08)",
            hovertemplate=f"{ticker} Price: $%{{y:.2f}}<extra></extra>",
        ), secondary_y=False)

        cur_price = hist_period["Close"].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[hist_period.index[-1]], y=[cur_price],
            mode="text",
            text=[f"<b>${cur_price:.2f}</b>"], textposition="top left",
            textfont=dict(size=10, color=BLUE, family="Arial Black"),
            showlegend=False, hoverinfo="skip", name="",
        ), secondary_y=False)

    # Indexed stock return — secondary y-axis (right), UP/DOWN colored
    if not stock_indexed.empty:
        fig.add_trace(go.Scatter(
            x=hist_period.index, y=stock_indexed,
            mode="lines", name=f"{ticker} Indexed",
            line=dict(color=ret_color, width=2.5),
            hovertemplate=f"{ticker} Indexed: %{{y:+.1f}}%<extra></extra>",
        ), secondary_y=True)

        cur_indexed = stock_indexed.iloc[-1]
        sign        = "▲" if cur_indexed >= 0 else "▼"
        fig.add_trace(go.Scatter(
            x=[hist_period.index[-1]], y=[cur_indexed],
            mode="text",
            text=[f"<b>{sign} {abs(cur_indexed):.1f}%</b>"], textposition="middle right",
            textfont=dict(size=10, color=ret_color, family="Arial Black"),
            showlegend=False, hoverinfo="skip", name="",
        ), secondary_y=True)

    # S&P 500 indexed — secondary y-axis (right), ORANGE
    if not sp500_indexed.empty:
        fig.add_trace(go.Scatter(
            x=sp500_period.index, y=sp500_indexed,
            mode="lines", name="S&P 500 Indexed",
            line=dict(color=ORANGE, width=2.5),
            hovertemplate="S&P 500 Indexed: %{y:+.1f}%<extra></extra>",
        ), secondary_y=True)

        sp500_current = sp500_indexed.iloc[-1]
        sp_sign       = "▲" if sp500_current >= 0 else "▼"
        fig.add_trace(go.Scatter(
            x=[sp500_period.index[-1]], y=[sp500_current],
            mode="text",
            text=[f"<b>{sp_sign} {abs(sp500_current):.1f}%</b>"], textposition="middle right",
            textfont=dict(size=10, color=ORANGE, family="Arial Black"),
            showlegend=False, hoverinfo="skip", name="",
        ), secondary_y=True)

    fig.update_xaxes(
        title="",
        showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
        tickfont=dict(color="#ffffff"),
        tickformat="%b '%y" if selected_period in ("1M","3M","6M","YTD","1Y") else "%Y",
    )
    fig.update_yaxes(
        title_text=f"{ticker} Price ($)", secondary_y=False,
        showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
        tickformat="$.2f", tickfont=dict(color="#ffffff"),
        title_font=dict(size=12, color=BLUE),
    )
    fig.update_yaxes(
        title_text="Indexed Performance (%)", secondary_y=True,
        showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
        tickformat="+.0f", tickfont=dict(color="#ffffff"),
        title_font=dict(size=12, color=ret_color),
    )
    fig.update_layout(
        **_CHART_LAYOUT,
        height=350,
        margin=dict(l=20, r=20, t=20, b=60),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    return perf_return

# ----------------------------
# Main render function
# ----------------------------
def render_tearsheet(ticker: str):
    _inject_css()
    st.markdown(
        f'<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:4px;">{ticker}</h1>',
        unsafe_allow_html=True,
    )

    # Get CIK for SEC fallback
    cik_map = get_cik_map()
    cik     = cik_map.get(ticker)

    # Get yfinance data
    info        = get_yf_info(ticker)
    yf_fin      = get_yf_financials(ticker)
    yf_balance  = get_yf_balance_sheet(ticker)
    yf_cashflow = get_yf_cashflow(ticker)
    q_income    = get_yf_quarterly_financials(ticker)
    q_balance   = get_yf_quarterly_balance(ticker)
    q_cashflow  = get_yf_quarterly_cashflow(ticker)

    # ── Peer Finding — mirrors Multiples page logic ───────────────────────────
    # Use the same Capital IQ-style automated finder (with SIC fallback) that
    # the Multiples page uses.  We request up to 12 candidates so we have
    # enough runway to select the best 4 that actually have data for each
    # specific metric in each tab.
    sic_map    = st.session_state.get("sic_map", {})
    company_df = st.session_state.get("company_df", pd.DataFrame())
    uploaded_universe = st.session_state.get("uploaded_universe", None)

    with st.spinner("Finding comparable companies…"):
        try:
            _peer_candidates = find_best_peers_automated(
                ticker=ticker,
                uploaded_universe=uploaded_universe,
                max_peers=12,
            )
        except Exception:
            _peer_candidates = []

        # SIC fallback — mirrors multiples.py exactly
        if not _peer_candidates and sic_map and not company_df.empty:
            _peer_candidates = find_peers_by_sic(
                ticker, sic_map, company_df, min_peers=4, max_peers=12
            )

    # Cache raw peer financials once so every tab reuses the same data
    # without re-fetching.  Keyed by ticker string.
    _peer_fin_cache     = {}
    _peer_balance_cache = {}
    _peer_cf_cache      = {}

    def _load_peer_data(peer: str):
        """Lazy-load and cache yfinance statements for a peer."""
        if peer not in _peer_fin_cache:
            _peer_fin_cache[peer]     = get_yf_financials(peer)
            _peer_balance_cache[peer] = get_yf_balance_sheet(peer)
            _peer_cf_cache[peer]      = get_yf_cashflow(peer)

    def _best_peers_for_metric(yf_label: str, source: str = "fin", max_peers: int = 4):
        """
        Return (labels, values) for the best ≤ max_peers peers that have
        a valid (non-NaN) value for `yf_label`.

        Peers are evaluated in the order returned by find_best_peers_automated
        (highest-quality first), so the first max_peers with data are kept —
        exactly mirroring how the Multiples page filters its peer bar chart.

        source: "fin" → income statement, "balance" → balance sheet,
                "cf"  → cash flow statement.
        """
        labels, values = [], []
        for peer in _peer_candidates:
            if len(labels) >= max_peers:
                break
            try:
                _load_peer_data(peer)
                if source == "fin":
                    df = _peer_fin_cache[peer]
                elif source == "balance":
                    df = _peer_balance_cache[peer]
                else:
                    df = _peer_cf_cache[peer]
                val = get_latest_value(df, yf_label)
                if is_valid_number(val):
                    labels.append(peer)
                    values.append(val)
            except Exception:
                continue
        return labels, values

    # Keep a flat list for sector-median computation (uses all candidates that
    # have data, not just the best 4).
    PEER_UNIVERSE = _peer_candidates

    # Get all years
    all_years      = get_all_years_from_dfs([yf_fin, yf_balance, yf_cashflow])
    non_peer_years = [y for y in all_years if y != 2021]
    target_sector  = info.get("sector")

    # 50/25/25 layout
    desc_col, tabs_col = st.columns([1, 1])

    with desc_col:
        st.write(info.get("longBusinessSummary", "No description available."))

    with tabs_col:
        tabs_host = st.container()

    st.markdown("")

    # ── Stock Performance ─────────────────────────────────────────────────────

    perf_period_options = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]
    selected_period = st.radio(
        "Select Period",
        options=perf_period_options,
        index=4,
        horizontal=True,
        key="period_selector",
        label_visibility="collapsed",
    )

    perf_return = render_stock_chart(ticker, selected_period)

    # ── LTM calculations (unchanged) ──────────────────────────────────────────
    def get_ltm_sum(df, field_name):
        if df is None or df.empty or field_name not in df.index:
            return np.nan
        values = df.loc[field_name].head(4)
        if len(values) < 4:
            return np.nan
        return float(values.sum())

    def get_latest_q(df, field_name):
        if df is None or df.empty or field_name not in df.index:
            return np.nan
        return float(df.loc[field_name].iloc[0])

    revenue      = get_ltm_sum(q_income,   "Total Revenue")
    gross_profit = get_ltm_sum(q_income,   "Gross Profit")
    ebit         = get_ltm_sum(q_income,   "Operating Income")
    ebitda       = get_ltm_sum(q_income,   "EBITDA")
    net_income   = get_ltm_sum(q_income,   "Net Income")
    ocf          = get_ltm_sum(q_cashflow, "Operating Cash Flow")
    capex        = get_ltm_sum(q_cashflow, "Capital Expenditure")
    fcf          = ocf - capex if is_valid_number(ocf) and is_valid_number(capex) else np.nan

    current_assets      = get_latest_q(q_balance, "Current Assets")
    current_liabilities = get_latest_q(q_balance, "Current Liabilities")
    inventory           = get_latest_q(q_balance, "Inventory")
    total_debt          = get_latest_q(q_balance, "Total Debt")
    equity              = get_latest_q(q_balance, "Stockholders Equity")
    total_assets        = get_latest_q(q_balance, "Total Assets")

    # SEC fallback
    if cik and (pd.isna(revenue) or pd.isna(ebitda)):
        try:
            facts = fetch_company_facts(cik)
            if pd.isna(revenue):
                rev_series = extract_quarterly_series(facts, GAAP_MAP.get("revenue", []))
                if not rev_series.empty:
                    revenue = float(rev_series.sort_index().iloc[-1])
            if pd.isna(ebitda):
                ebitda_series = extract_quarterly_series(facts, GAAP_MAP.get("ebitda", []))
                if not ebitda_series.empty:
                    ebitda = float(ebitda_series.sort_index().iloc[-1])
        except Exception:
            pass

    # Derived ratios
    gross_margin   = safe_divide(gross_profit, revenue)
    ebit_margin    = safe_divide(ebit,         revenue)
    ebitda_margin  = safe_divide(ebitda,       revenue)
    net_margin_r   = safe_divide(net_income,   revenue)
    roa            = safe_divide(net_income,   total_assets)
    roe            = safe_divide(net_income,   equity)
    roic           = safe_divide(ebit, (total_debt or 0) + (equity or 0)) if is_valid_number(ebit) else None
    current_ratio  = safe_divide(current_assets, current_liabilities)
    quick_assets   = current_assets - inventory if is_valid_number(inventory) else current_assets
    quick_ratio    = safe_divide(quick_assets, current_liabilities)
    debt_to_equity = safe_divide(total_debt, equity)

    eps        = info.get("trailingEps")
    pe         = info.get("trailingPE")
    wk_low     = info.get("fiftyTwoWeekLow")
    wk_high    = info.get("fiftyTwoWeekHigh")
    market_cap = info.get("marketCap")

    # ── Key Financials — restyled with _metric_card ───────────────────────────

    def _pct(v):
        return f"{v*100:.1f}%" if is_valid_number(v) else "N/A"

    def _x(v):
        return f"{v:.2f}x" if is_valid_number(v) else "N/A"

    def _margin_color(v):
        if not is_valid_number(v): return GREY
        return UP if v > 0.10 else (ORANGE if v > 0 else DOWN)

    def _ratio_color(v, good, ok):
        if not is_valid_number(v): return GREY
        return UP if v >= good else (ORANGE if v >= ok else DOWN)

    def _de_color(v):
        if not is_valid_number(v): return GREY
        return UP if v < 1 else (ORANGE if v < 2 else DOWN)

    # Row label helper
    def _row_label(text):
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;color:#ffffff;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin:16px 0 8px 0;opacity:0.7;">{text}</div>',
            unsafe_allow_html=True)

    def _render_metric_grid(items, cols=3):
        for i in range(0, len(items), cols):
            row = items[i:i + cols]
            row_cols = st.columns(cols)
            for col, item in zip(row_cols, row):
                with col:
                    st.markdown(
                        _metric_card(item["label"], item["value"], item["color"], tooltip=item["tooltip"]),
                        unsafe_allow_html=True,
                    )

    ret_color = UP if is_valid_number(perf_return) and perf_return >= 0 else (DOWN if is_valid_number(perf_return) else GREY)
    pe_color = UP if is_valid_number(pe) and pe > 0 else (DOWN if is_valid_number(pe) and pe < 0 else GREY)
    eps_color = UP if is_valid_number(eps) and eps > 0 else (DOWN if is_valid_number(eps) and eps < 0 else GREY)
    market_items = [
        {"label": "Return", "value": fmt_pct(perf_return) if is_valid_number(perf_return) else "N/A", "color": ret_color, "tooltip": f"Price return for the selected {selected_period} period."},
        {"label": "Market Cap", "value": format_market_cap(market_cap) if market_cap else "N/A", "color": BLUE, "tooltip": "Total market value of all outstanding shares."},
        {"label": "52W High", "value": f"${fmt_one_decimal(wk_high)}" if wk_high else "N/A", "color": GREY, "tooltip": "Highest closing price in the last 52 weeks."},
        {"label": "52W Low", "value": f"${fmt_one_decimal(wk_low)}" if wk_low else "N/A", "color": GREY, "tooltip": "Lowest closing price in the last 52 weeks."},
        {"label": "P/E", "value": f"{fmt_int(pe)}x" if is_valid_number(pe) else "N/A", "color": pe_color, "tooltip": "Price-to-earnings ratio on trailing twelve months."},
        {"label": "EPS", "value": f"${fmt_one_decimal(eps)}" if is_valid_number(eps) else "N/A", "color": eps_color, "tooltip": "Trailing twelve-month earnings per share."},
    ]
    profitability_items = [
        {"label": "Gross Margin", "value": _pct(gross_margin), "color": _margin_color(gross_margin), "tooltip": "Revenue left after direct production costs."},
        {"label": "EBIT Margin", "value": _pct(ebit_margin), "color": _margin_color(ebit_margin), "tooltip": "Operating profit per dollar of revenue."},
        {"label": "EBITDA Margin", "value": _pct(ebitda_margin), "color": _margin_color(ebitda_margin), "tooltip": "Earnings before interest, tax, D&A divided by revenue."},
        {"label": "Net Margin", "value": _pct(net_margin_r), "color": _margin_color(net_margin_r), "tooltip": "Bottom-line profit per dollar of revenue."},
        {"label": "ROA", "value": _pct(roa), "color": _ratio_color(roa, 0.05, 0), "tooltip": "Net income divided by total assets."},
        {"label": "ROE", "value": _pct(roe), "color": _ratio_color(roe, 0.10, 0), "tooltip": "Net income divided by shareholders' equity."},
        {"label": "ROIC", "value": _pct(roic), "color": _ratio_color(roic, 0.10, 0), "tooltip": "EBIT divided by (debt + equity). Measures capital efficiency."},
    ]
    liquidity_items = [
        {"label": "Current Ratio", "value": _x(current_ratio), "color": _ratio_color(current_ratio, 1.5, 1.0), "tooltip": "Current assets divided by current liabilities. Above 1x = bills are covered."},
        {"label": "Quick Ratio", "value": _x(quick_ratio), "color": _ratio_color(quick_ratio, 1.0, 0.5), "tooltip": "Like current ratio but strips out inventory. Stricter solvency test."},
        {"label": "Debt / Equity", "value": _x(debt_to_equity), "color": _de_color(debt_to_equity), "tooltip": "Total debt divided by shareholders' equity. Lower = less leveraged."},
    ]

    col_market, col_profitability, col_liquidity = st.columns(3)
    with col_market:
        _row_label("Market")
        _render_metric_grid(market_items, cols=1)
    with col_profitability:
        _row_label("Profitability")
        _render_metric_grid(profitability_items, cols=1)
    with col_liquidity:
        _row_label("Liquidity & Solvency")
        _render_metric_grid(liquidity_items, cols=1)

    st.markdown("")

    # ── Tabbed Interface (completely unchanged from original) ─────────────────
    with tabs_host:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Income Statement",
            "Balance Sheet",
            "Cash Flow",
            "Margin Analysis",
        ])

    highlight_color = "#1f77b4"
    other_color     = "#87CEEB"

    YF_LABELS = {
        "revenue": "Total Revenue", "cogs": "Cost Of Revenue",
        "gross_profit": "Gross Profit", "opex": "Operating Expense",
        "ebit": "Operating Income", "ebitda": "EBITDA",
        "interest": "Interest Expense", "net_income": "Net Income",
        "cash": "Cash And Cash Equivalents", "current_assets": "Current Assets",
        "current_debt": "Current Debt", "current_liabilities": "Current Liabilities",
        "lt_debt": "Long Term Debt",
        "total_liab": "Total Liabilities Net Minority Interest",
        "equity": "Stockholders Equity", "net_debt": "Net Debt",
        "da": "Depreciation And Amortization", "capex": "Capital Expenditure",
        "investing_cf": "Investing Cash Flow", "dividends": "Cash Dividends Paid",
        "financing_cf": "Financing Cash Flow", "fcf": "Free Cash Flow",
    }

    # TAB 1: Income Statement
    with tab1:
        metric_options = ["Revenue", "COGS", "Gross Profit", "Operating Expenses", "EBIT", "EBITDA", "Interest Expense", "Net Income"]
        col_a, col_b = st.columns(2)
        with col_a:
            selected_metric = st.selectbox("Select metric", metric_options, index=0, key="is_metric", label_visibility="collapsed")
        with col_b:
            selected_metric_compare = st.selectbox("Compare to", metric_options, index=2, key="is_compare", label_visibility="collapsed")

        metric_map = {
            "Revenue": "Total Revenue", "COGS": "Cost Of Revenue", "Gross Profit": "Gross Profit",
            "Operating Expenses": "Operating Expense", "EBIT": "Operating Income", "EBITDA": "EBITDA",
            "Interest Expense": "Interest Expense", "Net Income": "Net Income",
        }

        yf_label         = metric_map[selected_metric]
        yf_label_compare = metric_map[selected_metric_compare]

        if len(non_peer_years) > 0:
            values         = [get_value_by_year(yf_fin, yf_label,         y) for y in non_peer_years]
            compare_values = [get_value_by_year(yf_fin, yf_label_compare, y) for y in non_peer_years]

            peer_labels, peer_values = _best_peers_for_metric(yf_label, source="fin")
            industry_median = median_ignore_nan(peer_values)
            sector_median   = compute_sector_median(peer_labels, peer_values, target_sector)

            y_title, y_divisor = format_yaxis_currency(values)
            scaled_values  = [v / y_divisor if not pd.isna(v) else np.nan for v in values]
            scaled_compare = [v / y_divisor if not pd.isna(v) else np.nan for v in compare_values]
            scaled_industry= industry_median / y_divisor if is_valid_number(industry_median) else None
            scaled_sector  = sector_median   / y_divisor if is_valid_number(sector_median)   else None

            col_left, col_right = st.columns(2)

            with col_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(non_peer_years), y=scaled_values,  mode="lines+markers", name=selected_metric,
                    line=dict(color="#1f77b4", width=2.5), marker=dict(size=8)))
                if scaled_compare is not None:
                    fig.add_trace(go.Scatter(x=list(non_peer_years), y=scaled_compare, mode="lines+markers", name=selected_metric_compare,
                        line=dict(color="red", width=2), marker=dict(size=6)))
                if is_valid_number(scaled_industry):
                    fig.add_hline(y=scaled_industry, line_dash="dash", line_color="#f2b632", annotation_text="Industry", annotation_position="right")
                if is_valid_number(scaled_sector):
                    fig.add_hline(y=scaled_sector,   line_dash="dash", line_color="#2F5D8A", annotation_text="Sector",   annotation_position="right")
                fig.update_layout(xaxis=dict(title="", type="category"),
                    yaxis=dict(title=y_title, showgrid=True, nticks=6, tickformat=".0f"),
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
                    margin=dict(l=10, r=10, t=10, b=40), height=300)
                st.plotly_chart(fig, width="stretch")

            with col_right:
                pvt = list(peer_values) + [values[-1] if len(values) > 0 else np.nan]
                plt = list(peer_labels) + [ticker]
                colors = [highlight_color if lbl == ticker else other_color for lbl in plt]
                plt, pvt, colors = append_median_bars(plt, pvt, colors, industry_median, sector_median)
                py_title, py_div = format_yaxis_currency(pvt)
                spv = [v / py_div if not pd.isna(v) else np.nan for v in pvt]
                _render_public_comps_style_bar(plt, spv, colors, py_title, ticker, height=300)
    with tab2:
        bs_options = ["Cash and Short-term Investments", "Current Asset", "Short-term Debt", "Current Liabilities", "Long-term Debt", "Total Liabilities", "Total Equity", "Net Debt"]
        col_a, col_b = st.columns(2)
        with col_a:
            selected_bs = st.selectbox("Select metric", bs_options, index=0, key="bs_metric", label_visibility="collapsed")
        with col_b:
            selected_bs_compare = st.selectbox("Compare to", bs_options, index=1, key="bs_compare", label_visibility="collapsed")

        bs_map = {
            "Cash and Short-term Investments": "Cash And Cash Equivalents", "Current Asset": "Current Assets",
            "Short-term Debt": "Current Debt", "Current Liabilities": "Current Liabilities",
            "Long-term Debt": "Long Term Debt", "Total Liabilities": "Total Liabilities Net Minority Interest",
            "Total Equity": "Stockholders Equity", "Net Debt": "Net Debt",
        }

        yf_bs_label         = bs_map[selected_bs]
        yf_bs_label_compare = bs_map[selected_bs_compare]

        if len(non_peer_years) > 0:
            bs_values         = [get_value_by_year(yf_balance, yf_bs_label,         y) for y in non_peer_years]
            compare_bs_values = [get_value_by_year(yf_balance, yf_bs_label_compare, y) for y in non_peer_years]

            peer_bs_labels, peer_bs_values = _best_peers_for_metric(yf_bs_label, source="balance")
            industry_median = median_ignore_nan(peer_bs_values)
            sector_median   = compute_sector_median(peer_bs_labels, peer_bs_values, target_sector)

            bs_y_title, bs_y_div = format_yaxis_currency(bs_values)
            scaled_bs      = [v / bs_y_div if not pd.isna(v) else np.nan for v in bs_values]
            scaled_compare = [v / bs_y_div if not pd.isna(v) else np.nan for v in compare_bs_values]
            scaled_industry= industry_median / bs_y_div if is_valid_number(industry_median) else None
            scaled_sector  = sector_median   / bs_y_div if is_valid_number(sector_median)   else None

            col_left, col_right = st.columns(2)

            with col_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(non_peer_years), y=scaled_bs, mode="lines+markers", name=selected_bs,
                    line=dict(color="#1f77b4", width=2.5), marker=dict(size=8)))
                if scaled_compare is not None:
                    fig.add_trace(go.Scatter(x=list(non_peer_years), y=scaled_compare, mode="lines+markers", name=selected_bs_compare,
                        line=dict(color="red", width=2), marker=dict(size=6)))
                if is_valid_number(scaled_industry): fig.add_hline(y=scaled_industry, line_dash="dash", line_color="#f2b632")
                if is_valid_number(scaled_sector):   fig.add_hline(y=scaled_sector,   line_dash="dash", line_color="#2F5D8A")
                fig.update_layout(xaxis=dict(title="", type="category"),
                    yaxis=dict(title=bs_y_title, showgrid=True, nticks=6, tickformat=".0f"),
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
                    margin=dict(l=10, r=10, t=10, b=40), height=300)
                st.plotly_chart(fig, width="stretch")

            with col_right:
                pvt = list(peer_bs_values) + [bs_values[-1] if len(bs_values) > 0 else np.nan]
                plt = list(peer_bs_labels) + [ticker]
                colors = [highlight_color if lbl == ticker else other_color for lbl in plt]
                plt, pvt, colors = append_median_bars(plt, pvt, colors, industry_median, sector_median)
                py_title, py_div = format_yaxis_currency(pvt)
                spv = [v / py_div if not pd.isna(v) else np.nan for v in pvt]
                _render_public_comps_style_bar(plt, spv, colors, py_title, ticker, height=300)

    # TAB 3: Cash Flow
    with tab3:
        cf_options = ["Depreciation & Amortization", "CapEx", "Cash from Investing", "Dividends", "Cash from Financing", "Free Cash Flow"]
        col_a, col_b = st.columns(2)
        with col_a:
            selected_cf = st.selectbox("Select metric", cf_options, index=0, key="cf_metric", label_visibility="collapsed")
        with col_b:
            selected_cf_compare = st.selectbox("Compare to", cf_options, index=1, key="cf_compare", label_visibility="collapsed")

        cf_map = {
            "Depreciation & Amortization": "Depreciation And Amortization", "CapEx": "Capital Expenditure",
            "Cash from Investing": "Investing Cash Flow", "Dividends": "Cash Dividends Paid",
            "Cash from Financing": "Financing Cash Flow", "Free Cash Flow": "Free Cash Flow",
        }

        yf_cf_label         = cf_map[selected_cf]
        yf_cf_label_compare = cf_map[selected_cf_compare]

        if len(non_peer_years) > 0:
            cf_values         = [get_value_by_year(yf_cashflow, yf_cf_label,         y) for y in non_peer_years]
            compare_cf_values = [get_value_by_year(yf_cashflow, yf_cf_label_compare, y) for y in non_peer_years]

            peer_cf_labels, peer_cf_values = _best_peers_for_metric(yf_cf_label, source="cf")
            industry_median = median_ignore_nan(peer_cf_values)
            sector_median   = compute_sector_median(peer_cf_labels, peer_cf_values, target_sector)

            cf_y_title, cf_y_div = format_yaxis_currency(cf_values)
            scaled_cf      = [v / cf_y_div if not pd.isna(v) else np.nan for v in cf_values]
            scaled_compare = [v / cf_y_div if not pd.isna(v) else np.nan for v in compare_cf_values]
            scaled_industry= industry_median / cf_y_div if is_valid_number(industry_median) else None
            scaled_sector  = sector_median   / cf_y_div if is_valid_number(sector_median)   else None

            col_left, col_right = st.columns(2)

            with col_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(non_peer_years), y=scaled_cf, mode="lines+markers", name=selected_cf,
                    line=dict(color="#1f77b4", width=2.5), marker=dict(size=8)))
                if scaled_compare is not None:
                    fig.add_trace(go.Scatter(x=list(non_peer_years), y=scaled_compare, mode="lines+markers", name=selected_cf_compare,
                        line=dict(color="red", width=2), marker=dict(size=6)))
                if is_valid_number(scaled_industry): fig.add_hline(y=scaled_industry, line_dash="dash", line_color="#f2b632")
                if is_valid_number(scaled_sector):   fig.add_hline(y=scaled_sector,   line_dash="dash", line_color="#2F5D8A")
                fig.update_layout(xaxis=dict(title="", type="category"),
                    yaxis=dict(title=cf_y_title, showgrid=True, nticks=6, tickformat=".0f"),
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
                    margin=dict(l=10, r=10, t=10, b=40), height=300)
                st.plotly_chart(fig, width="stretch")

            with col_right:
                pvt = list(peer_cf_values) + [cf_values[-1] if len(cf_values) > 0 else np.nan]
                plt = list(peer_cf_labels) + [ticker]
                colors = [highlight_color if lbl == ticker else other_color for lbl in plt]
                plt, pvt, colors = append_median_bars(plt, pvt, colors, industry_median, sector_median)
                py_title, py_div = format_yaxis_currency(pvt)
                spv = [v / py_div if not pd.isna(v) else np.nan for v in pvt]
                _render_public_comps_style_bar(plt, spv, colors, py_title, ticker, height=300)

    # TAB 4: Margin Analysis
    with tab4:
        margin_options = ["Gross Margin %", "EBITDA Margin %", "EBIT Margin %", "Net Income Margin %", "ROA %", "ROE %", "ROIC %"]
        col_a, col_b = st.columns(2)
        with col_a:
            selected_margin = st.selectbox("Select metric", margin_options, index=0, key="margin_metric", label_visibility="collapsed")
        with col_b:
            selected_margin_compare = st.selectbox("Compare to", margin_options, index=1, key="margin_compare", label_visibility="collapsed")

        if len(non_peer_years) > 0:
            margin_values, compare_margin_values = [], []

            for y in non_peer_years:
                rev_val    = get_value_by_year(yf_fin,    "Total Revenue",       y)
                gp_val     = get_value_by_year(yf_fin,    "Gross Profit",        y)
                ebitda_val = get_value_by_year(yf_fin,    "EBITDA",              y)
                ebit_val   = get_value_by_year(yf_fin,    "Operating Income",    y)
                ni_val     = get_value_by_year(yf_fin,    "Net Income",          y)
                assets_val = get_value_by_year(yf_balance,"Total Assets",        y)
                equity_val = get_value_by_year(yf_balance,"Stockholders Equity", y)
                debt_val   = get_value_by_year(yf_balance,"Total Debt",          y)

                def _m(metric, rev, gp, ebitda, ebit, ni, assets, eq, debt):
                    ic = eq + debt if is_valid_number(eq) and is_valid_number(debt) else np.nan
                    lookup = {
                        "Gross Margin %":      safe_divide(gp,     rev),
                        "EBITDA Margin %":     safe_divide(ebitda, rev),
                        "EBIT Margin %":       safe_divide(ebit,   rev),
                        "Net Income Margin %": safe_divide(ni,     rev),
                        "ROA %":               safe_divide(ni,     assets),
                        "ROE %":               safe_divide(ni,     eq),
                        "ROIC %":              safe_divide(ebit,   ic),
                    }
                    raw = lookup.get(metric, np.nan)
                    return raw * 100 if is_valid_number(raw) else np.nan

                args = (rev_val, gp_val, ebitda_val, ebit_val, ni_val, assets_val, equity_val, debt_val)
                margin_values.append(_m(selected_margin,         *args))
                compare_margin_values.append(_m(selected_margin_compare, *args))

            peer_margin_labels, peer_margin_values = [], []
            for peer in _peer_candidates:
                if len(peer_margin_labels) >= 4:
                    break
                try:
                    _load_peer_data(peer)
                    pf = _peer_fin_cache[peer]
                    pb = _peer_balance_cache[peer]
                    rev    = get_latest_value(pf, "Total Revenue")
                    gp     = get_latest_value(pf, "Gross Profit")
                    ebitda = get_latest_value(pf, "EBITDA")
                    ebit   = get_latest_value(pf, "Operating Income")
                    ni     = get_latest_value(pf, "Net Income")
                    assets = get_latest_value(pb, "Total Assets")
                    eq     = get_latest_value(pb, "Stockholders Equity")
                    debt   = get_latest_value(pb, "Total Debt")
                    ic     = eq + debt if is_valid_number(eq) and is_valid_number(debt) else np.nan

                    lookup = {
                        "Gross Margin %":      safe_divide(gp,     rev),
                        "EBITDA Margin %":     safe_divide(ebitda, rev),
                        "EBIT Margin %":       safe_divide(ebit,   rev),
                        "Net Income Margin %": safe_divide(ni,     rev),
                        "ROA %":               safe_divide(ni,     assets),
                        "ROE %":               safe_divide(ni,     eq),
                        "ROIC %":              safe_divide(ebit,   ic),
                    }
                    raw = lookup.get(selected_margin, np.nan)
                    val = raw * 100 if is_valid_number(raw) else np.nan
                    if is_valid_number(val):
                        peer_margin_labels.append(peer)
                        peer_margin_values.append(val)
                except Exception:
                    continue

            industry_median = median_ignore_nan(peer_margin_values)
            sector_median   = compute_sector_median(peer_margin_labels, peer_margin_values, target_sector)

            col_left, col_right = st.columns(2)

            with col_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(non_peer_years), y=margin_values, mode="lines+markers", name=selected_margin,
                    line=dict(color="#1f77b4", width=2.5), marker=dict(size=8)))
                if compare_margin_values is not None:
                    fig.add_trace(go.Scatter(x=list(non_peer_years), y=compare_margin_values, mode="lines+markers", name=selected_margin_compare,
                        line=dict(color="red", width=2), marker=dict(size=6)))
                if is_valid_number(industry_median): fig.add_hline(y=industry_median, line_dash="dash", line_color="#f2b632")
                if is_valid_number(sector_median):   fig.add_hline(y=sector_median,   line_dash="dash", line_color="#2F5D8A")
                fig.update_layout(xaxis=dict(title="", type="category"),
                    yaxis=dict(title="Percentage (%)", showgrid=True, nticks=6, tickformat=".0f"),
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
                    margin=dict(l=10, r=10, t=10, b=40), height=300)
                st.plotly_chart(fig, width="stretch")

            with col_right:
                pvt = list(peer_margin_values) + [margin_values[-1] if len(margin_values) > 0 else np.nan]
                plt = list(peer_margin_labels) + [ticker]
                colors = [highlight_color if lbl == ticker else other_color for lbl in plt]
                plt, pvt, colors = append_median_bars(plt, pvt, colors, industry_median, sector_median)
                _render_public_comps_style_bar(plt, pvt, colors, "Percentage (%)", ticker, tick_suffix="%", height=300)
