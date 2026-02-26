# ============================
# MULTIPLES.PY - Valuation Multiples Analysis
# ============================
"""
Valuation Multiples Page — Plain-English Edition
================================================
Design philosophy: "Your company's value, told simply."
Mirrors the layout and formatting of performance.py and ratios.py.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sec_engine.cik_loader import load_full_cik_map
from sec_engine.sec_fetch import fetch_company_facts
from sec_engine.normalize import GAAP_MAP
from sec_engine.ltm import extract_annual_series
from sec_engine.capital_iq_style_peer_finder import find_best_peers_automated
from sec_engine.peer_finder import find_peers_by_sic

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Design tokens (matches performance.py and ratios.py exactly) ──────────────
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

# ── Metric display names ──────────────────────────────────────────────────────
METRIC_DISPLAY = {
    "TEV/Revenue":   "TEV/Revenue",
    "TEV/EBITDA":    "TEV/EBITDA",
    "TEV/EBIT":      "TEV/EBIT",
    "P/E":           "P/E",
    "P/B":           "P/B",
    "MCap/Revenue":  "MCap/Revenue",
}

METRIC_DESCRIPTIONS: Dict[str, str] = {
    "TEV/Revenue":   "Enterprise value relative to revenue. Lower multiples may signal undervaluation; useful for early-stage or low-margin businesses.",
    "TEV/EBITDA":    "Enterprise value to operating cash earnings. The most widely used valuation multiple in M&A and equity research.",
    "TEV/EBIT":      "Enterprise value to operating profit. Accounts for depreciation — useful when capex intensity differs across peers.",
    "P/E":           "Price-to-earnings. Measures how much investors pay per dollar of net profit. Lower P/E implies cheaper relative to earnings.",
    "P/B":           "Price-to-book. Compares market value to accounting net assets. Favored for banks and asset-heavy businesses.",
    "MCap/Revenue":  "Market cap relative to revenue. A simple, leverage-neutral alternative to TEV/Revenue.",
    "EPS":           "Earnings per share. Bottom-line profit attributable to each share. Drives P/E and dividend capacity.",
}

INSIGHT_THRESHOLDS = {
    "neutral":         15.0,
    "moderate":        30.0,
    "range_tail_low":  20.0,
    "range_tail_high": 80.0,
}


# ── UI helpers (mirrors performance.py and ratios.py) ─────────────────────────

def _metric_card(label: str, value_str: str, color: str = "#ffffff",
                 emoji: str = "", tooltip: str = "") -> str:
    return f"""
<div title="{tooltip}" style="background:{CARD_BG};border-radius:12px;padding:16px 18px;
     cursor:{'help' if tooltip else 'default'};border:1px solid {BORDER};height:100%;">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.05em;color:#ffffff;
              text-transform:uppercase;margin-bottom:6px;">{(emoji + ' ') if emoji else ''}{label}</div>
  <div style="font-size:26px;font-weight:700;line-height:1.1;color:{color};">{value_str}</div>
  {f'<div style="font-size:11px;color:#ffffff;margin-top:6px;line-height:1.4;opacity:0.7;">{tooltip}</div>' if tooltip else ''}
</div>"""


def _verdict_card(title: str, verdict: str, verdict_color: str, body: str,
                  fixed_height: Optional[int] = None) -> str:
    height_style = f"height:{fixed_height}px;" if fixed_height else ""
    return f"""
<div style="background:{CARD_BG};border-radius:12px;padding:18px 20px;border:1px solid {BORDER};
     border-left:4px solid {verdict_color};margin-bottom:12px;{height_style}overflow:hidden;">
  <div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:4px;">{title}</div>
  <div style="font-size:15px;font-weight:700;color:{verdict_color};margin-bottom:6px;">{verdict}</div>
  <div style="font-size:13px;color:#ffffff;opacity:0.85;line-height:1.6;">{body}</div>
</div>"""


def _section_header(title: str, subtitle: str = "") -> None:
    sub = (f'<p style="color:#ffffff;opacity:0.7;font-size:14px;margin:2px 0 0 0;">{subtitle}</p>'
           if subtitle else "")
    st.markdown(
        f'<div style="margin:28px 0 12px 0;">'
        f'<h3 style="font-size:20px;font-weight:700;color:#ffffff;margin:0;">{title}</h3>{sub}</div>',
        unsafe_allow_html=True)


def _divider() -> None:
    st.markdown('<hr style="border:none;border-top:1px solid #E5E5EA;margin:20px 0;">',
                unsafe_allow_html=True)


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
    div[data-testid="metric-container"] {{ background: transparent !important; border: 1px solid {BORDER};
        border-radius: 10px; padding: 12px 14px; }}
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label {{ color: #ffffff !important; }}
    div[data-testid="stMetricValue"] > div {{ color: #ffffff !important; }}
    div[data-testid="stExpander"] {{ background: transparent !important; border: 1px solid {BORDER} !important;
        border-radius: 10px; }}
    div[data-testid="stExpander"] > details > summary {{ color: #ffffff !important; font-weight: 600; }}
    div[data-testid="stExpander"] > details > summary svg {{ fill: #ffffff !important; }}
    div[data-testid="stAlert"] {{ background: transparent !important; }}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] div {{ color: #ffffff !important; }}
    div[data-testid="stDataFrame"] {{ background: transparent !important; }}
    .stRadio > label, .stMultiSelect > label {{ color: #ffffff !important; }}
    .stRadio div[role="radiogroup"] label, .stRadio div[role="radiogroup"] span {{ color: #ffffff !important; }}
    .stCaption p, small {{ color: #ffffff !important; opacity: 0.7; }}
    .stDownloadButton button {{ background: transparent !important; border: 1px solid {BORDER} !important;
        color: #ffffff !important; }}
    .block-container, section[data-testid="stSidebar"] > div {{ background: transparent !important; }}
    p, span, label, div {{ color: #ffffff; }}
    .stSelectbox label {{ color: #ffffff !important; }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{ color: #ffffff !important; }}
    div[role="radiogroup"] {{
        gap: 0.25rem; padding: 0.2rem;
        border: 1px solid rgba(120,130,150,0.28);
        border-radius: 10px; background: rgba(255,255,255,0.02);
        width: fit-content; margin-bottom: 0.55rem;
    }}
    div[role="radiogroup"] > label {{
        border: 1px solid transparent; border-radius: 8px;
        padding: 0.2rem 0.55rem !important; transition: all 0.15s ease; background: transparent;
    }}
    div[role="radiogroup"] > label:hover {{
        border-color: rgba(59,130,246,0.45); background: rgba(59,130,246,0.08);
    }}
    div[role="radiogroup"] > label[data-checked="true"] {{
        border-color: rgba(59,130,246,0.75); background: rgba(59,130,246,0.18);
        box-shadow: inset 0 0 0 1px rgba(59,130,246,0.35);
    }}
    </style>""", unsafe_allow_html=True)


# ── Cached YF helpers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_yf_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_yf_financials(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        return t.financials if hasattr(t, "financials") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_yf_balance_sheet(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        return t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_yf_history(ticker: str, period: str = "6y") -> Optional[pd.DataFrame]:
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return None

@st.cache_data
def get_cik_map() -> dict:
    return load_full_cik_map()


# ── Data helpers ──────────────────────────────────────────────────────────────

def safe_divide(a, b):
    try:
        if a is None or b is None:
            return np.nan
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def get_last_5_years(yf_data: pd.DataFrame) -> pd.Index:
    if yf_data.empty:
        return pd.Index([])
    years = pd.Index([pd.to_datetime(col).year for col in yf_data.columns])
    years = pd.Index(sorted(set(years), reverse=True))
    years = [y for y in years if y != 2021]
    return pd.Index(years[:5][::-1])


def get_last_n_years_from_dfs(dfs: List[pd.DataFrame], n: int = 5) -> pd.Index:
    years: set = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        try:
            years.update(pd.to_datetime(df.columns).year.tolist())
        except Exception:
            continue
    if not years:
        return pd.Index([])
    years = [y for y in years if y != 2021]
    return pd.Index(sorted(years, reverse=True)[:n][::-1])


def _fmt_multiple(val, is_eps: bool = False) -> str:
    if val == "" or pd.isna(val):
        return "—"
    try:
        num = float(val)
        return f"${num:.2f}" if is_eps else f"{num:.1f}x"
    except Exception:
        return "—"


# ── Multiples calculation ─────────────────────────────────────────────────────

def calculate_multiples_for_year(
    ticker: str, year: int,
    yf_fin: pd.DataFrame, yf_balance: pd.DataFrame,
    current_info: Optional[dict] = None,
    yf_history: Optional[pd.DataFrame] = None,
) -> Tuple[Dict, set]:
    multiples: Dict = {}
    approx_flags: set = set()

    revenue = ebitda = ebit = net_income = np.nan
    total_assets = total_equity = total_debt = cash = shares = np.nan

    for col in yf_fin.columns:
        if pd.to_datetime(col).year == year:
            for field, var in [
                ("Total Revenue", "revenue"), ("EBITDA", "ebitda"),
                ("Operating Income", "ebit"), ("Net Income", "net_income"),
            ]:
                if field in yf_fin.index:
                    locals()[var]  # reference to suppress lint; assign below
            revenue      = pd.to_numeric(yf_fin.loc["Total Revenue",       col], errors="coerce") if "Total Revenue"       in yf_fin.index else np.nan
            ebitda       = pd.to_numeric(yf_fin.loc["EBITDA",              col], errors="coerce") if "EBITDA"              in yf_fin.index else np.nan
            ebit         = pd.to_numeric(yf_fin.loc["Operating Income",    col], errors="coerce") if "Operating Income"    in yf_fin.index else np.nan
            net_income   = pd.to_numeric(yf_fin.loc["Net Income",          col], errors="coerce") if "Net Income"          in yf_fin.index else np.nan
            break

    for col in yf_balance.columns:
        if pd.to_datetime(col).year == year:
            total_assets  = pd.to_numeric(yf_balance.loc["Total Assets",              col], errors="coerce") if "Total Assets"              in yf_balance.index else np.nan
            total_equity  = pd.to_numeric(yf_balance.loc["Stockholders Equity",       col], errors="coerce") if "Stockholders Equity"       in yf_balance.index else np.nan
            total_debt    = pd.to_numeric(yf_balance.loc["Total Debt",                col], errors="coerce") if "Total Debt"                in yf_balance.index else np.nan
            cash          = pd.to_numeric(yf_balance.loc["Cash And Cash Equivalents", col], errors="coerce") if "Cash And Cash Equivalents" in yf_balance.index else np.nan
            shares        = pd.to_numeric(yf_balance.loc["Ordinary Shares Number",    col], errors="coerce") if "Ordinary Shares Number"    in yf_balance.index else np.nan
            break

    if pd.isna(shares) and current_info:
        shares = current_info.get("sharesOutstanding")
        if shares is not None:
            approx_flags.add("shares_fallback")

    historical_price = None
    if yf_history is not None:
        try:
            prices_in_year = yf_history[yf_history.index.year == year]
            if not prices_in_year.empty:
                historical_price = prices_in_year["Close"].iloc[-1]
        except Exception:
            pass

    current_price = current_info.get("currentPrice") if current_info else None
    price_to_use  = historical_price if historical_price is not None else current_price
    if historical_price is None and current_price is not None:
        approx_flags.add("price_fallback")

    market_cap = safe_divide(shares * price_to_use, 1) if not pd.isna(shares) and price_to_use else np.nan
    if pd.isna(market_cap) and current_info:
        market_cap = current_info.get("marketCap")
        if market_cap is not None:
            approx_flags.add("market_cap_fallback")

    net_debt = 0
    if pd.isna(total_debt) or pd.isna(cash):
        approx_flags.add("net_debt_fallback")
    else:
        net_debt = total_debt - cash
    enterprise_value = (market_cap + net_debt) if not pd.isna(market_cap) else np.nan

    eps               = safe_divide(net_income, shares) if not pd.isna(shares) else np.nan
    book_value_ps     = safe_divide(total_equity, shares) if not pd.isna(shares) else np.nan

    multiples["TEV/Revenue"]  = safe_divide(enterprise_value, revenue)
    multiples["TEV/EBITDA"]   = safe_divide(enterprise_value, ebitda)
    multiples["TEV/EBIT"]     = safe_divide(enterprise_value, ebit)
    multiples["P/E"]          = safe_divide(price_to_use, eps)    if price_to_use and not pd.isna(eps)          else np.nan
    multiples["P/B"]          = safe_divide(price_to_use, book_value_ps) if price_to_use and not pd.isna(book_value_ps) else np.nan
    multiples["MCap/Revenue"] = safe_divide(market_cap, revenue)
    multiples["EPS"]          = eps

    return multiples, approx_flags


# ── Signal / insight helpers ──────────────────────────────────────────────────

def compute_deviation(current, baseline) -> float:
    try:
        if current is None or baseline is None:
            return np.nan
        if pd.isna(current) or pd.isna(baseline) or baseline == 0:
            return np.nan
        return ((current / baseline) - 1.0) * 100.0
    except Exception:
        return np.nan


def classify_signal(dev_pct: float, thresholds=None) -> dict:
    if thresholds is None:
        thresholds = INSIGHT_THRESHOLDS
    if pd.isna(dev_pct):
        return {"sentiment": "neutral", "strength": "neutral", "label": "Insufficient data", "score": 0}
    abs_dev = abs(dev_pct)
    if abs_dev <= thresholds["neutral"]:
        return {"sentiment": "neutral", "strength": "neutral", "label": "Fairly valued", "score": 0}
    if abs_dev <= thresholds["moderate"]:
        if dev_pct > 0:
            return {"sentiment": "negative", "strength": "moderate", "label": "Moderate premium", "score": -1}
        return {"sentiment": "positive", "strength": "moderate", "label": "Moderate discount", "score": 1}
    if dev_pct > 0:
        return {"sentiment": "negative", "strength": "strong", "label": "Strong premium", "score": -2}
    return {"sentiment": "positive", "strength": "strong", "label": "Strong discount", "score": 2}


def _make_signal(metric, current_value, baseline_value, baseline_label,
                 deviation_pct, sentiment, strength, message, data_quality, unit="x") -> dict:
    return {
        "metric":         metric,
        "current_value":  current_value,
        "baseline_value": baseline_value,
        "baseline_label": baseline_label,
        "deviation_pct":  deviation_pct,
        "sentiment":      sentiment,
        "strength":       strength,
        "message":        message,
        "data_quality":   data_quality,
        "extra_context":  "",
        "unit":           unit,
    }


def _sentiment_to_color(sentiment: str) -> str:
    return {
        "positive": UP,
        "negative": DOWN,
        "neutral":  ORANGE,
    }.get(sentiment, GREY)


def _deviation_to_color(dev_pct: float) -> str:
    if pd.isna(dev_pct):
        return ORANGE
    if dev_pct > 0:
        return UP
    if dev_pct < 0:
        return DOWN
    return ORANGE


def build_historical_signals(df_multiples: pd.DataFrame) -> List[dict]:
    signals = []
    for metric in ["TEV/EBITDA", "P/E", "TEV/Revenue", "TEV/EBIT"]:
        if metric not in df_multiples.columns:
            continue
        series = df_multiples[metric].dropna()
        if len(series) < 2:
            continue
        current = series.iloc[-1]
        avg     = series.mean()
        dev     = compute_deviation(current, avg)
        cl      = classify_signal(dev)
        if cl["sentiment"] == "positive":
            msg = "Trading below its 5Y norm; current setup implies a valuation discount to history."
        elif cl["sentiment"] == "negative":
            msg = "Trading above its 5Y norm; valuation already reflects stronger expectations."
        else:
            msg = "Trading near its 5Y norm; valuation is broadly aligned with history."
        signals.append(_make_signal(
            metric=METRIC_DISPLAY.get(metric, metric), current_value=current,
            baseline_value=avg, baseline_label="5Y avg", deviation_pct=dev,
            sentiment=cl["sentiment"], strength=cl["strength"], message=msg,
            data_quality=len(series), unit="x",
        ))
    return signals


def build_peer_signals(df_peers: pd.DataFrame, ticker: str) -> List[dict]:
    signals = []
    if ticker not in df_peers.index:
        return signals
    peer_only = df_peers[df_peers.index != ticker]
    if peer_only.empty:
        return signals
    for metric in ["TEV/EBITDA", "P/E", "TEV/Revenue", "TEV/EBIT"]:
        if metric not in df_peers.columns:
            continue
        target_val = df_peers.loc[ticker, metric]
        peer_vals  = peer_only[metric].dropna()
        if pd.isna(target_val) or len(peer_vals) == 0:
            continue
        peer_median  = peer_vals.median()
        industry_avg = peer_vals.mean()
        dev          = compute_deviation(target_val, peer_median)
        cl           = classify_signal(dev)
        if cl["sentiment"] == "positive":
            msg = "Priced below peer and industry references; relative valuation looks discounted."
        elif cl["sentiment"] == "negative":
            msg = "Priced above peer and industry references; market is assigning a premium."
        else:
            msg = "Priced close to peer and industry references; valuation is in-line."
        sig = _make_signal(
            metric=METRIC_DISPLAY.get(metric, metric), current_value=target_val,
            baseline_value=peer_median, baseline_label="Peer median", deviation_pct=dev,
            sentiment=cl["sentiment"], strength=cl["strength"], message=msg,
            data_quality=len(peer_vals), unit="x",
        )
        sig["extra_context"] = f"Industry avg: {industry_avg:.1f}x"
        signals.append(sig)
    return signals


def rank_signals(signals: List[dict]) -> List[dict]:
    strength_order = {"strong": 3, "moderate": 2, "neutral": 1}
    return sorted(
        signals,
        key=lambda s: (
            strength_order.get(s.get("strength", "neutral"), 0),
            abs(0 if pd.isna(s.get("deviation_pct", np.nan)) else s.get("deviation_pct", 0)),
            s.get("data_quality", 0),
        ),
        reverse=True,
    )


def build_overview_signals(historical_signals: List[dict], peer_signals: List[dict]) -> List[dict]:
    return []


def build_selected_historical_signal(df_multiples: pd.DataFrame, selected_metric: str) -> Optional[dict]:
    if selected_metric not in df_multiples.columns:
        return None
    series = df_multiples[selected_metric].dropna()
    if len(series) < 2:
        return None
    current = series.iloc[-1]
    avg     = series.mean()
    dev     = compute_deviation(current, avg)
    cl      = classify_signal(dev)
    disp    = METRIC_DISPLAY.get(selected_metric, selected_metric)
    is_eps  = (selected_metric == "EPS")

    if is_eps:
        if cl["sentiment"] == "positive":
            msg = "EPS is running above its 5Y average, supporting stronger earnings quality."
        elif cl["sentiment"] == "negative":
            msg = "EPS is below its 5Y average, indicating weaker recent earnings power."
        else:
            msg = "EPS is close to its 5Y average, indicating stable earnings power."
        return _make_signal("EPS", current, avg, "5Y avg EPS", dev,
                            cl["sentiment"], cl["strength"], msg, len(series), unit="$")

    if cl["sentiment"] == "positive":
        msg = f"{disp} is below its 5Y average, pointing to a valuation discount versus history."
    elif cl["sentiment"] == "negative":
        msg = f"{disp} is above its 5Y average, indicating valuation is pricing in stronger expectations."
    else:
        msg = f"{disp} is near its 5Y average, suggesting valuation is broadly in line with history."
    return _make_signal(disp, current, avg, "5Y avg", dev,
                        cl["sentiment"], cl["strength"], msg, len(series), unit="x")


def build_selected_peer_signal(df_peers: pd.DataFrame, ticker: str,
                                selected_metric: str) -> Optional[dict]:
    if selected_metric == "EPS":
        return _make_signal("EPS", np.nan, np.nan, "", np.nan, "neutral", "neutral",
                            "Peer insight for EPS is not shown because the peer section is multiple-based.",
                            0, unit="")
    if ticker not in df_peers.index or selected_metric not in df_peers.columns:
        return None
    peer_only  = df_peers[df_peers.index != ticker]
    if peer_only.empty:
        return None
    target_val   = df_peers.loc[ticker, selected_metric]
    peer_vals    = peer_only[selected_metric].dropna()
    if pd.isna(target_val) or len(peer_vals) == 0:
        return None
    peer_median  = peer_vals.median()
    industry_avg = peer_vals.mean()
    dev          = compute_deviation(target_val, peer_median)
    cl           = classify_signal(dev)
    disp         = METRIC_DISPLAY.get(selected_metric, selected_metric)
    if cl["sentiment"] == "positive":
        msg = f"{disp} screens cheaper than peers and industry."
    elif cl["sentiment"] == "negative":
        msg = f"{disp} screens richer than peers and industry."
    else:
        msg = f"{disp} is broadly aligned with peers and industry."
    sig = _make_signal(disp, target_val, peer_median, "Peer median", dev,
                       cl["sentiment"], cl["strength"], msg, len(peer_vals), unit="x")
    sig["extra_context"] = f"Industry avg: {industry_avg:.1f}x"
    return sig


# ── Insight rendering (using _verdict_card, consistent with ratios.py) ────────

def render_insight_cards(signals: List[dict], limit: int = 6) -> None:
    """Render Key Insight verdict cards using the shared _verdict_card component."""
    if not signals:
        st.info("Insufficient data to generate insights.")
        return

    for signal in signals[:limit]:
        sentiment     = signal.get("sentiment", "neutral")
        current_val   = signal.get("current_value", np.nan)
        baseline_val  = signal.get("baseline_value", np.nan)
        baseline_lbl  = signal.get("baseline_label", "")
        unit          = signal.get("unit", "x")
        extra_context = signal.get("extra_context", "")
        verdict_color = _deviation_to_color(signal.get("deviation_pct", np.nan))

        # Format current value for verdict line
        if not pd.isna(current_val):
            if unit == "$":
                current_str = f"${current_val:.2f}"
            elif unit == "x":
                current_str = f"{current_val:.1f}x"
            else:
                current_str = ""
        else:
            current_str = ""

        # Format baseline for body
        if not pd.isna(baseline_val):
            if unit == "$":
                baseline_str = f"${baseline_val:.2f}"
            else:
                baseline_str = f"{baseline_val:.1f}x"
            baseline_part = f" {baseline_lbl}: {baseline_str}." if baseline_lbl else ""
        else:
            baseline_part = ""

        body_parts = [signal.get("message", "")]
        if baseline_part:
            body_parts.insert(0, baseline_part.strip())
        if extra_context:
            body_parts.append(extra_context)
        body = " ".join(p for p in body_parts if p)

        verdict_label = current_str if current_str else classify_signal(
            signal.get("deviation_pct", np.nan))["label"]

        st.markdown(_verdict_card(
            title=signal.get("metric", "Metric"),
            verdict=verdict_label,
            verdict_color=verdict_color,
            body=body,
        ), unsafe_allow_html=True)


# ── KPI strip ─────────────────────────────────────────────────────────────────

def _render_kpi_strip(info: dict, df_multiples: pd.DataFrame, ticker: str) -> None:
    """Top-of-page metric cards — intentionally empty for this view."""
    return


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_trend(year_labels: List[str], values: List[float],
                 metric: str, is_eps: bool = False, height: int = 500) -> go.Figure:
    """Line + fill chart for historical multiples — mirrors ratios.py style."""
    unit      = "$" if is_eps else "x"
    clean     = [v for v in values if not pd.isna(v)]
    avg       = float(np.mean(clean)) if clean else None
    final     = values[-1] if values else np.nan

    line_color = BLUE
    if avg is not None and not pd.isna(final):
        line_color = UP if final >= avg else DOWN

    r_hex = line_color.lstrip("#")
    rc, gc, bc = int(r_hex[0:2], 16), int(r_hex[2:4], 16), int(r_hex[4:6], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_labels, y=values,
        mode="lines+markers",
        name=metric,
        line=dict(color=line_color, width=3),
        marker=dict(size=8, color=line_color),
        fill="tozeroy",
        fillcolor=f"rgba({rc},{gc},{bc},0.08)",
        hovertemplate=f"<b>%{{x}}</b><br>{metric}: <b>%{{y:.2f}}{unit}</b><extra></extra>",
    ))

    if avg is not None:
        avg_label = f"${avg:.2f}" if is_eps else f"{avg:.1f}x"
        fig.add_hline(y=avg, line=dict(color=ORANGE, width=1.5, dash="dash"))
        fig.add_annotation(
            xref="paper", x=1.02, yref="y", y=avg,
            text=f"Avg: {avg_label}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11, color=ORANGE, family="'SF Pro Display','Segoe UI',sans-serif"),
            bgcolor="rgba(0,0,0,0.45)", borderpad=3,
        )

    if not pd.isna(final):
        sign         = "▲" if final >= (avg or final) else "▼"
        annot_color  = UP if final >= (avg or final) else DOWN
        final_label  = f"${final:.2f}" if is_eps else f"{final:.1f}x"
        fig.add_annotation(
            xref="paper", x=1.02, yref="y", y=final,
            text=f"{sign} {final_label}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11, color=annot_color, family="'SF Pro Display','Segoe UI',sans-serif"),
            bgcolor="rgba(0,0,0,0.45)", borderpad=3,
        )

    tick_suffix = "$" if is_eps else "x"
    fig.update_layout(
        **_CHART_LAYOUT,
        height=height,
        margin=dict(l=55, r=90, t=30, b=40),
        showlegend=False,
        xaxis=dict(
            type="category", showgrid=True,
            gridcolor="rgba(255,255,255,0.1)", zeroline=False,
            tickfont=dict(color="#ffffff"),
        ),
        yaxis=dict(
            tickprefix="$" if is_eps else "",
            ticksuffix="" if is_eps else "x",
            showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
            tickfont=dict(color="#ffffff"),
            title=dict(text="", font=dict(color="#ffffff")),
        ),
    )
    return fig


def _chart_peer_bar(df_peers: pd.DataFrame, ticker: str,
                    metric: str, df_peers_only: pd.DataFrame,
                    height: int = 500) -> go.Figure:
    """Horizontal bar chart for peer comparison — consistent with design system."""
    df_sorted = df_peers.sort_values(by=metric, ascending=True, na_position="last")
    labels    = df_sorted.index.tolist()
    vals      = df_sorted[metric].tolist()

    colors = []
    for lbl in labels:
        if lbl == ticker:
            colors.append(BLUE)
        else:
            colors.append(f"rgba(10,124,255,0.35)")

    # Add industry average
    if not df_peers_only.empty and metric in df_peers_only.columns:
        ind_avg = df_peers_only[metric].mean()
        if not pd.isna(ind_avg):
            labels.append("Industry Avg")
            vals.append(ind_avg)
            colors.append(ORANGE)

    fig = go.Figure(go.Bar(
        y=labels, x=vals,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.1)", width=0.5)),
        text=[f"{v:.1f}x" if (not pd.isna(v)) else "N/A" for v in vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}x<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_LAYOUT,
        height=height,
        margin=dict(l=120, r=20, t=10, b=20),
        xaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   zeroline=False, tickfont=dict(color="#ffffff"), ticksuffix="x"),
        yaxis=dict(showgrid=False, tickfont=dict(color="#ffffff")),
        showlegend=False,
    )
    return fig


# ── Stats table helper ────────────────────────────────────────────────────────

def _build_stats_table(df_multiples: pd.DataFrame, metric: str,
                        last5: pd.Index, is_eps: bool = False) -> pd.DataFrame:
    year_cols  = [str(y) for y in last5]
    rows: List[dict] = []

    main_row = {"Metric": metric}
    for y in last5:
        main_row[str(y)] = _fmt_multiple(df_multiples.loc[y, metric] if y in df_multiples.index else np.nan, is_eps)
    rows.append(main_row)

    values = df_multiples[metric].dropna() if metric in df_multiples.columns else pd.Series(dtype=float)
    if len(values) > 0:
        for stat_label, fn in [("Average", "mean"), ("High", "max"), ("Low", "min"), ("Median", "median")]:
            row = {"Metric": stat_label}
            for i, y in enumerate(last5):
                slice_ = df_multiples[metric].iloc[:i + 1].dropna()
                row[str(y)] = _fmt_multiple(getattr(slice_, fn)() if len(slice_) > 0 else np.nan, is_eps)
            rows.append(row)

    return pd.DataFrame(rows)[["Metric"] + year_cols]


# ── Main render ───────────────────────────────────────────────────────────────

def render_multiples(ticker: str) -> None:
    _inject_css()

    st.markdown(
        f'<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:4px;">'
        f'{ticker}',
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner(f"Loading financial data for {ticker}…"):
        info       = get_yf_info(ticker)
        yf_fin     = get_yf_financials(ticker)
        yf_balance = get_yf_balance_sheet(ticker)
        yf_history = get_yf_history(ticker, period="6y")

    last5 = get_last_n_years_from_dfs([yf_fin, yf_balance], n=5)
    if len(last5) == 0:
        st.warning("No historical data available for multiples analysis.")
        return

    # ── Calculate historical multiples ────────────────────────────────────────
    with st.spinner("Computing multiples…"):
        historical_multiples: Dict[int, Dict] = {}
        approx_flags_all: set = set()
        for year in last5:
            multiples, flags = calculate_multiples_for_year(
                ticker, year, yf_fin, yf_balance, info, yf_history)
            historical_multiples[year] = multiples
            approx_flags_all.update(flags)

    df_multiples = pd.DataFrame(historical_multiples).T
    df_multiples.index.name = "Year"

    year_labels  = [f"FY{str(y)[2:]}" for y in last5]
    core_metrics = ["TEV/Revenue", "TEV/EBITDA", "TEV/EBIT", "P/E", "MCap/Revenue"]
    avail_metrics = [m for m in core_metrics if m in df_multiples.columns]
    if "EPS" in df_multiples.columns:
        avail_metrics.append("EPS")

    # ── Pre-load peer data (before tabs so spinners don't fight tab rendering) ─
    uploaded_universe = st.session_state.get("uploaded_universe", None)
    company_df        = st.session_state.get("company_df", pd.DataFrame())
    sic_map           = st.session_state.get("sic_map", {})

    with st.spinner("Finding comparable companies…"):
        try:
            peer_tickers = find_best_peers_automated(
                ticker=ticker, uploaded_universe=uploaded_universe, max_peers=12)
        except Exception:
            peer_tickers = []
        if not peer_tickers and (not company_df.empty and sic_map):
            peer_tickers = find_peers_by_sic(ticker, sic_map, company_df,
                                             min_peers=4, max_peers=8)

    df_peers      = pd.DataFrame()
    df_peers_only = pd.DataFrame()
    peer_data: List[dict] = []

    if peer_tickers:
        with st.spinner("Loading peer data…"):
            for peer in peer_tickers:
                try:
                    p_info    = get_yf_info(peer)
                    p_fin     = get_yf_financials(peer)
                    p_balance = get_yf_balance_sheet(peer)
                    p_history = get_yf_history(peer, period="6y")
                    p_years   = get_last_5_years(p_fin)
                    if len(p_years) == 0:
                        continue
                    p_multiples, _ = calculate_multiples_for_year(
                        peer, p_years[-1], p_fin, p_balance, p_info, p_history)
                    if any(pd.notna(v) for k, v in p_multiples.items() if k != "EPS"):
                        p_multiples["Ticker"] = peer
                        peer_data.append(p_multiples)
                except Exception:
                    continue

        target_latest = historical_multiples[last5[-1]].copy()
        target_latest["Ticker"] = ticker
        peer_data.append(target_latest)

        if len(peer_data) > 1:
            df_peers      = pd.DataFrame(peer_data).set_index("Ticker")
            peer_cols     = [m for m in core_metrics if m in df_peers.columns]
            df_peers      = df_peers[peer_cols].dropna(how="all").dropna(axis=1, how="all")
            df_peers_only = df_peers[df_peers.index != ticker]

    # Pre-compute historical signals (shared between both tabs)
    historical_signals = build_historical_signals(df_multiples)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Historical Multiples", "Public Comparables"])

    # ════════════ TAB 0 — HISTORICAL MULTIPLES ════════════════════════════════
    with tabs[0]:
        col_left, col_right = st.columns([0.5, 0.5])

        with col_left:
            selected_metric = st.radio(
                "Metric selector",
                options=avail_metrics,
                index=0,
                horizontal=True,
                key="historical_metric_selector",
                label_visibility="collapsed",
            )
            is_eps   = (selected_metric == "EPS")
            df_stats = _build_stats_table(df_multiples, selected_metric, last5, is_eps)
            table_h  = min(240, 56 + 36 * len(df_stats))
            st.dataframe(df_stats, hide_index=True, use_container_width=True, height=table_h)

            # What does this metric mean?
            desc = METRIC_DESCRIPTIONS.get(selected_metric, "")
            if desc:
                st.markdown(_verdict_card(
                    title="What does this measure?",
                    verdict=selected_metric,
                    verdict_color=BLUE,
                    body=desc,
                ), unsafe_allow_html=True)

            # Key Insights
            ranked_hist          = rank_signals(historical_signals)
            overview_signals     = build_overview_signals(ranked_hist, [])
            selected_hist_signal = build_selected_historical_signal(df_multiples, selected_metric)
            cards = list(overview_signals)
            if selected_hist_signal is not None:
                cards.append(selected_hist_signal)
            render_insight_cards(cards, limit=3)

        with col_right:
            selected_values = (df_multiples[selected_metric].tolist()
                               if selected_metric in df_multiples.columns else [])
            n_hist_cards    = max(1, min(3, len(cards)))
            chart_height    = max(360, table_h + 60 + (115 if desc else 0) + (n_hist_cards * 115))

            if all(pd.isna(v) for v in selected_values):
                st.info("Insufficient data to plot a trend for this metric.")
            else:
                fig = _chart_trend(year_labels, selected_values, selected_metric, is_eps,
                                   height=chart_height)
                st.plotly_chart(fig, use_container_width=True)

    # ════════════ TAB 1 — PUBLIC COMPARABLES ══════════════════════════════════
    with tabs[1]:
        if df_peers.empty:
            st.info("No peers found with sufficient data for comparison.")
        else:
            # Metric selector for peer tab — mirrors the radio pill pattern
            peer_metric = st.radio(
                "Peer metric selector",
                options=list(df_peers.columns),
                index=0,
                horizontal=True,
                key="peer_metric_selector",
                label_visibility="collapsed",
            )

            col_left, col_right = st.columns([0.5, 0.5])

            with col_left:
                # Peer table with industry average row
                if not df_peers_only.empty:
                    industry_row     = pd.DataFrame([df_peers_only.mean()], index=["Industry Avg"])
                    df_peers_display = pd.concat([df_peers, industry_row])
                else:
                    df_peers_display = df_peers.copy()

                df_fmt = df_peers_display.copy()
                for col in df_fmt.columns:
                    df_fmt[col] = df_fmt[col].apply(lambda x: _fmt_multiple(x))
                df_fmt = df_fmt[df_fmt.apply(lambda row: any(v != "—" for v in row), axis=1)]

                peer_table_h = min(240, 56 + 36 * len(df_fmt))
                st.dataframe(df_fmt, hide_index=False, use_container_width=True, height=peer_table_h)

                desc = METRIC_DESCRIPTIONS.get(peer_metric, "")
                if desc:
                    st.markdown(_verdict_card(
                        title="What does this measure?",
                        verdict=peer_metric,
                        verdict_color=BLUE,
                        body=desc,
                    ), unsafe_allow_html=True)

                # Key Insights (peer)
                peer_signals         = build_peer_signals(df_peers, ticker)
                ranked_peer          = rank_signals(peer_signals)
                peer_overview        = build_overview_signals(historical_signals, ranked_peer)
                # Use whichever metric the user had selected in Tab 0 as focus
                selected_peer_signal = build_selected_peer_signal(
                    df_peers, ticker,
                    peer_metric if peer_metric in df_peers.columns else list(df_peers.columns)[0],
                )
                peer_cards = list(peer_overview)
                if selected_peer_signal is not None:
                    peer_cards.append(selected_peer_signal)
                render_insight_cards(peer_cards, limit=3)

            with col_right:
                if peer_metric in df_peers.columns:
                    n_peer_cards = max(1, min(3, len(peer_cards)))
                    peer_chart_height = max(360, peer_table_h + 60 + (115 if desc else 0) + (n_peer_cards * 115))
                    fig_peer = _chart_peer_bar(
                        df_peers, ticker, peer_metric, df_peers_only, height=peer_chart_height)
                    st.plotly_chart(fig_peer, use_container_width=True)
                else:
                    st.info("No data available for the selected metric.")


if __name__ == "__main__":
    render_multiples("AAPL")
