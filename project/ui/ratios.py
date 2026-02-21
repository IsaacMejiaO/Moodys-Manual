# ui/ratios.py
"""
Financial Ratios Page — Plain-English Edition
=============================================
Design philosophy: "Your company's health, told simply."
Mirrors the layout and formatting of performance.py.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sec_engine.aggregation import build_company_summary

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Design tokens (matches performance.py exactly) ────────────────────────────
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

# ── Category definitions ──────────────────────────────────────────────────────
CATEGORIES: List[str] = ["Profitability", "Margins", "Efficiency", "Liquidity", "Leverage", "Growth"]

CATEGORY_METRICS: Dict[str, List[str]] = {
    "Profitability": ["ROA", "ROIC", "ROE", "RCE"],
    "Margins":       ["Gross Margin", "SG&A Margin", "R&D Margin", "EBITDA Margin", "EBIT Margin", "Net Margin"],
    "Efficiency":    ["Asset Turnover", "AR Turnover", "Inventory Turnover", "DSO", "DIO", "DPO"],
    "Liquidity":     ["Current Ratio", "Quick Ratio"],
    "Leverage":      ["Total D/E", "Total D/Capital", "LT D/E", "Total Liab/Assets"],
    "Growth":        ["Revenue Growth", "Gross Profit Growth", "EBIT Growth", "Net Income Growth"],
}

UNIT_MAP: Dict[str, str] = {
    "ROA": "%", "ROIC": "%", "ROE": "%", "RCE": "%",
    "Gross Margin": "%", "SG&A Margin": "%", "R&D Margin": "%",
    "EBITDA Margin": "%", "EBIT Margin": "%", "Net Margin": "%",
    "Asset Turnover": "x", "AR Turnover": "x", "Inventory Turnover": "x",
    "DSO": "days", "DIO": "days", "DPO": "days",
    "Current Ratio": "x", "Quick Ratio": "x",
    "Total D/E": "x", "Total D/Capital": "x", "LT D/E": "x", "Total Liab/Assets": "x",
    "Revenue Growth": "%", "Gross Profit Growth": "%", "EBIT Growth": "%", "Net Income Growth": "%",
}

# Metrics where lower is better (inverse sentiment)
INVERSE_MAP: Dict[str, bool] = {
    "DSO": True, "DIO": True, "DPO": False,
    "Total D/E": True, "Total D/Capital": True, "LT D/E": True, "Total Liab/Assets": True,
    "SG&A Margin": True, "R&D Margin": True,
}

# Plain-English descriptions for Key Insight cards
METRIC_DESCRIPTIONS: Dict[str, str] = {
    "ROA":                  "Return on Assets — how many dollars of profit the company generates per dollar of assets.",
    "ROIC":                 "Return on Invested Capital — the yield on every dollar shareholders and lenders put in.",
    "ROE":                  "Return on Equity — profits earned per dollar of shareholder equity.",
    "RCE":                  "Return on Capital Employed — operating profit relative to the long-term capital base.",
    "Gross Margin":         "Revenue left after paying for direct production costs. Higher = more pricing power.",
    "SG&A Margin":          "How much of revenue is spent on selling, general, and admin costs. Lower = leaner.",
    "R&D Margin":           "Investment in innovation as a share of revenue.",
    "EBITDA Margin":        "Operating earnings before non-cash charges. A proxy for underlying cash profitability.",
    "EBIT Margin":          "Operating profit margin — what's left after all operating costs.",
    "Net Margin":           "Bottom-line profit margin — how many cents of net profit per dollar of revenue.",
    "Asset Turnover":       "How efficiently assets generate sales. Higher = more productive asset base.",
    "AR Turnover":          "How quickly the company collects money owed by customers.",
    "Inventory Turnover":   "How quickly inventory is sold and replaced. Higher = faster-moving stock.",
    "DSO":                  "Days Sales Outstanding — average days to collect a payment. Fewer days is better.",
    "DIO":                  "Days Inventory Outstanding — how long goods sit before being sold. Fewer is better.",
    "DPO":                  "Days Payable Outstanding — how long the company takes to pay suppliers.",
    "Current Ratio":        "Short-term assets vs. short-term debts. Above 1x means the company can cover near-term bills.",
    "Quick Ratio":          "Like Current Ratio but strips out inventory. A stricter measure of near-term solvency.",
    "Total D/E":            "Total debt relative to equity. Higher = more leveraged, more financial risk.",
    "Total D/Capital":      "Debt as a fraction of total capital. Above 50% signals heavy reliance on borrowing.",
    "LT D/E":               "Long-term debt vs. equity. Focuses on structural leverage, not short-term lines.",
    "Total Liab/Assets":    "What fraction of assets are funded by liabilities. Above 0.7x is elevated.",
    "Revenue Growth":       "Year-over-year top-line growth. The engine of everything else.",
    "Gross Profit Growth":  "Growth in gross dollars after production costs — shows pricing & volume combined.",
    "EBIT Growth":          "Growth in operating profit — measures operating leverage.",
    "Net Income Growth":    "Bottom-line profit growth — what flows to shareholders.",
}


# ── UI helpers (mirrors performance.py) ──────────────────────────────────────

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
    div[data-testid="stDataFrame"] {{ background: transparent !important; }}
    .stRadio > label, .stMultiSelect > label {{ color: #ffffff !important; }}
    .stRadio div[role="radiogroup"] label, .stRadio div[role="radiogroup"] span {{ color: #ffffff !important; }}
    .stCaption p, small {{ color: #ffffff !important; opacity: 0.7; }}
    .block-container {{ background: transparent !important; }}
    p, span, label, div {{ color: #ffffff; }}
    .stSelectbox label {{ color: #ffffff !important; }}
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


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_data(ticker: str) -> Dict:
    """Fetch all financial data for a ticker. yfinance primary."""
    ltm_data, balance_data, metadata, historical_data = {}, {}, {}, {}
    income_stmt = pd.DataFrame()
    balance_sheet = pd.DataFrame()

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        income_stmt        = getattr(stock, "income_stmt",           pd.DataFrame())
        balance_sheet      = getattr(stock, "balance_sheet",         pd.DataFrame())
        cashflow           = getattr(stock, "cashflow",              pd.DataFrame())
        q_income           = getattr(stock, "quarterly_income_stmt", pd.DataFrame())
        q_balance          = getattr(stock, "quarterly_balance_sheet",pd.DataFrame())
        q_cashflow         = getattr(stock, "quarterly_cashflow",    pd.DataFrame())

        metadata = {
            "name":               info.get("longName", info.get("shortName", ticker)),
            "industry":           info.get("industry", "Unknown"),
            "sector":             info.get("sector",   "Unknown"),
            "market_cap":         info.get("marketCap", np.nan),
            "pe_ltm":             info.get("trailingPE", np.nan),
            "eps_growth_pct":     (info.get("earningsGrowth", 0) or 0) * 100,
            "dividend_yield_pct": (info.get("dividendYield",  0) or 0) * 100,
        }

        def ltm(df, field):
            if df is None or df.empty or field not in df.index: return np.nan
            vals = df.loc[field].head(4)
            return float(vals.sum()) if len(vals) >= 4 else np.nan

        def latest(df, field):
            if df is None or df.empty or field not in df.index: return np.nan
            return float(df.loc[field].iloc[0])

        ltm_data = {
            "revenue":          ltm(q_income,   "Total Revenue"),
            "gross_profit":     ltm(q_income,   "Gross Profit"),
            "operating_income": ltm(q_income,   "Operating Income"),
            "net_income":       ltm(q_income,   "Net Income"),
            "sga":              ltm(q_income,   "Selling General And Administration"),
            "rd":               ltm(q_income,   "Research And Development"),
            "cogs":             ltm(q_income,   "Cost Of Revenue"),
            "interest_expense": ltm(q_income,   "Interest Expense"),
            "ebitda":           ltm(q_income,   "EBITDA"),
            "depreciation":     ltm(q_cashflow, "Depreciation And Amortization"),
            "amortization":     np.nan,
            "ocf":              ltm(q_cashflow, "Operating Cash Flow"),
            "capex":            ltm(q_cashflow, "Capital Expenditure"),
        }

        balance_data = {
            "total_assets":       latest(q_balance, "Total Assets"),
            "current_assets":     latest(q_balance, "Current Assets"),
            "cash":               latest(q_balance, "Cash And Cash Equivalents"),
            "accounts_receivable":latest(q_balance, "Accounts Receivable"),
            "inventory":          latest(q_balance, "Inventory"),
            "ppe":                latest(q_balance, "Net PPE"),
            "total_liabilities":  latest(q_balance, "Total Liabilities Net Minority Interest"),
            "current_liabilities":latest(q_balance, "Current Liabilities"),
            "accounts_payable":   latest(q_balance, "Accounts Payable"),
            "long_term_debt":     latest(q_balance, "Long Term Debt"),
            "debt":               latest(q_balance, "Total Debt"),
            "equity":             latest(q_balance, "Stockholders Equity"),
            "retained_earnings":  latest(q_balance, "Retained Earnings"),
        }

        def annual_series(df, field):
            if df is None or df.empty or field not in df.index: return None
            s = df.loc[field].sort_index()
            return s if not s.empty else None

        ocf_h   = annual_series(cashflow, "Operating Cash Flow")
        capex_h = annual_series(cashflow, "Capital Expenditure")
        lfcf_h  = None
        if ocf_h is not None and capex_h is not None:
            combined = pd.concat([ocf_h, capex_h], axis=1, keys=["ocf","capex"]).dropna()
            if not combined.empty:
                lfcf_h = combined["ocf"] - combined["capex"]

        historical_data = {
            "revenue_history":          annual_series(income_stmt,   "Total Revenue"),
            "gross_profit_history":     annual_series(income_stmt,   "Gross Profit"),
            "ebit_history":             annual_series(income_stmt,   "Operating Income"),
            "ebitda_history":           annual_series(income_stmt,   "EBITDA"),
            "net_income_history":       annual_series(income_stmt,   "Net Income"),
            "ar_history":               annual_series(balance_sheet, "Accounts Receivable"),
            "inventory_history":        annual_series(balance_sheet, "Inventory"),
            "ppe_history":              annual_series(balance_sheet, "Net PPE"),
            "total_assets_history":     annual_series(balance_sheet, "Total Assets"),
            "total_liabilities_history":annual_series(balance_sheet, "Total Liabilities Net Minority Interest"),
            "equity_history":           annual_series(balance_sheet, "Stockholders Equity"),
            "lfcf_history":             lfcf_h,
            "eps_history":              None,
            "diluted_eps_history":      annual_series(income_stmt,   "Diluted EPS"),
        }

    except Exception as e:
        st.warning(f"⚠️ Data fetch incomplete for {ticker}: {e}")

    return {
        "ltm_data":         ltm_data,
        "balance_data":     balance_data,
        "metadata":         metadata,
        "income_stmt":      income_stmt,
        "balance_sheet_stmt": balance_sheet,
        **historical_data,
    }


# ── Ratio calculations ────────────────────────────────────────────────────────

def _get_val(df: pd.DataFrame, field: str, year: int) -> float:
    """Extract a value from an annual DataFrame column matching the given year."""
    if df is None or df.empty or field not in df.index:
        return np.nan
    for col in df.columns:
        if pd.to_datetime(col).year == year:
            return pd.to_numeric(df.loc[field, col], errors="coerce")
    return np.nan


def _calc_ratio(year: int, income_stmt: pd.DataFrame,
                balance_sheet: pd.DataFrame, metric: str) -> float:
    """Calculate a single ratio for a given fiscal year."""
    g = lambda f: _get_val(income_stmt,   f, year)
    b = lambda f: _get_val(balance_sheet, f, year)

    revenue      = g("Total Revenue");          gross_profit = g("Gross Profit")
    ebit         = g("Operating Income");        ebitda       = g("EBITDA")
    net_income   = g("Net Income");              sga          = g("Selling General And Administration")
    rd           = g("Research And Development"); cogs         = g("Cost Of Revenue")
    total_assets = b("Total Assets");            equity       = b("Stockholders Equity")
    curr_assets  = b("Current Assets");          curr_liab    = b("Current Liabilities")
    inventory    = b("Inventory");               debt         = b("Total Debt")
    lt_debt      = b("Long Term Debt");          ar           = b("Accounts Receivable")
    ap           = b("Accounts Payable");        total_liab   = b("Total Liabilities Net Minority Interest")

    def safe_div(n, d): return (n / d) if (d and d != 0 and not pd.isna(n) and not pd.isna(d)) else np.nan

    if metric == "ROA":               return safe_div(net_income,    total_assets) * 100
    if metric == "ROIC":
        ic = (debt or 0) + (equity or 0)
        return safe_div(ebit, ic) * 100
    if metric == "ROE":               return safe_div(net_income,    equity) * 100
    if metric == "RCE":
        ce = (total_assets or 0) - (curr_liab or 0)
        return safe_div(ebit, ce) * 100
    if metric == "Gross Margin":      return safe_div(gross_profit,  revenue) * 100
    if metric == "SG&A Margin":       return safe_div(sga,           revenue) * 100
    if metric == "R&D Margin":        return safe_div(rd,            revenue) * 100
    if metric == "EBITDA Margin":     return safe_div(ebitda,        revenue) * 100
    if metric == "EBIT Margin":       return safe_div(ebit,          revenue) * 100
    if metric == "Net Margin":        return safe_div(net_income,    revenue) * 100
    if metric == "Asset Turnover":    return safe_div(revenue,       total_assets)
    if metric == "AR Turnover":       return safe_div(revenue,       ar)
    if metric == "Inventory Turnover":return safe_div(cogs,          inventory)
    if metric == "DSO":               return safe_div(ar,            revenue) * 365
    if metric == "DIO":               return safe_div(inventory,     cogs)    * 365
    if metric == "DPO":               return safe_div(ap,            cogs)    * 365
    if metric == "Current Ratio":     return safe_div(curr_assets,   curr_liab)
    if metric == "Quick Ratio":
        qa = (curr_assets or 0) - (inventory or 0)
        return safe_div(qa, curr_liab)
    if metric == "Total D/E":         return safe_div(debt,          equity)
    if metric == "Total D/Capital":
        capital = (debt or 0) + (equity or 0)
        return safe_div(debt, capital)
    if metric == "LT D/E":            return safe_div(lt_debt,       equity)
    if metric == "Total Liab/Assets": return safe_div(total_liab,    total_assets)
    return np.nan


def _calc_growth(series: Optional[pd.Series], year: int) -> float:
    """YoY growth for a given annual series."""
    if series is None or series.empty: return np.nan
    def _val(y):
        for idx in series.index:
            if idx.year == y: return series.loc[idx]
        return np.nan
    curr, prev = _val(year), _val(year - 1)
    if pd.isna(curr) or pd.isna(prev) or prev == 0: return np.nan
    return ((curr - prev) / abs(prev)) * 100


GROWTH_SERIES_MAP = {
    "Revenue Growth":       "revenue_history",
    "Gross Profit Growth":  "gross_profit_history",
    "EBIT Growth":          "ebit_history",
    "Net Income Growth":    "net_income_history",
}


def _get_value_for_metric(metric: str, year: int, income_stmt: pd.DataFrame,
                           balance_sheet: pd.DataFrame, data: Dict) -> float:
    if metric in GROWTH_SERIES_MAP:
        return _calc_growth(data.get(GROWTH_SERIES_MAP[metric]), year)
    return _calc_ratio(year, income_stmt, balance_sheet, metric)


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt(value: float, unit: str) -> str:
    if pd.isna(value): return "—"
    if unit == "%":    return f"{value:.1f}%"
    if unit == "x":    return f"{value:.2f}x"
    if unit == "days": return f"{value:.0f}d"
    return f"{value:.2f}"


def _fiscal_years(data: Dict) -> List[int]:
    """Deduce the last 4 available fiscal years from any available series."""
    all_years: set = set()
    for key, series in data.items():
        if isinstance(series, pd.Series) and not series.empty:
            all_years.update(idx.year for idx in series.index)
    return sorted(all_years, reverse=True)[:4]


# ── Verdict / signal helpers ──────────────────────────────────────────────────

def _signal(value: float, values: List[float], inverse: bool = False) -> Tuple[str, str]:
    """
    Return (verdict_text, color) by comparing current value to its own history.
    """
    clean = [v for v in values if not pd.isna(v)]
    if len(clean) < 2 or pd.isna(value):
        return "Not enough data", GREY
    avg = float(np.mean(clean))
    if avg == 0:
        return "Stable", GREY
    dev = ((value / avg) - 1) * 100
    if dev >= 15:  return "Improving",    UP
    if dev >= 5:   return "Above average", UP
    if dev >= -5:  return "Stable",        ORANGE
    if dev >= -15: return "Below average", ORANGE
    return "Declining", DOWN


def _ltm_signal(ltm_val: float, label: str, unit: str) -> Tuple[str, str]:
    """Color the LTM value card based on known healthy ranges."""
    if pd.isna(ltm_val): return GREY, "N/A"
    thresholds: Dict[str, Tuple[float, float]] = {
        "ROA": (5, 15), "ROIC": (10, 20), "ROE": (10, 20), "RCE": (10, 20),
        "Gross Margin": (30, 60), "EBITDA Margin": (15, 35),
        "EBIT Margin": (10, 25), "Net Margin": (5, 20),
        "Current Ratio": (1.5, 3.0), "Quick Ratio": (1.0, 2.5),
    }
    inverse_labels = {"DSO", "DIO", "Total D/E", "Total D/Capital",
                      "LT D/E", "Total Liab/Assets", "SG&A Margin"}
    lo, hi = thresholds.get(label, (None, None))
    if lo is None:
        return BLUE, _fmt(ltm_val, unit)
    if label in inverse_labels:
        color = UP if ltm_val <= lo else (ORANGE if ltm_val <= hi else DOWN)
    else:
        color = UP if ltm_val >= hi else (ORANGE if ltm_val >= lo else DOWN)
    return color, _fmt(ltm_val, unit)


# ── Chart ─────────────────────────────────────────────────────────────────────

def _chart_ratio_trend(year_labels: List[str], values: List[float],
                        metric: str, unit: str, height: int = 420) -> go.Figure:
    clean = [v for v in values if not pd.isna(v)]
    avg   = float(np.mean(clean)) if clean else None
    is_down = INVERSE_MAP.get(metric, False)

    is_leverage_metric = metric in CATEGORY_METRICS.get("Leverage", [])

    # Color: follow trend direction for leverage (up=green, down=red)
    if avg and not pd.isna(values[-1]):
        if is_leverage_metric:
            final_better = values[-1] > avg
        else:
            final_better = (values[-1] < avg) if is_down else (values[-1] > avg)
        line_color = UP if final_better else DOWN
    else:
        line_color = BLUE

    r_hex = line_color.lstrip("#")
    rc, gc, bc = int(r_hex[0:2], 16), int(r_hex[2:4], 16), int(r_hex[4:6], 16)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=year_labels,
        y=values,
        mode="lines+markers",
        name=metric,
        line=dict(color=line_color, width=3),
        marker=dict(size=8, color=line_color),
        fill="tozeroy",
        fillcolor=f"rgba({rc},{gc},{bc},0.08)",
        hovertemplate=f"<b>%{{x}}</b><br>{metric}: <b>%{{y:.2f}}</b><extra></extra>",
    ))

    if avg is not None:
        fig.add_hline(
            y=avg,
            line=dict(color=ORANGE, width=1.5, dash="dash"),
        )
        fig.add_annotation(
            xref="paper", x=1.02,
            yref="y",     y=avg,
            text=f"Avg: {_fmt(avg, unit)}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11, color=ORANGE,
                      family="'SF Pro Display','Segoe UI',sans-serif"),
            bgcolor="rgba(0,0,0,0.45)", borderpad=3,
        )

    # Final-value annotation on the right axis
    if not pd.isna(values[-1]):
        sign = "▲" if (values[-1] >= avg if avg else True) else "▼"
        is_positive_sign = (values[-1] >= avg) if avg else True
        if is_down and not is_leverage_metric:
            is_positive_sign = not is_positive_sign
        annot_color = UP if is_positive_sign else DOWN
        fig.add_annotation(
            xref="paper", x=1.02,
            yref="y",     y=values[-1],
            text=f"{sign} {_fmt(values[-1], unit)}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11, color=annot_color,
                      family="'SF Pro Display','Segoe UI',sans-serif"),
            bgcolor="rgba(0,0,0,0.45)", borderpad=3,
        )

    tick_suffix = "%" if unit == "%" else ("x" if unit == "x" else "d" if unit == "days" else "")
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
            ticksuffix=tick_suffix,
            showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
            tickfont=dict(color="#ffffff"),
            title=dict(text="", font=dict(color="#ffffff")),
        ),
    )
    return fig


# ── Insight cards ─────────────────────────────────────────────────────────────

def _render_insights(metric: str, values: List[float],
                     year_labels: List[str], unit: str) -> None:
    """Render Key Insight verdict cards below the table, mirroring performance.py."""
    clean = [v for v in values if not pd.isna(v)]
    inverse = INVERSE_MAP.get(metric, False)

    # 1. What is this metric?
    description = METRIC_DESCRIPTIONS.get(metric, "")
    if description:
        st.markdown(_verdict_card(
            title="What does this measure?",
            verdict=metric,
            verdict_color=BLUE,
            body=description,
        ), unsafe_allow_html=True)

    if len(clean) < 2:
        st.markdown(_verdict_card(
            title="Historical Trend",
            verdict="Not enough data",
            verdict_color=GREY,
            body="Fewer than 2 years of data are available for trend analysis.",
        ), unsafe_allow_html=True)
        return

    latest_val = clean[-1]
    avg_val    = float(np.mean(clean))
    verdict, color = _signal(latest_val, values, inverse)

    # 2. Trend verdict
    if avg_val != 0:
        dev_pct = ((latest_val / avg_val) - 1) * 100
        dev_label = f"{dev_pct:+.1f}% vs. {len(clean)}-year average ({_fmt(avg_val, unit)})"
    else:
        dev_label = f"Average: {_fmt(avg_val, unit)}"

    trend_body = (
        f"Most recent value is {_fmt(latest_val, unit)}. "
        f"{dev_label}. "
        f"{'Lower is better for this metric.' if inverse else 'Higher is generally better.'}"
    )
    st.markdown(_verdict_card(
        title="Historical Trend",
        verdict=verdict,
        verdict_color=color,
        body=trend_body,
    ), unsafe_allow_html=True)

    # 3. Best / worst years
    if len(clean) >= 3:
        best_idx  = int(np.argmin(values) if inverse else np.argmax(values))
        worst_idx = int(np.argmax(values) if inverse else np.argmin(values))
        if best_idx != worst_idx and not pd.isna(values[best_idx]) and not pd.isna(values[worst_idx]):
            st.markdown(_verdict_card(
                title="Peak vs. Trough",
                verdict=f"Best: {year_labels[best_idx]} ({_fmt(values[best_idx], unit)})",
                verdict_color=UP,
                body=f"Weakest year was {year_labels[worst_idx]} at {_fmt(values[worst_idx], unit)}. "
                     f"Range of {_fmt(abs(values[best_idx] - values[worst_idx]), unit)} across the period.",
            ), unsafe_allow_html=True)

    # 4. Momentum: last year vs prior year
    if len(clean) >= 2:
        last, prior = clean[-1], clean[-2]
        if prior != 0:
            chg = ((last - prior) / abs(prior)) * 100
            mom_positive = chg > 0
            mom_color    = UP if mom_positive else DOWN
            mom_label    = "Improving" if mom_positive else "Weakening"
            st.markdown(_verdict_card(
                title="Year-over-Year Momentum",
                verdict=f"{mom_label} ({chg:+.1f}%)",
                verdict_color=mom_color,
                body=f"From {_fmt(prior, unit)} to {_fmt(last, unit)} — "
                     f"a change of {_fmt(abs(last - prior), unit)} in the latest year.",
            ), unsafe_allow_html=True)


# ── Summary KPI strip ─────────────────────────────────────────────────────────

def _render_kpi_strip(metadata: Dict, ltm_data: Dict, balance_data: Dict) -> None:
    """Top-of-page metric cards: key LTM figures at a glance."""
    mktcap = metadata.get("market_cap", np.nan)
    rev    = ltm_data.get("revenue",          np.nan)
    ni     = ltm_data.get("net_income",       np.nan)
    equity = balance_data.get("equity",        np.nan)
    debt   = balance_data.get("debt",          np.nan)
    pe     = metadata.get("pe_ltm",           np.nan)

    def _abbr(v):
        if pd.isna(v): return "N/A"
        if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:  return f"${v/1e9:.1f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
        return f"${v:,.0f}"

    net_margin = (ni / rev * 100) if (not pd.isna(ni) and not pd.isna(rev) and rev != 0) else np.nan
    de_ratio   = (debt / equity)  if (not pd.isna(debt) and not pd.isna(equity) and equity != 0) else np.nan

    nm_color = UP if (not pd.isna(net_margin) and net_margin > 10) else (
               ORANGE if (not pd.isna(net_margin) and net_margin > 0) else DOWN)
    de_color = UP if (not pd.isna(de_ratio) and de_ratio < 1) else (
               ORANGE if (not pd.isna(de_ratio) and de_ratio < 2) else DOWN)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_metric_card("Market Cap",  _abbr(mktcap),
            UP if not pd.isna(mktcap) else GREY,
            tooltip="Total market value of all outstanding shares."), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_card("LTM Revenue", _abbr(rev),
            BLUE,
            tooltip="Last-twelve-months revenue."), unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_card("Net Margin",
            f"{net_margin:.1f}%" if not pd.isna(net_margin) else "N/A",
            nm_color,
            tooltip="LTM net income as a percentage of revenue."), unsafe_allow_html=True)
    with c4:
        st.markdown(_metric_card("P/E (LTM)",
            f"{pe:.1f}x" if not pd.isna(pe) else "N/A",
            BLUE,
            tooltip="Price-to-earnings ratio on trailing twelve months earnings."), unsafe_allow_html=True)
    with c5:
        st.markdown(_metric_card("Total D/E",
            f"{de_ratio:.2f}x" if not pd.isna(de_ratio) else "N/A",
            de_color,
            tooltip="Total debt divided by shareholders' equity."), unsafe_allow_html=True)


# ── Main render ───────────────────────────────────────────────────────────────

def render_ratios(ticker: str) -> None:
    _inject_css()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:4px;">'
        f'{ticker}</h1>',
        unsafe_allow_html=True,
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner(f"Loading financial data for {ticker}…"):
        try:
            data = _fetch_data(ticker)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    metadata      = data.get("metadata",       {})
    ltm_data      = data.get("ltm_data",       {})
    balance_data  = data.get("balance_data",   {})
    income_stmt   = data.get("income_stmt",    pd.DataFrame())
    balance_sheet = data.get("balance_sheet_stmt", pd.DataFrame())

    try:
        build_company_summary(
            ticker=ticker,
            ltm_data=ltm_data,
            balance_data=balance_data,
            metadata=metadata,
            revenue_history=data.get("revenue_history"),
            lfcf_history=data.get("lfcf_history"),
            gross_profit_history=data.get("gross_profit_history"),
            ebit_history=data.get("ebit_history"),
            ebitda_history=data.get("ebitda_history"),
            net_income_history=data.get("net_income_history"),
            eps_history=data.get("eps_history"),
            diluted_eps_history=data.get("diluted_eps_history"),
            ar_history=data.get("ar_history"),
            inventory_history=data.get("inventory_history"),
            ppe_history=data.get("ppe_history"),
            total_assets_history=data.get("total_assets_history"),
            total_liabilities_history=data.get("total_liabilities_history"),
            equity_history=data.get("equity_history"),
        )
    except Exception as e:
        st.warning(f"Could not build full company summary: {e}")

    # ── KPI strip ─────────────────────────────────────────────────────────────

    # ── Fiscal year axis ──────────────────────────────────────────────────────
    fiscal_years = _fiscal_years({
        "revenue": data.get("revenue_history"),
        "equity":  data.get("equity_history"),
        "assets":  data.get("total_assets_history"),
    })

    if not fiscal_years:
        st.warning("No historical data available for this ticker.")
        return

    fiscal_years  = sorted(fiscal_years)          # oldest → newest
    year_labels   = [f"FY{str(y)[2:]}" for y in fiscal_years]

    # ── Narrative glancebox ───────────────────────────────────────────────────

    # ── Category & metric selectors ───────────────────────────────────────────
    if "ratio_category" not in st.session_state:
        st.session_state.ratio_category = "Profitability"

    sel_col_left, sel_col_right = st.columns([0.55, 0.45])
    with sel_col_left:
        category = st.radio(
            "Category",
            options=CATEGORIES,
            horizontal=True,
            key="ratio_category",
            label_visibility="collapsed",
        )
    metrics = CATEGORY_METRICS[category]
    with sel_col_right:
        selected_metric = st.selectbox(
            "Focus Metric",
            options=metrics,
            index=0,
            key=f"{category}_focus",
            label_visibility="collapsed",
        )


    # ── Build value matrix ────────────────────────────────────────────────────
    unit    = UNIT_MAP.get(selected_metric, "%")
    inverse = INVERSE_MAP.get(selected_metric, False)

    value_matrix: Dict[str, List[float]] = {
        m: [_get_value_for_metric(m, y, income_stmt, balance_sheet, data) for y in fiscal_years]
        for m in metrics
    }
    selected_values = value_matrix[selected_metric]

    # ── Two-column layout: left = table + insights, right = chart ─────────────
    col_left, col_right = st.columns([0.5, 0.5])

    with col_left:
        # ── Section header ────────────────────────────────────────────────────

        # ── Historical table ──────────────────────────────────────────────────
        table_data = {"Metric": metrics}
        for year, label in zip(fiscal_years, year_labels):
            table_data[label] = [
                _fmt(_get_value_for_metric(m, year, income_stmt, balance_sheet, data),
                     UNIT_MAP.get(m, "%"))
                for m in metrics
            ]
        table_df     = pd.DataFrame(table_data)
        table_height = min(360, 56 + 36 * len(metrics))
        st.dataframe(table_df, hide_index=True, use_container_width=True, height=table_height)

        # ── Key Insights ──────────────────────────────────────────────────────
        _render_insights(selected_metric, selected_values, year_labels, unit)

    with col_right:

        # Match chart height to left column:
        # left = section_header (~50) + table + section_header (~50) + insight cards (~115 each)
        clean_for_height = [v for v in selected_values if not pd.isna(v)]
        n_insight_cards = 1  # "What does this measure?" always shows
        if len(clean_for_height) >= 2: n_insight_cards += 1  # Trend
        if len(clean_for_height) >= 3: n_insight_cards += 1  # Peak vs Trough
        if len(clean_for_height) >= 2: n_insight_cards += 1  # Momentum
        chart_height = table_height + 100 + (n_insight_cards * 115)

        all_nan = all(pd.isna(v) for v in selected_values)
        if all_nan:
            st.info("Insufficient data to plot a trend for this metric.")
        else:
            fig = _chart_ratio_trend(year_labels, selected_values, selected_metric, unit,
                                     height=chart_height)
            st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    render_ratios("AAPL")
