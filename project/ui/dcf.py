# ui/dcf.py
# ──────────────────────────────────────────────────────────────────────────────
# Valuation Page
# ──────────────────────────────────────────────────────────────────────────────
#
# Layout
# ------
#  Header:    Ticker (no subtitle)
#  Section A: "Model Assumptions" expander  (Configure-optimizer style, 3 columns)
#  Tabs:
#    ├─ Your Model         — WACC Bridge cards + Equity Value Bridge
#    ├─ Wall St. Consensus — Street assumptions summary + Street Equity Value Bridge
#    └─ Charts             — PV waterfall + Revenue/EBITDA/FCF
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sec_engine.cik_loader import load_full_cik_map
from sec_engine.sec_fetch import fetch_company_facts, fetch_company_submissions
from sec_engine.dcf import (
    DCFAssumptions,
    build_dcf_assumptions,
    run_dcf,
    BetaResult,
    RISK_FREE_RATE_DEFAULT,
    EQUITY_RISK_PREMIUM_DEFAULT,
)

warnings.filterwarnings("ignore", category=FutureWarning)
_logger = logging.getLogger(__name__)

UP     = "#00C805"
DOWN   = "#FF3B30"
BLUE   = "#0A7CFF"
ORANGE = "#FF9F0A"
GREY   = "#6E6E73"
PURPLE = "#BF5AF2"

# Tearsheet card constants
CARD_BG = "transparent"
BORDER  = "rgba(255,255,255,0.12)"


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;}}

/* ── Tabs — matches financials.py ── */
.stTabs [data-baseweb="tab-list"]{{gap:0;border-bottom:1px solid rgba(255,255,255,0.1);background:transparent!important;}}
.stTabs [data-baseweb="tab"]{{font-weight:600;font-size:13px;padding:10px 22px;color:rgba(255,255,255,0.45);background:transparent!important;border-bottom:2px solid transparent;border-radius:0;}}
.stTabs [aria-selected="true"]{{color:#fff!important;border-bottom:2px solid rgba(255,255,255,0.6)!important;background:transparent!important;}}

/* ── Tearsheet-style metric cards ── */
.ts-card-grid{{display:grid;gap:12px;margin:16px 0 8px 0;}}
.ts-card-grid.cols-2{{grid-template-columns:repeat(2,1fr);}}
.ts-card-grid.cols-3{{grid-template-columns:repeat(3,1fr);}}
.ts-card-grid.cols-4{{grid-template-columns:repeat(4,1fr);}}
.ts-card-grid.cols-5{{grid-template-columns:repeat(5,1fr);}}
.ts-card{{background:{CARD_BG};border-radius:12px;padding:16px 18px;border:1px solid {BORDER};height:100%;}}
.ts-card .ts-label{{font-size:11px;font-weight:600;letter-spacing:0.05em;color:#ffffff;text-transform:uppercase;margin-bottom:6px;}}
.ts-card .ts-value{{font-size:22px;font-weight:700;line-height:1.1;}}
.ts-card .ts-sub{{font-size:10.5px;color:rgba(255,255,255,.38);margin-top:5px;}}
.ts-card.accent-blue{{border-color:rgba(10,124,255,0.4);background:rgba(10,124,255,0.06);}}
.ts-card.accent-green{{border-color:rgba(0,200,5,0.35);background:rgba(0,200,5,0.05);}}
.ts-card.accent-red{{border-color:rgba(255,59,48,0.35);background:rgba(255,59,48,0.05);}}
.ts-card.accent-orange{{border-color:rgba(255,159,10,0.35);background:rgba(255,159,10,0.05);}}
.ts-card.accent-purple{{border-color:rgba(191,90,242,0.35);background:rgba(191,90,242,0.05);}}
.ts-card.accent-street{{border-color:rgba(0,200,5,0.25);background:rgba(0,200,5,0.04);}}

/* ── Configure-optimizer style labels ── */
.cfg-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: rgba(255,255,255,1.0);
    margin: 0 0 4px 0; line-height: 1;
}}
.cfg-sublabel {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; color: rgba(255,255,255,0.45);
    margin: 10px 0 4px 0; line-height: 1;
}}

/* ── Source badge ── */
.src-badge{{display:inline-block;font-size:8.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;border-radius:4px;padding:2px 6px;margin-left:6px;vertical-align:middle;}}
.src-badge.sec{{color:rgba(10,124,255,.9);background:rgba(10,124,255,.12);border:1px solid rgba(10,124,255,.2);}}
.src-badge.yf{{color:rgba(255,159,10,.9);background:rgba(255,159,10,.12);border:1px solid rgba(255,159,10,.2);}}
.src-badge.reg{{color:rgba(191,90,242,.9);background:rgba(191,90,242,.12);border:1px solid rgba(191,90,242,.2);}}
.src-badge.calc{{color:rgba(255,255,255,.5);background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);}}
.src-badge.street{{color:rgba(0,200,5,.9);background:rgba(0,200,5,.1);border:1px solid rgba(0,200,5,.2);}}

/* ── Analyst rating badge ── */
.analyst-rating{{display:inline-block;font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px;}}
.rating-buy{{background:rgba(0,200,5,.15);color:{UP};border:1px solid rgba(0,200,5,.25);}}
.rating-hold{{background:rgba(255,159,10,.12);color:{ORANGE};border:1px solid rgba(255,159,10,.22);}}
.rating-sell{{background:rgba(255,59,48,.12);color:{DOWN};border:1px solid rgba(255,59,48,.22);}}

/* ── Divider ── */
.val-divider{{border:none;border-top:1px solid rgba(255,255,255,0.07);margin:24px 0;}}

/* ── Warning banner ── */
.val-warn{{background:rgba(255,159,10,.07);border:1px solid rgba(255,159,10,.22);border-radius:10px;padding:12px 16px;margin:10px 0;font-size:12px;color:rgba(255,255,255,.75);line-height:1.7;}}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _cik_map() -> dict:
    try:
        return load_full_cik_map()
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_facts(cik: str) -> dict:
    try:
        return fetch_company_facts(cik)
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_sic(cik: str) -> str:
    try:
        subs = fetch_company_submissions(cik)
        return str(subs.get("sic", ""))
    except Exception:
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def _get_analyst_estimates(ticker: str) -> dict:
    result = {
        "price_targets": {},
        "rating": None,
        "n_analysts": 0,
        "rev_growth_fwd": np.nan,
        "eps_growth_fwd": np.nan,
        "ebitda_margin": np.nan,
        "street_rev_g_y1": np.nan,
        "street_rev_g_y5": np.nan,
        "street_ebitda_m": np.nan,
    }
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}

        current = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
        result["price_targets"] = {
            "current": current,
            "low":     _safe(info.get("targetLowPrice")),
            "mean":    _safe(info.get("targetMeanPrice")),
            "high":    _safe(info.get("targetHighPrice")),
        }
        result["n_analysts"] = int(info.get("numberOfAnalystOpinions") or 0)

        _rec_map = {
            "strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
            "underperform": "Sell", "sell": "Strong Sell",
        }
        rec = info.get("recommendationKey", "")
        result["rating"] = _rec_map.get(rec.lower().replace(" ", "_"), rec.capitalize() if rec else None)

        ebitda = info.get("ebitda")
        rev    = info.get("totalRevenue")
        if ebitda and rev and float(rev) > 0:
            result["ebitda_margin"]   = float(ebitda) / float(rev)
            result["street_ebitda_m"] = float(ebitda) / float(rev)

        try:
            rev_est = t.revenue_estimate
            if rev_est is not None and not rev_est.empty and len(rev_est) >= 2:
                g0 = _safe(rev_est.iloc[0].get("growth"))
                g1 = _safe(rev_est.iloc[1].get("growth"))
                result["rev_growth_fwd"]   = g0
                result["street_rev_g_y1"]  = g0
                result["street_rev_g_y5"]  = g1
        except Exception:
            pass

        try:
            eps_est = t.earnings_estimate
            if eps_est is not None and not eps_est.empty and len(eps_est) >= 2:
                result["eps_growth_fwd"] = _safe(eps_est.iloc[1].get("growth"))
        except Exception:
            pass

    except Exception as exc:
        _logger.warning("_get_analyst_estimates failed for %s: %s", ticker, exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v, default: float = np.nan) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp(v, lo, hi, fallback):
    try:
        f = float(v)
        return max(lo, min(hi, f)) if np.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _fmt_m(v: float, decimals: int = 1) -> str:
    if np.isnan(v):
        return "—"
    if abs(v) >= 1e12:
        return f"${v/1e12:,.{decimals}f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:,.{decimals}f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:,.{decimals}f}M"
    return f"${v:,.0f}"


def _fmt_price(v: float) -> str:
    return "—" if np.isnan(v) else f"${v:,.2f}"


def _fmt_pct(v: float, decimals: int = 1, show_plus: bool = False) -> str:
    if np.isnan(v):
        return "—"
    sign = "+" if show_plus and v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Tearsheet-style card helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_card(label: str, value: str, color: str = "#ffffff",
             sub: str = "", accent: str = "") -> str:
    """Render a single tearsheet-style metric card."""
    cls = f"ts-card{' ' + accent if accent else ''}"
    sub_html = f'<div class="ts-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{cls}">'
        f'<div class="ts-label">{label}</div>'
        f'<div class="ts-value" style="color:{color};">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _ts_grid(*cards: str, cols: int = 4) -> str:
    """Wrap cards in a CSS grid."""
    inner = "".join(cards)
    return f'<div class="ts-card-grid cols-{cols}">{inner}</div>'


def _badge(text: str, kind: str = "calc") -> str:
    return f'<span class="src-badge {kind}">{text}</span>'


def _beta_badge(br: BetaResult) -> str:
    m = {"fundamental": ("sec","SEC"), "regression": ("reg","Regression"),
         "yfinance": ("yf","yfinance"), "fallback": ("calc","Default")}
    kind, label = m.get(br.method, ("calc", br.method))
    return _badge(label, kind)


def _rating_badge(rating: Optional[str]) -> str:
    if not rating:
        return ""
    r = rating.lower()
    cls = "rating-buy" if "buy" in r else ("rating-sell" if ("sell" in r or "underperform" in r) else "rating-hold")
    return f'<span class="analyst-rating {cls}">{rating}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# WACC Bridge — tearsheet-style cards  (replaces old _render_wacc_bridge)
# ─────────────────────────────────────────────────────────────────────────────

def _render_wacc_bridge_cards(
    result, assumptions: DCFAssumptions,
    rf_rate: float, beta_val: float, erp: float,
    ke: float, kd: float, wacc: float,
    cod: float, tax_rate: float, debt_w: float,
    equity_w: float, current_price: float,
    beta_result: BetaResult,
    terminal_g: float,
) -> None:
    """
    Full WACC Bridge rendered as tearsheet-style card rows.

    Row 1 — WACC inputs (6 cards)
    Row 2 — CAPM / WACC outputs (5 cards)
    Row 3 — Terminal value bridge (4 cards)
    Row 4 — Enterprise → equity value bridge (5 cards)
    Row 5 — Implied price vs current (4 cards)
    """

    # ── Derived values ──────────────────────────────────────────────────────
    capm            = rf_rate + beta_val * erp
    terminal_capm   = rf_rate + beta_val * erp          # same CAPM; terminal uses same rate
    terminal_rf     = rf_rate                            # could be overridden; same for now
    terminal_wacc   = wacc                               # same WACC in terminal year
    tv_undiscounted = result.terminal_value
    pv_tv           = result.pv_terminal_value
    sum_pv_fcf      = result.sum_pv_ufcf
    ev              = result.enterprise_value
    total_debt      = assumptions.net_debt + (
        # net_debt = total_debt – cash, so total_debt = net_debt + cash
        # We'll display net_debt as the bridge item; total_debt needs separate storage.
        # Use net_debt as a proxy here unless a separate total_debt field exists.
        0
    )
    # Try to get total_debt and cash from assumptions directly
    raw_total_debt = getattr(assumptions, "total_debt", np.nan)
    raw_cash       = getattr(assumptions, "cash", np.nan)
    raw_other      = getattr(assumptions, "other_claims", np.nan)
    if np.isnan(raw_total_debt):
        raw_total_debt = assumptions.net_debt  # fallback
    if np.isnan(raw_cash):
        raw_cash = 0.0

    market_cap    = result.equity_value
    shares        = assumptions.shares_outstanding
    implied_price = result.intrinsic_price
    is_underval   = (not np.isnan(implied_price) and not np.isnan(current_price)
                     and current_price > 0 and implied_price > current_price)

    # ── Row 1: WACC Inputs ──────────────────────────────────────────────────
    st.markdown('<p class="cfg-label" style="margin:18px 0 6px 0;">WACC Inputs</p>',
                unsafe_allow_html=True)
    st.markdown(_ts_grid(
        _ts_card("Tax Rate",          f"{tax_rate:.1%}",  "#fff"),
        _ts_card("Risk-Free Rate",    f"{rf_rate:.2%}",   "#fff"),
        _ts_card("Beta (β)",          f"{beta_val:.2f}x", "#fff",
                 sub=f"{beta_result.method.capitalize()} method"),
        _ts_card("Market Risk Premium", f"{erp:.2%}",     "#fff"),
        _ts_card("% Equity",          f"{equity_w:.0%}",  "#fff"),
        _ts_card("% Debt",            f"{debt_w:.0%}",    "#fff"),
        _ts_card("Cost of Debt",      f"{cod:.2%}",       "#fff",
                 sub="Pre-tax"),
        cols=4,
    ), unsafe_allow_html=True)

    # ── Row 2: CAPM / WACC ─────────────────────────────────────────────────
    st.markdown('<p class="cfg-label" style="margin:18px 0 6px 0;">CAPM & WACC</p>',
                unsafe_allow_html=True)
    st.markdown(_ts_grid(
        _ts_card("CAPM",              f"{capm:.2%}",    BLUE,   accent="accent-blue",
                 sub="Rf + β × ERP"),
        _ts_card("WACC",              f"{wacc:.2%}",    BLUE,   accent="accent-blue",
                 sub=f"Ke {ke:.2%} · Kd {kd:.2%}"),
        _ts_card("Terminal Risk-Free", f"{terminal_rf:.2%}", "#fff"),
        _ts_card("Terminal CAPM",     f"{terminal_capm:.2%}", "#fff"),
        _ts_card("Terminal WACC",     f"{terminal_wacc:.2%}", "#fff"),
        cols=5,
    ), unsafe_allow_html=True)

    # ── Row 3: Terminal Value Bridge ────────────────────────────────────────
    st.markdown('<p class="cfg-label" style="margin:18px 0 6px 0;">Terminal Value</p>',
                unsafe_allow_html=True)
    st.markdown(_ts_grid(
        _ts_card("Terminal Growth Rate", f"{terminal_g:.1%}",  ORANGE, accent="accent-orange"),
        _ts_card("Terminal Value",        _fmt_m(tv_undiscounted), ORANGE, accent="accent-orange",
                 sub="Undiscounted"),
        _ts_card("PV of Terminal Value",  _fmt_m(pv_tv),       "#fff",
                 sub=f"{result.pv_tv_pct:.0f}% of EV"),
        _ts_card("Sum of PV Free Cash Flows", _fmt_m(sum_pv_fcf), "#fff",
                 sub=f"{result.pv_ufcf_pct:.0f}% of EV"),
        cols=4,
    ), unsafe_allow_html=True)

    # ── Row 4: Enterprise → Equity Bridge ──────────────────────────────────
    st.markdown('<p class="cfg-label" style="margin:18px 0 6px 0;">Equity Value Bridge</p>',
                unsafe_allow_html=True)

    other_val = raw_other if not np.isnan(raw_other) else 0.0
    st.markdown(_ts_grid(
        _ts_card("Enterprise Value",   _fmt_m(ev),            BLUE,   accent="accent-blue",
                 sub="DCF-implied"),
        _ts_card("Total Debt",         _fmt_m(raw_total_debt), "#fff",
                 sub="Balance sheet"),
        _ts_card("Other Claims",       _fmt_m(other_val),     "#fff",
                 sub="Minorities / preferred" if other_val != 0 else "—"),
        _ts_card("Cash & Equivalents", _fmt_m(raw_cash),      UP,     accent="accent-green",
                 sub="Added back"),
        _ts_card("Market Capitalization", _fmt_m(market_cap), "#fff",
                 sub="EV – Net Debt"),
        cols=5,
    ), unsafe_allow_html=True)

    # ── Row 5: Implied Price ────────────────────────────────────────────────
    st.markdown('<p class="cfg-label" style="margin:18px 0 6px 0;">Implied Price</p>',
                unsafe_allow_html=True)

    shares_str = f"{shares/1e6:,.1f}M" if not np.isnan(shares) else "—"
    upside_val = ((implied_price - current_price) / current_price * 100
                  if not np.isnan(implied_price) and not np.isnan(current_price) and current_price > 0
                  else np.nan)
    upside_str  = _fmt_pct(upside_val, show_plus=True) if not np.isnan(upside_val) else "—"
    upside_color = UP if is_underval else DOWN
    verdict_str  = "Undervalued" if is_underval else "Overvalued"
    verdict_accent = "accent-green" if is_underval else "accent-red"
    verdict_color  = UP if is_underval else DOWN

    st.markdown(_ts_grid(
        _ts_card("Fully Diluted Shares", shares_str,          "#fff"),
        _ts_card("Implied Price",         _fmt_price(implied_price), BLUE, accent="accent-blue",
                 sub="Your model"),
        _ts_card("Current Price",         _fmt_price(current_price), "#fff"),
        _ts_card(verdict_str,             upside_str,          verdict_color, accent=verdict_accent,
                 sub="vs. current price"),
        cols=4,
    ), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _chart_waterfall(result, assumptions: DCFAssumptions) -> go.Figure:
    n      = assumptions.n_years
    labels = [f"Y{i+1}" for i in range(n)] + ["Terminal\nValue"]
    values = list(result.pv_ufcf) + [result.pv_terminal_value]
    colors = [f"rgba(10,124,255,{0.45+0.04*i})" for i in range(n)] + ["rgba(255,159,10,0.70)"]
    fig    = go.Figure(go.Bar(
        x=labels, y=[v/1e9 for v in values],
        marker_color=colors,
        text=[f"${v/1e9:.2f}B" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="rgba(255,255,255,0.65)"),
    ))
    fig.add_annotation(
        x=labels[-1], y=result.pv_terminal_value/1e9,
        text=f"{result.pv_tv_pct:.0f}% of EV",
        showarrow=False, yanchor="bottom", yshift=22,
        font=dict(size=9.5, color=ORANGE),
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,-apple-system,sans-serif", size=11, color="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=24, b=0), height=280, showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="$B", gridcolor="rgba(255,255,255,0.05)", tickformat="$.1f"),
        bargap=0.25,
    )
    return fig


def _chart_revenue_fcf(result, assumptions: DCFAssumptions) -> go.Figure:
    n     = assumptions.n_years
    years = [f"Y{i+1}" for i in range(n)]
    fig   = go.Figure()
    fig.add_trace(go.Bar(
        name="Revenue", x=years,
        y=[v/1e9 for v in result.projected_revenues],
        marker_color="rgba(10,124,255,0.30)",
        marker_line=dict(color="rgba(10,124,255,0.60)", width=1),
    ))
    fig.add_trace(go.Scatter(
        name="Unlevered FCF", x=years,
        y=[v/1e9 for v in result.projected_ufcf],
        mode="lines+markers", line=dict(color=UP, width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        name="EBITDA", x=years,
        y=[v/1e9 for v in result.projected_ebitda],
        mode="lines+markers", line=dict(color=ORANGE, width=1.5, dash="dot"), marker=dict(size=4),
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,-apple-system,sans-serif", size=11, color="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=24, b=0), height=280,
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="$B", gridcolor="rgba(255,255,255,0.05)", tickformat="$.1f"),
        bargap=0.30, hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_dcf(
    ticker: str,
    summary: Optional[dict] = None,
    data: Optional[dict] = None,
) -> None:
    _inject_css()
    ticker = ticker.upper().strip()

    # ── Fetch / reuse cached data ─────────────────────────────────────────────
    cik = _cik_map().get(ticker, "")

    if data is None:
        data = st.session_state.get("ticker_data_cache", {}).get(ticker)
    if summary is None:
        summary = st.session_state.get("ticker_summary_cache", {}).get(ticker)

    if data is None:
        with st.spinner(f"Fetching {ticker} data…"):
            try:
                from app import fetch_company_data_unified
                data = fetch_company_data_unified(ticker, cik)
            except Exception:
                from sec_engine.sec_fetch import fetch_company_facts as _fcf
                from sec_engine.ltm import build_ltm_financials
                from sec_engine.normalize import GAAP_MAP
                try:
                    data = {"ltm_data": build_ltm_financials(_fcf(cik) if cik else {}, GAAP_MAP),
                            "balance_data": {}, "metadata": {}}
                except Exception:
                    data = {"ltm_data": {}, "balance_data": {}, "metadata": {}}

    if summary is None:
        with st.spinner("Computing metrics…"):
            try:
                from aggregation import build_company_summary
                summary = build_company_summary(
                    ticker=ticker,
                    ltm_data=data.get("ltm_data", {}),
                    balance_data=data.get("balance_data", {}),
                    metadata=data.get("metadata", {}),
                    revenue_history=data.get("revenue_history"),
                    lfcf_history=data.get("lfcf_history"),
                    gross_profit_history=data.get("gross_profit_history"),
                    ebit_history=data.get("ebit_history"),
                    ebitda_history=data.get("ebitda_history"),
                    net_income_history=data.get("net_income_history"),
                )
            except Exception:
                summary = {}
    if summary is None:
        summary = {}

    # ── SIC, price, analyst estimates ─────────────────────────────────────────
    sic_code = _load_sic(cik) if cik else ""
    with st.spinner("Fetching market data & analyst estimates…"):
        estimates    = _get_analyst_estimates(ticker)
    current_price = _safe(estimates["price_targets"].get("current", np.nan))

    meta = data.get("metadata", {})

    # ── Page header (ticker only, no subtitle) ────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:30px;font-weight:800;color:#fff;margin-bottom:20px;">{ticker}</h1>',
        unsafe_allow_html=True,
    )

    # ── Build DCF assumptions from SEC data ───────────────────────────────────
    with st.spinner("Building assumptions from SEC data…"):
        facts_for_beta = _load_facts(cik) if cik else {}
        default_asm, beta_result = build_dcf_assumptions(
            ticker=ticker, summary=summary, data=data,
            sic_code=sic_code, facts=facts_for_beta,
        )

    # Blend in Street estimates where richer
    cons_g1 = estimates.get("street_rev_g_y1", np.nan)
    cons_g5 = estimates.get("street_rev_g_y5", np.nan)
    cons_em = estimates.get("street_ebitda_m", np.nan)

    def_g1 = cons_g1 if not np.isnan(cons_g1) else default_asm.revenue_growth_y1
    def_g5 = cons_g5 if not np.isnan(cons_g5) else default_asm.revenue_growth_y5
    def_em = cons_em if not np.isnan(cons_em) else default_asm.ebitda_margin

    # ── Session-state helpers ─────────────────────────────────────────────────
    pfx = f"dcf_{ticker}_"

    def _state(key: str, default):
        full = pfx + key
        if full not in st.session_state:
            st.session_state[full] = default
        return full

    # ─────────────────────────────────────────────────────────────────────────
    # ASSUMPTION EXPANDER  — Configure-optimizer style
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("Model Assumptions", expanded=True):
        col_g, col_m, col_w = st.columns([1, 1, 1], gap="large")

        with col_g:
            st.markdown('<p class="cfg-label">Growth</p>', unsafe_allow_html=True)

            st.markdown('<p class="cfg-sublabel">Revenue Growth — Year 1</p>', unsafe_allow_html=True)
            rev_g_y1 = st.slider(
                "rev_g_y1", min_value=-20.0, max_value=80.0,
                value=_clamp(round(def_g1 * 100, 1), -20.0, 80.0, 10.0),
                step=0.5, format="%.1f%%",
                key=_state("rev_g_y1", _clamp(round(def_g1 * 100, 1), -20.0, 80.0, 10.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">Revenue Growth — Year 5</p>', unsafe_allow_html=True)
            rev_g_y5 = st.slider(
                "rev_g_y5", min_value=-10.0, max_value=50.0,
                value=_clamp(round(def_g5 * 100, 1), -10.0, 50.0, 8.0),
                step=0.5, format="%.1f%%",
                key=_state("rev_g_y5", _clamp(round(def_g5 * 100, 1), -10.0, 50.0, 8.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">Terminal Growth Rate</p>', unsafe_allow_html=True)
            terminal_g = st.slider(
                "terminal_g", min_value=0.0, max_value=5.0,
                value=_clamp(round(default_asm.terminal_growth_rate * 100, 1), 0.0, 5.0, 2.5),
                step=0.1, format="%.1f%%",
                key=_state("terminal_g", _clamp(round(default_asm.terminal_growth_rate * 100, 1), 0.0, 5.0, 2.5)),
                label_visibility="collapsed",
            ) / 100

        with col_m:
            st.markdown('<p class="cfg-label">Profitability & CapEx</p>', unsafe_allow_html=True)

            st.markdown('<p class="cfg-sublabel">EBITDA Margin</p>', unsafe_allow_html=True)
            ebitda_m = st.slider(
                "ebitda_m", min_value=0.0, max_value=85.0,
                value=_clamp(round(def_em * 100, 1), 0.0, 85.0, 20.0),
                step=0.5, format="%.1f%%",
                key=_state("ebitda_m", _clamp(round(def_em * 100, 1), 0.0, 85.0, 20.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">CapEx % Revenue</p>', unsafe_allow_html=True)
            capex_pct = st.slider(
                "capex_pct", min_value=0.0, max_value=40.0,
                value=_clamp(round(default_asm.capex_pct_revenue * 100, 1), 0.0, 40.0, 5.0),
                step=0.5, format="%.1f%%",
                key=_state("capex_pct", _clamp(round(default_asm.capex_pct_revenue * 100, 1), 0.0, 40.0, 5.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">D&A % Revenue</p>', unsafe_allow_html=True)
            da_pct = st.slider(
                "da_pct", min_value=0.0, max_value=25.0,
                value=_clamp(round(default_asm.da_pct_revenue * 100, 1), 0.0, 25.0, 3.0),
                step=0.5, format="%.1f%%",
                key=_state("da_pct", _clamp(round(default_asm.da_pct_revenue * 100, 1), 0.0, 25.0, 3.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">NWC Change % Rev Δ</p>', unsafe_allow_html=True)
            nwc_pct = st.slider(
                "nwc_pct", min_value=-5.0, max_value=15.0,
                value=_clamp(round(default_asm.nwc_pct_revenue * 100, 1), -5.0, 15.0, 1.0),
                step=0.5, format="%.1f%%",
                key=_state("nwc_pct", _clamp(round(default_asm.nwc_pct_revenue * 100, 1), -5.0, 15.0, 1.0)),
                label_visibility="collapsed",
            ) / 100

        with col_w:
            st.markdown('<p class="cfg-label">WACC Inputs</p>', unsafe_allow_html=True)

            st.markdown(f'<p class="cfg-sublabel">Beta {_beta_badge(beta_result)}</p>', unsafe_allow_html=True)
            beta_val = st.slider(
                "beta", min_value=0.10, max_value=3.50,
                value=_clamp(round(default_asm.beta, 2), 0.10, 3.50, 1.0),
                step=0.05,
                key=_state("beta", _clamp(round(default_asm.beta, 2), 0.10, 3.50, 1.0)),
                label_visibility="collapsed",
            )

            st.markdown('<p class="cfg-sublabel">Risk-Free Rate</p>', unsafe_allow_html=True)
            rf_rate = st.slider(
                "rf_rate", min_value=0.0, max_value=8.0,
                value=_clamp(round(default_asm.risk_free_rate * 100, 2), 0.0, 8.0, 4.5),
                step=0.05, format="%.2f%%",
                key=_state("rf_rate", _clamp(round(default_asm.risk_free_rate * 100, 2), 0.0, 8.0, 4.5)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">Equity Risk Premium</p>', unsafe_allow_html=True)
            erp = st.slider(
                "erp", min_value=2.0, max_value=10.0,
                value=_clamp(round(default_asm.equity_risk_premium * 100, 2), 2.0, 10.0, 5.5),
                step=0.25, format="%.2f%%",
                key=_state("erp", _clamp(round(default_asm.equity_risk_premium * 100, 2), 2.0, 10.0, 5.5)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">Cost of Debt (pre-tax)</p>', unsafe_allow_html=True)
            cod = st.slider(
                "cod", min_value=0.0, max_value=20.0,
                value=_clamp(round(default_asm.cost_of_debt_pretax * 100, 2), 0.0, 20.0, 5.0),
                step=0.25, format="%.2f%%",
                key=_state("cod", _clamp(round(default_asm.cost_of_debt_pretax * 100, 2), 0.0, 20.0, 5.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">Debt Weight</p>', unsafe_allow_html=True)
            debt_w = st.slider(
                "debt_w", min_value=0.0, max_value=90.0,
                value=_clamp(round(default_asm.debt_weight * 100, 1), 0.0, 90.0, 20.0),
                step=1.0, format="%.0f%%",
                key=_state("debt_w", _clamp(round(default_asm.debt_weight * 100, 1), 0.0, 90.0, 20.0)),
                label_visibility="collapsed",
            ) / 100

            st.markdown('<p class="cfg-sublabel">Tax Rate</p>', unsafe_allow_html=True)
            tax_rate = st.slider(
                "tax_rate", min_value=0.0, max_value=50.0,
                value=_clamp(round(default_asm.tax_rate * 100, 1), 0.0, 50.0, 21.0),
                step=0.5, format="%.1f%%",
                key=_state("tax_rate", _clamp(round(default_asm.tax_rate * 100, 1), 0.0, 50.0, 21.0)),
                label_visibility="collapsed",
            ) / 100

    # ── Assemble your model's assumptions ────────────────────────────────────
    equity_w = 1.0 - debt_w
    assumptions = DCFAssumptions(
        n_years=10, stage1_years=5,
        revenue_growth_y1=rev_g_y1, revenue_growth_y5=rev_g_y5,
        ebitda_margin=ebitda_m, capex_pct_revenue=capex_pct,
        da_pct_revenue=da_pct, nwc_pct_revenue=nwc_pct,
        terminal_growth_rate=terminal_g, beta=beta_val,
        risk_free_rate=rf_rate, equity_risk_premium=erp,
        cost_of_debt_pretax=cod, tax_rate=tax_rate,
        debt_weight=debt_w, equity_weight=equity_w,
        base_ufcf=default_asm.base_ufcf,
        base_revenue=default_asm.base_revenue,
        net_debt=default_asm.net_debt,
        shares_outstanding=default_asm.shares_outstanding,
    )
    result = run_dcf(assumptions)

    for w in result.warnings:
        st.markdown(f'<div class="val-warn">⚠ {w}</div>', unsafe_allow_html=True)
    if result.warnings and "undefined" in str(result.warnings):
        return

    ke, kd, wacc = result.cost_of_equity, result.cost_of_debt_aftertax, result.wacc

    # ── Build Street Consensus assumptions ───────────────────────────────────
    street_g1 = _clamp(cons_g1, -0.20, 0.80, rev_g_y1)
    street_g5 = _clamp(cons_g5, -0.10, 0.50, max(rev_g_y1 * 0.5, terminal_g + 0.01))
    street_em = _clamp(cons_em, 0.01, 0.85, ebitda_m)

    street_asm = DCFAssumptions(
        n_years=10, stage1_years=5,
        revenue_growth_y1=street_g1, revenue_growth_y5=street_g5,
        ebitda_margin=street_em, capex_pct_revenue=capex_pct,
        da_pct_revenue=da_pct, nwc_pct_revenue=nwc_pct,
        terminal_growth_rate=terminal_g, beta=beta_val,
        risk_free_rate=rf_rate, equity_risk_premium=erp,
        cost_of_debt_pretax=cod, tax_rate=tax_rate,
        debt_weight=debt_w, equity_weight=equity_w,
        base_ufcf=default_asm.base_ufcf,
        base_revenue=default_asm.base_revenue,
        net_debt=default_asm.net_debt,
        shares_outstanding=default_asm.shares_outstanding,
    )
    street_result = run_dcf(street_asm)
    street_intrinsic = street_result.intrinsic_price

    # ── Analyst data ──────────────────────────────────────────────────────────
    pt          = estimates.get("price_targets", {})
    mean_target = pt.get("mean", np.nan)
    rat         = estimates.get("rating")
    n_an        = estimates.get("n_analysts", 0)

    # ─────────────────────────────────────────────────────────────────────────
    # TABS  (no more WACC Bridge tab — removed)
    # ─────────────────────────────────────────────────────────────────────────
    tab_your, tab_street, tab_charts = st.tabs([
        "Your Model",
        "Wall St. Consensus",
        "Charts",
    ])

    # ── Tab: Your Model ───────────────────────────────────────────────────────
    with tab_your:
        _render_wacc_bridge_cards(
            result, assumptions,
            rf_rate, beta_val, erp, ke, kd, wacc,
            cod, tax_rate, debt_w, equity_w, current_price,
            beta_result, terminal_g,
        )

    # ── Tab: Wall St. Consensus ───────────────────────────────────────────────
    with tab_street:
        # Price target cards
        col_r, col_lo, col_mn, col_hi = st.columns(4)
        with col_r:
            rat_html = _rating_badge(rat) if rat else "<span style='color:rgba(255,255,255,.3)'>—</span>"
            st.markdown(
                f'<div class="ts-card"><div class="ts-label">Analyst Rating</div>'
                f'<div class="ts-value" style="font-size:17px;padding-top:4px;">{rat_html}</div>'
                f'<div class="ts-sub">{n_an} analysts</div></div>',
                unsafe_allow_html=True,
            )
        for col, key, lbl, accent in [
            (col_lo, "low",  "Price Target (Low)",  ""),
            (col_mn, "mean", "Price Target (Mean)",
             "accent-green" if not np.isnan(mean_target) and not np.isnan(current_price) and mean_target > current_price else "accent-red"),
            (col_hi, "high", "Price Target (High)", "accent-green"),
        ]:
            with col:
                pv = pt.get(key, np.nan)
                up = (pv - current_price) / current_price * 100 if (
                    not np.isnan(pv) and not np.isnan(current_price) and current_price > 0
                ) else np.nan
                st.markdown(
                    _ts_card(lbl, _fmt_price(pv),
                             UP if accent == "accent-green" else (DOWN if accent == "accent-red" else "#fff"),
                             sub=f"{_fmt_pct(up, show_plus=True)} vs current" if not np.isnan(up) else "",
                             accent=accent),
                    unsafe_allow_html=True,
                )

        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)

        # Street assumptions summary
        st.markdown(
            f'<p style="font-size:11.5px;color:rgba(255,255,255,.5);margin-bottom:12px;">'
            f'Consensus assumptions modeled through the same DCF engine: '
            f'<b style="color:#fff;">Revenue growth Y1 {street_g1*100:+.1f}%</b>, '
            f'<b style="color:#fff;">Y5 {street_g5*100:+.1f}%</b>, '
            f'<b style="color:#fff;">EBITDA margin {street_em*100:.1f}%</b>. '
            f'WACC, CapEx, D&A, and terminal growth are identical to your model.</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="cfg-label" style="margin-bottom:12px;">Street Equity Value Bridge</p>',
            unsafe_allow_html=True,
        )
        _render_wacc_bridge_cards(
            street_result, street_asm,
            rf_rate, beta_val, erp,
            street_result.cost_of_equity, street_result.cost_of_debt_aftertax, street_result.wacc,
            cod, tax_rate, debt_w, equity_w, current_price,
            beta_result, terminal_g,
        )

    # ── Tab: Charts ───────────────────────────────────────────────────────────
    with tab_charts:
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            st.markdown(
                '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);margin-bottom:4px;">PV of Cash Flows by Year — Your Model</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_chart_waterfall(result, assumptions),
                            use_container_width=True, config={"displayModeBar": False})
        with cc2:
            st.markdown(
                '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);margin-bottom:4px;">Revenue · EBITDA · Unlevered FCF — Your Model</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_chart_revenue_fcf(result, assumptions),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2, gap="large")
        with sc1:
            st.markdown(
                '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);margin-bottom:4px;">PV of Cash Flows by Year — Street Consensus</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_chart_waterfall(street_result, street_asm),
                            use_container_width=True, config={"displayModeBar": False})
        with sc2:
            st.markdown(
                '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);margin-bottom:4px;">Revenue · EBITDA · Unlevered FCF — Street Consensus</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_chart_revenue_fcf(street_result, street_asm),
                            use_container_width=True, config={"displayModeBar": False})


# Backwards-compatibility alias
render_valuation = render_dcf
