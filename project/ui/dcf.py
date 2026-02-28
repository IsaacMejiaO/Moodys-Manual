# ui/dcf.py
# ──────────────────────────────────────────────────────────────────────────────
# DCF Valuation Page
# ──────────────────────────────────────────────────────────────────────────────
#
# Architecture
# ------------
# PRIMARY data source : SEC EDGAR (same as Financials page)
#   - Revenue, EBITDA, CapEx, D&A, Debt, Cash — all from EDGAR via the
#     unified fetch_company_data_unified() / build_company_summary() pipeline
#   - Beta: SEC SIC-derived fundamental beta (Hamada re-levering) → regression
#     fallback → yfinance fallback (see sec_engine/dcf.py)
#
# Connection to Financials
# ------------------------
# This page reads from the SAME cached data that the Financials page uses
# (fetch_company_data_unified + build_company_summary). All historical series
# used to pre-populate growth rates come from the Financials page data layer.
# The sec_engine.dcf.build_dcf_assumptions() function is the bridge.
#
# Layout
# ------
#   Header: company name, current price, beta badge, WACC badge
#   Row 1:  Assumption sliders (3 columns)
#   Row 2:  WACC breakdown card
#   Row 3:  DCF output — Intrinsic price card + projected CF waterfall chart
#   Row 4:  Sensitivity tables (WACC×TG and Growth×Margin)
#   Row 5:  Projection table (10-year detailed model)
#   Footer: methodology note
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
    sensitivity_table,
    sensitivity_table_growth,
    BetaResult,
    RISK_FREE_RATE_DEFAULT,
    EQUITY_RISK_PREMIUM_DEFAULT,
)

warnings.filterwarnings("ignore", category=FutureWarning)
_logger = logging.getLogger(__name__)

# ── Design tokens (identical to financials.py and ratios.py) ──────────────────
UP      = "#00C805"
DOWN    = "#FF3B30"
BLUE    = "#0A7CFF"
ORANGE  = "#FF9F0A"
GREY    = "#6E6E73"
PURPLE  = "#BF5AF2"

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;}}

/* ── Tabs (identical to financials) ── */
.stTabs [data-baseweb="tab-list"]{{gap:0;border-bottom:1px solid rgba(255,255,255,0.1);background:transparent!important;}}
.stTabs [data-baseweb="tab"]{{font-weight:600;font-size:13px;padding:10px 22px;color:rgba(255,255,255,0.45);background:transparent!important;border-bottom:2px solid transparent;border-radius:0;}}
.stTabs [aria-selected="true"]{{color:#fff!important;border-bottom:2px solid rgba(255,255,255,0.6)!important;background:transparent!important;}}

/* ── Metric card row ── */
.dcf-cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 20px 0;}}
.dcf-card{{flex:1;min-width:140px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 18px;}}
.dcf-card .label{{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:6px;}}
.dcf-card .value{{font-size:22px;font-weight:800;color:#fff;line-height:1.1;}}
.dcf-card .sub{{font-size:11px;color:rgba(255,255,255,.4);margin-top:4px;}}
.dcf-card.highlight{{border-color:rgba(10,124,255,0.4);background:rgba(10,124,255,0.06);}}
.dcf-card.up{{border-color:rgba(0,200,5,0.35);background:rgba(0,200,5,0.05);}}
.dcf-card.down{{border-color:rgba(255,59,48,0.35);background:rgba(255,59,48,0.05);}}
.dcf-card.orange{{border-color:rgba(255,159,10,0.35);background:rgba(255,159,10,0.05);}}

/* ── Assumption section header ── */
.asm-header{{font-size:10.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:rgba(255,255,255,.35);margin:20px 0 6px 0;}}

/* ── Source badge ── */
.src-badge{{display:inline-block;font-size:8.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;border-radius:4px;padding:2px 6px;margin-left:6px;vertical-align:middle;}}
.src-badge.sec{{color:rgba(10,124,255,.9);background:rgba(10,124,255,.12);border:1px solid rgba(10,124,255,.2);}}
.src-badge.yf{{color:rgba(255,159,10,.9);background:rgba(255,159,10,.12);border:1px solid rgba(255,159,10,.2);}}
.src-badge.reg{{color:rgba(191,90,242,.9);background:rgba(191,90,242,.12);border:1px solid rgba(191,90,242,.2);}}
.src-badge.calc{{color:rgba(255,255,255,.5);background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);}}

/* ── Sensitivity table ── */
.sens-wrap{{overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.015);margin-top:8px;}}
.sens-table{{width:100%;border-collapse:collapse;font-size:12.5px;}}
.sens-table thead th{{background:rgba(255,255,255,0.04);color:#fff;font-weight:700;font-size:10px;letter-spacing:.07em;text-transform:uppercase;padding:9px 14px;border-bottom:1px solid rgba(255,255,255,0.09);text-align:center;white-space:nowrap;}}
.sens-table thead th.row-label{{text-align:left;min-width:100px;}}
.sens-table td{{padding:7px 14px;border-bottom:1px solid rgba(255,255,255,.035);color:#fff;text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap;}}
.sens-table td.row-label{{text-align:left;font-weight:600;color:rgba(255,255,255,.65);}}
.sens-table tbody tr:hover td{{background:rgba(255,255,255,.022);}}
.sens-table .base-col{{background:rgba(10,124,255,.07)!important;font-weight:700;}}
.sens-table .cell-hi{{color:{UP}!important;font-weight:600;}}
.sens-table .cell-lo{{color:{DOWN}!important;font-weight:600;}}
.sens-table .cell-mid{{color:rgba(255,255,255,.85);}}

/* ── Projection table ── */
.proj-wrap{{overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.015);margin-top:8px;}}
.proj-table{{width:100%;border-collapse:collapse;font-size:12px;}}
.proj-table thead th{{background:rgba(255,255,255,0.04);color:#fff;font-weight:700;font-size:10px;letter-spacing:.07em;text-transform:uppercase;padding:9px 14px 8px;border-bottom:1px solid rgba(255,255,255,.09);text-align:right;white-space:nowrap;}}
.proj-table thead th.hl{{text-align:left;min-width:200px;}}
.proj-table td{{padding:6px 14px;border-bottom:1px solid rgba(255,255,255,.03);color:#fff;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}}
.proj-table td.hl{{text-align:left;color:rgba(255,255,255,.65);font-size:11.5px;}}
.proj-table .rt td{{font-weight:700;border-top:1px solid rgba(255,255,255,.12);border-bottom:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.018)!important;}}
.proj-table .rs td{{font-size:9.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#fff;background:rgba(255,255,255,.04)!important;padding-top:10px;padding-bottom:4px;border-top:1px solid rgba(255,255,255,.07);border-bottom:none;}}
.proj-table tbody tr:hover td{{background:rgba(255,255,255,.022);}}

/* ── Warning banner ── */
.dcf-warn{{background:rgba(255,159,10,.07);border:1px solid rgba(255,159,10,.22);border-radius:10px;padding:12px 16px;margin:10px 0;font-size:12px;color:rgba(255,255,255,.75);line-height:1.7;}}

/* ── Methodology note ── */
.dcf-note{{margin-top:28px;padding:14px 18px;background:rgba(255,255,255,.015);border-radius:10px;font-size:11px;color:rgba(255,255,255,.45);line-height:1.9;}}
.dcf-note b{{color:rgba(255,255,255,.65);}}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (cached, mirrors financials.py pattern)
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
def _get_current_price(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).info or {}
        p = info.get("currentPrice") or info.get("regularMarketPrice")
        return float(p) if p else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_m(v: float, decimals: int = 1) -> str:
    """Format dollar value in billions/millions."""
    if np.isnan(v):
        return "—"
    if abs(v) >= 1e12:
        return f"${v/1e12:,.{decimals}f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:,.{decimals}f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:,.{decimals}f}M"
    return f"${v:,.0f}"


def _fmt_pct(v: float, decimals: int = 1) -> str:
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}%"


def _fmt_price(v: float) -> str:
    if np.isnan(v):
        return "—"
    return f"${v:,.2f}"


def _safe(v, default: float = np.nan) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# HTML component builders
# ─────────────────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str, sub: str = "", css_class: str = "") -> str:
    return f"""
<div class="dcf-card {css_class}">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {"" if not sub else f'<div class="sub">{sub}</div>'}
</div>"""


def _badge(text: str, kind: str = "calc") -> str:
    return f'<span class="src-badge {kind}">{text}</span>'


def _beta_badge(br: BetaResult) -> str:
    method_map = {
        "fundamental": ("sec", "SEC"),
        "regression": ("reg", "Regression"),
        "yfinance": ("yf", "yfinance"),
        "fallback": ("calc", "Default"),
    }
    kind, label = method_map.get(br.method, ("calc", br.method))
    return _badge(label, kind)


def _render_sensitivity(df: pd.DataFrame, current_price: float, base_wacc_str: str = None, base_tg_str: str = None) -> str:
    """Render sensitivity table as styled HTML with heat map coloring."""
    if df.empty:
        return "<p style='color:rgba(255,255,255,.3)'>No sensitivity data.</p>"

    # Gather all numeric values for heat map range
    flat = []
    for col in df.columns:
        for v in df[col]:
            if not pd.isna(v):
                flat.append(v)
    if not flat:
        return "<p>No data.</p>"
    p10, p90 = np.percentile(flat, 10), np.percentile(flat, 90)

    # Column headers
    h = f'<th class="row-label">{df.index.name or ""}</th>'
    for col in df.columns:
        is_base = (base_tg_str and col == base_tg_str)
        cls = " class='base-col'" if is_base else ""
        h += f"<th{cls}>{col}</th>"

    html = f'<div class="sens-wrap"><table class="sens-table"><thead><tr>{h}</tr></thead><tbody>'

    for idx, row in df.iterrows():
        is_base_row = (base_wacc_str and str(idx) == base_wacc_str)
        row_open = '<tr style="background:rgba(10,124,255,0.04);">' if is_base_row else "<tr>"
        cells = f'<td class="row-label">{idx}</td>'
        for col in df.columns:
            v = row[col]
            is_base_col = (base_tg_str and col == base_tg_str)
            base_cls = "base-col " if is_base_col else ""
            if pd.isna(v):
                cells += f'<td class="{base_cls}cell-mid">—</td>'
            else:
                if v >= p90:
                    color_cls = "cell-hi"
                elif v <= p10:
                    color_cls = "cell-lo"
                else:
                    color_cls = "cell-mid"
                # Upside/downside vs current price
                if not np.isnan(current_price) and current_price > 0:
                    upside = (v - current_price) / current_price * 100
                    tip = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
                    label = f"${v:,.2f} <span style='font-size:9.5px;opacity:.5'>({tip})</span>"
                else:
                    label = f"${v:,.2f}"
                cells += f'<td class="{base_cls}{color_cls}">{label}</td>'
        html += f"{row_open}{cells}</tr>"

    return html + "</tbody></table></div>"


def _render_projection_table(result, assumptions: DCFAssumptions, divisor: float = 1e9, unit: str = "$B") -> str:
    """Render the 10-year projection table as HTML."""
    n = assumptions.n_years
    years = [f"Y{i}" for i in range(1, n + 1)]

    rows_data = [
        ("REVENUE DRIVERS", None, "rs"),
        ("Revenue", result.projected_revenues, "rt"),
        ("Revenue Growth %", None, ""),
        ("EBITDA", result.projected_ebitda, ""),
        ("EBITDA Margin %", None, ""),
        ("FREE CASH FLOW BUILD", None, "rs"),
        ("EBITDA", result.projected_ebitda, ""),
        ("less: D&A (add back)", None, ""),
        ("= EBIT", None, ""),
        ("less: Taxes on EBIT", None, ""),
        ("= NOPAT", None, ""),
        ("+ D&A", None, ""),
        ("– CapEx", None, ""),
        ("– ΔNWC", None, ""),
        ("Unlevered FCF", result.projected_ufcf, "rt"),
        ("PRESENT VALUE", None, "rs"),
        ("Discount Factor", result.discount_factors, ""),
        ("PV of UFCF", result.pv_ufcf, ""),
    ]

    # Precompute derived rows
    prev_rev = assumptions.base_revenue
    rev_growths = []
    ebitda_margins = []
    ebits = []
    taxes = []
    nopats = []
    das = []
    capexs = []
    dnwcs = []

    for i, rev in enumerate(result.projected_revenues):
        rev_growths.append((rev / prev_rev - 1) * 100 if prev_rev > 0 else np.nan)
        prev_rev = rev
        ebitda_margins.append(result.projected_ebitda[i] / rev * 100 if rev > 0 else np.nan)
        da = rev * assumptions.da_pct_revenue
        ebit = result.projected_ebitda[i] - da
        tax = ebit * assumptions.tax_rate if ebit > 0 else 0.0
        nopat = ebit - tax
        capex = rev * assumptions.capex_pct_revenue
        if i == 0:
            prev_r = assumptions.base_revenue
        else:
            prev_r = result.projected_revenues[i - 1]
        dnwc = (rev - prev_r) * assumptions.nwc_pct_revenue
        das.append(da)
        ebits.append(ebit)
        taxes.append(tax)
        nopats.append(nopat)
        capexs.append(capex)
        dnwcs.append(dnwc)

    derived = {
        "Revenue Growth %": (rev_growths, True, "%"),
        "EBITDA Margin %":  (ebitda_margins, True, "%"),
        "less: D&A (add back)": (das, False, "$"),
        "= EBIT":           (ebits, False, "$"),
        "less: Taxes on EBIT": (taxes, False, "$"),
        "= NOPAT":          (nopats, False, "$"),
        "+ D&A":            (das, False, "$"),
        "– CapEx":          (capexs, False, "$"),
        "– ΔNWC":           (dnwcs, False, "$"),
        "Discount Factor":  (result.discount_factors, True, "x"),
    }

    # Header
    hdrs = "".join(f"<th>{y}</th>" for y in years)
    html = f'<div class="proj-wrap"><table class="proj-table"><thead><tr><th class="hl">{unit}</th>{hdrs}</tr></thead><tbody>'

    for label, data_list, row_cls in rows_data:
        if row_cls == "rs":
            html += f'<tr class="rs"><td class="hl" colspan="{n+1}">{label}</td></tr>'
            continue

        # Use precomputed if no direct data
        if data_list is None:
            if label in derived:
                data_list, is_pct, fmt_type = derived[label]
            else:
                continue

        else:
            is_pct = False
            fmt_type = "$"
            if label in derived:
                _, is_pct, fmt_type = derived[label]

        cells = f'<td class="hl">{label}</td>'
        for v in data_list:
            if np.isnan(v) if isinstance(v, float) else False:
                cells += "<td>—</td>"
            elif fmt_type == "%":
                sign = "+" if v >= 0 else ""
                cls = ""
                if "Growth" in label or "Margin" in label:
                    cls = ""
                cells += f'<td class="{cls}">{sign}{v:.1f}%</td>'
            elif fmt_type == "x":
                cells += f"<td>{v:.4f}x</td>"
            else:
                # dollar value
                scaled = v / divisor
                if abs(scaled) >= 100:
                    s = f"{abs(scaled):,.0f}"
                else:
                    s = f"{abs(scaled):,.2f}"
                neg_cls = ' style="color:#FF3B30"' if v < 0 else ""
                txt = f"({s})" if v < 0 else s
                cells += f"<td{neg_cls}>{txt}</td>"

        tr_cls = f' class="{row_cls}"' if row_cls else ""
        html += f"<tr{tr_cls}>{cells}</tr>"

    return html + "</tbody></table></div>"


# ─────────────────────────────────────────────────────────────────────────────
# Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

def _chart_waterfall(result, assumptions: DCFAssumptions) -> go.Figure:
    """Waterfall chart: PV of each year's UFCF + PV terminal value."""
    n = assumptions.n_years
    labels = [f"Y{i+1}" for i in range(n)] + ["Terminal\nValue"]
    values = list(result.pv_ufcf) + [result.pv_terminal_value]

    colors = [BLUE] * n + [ORANGE]
    bar_colors = [f"rgba(10,124,255,{0.45 + 0.04*i})" for i in range(n)] + [f"rgba(255,159,10,0.70)"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=[v / 1e9 for v in values],
        marker_color=bar_colors,
        text=[f"${v/1e9:.2f}B" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="rgba(255,255,255,0.65)"),
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=24, b=0),
        height=280,
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(
            title="$B",
            gridcolor="rgba(255,255,255,0.05)",
            tickformat="$.1f",
        ),
        bargap=0.25,
    )
    # TV annotation
    fig.add_annotation(
        x=labels[-1], y=result.pv_terminal_value / 1e9,
        text=f"{result.pv_tv_pct:.0f}% of EV",
        showarrow=False,
        yanchor="bottom",
        yshift=22,
        font=dict(size=9.5, color=ORANGE),
    )
    return fig


def _chart_revenue_fcf(result, assumptions: DCFAssumptions) -> go.Figure:
    """Dual-axis chart: revenue bars + UFCF line."""
    n = assumptions.n_years
    years = [f"Y{i+1}" for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Revenue",
        x=years,
        y=[v / 1e9 for v in result.projected_revenues],
        marker_color="rgba(10,124,255,0.30)",
        marker_line=dict(color="rgba(10,124,255,0.60)", width=1),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="Unlevered FCF",
        x=years,
        y=[v / 1e9 for v in result.projected_ufcf],
        mode="lines+markers",
        line=dict(color=UP, width=2),
        marker=dict(size=5),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="EBITDA",
        x=years,
        y=[v / 1e9 for v in result.projected_ebitda],
        mode="lines+markers",
        line=dict(color=ORANGE, width=1.5, dash="dot"),
        marker=dict(size=4),
        yaxis="y",
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=24, b=0),
        height=280,
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(
            title="$B",
            gridcolor="rgba(255,255,255,0.05)",
            tickformat="$.1f",
        ),
        bargap=0.30,
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_dcf(
    ticker: str,
    summary: Optional[dict] = None,
    data: Optional[dict] = None,
) -> None:
    """
    Render the DCF valuation page for a given ticker.

    Args:
        ticker  : Stock ticker (e.g. "AAPL")
        summary : Output of build_company_summary() — if None, will be fetched
        data    : Output of fetch_company_data_unified() — if None, will be fetched
    """
    _inject_css()
    ticker = ticker.upper().strip()

    # ── Fetch data (reuse from session cache if available) ────────────────────
    cik = _cik_map().get(ticker, "")

    # Try to pull from session state first (set by app.py data pipeline)
    if data is None:
        cache = st.session_state.get("ticker_data_cache", {})
        data = cache.get(ticker)

    if summary is None:
        scache = st.session_state.get("ticker_summary_cache", {})
        summary = scache.get(ticker)

    # If still missing, fetch fresh (spinner shown)
    if data is None:
        with st.spinner(f"Fetching {ticker} data…"):
            try:
                # Import here to avoid circular dependency with app.py
                from app import fetch_company_data_unified
                data = fetch_company_data_unified(ticker, cik)
            except Exception:
                # Standalone fallback
                from sec_engine.sec_fetch import fetch_company_facts
                from sec_engine.ltm import build_ltm_financials, extract_annual_series
                from sec_engine.normalize import GAAP_MAP
                try:
                    facts = fetch_company_facts(cik) if cik else {}
                    ltm = build_ltm_financials(facts, GAAP_MAP)
                    data = {"ltm_data": ltm, "balance_data": ltm, "metadata": {}}
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

    # ── SIC code (for beta) ───────────────────────────────────────────────────
    sic_code = ""
    if cik:
        with st.spinner("Loading SIC code…"):
            sic_code = _load_sic(cik)

    # ── Current price ─────────────────────────────────────────────────────────
    current_price = _get_current_price(ticker)

    # ── Company name ──────────────────────────────────────────────────────────
    meta = data.get("metadata", {})
    name = meta.get("name") or ticker

    # ── Page title ────────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:30px;font-weight:800;color:#fff;margin-bottom:2px;">'
        f'{ticker}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:13px;color:rgba(255,255,255,.4);margin-top:0;margin-bottom:20px;">'
        f'{name} &nbsp;·&nbsp; DCF Valuation Model</p>',
        unsafe_allow_html=True,
    )

    # ── Build default assumptions from financial data ─────────────────────────
    with st.spinner("Building assumptions from SEC data…"):
        facts_for_beta = _load_facts(cik) if cik else {}
        default_assumptions, beta_result = build_dcf_assumptions(
            ticker=ticker,
            summary=summary,
            data=data,
            sic_code=sic_code,
            facts=facts_for_beta,
        )

    # ── Session state key prefix (per-ticker so different tickers don't clash) ─
    pfx = f"dcf_{ticker}_"

    def _state(key: str, default):
        full = pfx + key
        if full not in st.session_state:
            st.session_state[full] = default
        return full

    # ─────────────────────────────────────────────────────────────────────────
    # ASSUMPTION CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="asm-header">Assumptions</div>', unsafe_allow_html=True)

    col_g, col_m, col_w = st.columns(3, gap="large")

    with col_g:
        st.markdown('<div class="asm-header" style="margin-top:4px;">Growth</div>', unsafe_allow_html=True)

        rev_g_y1 = st.slider(
            "Revenue Growth — Year 1",
            min_value=-20.0, max_value=80.0,
            value=float(round(default_assumptions.revenue_growth_y1 * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("rev_g_y1", round(default_assumptions.revenue_growth_y1 * 100, 1)),
            help="Year 1 revenue growth rate. Pre-filled from your 3yr historical CAGR."
        ) / 100

        rev_g_y5 = st.slider(
            "Revenue Growth — Year 5",
            min_value=-10.0, max_value=50.0,
            value=float(round(default_assumptions.revenue_growth_y5 * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("rev_g_y5", round(default_assumptions.revenue_growth_y5 * 100, 1)),
            help="Year 5 revenue growth rate. Growth interpolates linearly from Y1→Y5, then fades to terminal."
        ) / 100

        terminal_g = st.slider(
            "Terminal Growth Rate",
            min_value=0.0, max_value=5.0,
            value=float(round(default_assumptions.terminal_growth_rate * 100, 1)),
            step=0.1, format="%.1f%%",
            key=_state("terminal_g", round(default_assumptions.terminal_growth_rate * 100, 1)),
            help="Perpetuity growth rate. Should not exceed long-run nominal GDP growth (~2.5%)."
        ) / 100

    with col_m:
        st.markdown('<div class="asm-header" style="margin-top:4px;">Profitability & CapEx</div>', unsafe_allow_html=True)

        ebitda_m = st.slider(
            "EBITDA Margin",
            min_value=0.0, max_value=85.0,
            value=float(round(default_assumptions.ebitda_margin * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("ebitda_m", round(default_assumptions.ebitda_margin * 100, 1)),
            help="Projected EBITDA as % of revenue. Pre-filled from LTM EBITDA margin."
        ) / 100

        capex_pct = st.slider(
            "CapEx % Revenue",
            min_value=0.0, max_value=40.0,
            value=float(round(default_assumptions.capex_pct_revenue * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("capex_pct", round(default_assumptions.capex_pct_revenue * 100, 1)),
            help="CapEx as % of projected revenue. Pre-filled from LTM actual."
        ) / 100

        da_pct = st.slider(
            "D&A % Revenue",
            min_value=0.0, max_value=25.0,
            value=float(round(default_assumptions.da_pct_revenue * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("da_pct", round(default_assumptions.da_pct_revenue * 100, 1)),
            help="Depreciation & Amortization as % of revenue."
        ) / 100

        nwc_pct = st.slider(
            "NWC Change % Rev Δ",
            min_value=-5.0, max_value=15.0,
            value=float(round(default_assumptions.nwc_pct_revenue * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("nwc_pct", round(default_assumptions.nwc_pct_revenue * 100, 1)),
            help="Incremental change in Net Working Capital as % of each year's revenue change. Positive = cash outflow."
        ) / 100

    with col_w:
        st.markdown('<div class="asm-header" style="margin-top:4px;">WACC Inputs</div>', unsafe_allow_html=True)

        beta_val = st.slider(
            f"Beta {_beta_badge(beta_result)}",
            min_value=0.10, max_value=3.50,
            value=float(round(default_assumptions.beta, 2)),
            step=0.05,
            key=_state("beta", round(default_assumptions.beta, 2)),
            help=f"Equity beta. Source: {beta_result.detail}"
        )

        rf_rate = st.slider(
            "Risk-Free Rate",
            min_value=0.0, max_value=8.0,
            value=float(round(default_assumptions.risk_free_rate * 100, 2)),
            step=0.05, format="%.2f%%",
            key=_state("rf_rate", round(default_assumptions.risk_free_rate * 100, 2)),
            help="10-year US Treasury yield. Update to current market rate."
        ) / 100

        erp = st.slider(
            "Equity Risk Premium",
            min_value=2.0, max_value=10.0,
            value=float(round(default_assumptions.equity_risk_premium * 100, 2)),
            step=0.25, format="%.2f%%",
            key=_state("erp", round(default_assumptions.equity_risk_premium * 100, 2)),
            help="Market ERP above risk-free rate. Damodaran implied ERP (US, 2025) ≈ 5.5%."
        ) / 100

        cod = st.slider(
            "Cost of Debt (pre-tax)",
            min_value=0.0, max_value=20.0,
            value=float(round(default_assumptions.cost_of_debt_pretax * 100, 2)),
            step=0.25, format="%.2f%%",
            key=_state("cod", round(default_assumptions.cost_of_debt_pretax * 100, 2)),
            help="Pre-tax cost of debt. Pre-filled as Interest Expense / Total Debt (LTM)."
        ) / 100

        debt_w = st.slider(
            "Debt Weight (% of capital)",
            min_value=0.0, max_value=90.0,
            value=float(round(default_assumptions.debt_weight * 100, 1)),
            step=1.0, format="%.0f%%",
            key=_state("debt_w", round(default_assumptions.debt_weight * 100, 1)),
            help="Debt / (Debt + Market Cap equity). Pre-filled from current capital structure."
        ) / 100

        tax_rate = st.slider(
            "Tax Rate",
            min_value=0.0, max_value=50.0,
            value=float(round(default_assumptions.tax_rate * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("tax_rate", round(default_assumptions.tax_rate * 100, 1)),
            help="Effective corporate tax rate. Pre-filled from constants.py registry."
        ) / 100

    # ── Assemble final assumptions from slider values ─────────────────────────
    assumptions = DCFAssumptions(
        n_years=10,
        stage1_years=5,
        revenue_growth_y1=rev_g_y1,
        revenue_growth_y5=rev_g_y5,
        ebitda_margin=ebitda_m,
        capex_pct_revenue=capex_pct,
        da_pct_revenue=da_pct,
        nwc_pct_revenue=nwc_pct,
        terminal_growth_rate=terminal_g,
        beta=beta_val,
        risk_free_rate=rf_rate,
        equity_risk_premium=erp,
        cost_of_debt_pretax=cod,
        tax_rate=tax_rate,
        debt_weight=debt_w,
        equity_weight=1.0 - debt_w,
        base_ufcf=default_assumptions.base_ufcf,
        base_revenue=default_assumptions.base_revenue,
        net_debt=default_assumptions.net_debt,
        shares_outstanding=default_assumptions.shares_outstanding,
    )

    # ── Run the model ─────────────────────────────────────────────────────────
    result = run_dcf(assumptions)

    # ── Warnings ──────────────────────────────────────────────────────────────
    for w in result.warnings:
        st.markdown(f'<div class="dcf-warn">⚠ {w}</div>', unsafe_allow_html=True)

    if result.warnings and "undefined" in str(result.warnings):
        return  # Fatal error — can't render results

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:20px 0;'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # KEY METRICS ROW
    # ─────────────────────────────────────────────────────────────────────────
    ke, kd, wacc = result.cost_of_equity, result.cost_of_debt_aftertax, result.wacc

    # Upside / downside
    if not np.isnan(result.intrinsic_price) and not np.isnan(current_price) and current_price > 0:
        upside_pct = (result.intrinsic_price - current_price) / current_price * 100
        upside_str = f"+{upside_pct:.1f}%" if upside_pct >= 0 else f"{upside_pct:.1f}%"
        upside_card_cls = "up" if upside_pct >= 0 else "down"
        upside_label = "Upside to Intrinsic" if upside_pct >= 0 else "Downside to Intrinsic"
    else:
        upside_str = "—"
        upside_card_cls = ""
        upside_label = "vs. Intrinsic Value"

    cards_html = '<div class="dcf-cards">'
    cards_html += _metric_card("Intrinsic Price", _fmt_price(result.intrinsic_price), f"vs. {_fmt_price(current_price)} current", "highlight")
    cards_html += _metric_card(upside_label, upside_str, "", upside_card_cls)
    cards_html += _metric_card("Enterprise Value", _fmt_m(result.enterprise_value), "DCF-implied", "")
    cards_html += _metric_card("WACC", f"{wacc:.2%}", f"Ke {ke:.2%} · Kd {kd:.2%}", "")
    cards_html += _metric_card("Terminal Value", f"{result.pv_tv_pct:.0f}% of EV", _fmt_m(result.pv_terminal_value), "orange")
    cards_html += _metric_card("Beta", f"{beta_val:.2f}", beta_result.method.capitalize(), "")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown(
            '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.35);margin-bottom:4px;">PV of Cash Flows by Year</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_chart_waterfall(result, assumptions), use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown(
            '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.35);margin-bottom:4px;">Revenue · EBITDA · Unlevered FCF</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_chart_revenue_fcf(result, assumptions), use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────────
    # TABS: Sensitivity / Projection Model / WACC Bridge
    # ─────────────────────────────────────────────────────────────────────────
    tab_sens1, tab_sens2, tab_proj, tab_wacc = st.tabs([
        "WACC × Terminal Growth",
        "Revenue Growth × EBITDA Margin",
        "10-Year Model",
        "WACC Bridge",
    ])

    with tab_sens1:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:12px;">Intrinsic price per share across WACC and terminal growth rate scenarios. '
            'Green = above current price. Red = below current price. Base case highlighted in blue.</p>',
            unsafe_allow_html=True,
        )
        # Build ±200bps WACC range around current WACC
        wacc_range = sorted(set([
            round(wacc - 0.04, 3), round(wacc - 0.03, 3), round(wacc - 0.02, 3),
            round(wacc - 0.01, 3), round(wacc, 3),
            round(wacc + 0.01, 3), round(wacc + 0.02, 3), round(wacc + 0.03, 3),
        ]))
        wacc_range = [w for w in wacc_range if 0.03 < w < 0.35]

        tg_range = sorted(set([
            round(terminal_g - 0.015, 3), round(terminal_g - 0.01, 3),
            round(terminal_g - 0.005, 3), round(terminal_g, 3),
            round(terminal_g + 0.005, 3), round(terminal_g + 0.01, 3),
            round(terminal_g + 0.015, 3),
        ]))
        tg_range = [t for t in tg_range if 0.0 < t < wacc - 0.01]

        with st.spinner("Computing sensitivity…"):
            sens_df = sensitivity_table(assumptions, wacc_range, tg_range)

        base_wacc_str = f"{wacc:.1%}"
        base_tg_str = f"{terminal_g:.1%}"
        st.markdown(
            _render_sensitivity(sens_df, current_price, base_wacc_str, base_tg_str),
            unsafe_allow_html=True,
        )

    with tab_sens2:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:12px;">Intrinsic price per share across revenue growth and EBITDA margin scenarios.</p>',
            unsafe_allow_html=True,
        )
        g_range = sorted(set([
            round(rev_g_y1 - 0.12, 2), round(rev_g_y1 - 0.08, 2), round(rev_g_y1 - 0.04, 2),
            round(rev_g_y1, 2), round(rev_g_y1 + 0.04, 2), round(rev_g_y1 + 0.08, 2),
            round(rev_g_y1 + 0.12, 2),
        ]))
        g_range = [g for g in g_range if -0.20 < g < 0.80]

        m_range = sorted(set([
            round(ebitda_m - 0.10, 2), round(ebitda_m - 0.05, 2),
            round(ebitda_m, 2), round(ebitda_m + 0.05, 2), round(ebitda_m + 0.10, 2),
        ]))
        m_range = [m for m in m_range if 0.0 < m < 0.90]

        with st.spinner("Computing sensitivity…"):
            sens_df2 = sensitivity_table_growth(assumptions, g_range, m_range)

        base_g_str = f"{rev_g_y1:.0%}"
        base_m_str = f"{ebitda_m:.0%}"
        st.markdown(
            _render_sensitivity(sens_df2, current_price, base_g_str, base_m_str),
            unsafe_allow_html=True,
        )

    with tab_proj:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:12px;">10-year projected income and free cash flow build. All values in $B unless otherwise noted.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _render_projection_table(result, assumptions, divisor=1e9, unit="$B"),
            unsafe_allow_html=True,
        )

        # Terminal value bridge below the table
        st.markdown("<br>", unsafe_allow_html=True)
        tv_col1, tv_col2, tv_col3 = st.columns(3)
        with tv_col1:
            st.markdown(
                f'<div class="dcf-card"><div class="label">Year 10 UFCF</div>'
                f'<div class="value">{_fmt_m(result.projected_ufcf[-1])}</div></div>',
                unsafe_allow_html=True,
            )
        with tv_col2:
            st.markdown(
                f'<div class="dcf-card orange"><div class="label">Terminal Value (undiscounted)</div>'
                f'<div class="value">{_fmt_m(result.terminal_value)}</div>'
                f'<div class="sub">Year 10 UFCF × (1+{terminal_g:.1%}) / ({wacc:.2%} − {terminal_g:.1%})</div></div>',
                unsafe_allow_html=True,
            )
        with tv_col3:
            st.markdown(
                f'<div class="dcf-card"><div class="label">PV of Terminal Value</div>'
                f'<div class="value">{_fmt_m(result.pv_terminal_value)}</div>'
                f'<div class="sub">{result.pv_tv_pct:.0f}% of total Enterprise Value</div></div>',
                unsafe_allow_html=True,
            )

    with tab_wacc:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:16px;">WACC component breakdown and equity value bridge.</p>',
            unsafe_allow_html=True,
        )

        wacc_col1, wacc_col2 = st.columns(2, gap="large")

        with wacc_col1:
            st.markdown('<div class="asm-header">WACC Build</div>', unsafe_allow_html=True)
            wacc_rows = [
                ("Risk-Free Rate", f"{rf_rate:.2%}", "10yr UST yield"),
                ("× Beta (β)", f"{beta_val:.2f}x", f"{beta_result.method.capitalize()} method"),
                ("× Equity Risk Premium", f"{erp:.2%}", "Damodaran implied ERP"),
                ("= Cost of Equity (Ke)", f"{ke:.2%}", f"Ke = Rf + β × ERP"),
                ("", "", ""),
                ("Cost of Debt (pre-tax)", f"{cod:.2%}", "Interest expense / total debt"),
                ("× (1 – Tax Rate)", f"{1-tax_rate:.2%}", f"After-tax shield at {tax_rate:.0%}"),
                ("= Cost of Debt (Kd)", f"{kd:.2%}", "After-tax cost of debt"),
                ("", "", ""),
                ("Equity Weight", f"{(1-debt_w):.0%}", "Market cap / total capital"),
                ("Debt Weight", f"{debt_w:.0%}", "Total debt / total capital"),
                ("= WACC", f"{wacc:.2%}", "Ke×We + Kd×Wd"),
            ]
            for label, value, note in wacc_rows:
                if not label:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    continue
                is_result = label.startswith("=")
                weight = "700" if is_result else "400"
                color = "#fff" if is_result else "rgba(255,255,255,.8)"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                    f'<span style="font-size:12.5px;color:{color};font-weight:{weight};">{label}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:{BLUE if is_result else "#fff"};">{value}'
                    f'<span style="font-size:10px;font-weight:400;color:rgba(255,255,255,.3);margin-left:8px;">{note}</span></span></div>',
                    unsafe_allow_html=True,
                )

        with wacc_col2:
            st.markdown('<div class="asm-header">Equity Value Bridge</div>', unsafe_allow_html=True)

            bridge_rows = [
                ("Sum of PV (UFCF, Y1–Y10)", result.sum_pv_ufcf, f"{result.pv_ufcf_pct:.0f}% of EV"),
                ("+ PV of Terminal Value", result.pv_terminal_value, f"{result.pv_tv_pct:.0f}% of EV"),
                ("= Enterprise Value", result.enterprise_value, "DCF-implied"),
                ("", np.nan, ""),
                ("– Net Debt", assumptions.net_debt, "Total Debt – Cash"),
                ("= Equity Value", result.equity_value, "EV – Net Debt"),
                ("", np.nan, ""),
                ("÷ Shares Outstanding", assumptions.shares_outstanding, "Diluted"),
                ("= Intrinsic Price per Share", result.intrinsic_price, ""),
                ("Current Market Price", current_price, ""),
            ]
            for label, value, note in bridge_rows:
                if not label:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    continue
                is_result = label.startswith("=")
                is_final = "Intrinsic Price" in label
                color = BLUE if is_final else ("#fff" if is_result else "rgba(255,255,255,.8)")
                weight = "800" if is_final else ("700" if is_result else "400")

                if isinstance(value, float) and not np.isnan(value):
                    if "Share" in label and "Intrinsic" not in label and "÷" in label:
                        val_str = f"{value/1e6:,.1f}M"
                    elif abs(value) < 1000:
                        val_str = f"${value:,.2f}"
                    else:
                        val_str = _fmt_m(value)
                else:
                    val_str = "—"

                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                    f'<span style="font-size:12.5px;color:{color};font-weight:{weight};">{label}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:{color};">{val_str}'
                    f'<span style="font-size:10px;font-weight:400;color:rgba(255,255,255,.3);margin-left:8px;">{note}</span></span></div>',
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Beta detail expander
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("Beta Calculation Detail", expanded=False):
        st.markdown(
            f"""
<div style="font-size:12px;color:rgba(255,255,255,.6);line-height:1.9;">
<b style="color:#fff;">Method used:</b> {beta_result.method.capitalize()}
{_beta_badge(beta_result)}<br>
<b style="color:#fff;">Detail:</b> {beta_result.detail or "—"}<br>
{"<b style='color:#fff;'>Unlevered β (industry):</b> " + f"{beta_result.unlevered_beta:.3f}" + "<br>" if not np.isnan(beta_result.unlevered_beta) else ""}
{"<b style='color:#fff;'>Re-levered β (Hamada):</b> " + f"{beta_result.relevered_beta:.3f}" + "<br>" if not np.isnan(beta_result.relevered_beta) else ""}
{"<b style='color:#fff;'>Regression β (raw OLS):</b> " + f"{beta_result.regression_beta_raw:.3f}" + "<br>" if not np.isnan(beta_result.regression_beta_raw) else ""}
{"<b style='color:#fff;'>Regression R²:</b> " + f"{beta_result.regression_r2:.3f}" + "<br>" if not np.isnan(beta_result.regression_r2) else ""}
{"<b style='color:#fff;'>Vasicek-adjusted β:</b> " + f"{beta_result.vasicek_beta:.3f}" + "<br>" if not np.isnan(beta_result.vasicek_beta) else ""}
<b style="color:#fff;">Final β applied:</b> {beta_result.beta:.3f}
</div>""",
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Methodology note
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="dcf-note">
<b>Data Sources</b> — All financial inputs (Revenue, EBITDA, CapEx, D&A, Debt, Cash) are sourced from SEC EDGAR
via the same pipeline as the Financials page (XBRL companyfacts, 10-K annual filings, LTM quarterly construction).
yfinance is used as a secondary fallback for fields missing from EDGAR. &nbsp;·&nbsp;
<b>Beta Hierarchy</b> — (1) SEC-derived fundamental beta using SIC-code industry unlevered beta (Damodaran methodology)
re-levered via Hamada equation with EDGAR debt/equity; (2) 5yr monthly OLS regression vs. SPY with Vasicek shrinkage (0.60×β + 0.40×1.0);
(3) Yahoo Finance published beta; (4) market beta β=1.0. &nbsp;·&nbsp;
<b>Model</b> — Two-stage DCF: Stage 1 (Y1–Y5) explicit growth; Stage 2 (Y6–Y10) linear fade to terminal growth.
Terminal value via Gordon Growth Model. WACC = Ke×We + Kd(1–t)×Wd. &nbsp;·&nbsp;
<b>Limitations</b> — This model is a starting framework. UFCF projections are highly sensitive to growth, margin,
and WACC assumptions. Terminal value represents a large fraction of enterprise value in growth companies.
Adjust all assumptions to reflect your own research before drawing investment conclusions.
</div>""",
        unsafe_allow_html=True,
    )
