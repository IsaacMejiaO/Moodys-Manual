# ui/dcf.py
# ──────────────────────────────────────────────────────────────────────────────
# Valuation Page
# ──────────────────────────────────────────────────────────────────────────────
#
# Layout
# ------
#  Header:    Ticker · Valuation Model
#  Section A: Key result cards  (Intrinsic / Upside / Street Target / EV / WACC / TV / Beta)
#  Section B: "Model Assumptions" expander  (Configure-optimizer style, 3 columns)
#  Tabs:
#    ├─ Your Model         — 10-year projection table + TV bridge cards
#    ├─ Wall St. Consensus — Street model run through same DCF engine + WACC bridge
#    ├─ Charts             — PV waterfall + Revenue/EBITDA/FCF
#    └─ WACC Bridge        — component build + equity value bridge
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

/* ── Metric cards ── */
.val-cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 20px 0;}}
.val-card{{flex:1;min-width:130px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 18px;}}
.val-card .label{{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:6px;}}
.val-card .value{{font-size:22px;font-weight:800;color:#fff;line-height:1.1;}}
.val-card .sub{{font-size:11px;color:rgba(255,255,255,.4);margin-top:4px;}}
.val-card.highlight{{border-color:rgba(10,124,255,0.4);background:rgba(10,124,255,0.06);}}
.val-card.up{{border-color:rgba(0,200,5,0.35);background:rgba(0,200,5,0.05);}}
.val-card.down{{border-color:rgba(255,59,48,0.35);background:rgba(255,59,48,0.05);}}
.val-card.orange{{border-color:rgba(255,159,10,0.35);background:rgba(255,159,10,0.05);}}
.val-card.purple{{border-color:rgba(191,90,242,0.35);background:rgba(191,90,242,0.05);}}
.val-card.street{{border-color:rgba(0,200,5,0.25);background:rgba(0,200,5,0.04);}}

/* ── Configure-optimizer style labels (matches portfolio_monte_carlo.py) ── */
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

/* ── Projection / sensitivity tables — matches financials.py fin-table ── */
.fin-wrap{{overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.015);margin-top:8px;}}
.fin-table{{width:100%;border-collapse:collapse;font-size:12.5px;}}
.fin-table thead th{{background:rgba(255,255,255,0.04);color:#fff;font-weight:700;font-size:10px;letter-spacing:.07em;text-transform:uppercase;padding:9px 14px;border-bottom:1px solid rgba(255,255,255,0.09);text-align:right;white-space:nowrap;}}
.fin-table thead th.hl{{text-align:left;min-width:200px;}}
.fin-table td{{padding:6px 14px;border-bottom:1px solid rgba(255,255,255,.035);color:#fff;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}}
.fin-table td.hl{{text-align:left;color:rgba(255,255,255,.65);font-size:11.5px;}}
.fin-table .rt td{{font-weight:700!important;border-top:1px solid rgba(255,255,255,.12)!important;border-bottom:1px solid rgba(255,255,255,.12)!important;background:rgba(255,255,255,.018)!important;}}
.fin-table .rs td{{font-size:9.5px!important;font-weight:800!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:#fff!important;background:rgba(255,255,255,.04)!important;padding-top:10px!important;padding-bottom:4px!important;border-top:1px solid rgba(255,255,255,.07)!important;border-bottom:none!important;}}
.fin-table tbody tr:hover td{{background:rgba(255,255,255,.022);}}

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
    """
    Pull Wall Street consensus from yfinance.
    Returns keys: price_targets, rating, n_analysts,
                  rev_growth_fwd, eps_growth_fwd, ebitda_margin,
                  street_rev_g_y1, street_rev_g_y5, street_ebitda_m,
                  rec_trend (DataFrame), upgrades_downgrades (DataFrame)
    """
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
        "rec_trend": pd.DataFrame(),
        "upgrades_downgrades": pd.DataFrame(),
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

        # EBITDA margin from info
        ebitda = info.get("ebitda")
        rev    = info.get("totalRevenue")
        if ebitda and rev and float(rev) > 0:
            result["ebitda_margin"]   = float(ebitda) / float(rev)
            result["street_ebitda_m"] = float(ebitda) / float(rev)

        # Forward revenue growth from revenue_estimate
        try:
            rev_est = t.revenue_estimate
            if rev_est is not None and not rev_est.empty and len(rev_est) >= 2:
                g0 = _safe(rev_est.iloc[0].get("growth"))
                g1 = _safe(rev_est.iloc[1].get("growth"))
                result["rev_growth_fwd"]   = g0
                result["street_rev_g_y1"]  = g0
                result["street_rev_g_y5"]  = g1  # use year+2 as approximate Y5 fade target
        except Exception:
            pass

        # EPS growth
        try:
            eps_est = t.earnings_estimate
            if eps_est is not None and not eps_est.empty and len(eps_est) >= 2:
                result["eps_growth_fwd"] = _safe(eps_est.iloc[1].get("growth"))
        except Exception:
            pass

        # Recommendation trend (strong buy / buy / hold / sell counts by period)
        try:
            rt = t.recommendations_summary if hasattr(t, "recommendations_summary") else None
            if rt is None or (isinstance(rt, pd.DataFrame) and rt.empty):
                rt = t.recommendations if hasattr(t, "recommendations") else None
            if rt is not None and isinstance(rt, pd.DataFrame) and not rt.empty:
                result["rec_trend"] = rt.copy()
        except Exception:
            pass

        # Upgrades / downgrades history
        try:
            ud = t.upgrades_downgrades if hasattr(t, "upgrades_downgrades") else None
            if ud is not None and isinstance(ud, pd.DataFrame) and not ud.empty:
                result["upgrades_downgrades"] = ud.copy()
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
# HTML components
# ─────────────────────────────────────────────────────────────────────────────

def _card(label: str, value: str, sub: str = "", css: str = "") -> str:
    return (
        f'<div class="val-card {css}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        + (f'<div class="sub">{sub}</div>' if sub else "")
        + "</div>"
    )


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
# Projection table renderer (shared by Your Model and Street Consensus tabs)
# ─────────────────────────────────────────────────────────────────────────────

def _render_projection_table(result, assumptions: DCFAssumptions,
                              divisor: float = 1e9, unit: str = "$B") -> str:
    n     = assumptions.n_years
    years = [f"Y{i}" for i in range(1, n + 1)]

    # Precompute derived rows
    prev_rev = assumptions.base_revenue
    rev_growths, ebitda_margins, das, ebits, taxes, nopats, capexs, dnwcs = (
        [], [], [], [], [], [], [], []
    )
    for i, rev in enumerate(result.projected_revenues):
        rev_growths.append((rev / prev_rev - 1) * 100 if prev_rev > 0 else np.nan)
        prev_rev = rev
        ebitda_margins.append(result.projected_ebitda[i] / rev * 100 if rev > 0 else np.nan)
        da    = rev * assumptions.da_pct_revenue
        ebit  = result.projected_ebitda[i] - da
        tax   = ebit * assumptions.tax_rate if ebit > 0 else 0.0
        capex = rev * assumptions.capex_pct_revenue
        prev_r = assumptions.base_revenue if i == 0 else result.projected_revenues[i - 1]
        dnwc   = (rev - prev_r) * assumptions.nwc_pct_revenue
        das.append(da); ebits.append(ebit); taxes.append(tax)
        nopats.append(ebit - tax); capexs.append(capex); dnwcs.append(dnwc)

    rows = [
        ("REVENUE DRIVERS",      None,                      "rs"),
        ("Revenue",              result.projected_revenues, "rt"),
        ("Revenue Growth %",     rev_growths,               "pct"),
        ("EBITDA",               result.projected_ebitda,   ""),
        ("EBITDA Margin %",      ebitda_margins,            "pct"),
        ("FREE CASH FLOW BUILD", None,                      "rs"),
        ("EBITDA",               result.projected_ebitda,   ""),
        ("less: D&A (add back)", das,                       ""),
        ("= EBIT",               ebits,                     ""),
        ("less: Taxes on EBIT",  taxes,                     ""),
        ("= NOPAT",              nopats,                    ""),
        ("+ D&A",                das,                       ""),
        ("– CapEx",              capexs,                    ""),
        ("– ΔNWC",               dnwcs,                     ""),
        ("Unlevered FCF",        result.projected_ufcf,     "rt"),
        ("PRESENT VALUE",        None,                      "rs"),
        ("Discount Factor",      result.discount_factors,   "factor"),
        ("PV of UFCF",           result.pv_ufcf,            ""),
    ]

    hdrs = "".join(f"<th>{y}</th>" for y in years)
    html = (
        f'<div class="fin-wrap"><table class="fin-table">'
        f'<thead><tr><th class="hl">{unit}</th>{hdrs}</tr></thead><tbody>'
    )

    for label, data_list, row_cls in rows:
        if row_cls == "rs":
            html += f'<tr class="rs"><td class="hl" colspan="{n+1}">{label}</td></tr>'
            continue
        if data_list is None:
            continue
        cells = f'<td class="hl">{label}</td>'
        for v in data_list:
            if isinstance(v, float) and np.isnan(v):
                cells += "<td>—</td>"
            elif row_cls == "pct":
                sign = "+" if v >= 0 else ""
                cells += f"<td>{sign}{v:.1f}%</td>"
            elif row_cls == "factor":
                cells += f"<td>{v:.4f}x</td>"
            else:
                scaled = v / divisor
                s   = f"{abs(scaled):,.0f}" if abs(scaled) >= 100 else f"{abs(scaled):,.2f}"
                neg = ' style="color:#FF3B30"' if v < 0 else ""
                txt = f"({s})" if v < 0 else s
                cells += f"<td{neg}>{txt}</td>"
        tr_c = "rt" if row_cls == "rt" else ""
        html += f'<tr class="{tr_c}">{cells}</tr>'

    return html + "</tbody></table></div>"


def _render_tv_cards(result, assumptions: DCFAssumptions, wacc: float, terminal_g: float) -> None:
    """Three terminal-value summary cards below a projection table."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="val-card"><div class="label">Year 10 UFCF</div>'
            f'<div class="value">{_fmt_m(result.projected_ufcf[-1])}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="val-card orange"><div class="label">Terminal Value (undiscounted)</div>'
            f'<div class="value">{_fmt_m(result.terminal_value)}</div>'
            f'<div class="sub">Y10 UFCF × (1+{terminal_g:.1%}) / ({wacc:.2%} − {terminal_g:.1%})</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="val-card"><div class="label">PV of Terminal Value</div>'
            f'<div class="value">{_fmt_m(result.pv_terminal_value)}</div>'
            f'<div class="sub">{result.pv_tv_pct:.0f}% of Enterprise Value</div></div>',
            unsafe_allow_html=True,
        )


def _render_equity_value_bridge(result, assumptions: DCFAssumptions,
                                rf_rate, beta_val, erp, ke, kd, wacc,
                                cod, tax_rate, debt_w, current_price,
                                beta_result: BetaResult,
                                terminal_g: float) -> None:
    """Full Street Equity Value Bridge: WACC inputs + valuation bridge side by side."""

    # ── Derived intermediates ─────────────────────────────────────────────────
    equity_w      = 1.0 - debt_w
    terminal_rf   = rf_rate          # terminal risk-free (same rate by default)
    terminal_capm = terminal_rf + beta_val * erp
    terminal_wacc = terminal_capm * equity_w + kd * debt_w

    total_debt    = assumptions.net_debt + (
        # approximate: net_debt = total_debt - cash;
        # we keep total_debt as net_debt for simplicity when cash is unknown
        0.0
    )
    # Use stored values where available, fall back to net_debt proxy
    _total_debt   = getattr(assumptions, "total_debt", assumptions.net_debt)
    _cash         = getattr(assumptions, "cash",       max(0.0, -assumptions.net_debt))
    _other_claims = getattr(assumptions, "other_claims", 0.0)

    mkt_cap       = result.equity_value
    shares        = assumptions.shares_outstanding
    implied_price = result.intrinsic_price

    # ── Undervalued / Overvalued label ────────────────────────────────────────
    if not np.isnan(implied_price) and not np.isnan(current_price) and current_price > 0:
        upside_pct = (implied_price - current_price) / current_price * 100
        uv_label   = f"{'Undervalued' if upside_pct >= 0 else 'Overvalued'} {abs(upside_pct):.1f}%"
        uv_color   = UP if upside_pct >= 0 else DOWN
    else:
        uv_label   = "—"
        uv_color   = "rgba(255,255,255,.4)"

    # ── Layout: two columns ───────────────────────────────────────────────────
    wc1, wc2 = st.columns(2, gap="large")

    def _row(lbl, val_str, note="", is_result=False, color=None):
        c  = color or ("#fff" if is_result else "rgba(255,255,255,.8)")
        w  = "700" if is_result else "400"
        vc = color or (BLUE if is_result else "#fff")
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<span style="font-size:12.5px;color:{c};font-weight:{w};">{lbl}</span>'
            f'<span style="font-size:13px;font-weight:{w};color:{vc};">{val_str}'
            f'<span style="font-size:10px;font-weight:400;color:rgba(255,255,255,.3);margin-left:8px;">{note}</span>'
            f'</span></div>'
        )

    def _spacer():
        return "<div style='height:8px'></div>"

    # ── Left: WACC inputs ─────────────────────────────────────────────────────
    with wc1:
        st.markdown('<p class="cfg-label" style="margin-bottom:10px;">WACC Inputs</p>', unsafe_allow_html=True)
        rows_left = "".join([
            _row("Tax Rate",                    f"{tax_rate:.1%}",      "Effective / statutory"),
            _row("Risk-Free Rate",               f"{rf_rate:.2%}",       "10yr UST yield"),
            _row("Beta (β)",                     f"{beta_val:.2f}x",     f"{beta_result.method.capitalize()} method"),
            _row("Market Risk Premium",          f"{erp:.2%}",           "Damodaran implied ERP"),
            _spacer(),
            _row("% Equity",                     f"{equity_w:.0%}",      "Market cap / total capital"),
            _row("% Debt",                       f"{debt_w:.0%}",        "Total debt / total capital"),
            _row("Cost of Debt",                 f"{cod:.2%}",           "Pre-tax cost of debt"),
            _spacer(),
            _row("= CAPM (Ke)",                  f"{ke:.2%}",            "Rf + β × ERP",              is_result=True),
            _row("= WACC",                       f"{wacc:.2%}",          "Ke×We + Kd×Wd",             is_result=True),
            _spacer(),
            _row("Terminal Risk-Free Rate",      f"{terminal_rf:.2%}",   "Same as entry Rf"),
            _row("Terminal CAPM",                f"{terminal_capm:.2%}", "Rf + β × ERP (terminal)"),
            _row("= Terminal WACC",              f"{terminal_wacc:.2%}", "Terminal Ke×We + Kd×Wd",    is_result=True),
            _spacer(),
            _row("Terminal Growth Rate",         f"{terminal_g:.2%}",    "Perpetuity growth rate"),
        ])
        st.markdown(rows_left, unsafe_allow_html=True)

    # ── Right: Equity Value Bridge ────────────────────────────────────────────
    with wc2:
        st.markdown('<p class="cfg-label" style="margin-bottom:10px;">Equity Value Bridge</p>', unsafe_allow_html=True)

        def _fmt_bridge(v):
            if isinstance(v, float) and np.isnan(v):
                return "—"
            if abs(v) < 1000:
                return f"${v:,.2f}"
            return _fmt_m(v)

        def _fmt_shares(v):
            if isinstance(v, float) and np.isnan(v):
                return "—"
            return f"{v/1e6:,.1f}M"

        rows_right = "".join([
            _row("Terminal Value",               _fmt_bridge(result.terminal_value),     "Undiscounted"),
            _row("PV of Terminal Value",         _fmt_bridge(result.pv_terminal_value),  f"{result.pv_tv_pct:.0f}% of EV"),
            _row("Sum of PV Free Cash Flows",    _fmt_bridge(result.sum_pv_ufcf),        f"{result.pv_ufcf_pct:.0f}% of EV"),
            _row("= Enterprise Value",           _fmt_bridge(result.enterprise_value),   "DCF-implied",                      is_result=True),
            _spacer(),
            _row("– Total Debt",                 _fmt_bridge(_total_debt),               "Balance sheet"),
            _row("– Other Claims of Value",      _fmt_bridge(_other_claims),             "Minorities / preferred"),
            _row("+ Cash & Cash Equivalents",    _fmt_bridge(_cash),                     "Most recent"),
            _row("= Market Capitalization",      _fmt_bridge(mkt_cap),                   "Implied equity value",             is_result=True),
            _spacer(),
            _row("÷ Fully Diluted Shares",       _fmt_shares(shares),                    "Diluted share count"),
            _row("= Implied Price",              _fmt_bridge(implied_price),             "",                                 is_result=True),
            _spacer(),
            _row("Current Price",                _fmt_bridge(current_price),             "Market price"),
            _row("Undervalued / Overvalued",     uv_label,                               "",                                 is_result=True, color=uv_color),
        ])
        st.markdown(rows_right, unsafe_allow_html=True)


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
# Analyst section renderers
# ─────────────────────────────────────────────────────────────────────────────

def _chart_analyst_recommendations(rec_trend: pd.DataFrame) -> Optional[go.Figure]:
    """
    Stacked bar chart of analyst recommendation counts by period.
    Expects columns like: strongBuy / strongbuy, buy, hold, sell / strongSell.
    Returns None if data is unavailable.
    """
    if rec_trend is None or rec_trend.empty:
        return None

    df = rec_trend.copy()
    df.columns = [c.lower() for c in df.columns]

    # Normalise column names to consistent keys
    col_map = {
        "strongbuy":  "Strong Buy",  "strong_buy":  "Strong Buy",
        "buy":        "Buy",
        "hold":       "Hold",
        "sell":       "Sell",
        "strongsell": "Strong Sell", "strong_sell": "Strong Sell",
        "underperform": "Sell",
    }
    rename = {c: col_map[c] for c in df.columns if c in col_map}
    df = df.rename(columns=rename)
    df = df[[c for c in ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"] if c in df.columns]]

    if df.empty:
        return None

    # Period label: use index if it's a date/string, else enumerate
    if hasattr(df.index, "strftime"):
        labels = df.index.strftime("%b '%y").tolist()
    else:
        labels = [str(i) for i in df.index]

    colors = {
        "Strong Buy":  "#00C805",
        "Buy":         "#34C759",
        "Hold":        "#FF9F0A",
        "Sell":        "#FF6B6B",
        "Strong Sell": "#FF3B30",
    }

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(
            name=col, x=labels, y=df[col].tolist(),
            marker_color=colors.get(col, "#888"),
            marker_line_width=0,
        ))

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,-apple-system,sans-serif", size=11, color="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=8, b=0), height=260,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="# Analysts", gridcolor="rgba(255,255,255,0.05)", tickformat=".0f"),
        bargap=0.25, hovermode="x unified",
    )
    return fig


def _chart_analyst_price_targets(pt: dict, current_price: float) -> Optional[go.Figure]:
    """
    Horizontal range chart showing low / current / mean / high price targets
    vs the current stock price — styled like a Yahoo Finance price target bar.
    Returns None if targets are unavailable.
    """
    low  = _safe(pt.get("low",  np.nan))
    mean = _safe(pt.get("mean", np.nan))
    high = _safe(pt.get("high", np.nan))

    if all(np.isnan(v) for v in [low, mean, high]):
        return None

    vals   = [v for v in [low, mean, high, current_price] if not np.isnan(v)]
    x_min  = min(vals) * 0.88
    x_max  = max(vals) * 1.12

    fig = go.Figure()

    # Range bar (low → high)
    if not np.isnan(low) and not np.isnan(high):
        fig.add_trace(go.Scatter(
            x=[low, high], y=[0.5, 0.5],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=14),
            showlegend=False, hoverinfo="skip",
        ))

    # Mean target marker
    if not np.isnan(mean):
        fig.add_trace(go.Scatter(
            x=[mean], y=[0.5],
            mode="markers+text",
            marker=dict(size=16, color=UP, symbol="diamond"),
            text=[f"<b>{_fmt_price(mean)}</b>"],
            textposition="top center",
            textfont=dict(size=11, color=UP),
            name="Mean Target",
        ))

    # Low target
    if not np.isnan(low):
        fig.add_trace(go.Scatter(
            x=[low], y=[0.5],
            mode="markers+text",
            marker=dict(size=10, color="rgba(255,255,255,0.55)", symbol="circle"),
            text=[_fmt_price(low)],
            textposition="bottom center",
            textfont=dict(size=10, color="rgba(255,255,255,0.55)"),
            name="Low Target",
        ))

    # High target
    if not np.isnan(high):
        fig.add_trace(go.Scatter(
            x=[high], y=[0.5],
            mode="markers+text",
            marker=dict(size=10, color="rgba(255,255,255,0.55)", symbol="circle"),
            text=[_fmt_price(high)],
            textposition="bottom center",
            textfont=dict(size=10, color="rgba(255,255,255,0.55)"),
            name="High Target",
        ))

    # Current price vertical line
    if not np.isnan(current_price):
        fig.add_vline(
            x=current_price,
            line_width=2, line_dash="dash",
            line_color="rgba(255,255,255,0.40)",
            annotation_text=f"Current {_fmt_price(current_price)}",
            annotation_position="top",
            annotation_font=dict(size=10, color="rgba(255,255,255,0.55)"),
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,-apple-system,sans-serif", size=11, color="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=36, b=20), height=180,
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[x_min, x_max], showgrid=False,
                   tickformat="$,.0f", tickfont=dict(size=10)),
        yaxis=dict(visible=False, range=[0, 1]),
        hovermode="x unified",
    )
    return fig


def _render_latest_ratings(upgrades_downgrades: pd.DataFrame) -> None:
    """
    Render a styled table of the most recent analyst rating actions —
    Date, Firm, Action (Upgrade/Downgrade/Init/Reiterate), Rating, Price Target.
    """
    if upgrades_downgrades is None or upgrades_downgrades.empty:
        st.markdown(
            '<p style="font-size:12px;color:rgba(255,255,255,.35);margin-top:8px;">No recent rating actions available.</p>',
            unsafe_allow_html=True,
        )
        return

    df = upgrades_downgrades.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Normalise index to date column
    if df.index.name and "date" in df.index.name.lower():
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Map column names (yfinance varies across versions)
    col_aliases = {
        "date":        ["date", "gradedate", "grade_date"],
        "firm":        ["firm", "tograde", "company"],
        "action":      ["action", "fromgrade", "from_grade"],
        "rating":      ["tograde", "to_grade", "grade", "rating"],
        "price_target":["price_target", "pricetarget", "target_price"],
    }

    def _find_col(aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    date_col   = _find_col(col_aliases["date"])
    firm_col   = _find_col(["firm", "company"])
    action_col = _find_col(["action"])
    from_col   = _find_col(["fromgrade", "from_grade"])
    to_col     = _find_col(["tograde", "to_grade", "grade"])
    pt_col     = _find_col(col_aliases["price_target"])

    # Parse and sort by date descending
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col, ascending=False)

    # Take most recent 15 rows
    df = df.head(15)

    # Action colour helper
    _action_color = {
        "upgrade":    UP,
        "downgrade":  DOWN,
        "init":       BLUE,
        "initiated":  BLUE,
        "reiterated": ORANGE,
        "maintained": "rgba(255,255,255,0.55)",
    }

    def _action_badge(val):
        if not val or str(val).lower() in ("nan", "none", ""):
            return "—"
        v = str(val).strip()
        c = _action_color.get(v.lower(), "rgba(255,255,255,0.55)")
        return (
            f'<span style="display:inline-block;font-size:10px;font-weight:700;'
            f'border-radius:4px;padding:2px 7px;'
            f'background:{c}22;color:{c};border:1px solid {c}55;">{v}</span>'
        )

    def _rating_color(val):
        if not val or str(val).lower() in ("nan", "none", ""):
            return "rgba(255,255,255,0.55)"
        v = str(val).lower()
        if "strong buy" in v or "outperform" in v or "buy" in v or "overweight" in v:
            return UP
        if "sell" in v or "underperform" in v or "underweight" in v:
            return DOWN
        if "hold" in v or "neutral" in v or "market" in v:
            return ORANGE
        return "rgba(255,255,255,0.75)"

    # Build HTML table
    html = (
        '<div class="fin-wrap" style="margin-top:8px;">'
        '<table class="fin-table" style="font-size:12px;">'
        '<thead><tr>'
        '<th class="hl" style="text-align:left;">Date</th>'
        '<th class="hl" style="text-align:left;">Firm</th>'
        '<th style="text-align:left;">Action</th>'
        '<th style="text-align:left;">From</th>'
        '<th style="text-align:left;">To</th>'
    )
    if pt_col:
        html += '<th>Price Target</th>'
    html += '</tr></thead><tbody>'

    for _, row in df.iterrows():
        date_str = row[date_col].strftime("%b %d, %Y") if date_col and pd.notna(row.get(date_col)) else "—"
        firm_str = str(row[firm_col]).strip() if firm_col and pd.notna(row.get(firm_col)) else "—"
        action_str = _action_badge(row.get(action_col, "")) if action_col else "—"
        from_str = str(row[from_col]).strip() if from_col and pd.notna(row.get(from_col)) and str(row.get(from_col)).lower() not in ("nan","none","") else "—"
        to_val   = row.get(to_col, "") if to_col else ""
        to_str   = str(to_val).strip() if pd.notna(to_val) and str(to_val).lower() not in ("nan","none","") else "—"
        to_color = _rating_color(to_val)

        pt_cell = ""
        if pt_col:
            pt_val = _safe(row.get(pt_col, np.nan))
            pt_cell = f'<td style="text-align:right;">{_fmt_price(pt_val)}</td>'

        html += (
            f'<tr>'
            f'<td class="hl" style="color:rgba(255,255,255,.45);font-size:11px;">{date_str}</td>'
            f'<td class="hl" style="color:rgba(255,255,255,.85);">{firm_str}</td>'
            f'<td style="text-align:left;">{action_str}</td>'
            f'<td style="text-align:left;color:rgba(255,255,255,.45);font-size:11px;">{from_str}</td>'
            f'<td style="text-align:left;font-weight:600;color:{to_color};">{to_str}</td>'
            f'{pt_cell}'
            f'</tr>'
        )

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)




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
    yf_info      = yf.Ticker(ticker).info or {} if True else {}
    current_price = _safe(estimates["price_targets"].get("current", np.nan))

    meta = data.get("metadata", {})
    name = meta.get("name") or ticker

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:30px;font-weight:800;color:#fff;margin-bottom:2px;">{ticker}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:13px;color:rgba(255,255,255,.4);margin-top:0;margin-bottom:20px;">'
        f'</p>',
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
    assumptions = DCFAssumptions(
        n_years=10, stage1_years=5,
        revenue_growth_y1=rev_g_y1, revenue_growth_y5=rev_g_y5,
        ebitda_margin=ebitda_m, capex_pct_revenue=capex_pct,
        da_pct_revenue=da_pct, nwc_pct_revenue=nwc_pct,
        terminal_growth_rate=terminal_g, beta=beta_val,
        risk_free_rate=rf_rate, equity_risk_premium=erp,
        cost_of_debt_pretax=cod, tax_rate=tax_rate,
        debt_weight=debt_w, equity_weight=1.0 - debt_w,
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

    # ── Build Street Consensus assumptions ────────────────────────────────────
    # Street model uses ONLY consensus growth/margin + the DEFAULT (SEC-derived)
    # WACC inputs. It is intentionally isolated from the user's slider tweaks so
    # that adjusting "Your Model" assumptions never changes the Street numbers.
    street_g1 = _clamp(cons_g1, -0.20, 0.80, default_asm.revenue_growth_y1)
    street_g5 = _clamp(cons_g5, -0.10, 0.50, max(default_asm.revenue_growth_y1 * 0.5, default_asm.terminal_growth_rate + 0.01))
    street_em = _clamp(cons_em, 0.01, 0.85, default_asm.ebitda_margin)

    street_asm = DCFAssumptions(
        n_years=10, stage1_years=5,
        revenue_growth_y1=street_g1,
        revenue_growth_y5=street_g5,
        ebitda_margin=street_em,
        capex_pct_revenue=default_asm.capex_pct_revenue,
        da_pct_revenue=default_asm.da_pct_revenue,
        nwc_pct_revenue=default_asm.nwc_pct_revenue,
        terminal_growth_rate=default_asm.terminal_growth_rate,
        beta=default_asm.beta,
        risk_free_rate=default_asm.risk_free_rate,
        equity_risk_premium=default_asm.equity_risk_premium,
        cost_of_debt_pretax=default_asm.cost_of_debt_pretax,
        tax_rate=default_asm.tax_rate,
        debt_weight=default_asm.debt_weight,
        equity_weight=default_asm.equity_weight,
        base_ufcf=default_asm.base_ufcf,
        base_revenue=default_asm.base_revenue,
        net_debt=default_asm.net_debt,
        shares_outstanding=default_asm.shares_outstanding,
    )
    street_result = run_dcf(street_asm)
    street_intrinsic = street_result.intrinsic_price

    # ── Key results row ───────────────────────────────────────────────────────
    pt   = estimates.get("price_targets", {})
    mean_target = pt.get("mean", np.nan)
    rat  = estimates.get("rating")
    n_an = estimates.get("n_analysts", 0)

    # (top summary cards removed — see Street Equity Value Bridge in each tab)

    # ─────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────
    tab_your, tab_street, tab_charts = st.tabs([
        "Your Model",
        "Wall St. Consensus",
        "Charts",
    ])

    # ── Tab: Your Model ───────────────────────────────────────────────────────
    with tab_your:
        st.markdown(_render_projection_table(result, assumptions), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="cfg-label" style="margin-bottom:12px;">Street Equity Value Bridge</p>',
            unsafe_allow_html=True,
        )
        _render_equity_value_bridge(
            result, assumptions,
            rf_rate, beta_val, erp,
            ke, kd, wacc,
            cod, tax_rate, debt_w, current_price, beta_result,
            terminal_g,
        )

    # ── Tab: Wall St. Consensus ───────────────────────────────────────────────
    with tab_street:
        rec_trend        = estimates.get("rec_trend",        pd.DataFrame())
        upgrades_df      = estimates.get("upgrades_downgrades", pd.DataFrame())

        # ── Row 1: Analyst Recommendations bar + Price Targets range ──────────
        ac1, ac2 = st.columns(2, gap="large")

        with ac1:
            st.markdown(
                '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);margin-bottom:4px;">Analyst Recommendations</p>',
                unsafe_allow_html=True,
            )
            fig_rec = _chart_analyst_recommendations(rec_trend)
            if fig_rec:
                st.plotly_chart(fig_rec, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown('<p style="font-size:12px;color:rgba(255,255,255,.3);margin-top:12px;">No recommendation trend data available.</p>', unsafe_allow_html=True)

        with ac2:
            st.markdown(
                '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);margin-bottom:4px;">Analyst Price Targets</p>',
                unsafe_allow_html=True,
            )
            fig_pt = _chart_analyst_price_targets(pt, current_price)
            if fig_pt:
                st.plotly_chart(fig_pt, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown('<p style="font-size:12px;color:rgba(255,255,255,.3);margin-top:12px;">No price target data available.</p>', unsafe_allow_html=True)

        # ── Latest Ratings table ──────────────────────────────────────────────
        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
            'color:rgba(255,255,255,.35);margin-bottom:4px;">Latest Ratings</p>',
            unsafe_allow_html=True,
        )
        _render_latest_ratings(upgrades_df)

        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)
        st.markdown(_render_projection_table(street_result, street_asm), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<hr class="val-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="cfg-label" style="margin-bottom:12px;">Street Equity Value Bridge</p>',
            unsafe_allow_html=True,
        )
        _render_equity_value_bridge(
            street_result, street_asm,
            rf_rate, beta_val, erp,
            street_result.cost_of_equity, street_result.cost_of_debt_aftertax, street_result.wacc,
            cod, tax_rate, debt_w, current_price, beta_result,
            terminal_g,
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
