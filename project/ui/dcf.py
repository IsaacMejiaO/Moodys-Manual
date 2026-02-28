# ui/valuation.py
# ──────────────────────────────────────────────────────────────────────────────
# Valuation Page  (replaces dcf.py)
# ──────────────────────────────────────────────────────────────────────────────
#
# Layout (top → bottom)
# ---------------------
#  Header:   Company name · Valuation Model  (matches financials.py style)
#  Section A: Analyst Consensus Panel
#             ├─ Price target table  (Low / Mean / High / Upside)
#             ├─ Revenue & EPS estimate table  (next 2 fiscal years)
#             └─ Growth & margin consensus cards used to pre-fill sliders
#  Section B: Your Assumptions  (3-column sliders, same layout as before)
#             └─ "Import Consensus" button copies Street estimates → sliders
#  Divider
#  Section C: Key results cards  (Intrinsic Price / Upside / EV / WACC / TV / Beta)
#  Section D: Charts (PV waterfall | Revenue·EBITDA·FCF)
#  Section E: Tabs
#             ├─ WACC × Terminal Growth  (sensitivity)
#             ├─ Revenue Growth × EBITDA Margin  (sensitivity)
#             ├─ 10-Year Model  (projection table)
#             └─ WACC Bridge
#  Footer: methodology + data sources note
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

# ── Design tokens (shared with financials.py / ratios.py) ─────────────────────
UP     = "#00C805"
DOWN   = "#FF3B30"
BLUE   = "#0A7CFF"
ORANGE = "#FF9F0A"
GREY   = "#6E6E73"
PURPLE = "#BF5AF2"


# ─────────────────────────────────────────────────────────────────────────────
# CSS  — base matches financials.py exactly; valuation-specific additions below
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;}}

/* ── Tabs — identical to financials.py ── */
.stTabs [data-baseweb="tab-list"]{{gap:0;border-bottom:1px solid rgba(255,255,255,0.1);background:transparent!important;}}
.stTabs [data-baseweb="tab"]{{font-weight:600;font-size:13px;padding:10px 22px;color:rgba(255,255,255,0.45);background:transparent!important;border-bottom:2px solid transparent;border-radius:0;}}
.stTabs [aria-selected="true"]{{color:#fff!important;border-bottom:2px solid rgba(255,255,255,0.6)!important;background:transparent!important;}}

/* ── financials.py fin-table styles (reused for analyst tables) ── */
.fin-wrap{{overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.015);margin-top:8px;}}
.fin-table{{width:100%;border-collapse:collapse;font-size:12.5px;}}
.fin-table thead th{{background:rgba(255,255,255,0.04);color:#ffffff;font-weight:700;font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;padding:10px 16px 9px;border-bottom:1px solid rgba(255,255,255,0.09);text-align:right;white-space:nowrap;}}
.fin-table thead th.hl{{text-align:left;min-width:200px;}}
.fin-table thead th.hy{{min-width:88px;}}
.fin-table td{{padding:6px 16px;border-bottom:1px solid rgba(255,255,255,.035);color:#ffffff;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}}
.fin-table td.tl{{text-align:left;}}
.fin-table tbody tr:hover td{{background:rgba(255,255,255,.022);}}
.rt td{{font-weight:700!important;color:#fff!important;border-top:1px solid rgba(255,255,255,.14)!important;border-bottom:1px solid rgba(255,255,255,.14)!important;background:rgba(255,255,255,.018)!important;}}
.rs td{{font-size:9.5px!important;font-weight:800!important;letter-spacing:.13em!important;text-transform:uppercase!important;color:#ffffff!important;background:rgba(255,255,255,.04)!important;padding-top:11px!important;padding-bottom:5px!important;border-top:1px solid rgba(255,255,255,.07)!important;border-bottom:none!important;}}
.ri1 td.tl{{padding-left:16px!important;}}
.ri2 td.tl{{padding-left:32px!important;color:#ffffff!important;font-size:12px!important;}}
.vp{{color:{UP}!important;}}.vn{{color:{DOWN}!important;}}.vna{{color:rgba(255,255,255,.35)!important;}}.vpc{{font-size:11px!important;color:#ffffff!important;}}
.calc-badge{{display:inline-block;font-size:8.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:rgba(255,255,255,.35);background:rgba(255,255,255,.07);border-radius:3px;padding:1px 4px;margin-left:6px;vertical-align:middle;}}
.fin-note{{margin-top:22px;padding:12px 16px;background:rgba(255,255,255,.015);border-radius:8px;font-size:10.5px;color:rgba(255,255,255,.5);line-height:1.85;}}

/* ── Metric cards ── */
.val-cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 20px 0;}}
.val-card{{flex:1;min-width:140px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 18px;}}
.val-card .label{{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:6px;}}
.val-card .value{{font-size:22px;font-weight:800;color:#fff;line-height:1.1;}}
.val-card .sub{{font-size:11px;color:rgba(255,255,255,.4);margin-top:4px;}}
.val-card.highlight{{border-color:rgba(10,124,255,0.4);background:rgba(10,124,255,0.06);}}
.val-card.up{{border-color:rgba(0,200,5,0.35);background:rgba(0,200,5,0.05);}}
.val-card.down{{border-color:rgba(255,59,48,0.35);background:rgba(255,59,48,0.05);}}
.val-card.orange{{border-color:rgba(255,159,10,0.35);background:rgba(255,159,10,0.05);}}
.val-card.purple{{border-color:rgba(191,90,242,0.35);background:rgba(191,90,242,0.05);}}

/* ── Analyst consensus section ── */
.consensus-header{{font-size:10.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:rgba(255,255,255,.35);margin:20px 0 6px 0;}}
.consensus-banner{{background:rgba(10,124,255,0.06);border:1px solid rgba(10,124,255,0.18);border-radius:12px;padding:16px 20px;margin:12px 0 20px 0;}}
.consensus-banner .cb-title{{font-size:11.5px;font-weight:700;color:rgba(255,255,255,.6);margin-bottom:12px;letter-spacing:.03em;}}
.analyst-rating{{display:inline-block;font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px;}}
.rating-buy{{background:rgba(0,200,5,.15);color:{UP};border:1px solid rgba(0,200,5,.25);}}
.rating-hold{{background:rgba(255,159,10,.12);color:{ORANGE};border:1px solid rgba(255,159,10,.22);}}
.rating-sell{{background:rgba(255,59,48,.12);color:{DOWN};border:1px solid rgba(255,59,48,.22);}}

/* ── Assumption section ── */
.asm-header{{font-size:10.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:rgba(255,255,255,.35);margin:20px 0 6px 0;}}

/* ── Source badges ── */
.src-badge{{display:inline-block;font-size:8.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;border-radius:4px;padding:2px 6px;margin-left:6px;vertical-align:middle;}}
.src-badge.sec{{color:rgba(10,124,255,.9);background:rgba(10,124,255,.12);border:1px solid rgba(10,124,255,.2);}}
.src-badge.yf{{color:rgba(255,159,10,.9);background:rgba(255,159,10,.12);border:1px solid rgba(255,159,10,.2);}}
.src-badge.reg{{color:rgba(191,90,242,.9);background:rgba(191,90,242,.12);border:1px solid rgba(191,90,242,.2);}}
.src-badge.calc{{color:rgba(255,255,255,.5);background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);}}
.src-badge.street{{color:rgba(0,200,5,.9);background:rgba(0,200,5,.1);border:1px solid rgba(0,200,5,.2);}}

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
.val-warn{{background:rgba(255,159,10,.07);border:1px solid rgba(255,159,10,.22);border-radius:10px;padding:12px 16px;margin:10px 0;font-size:12px;color:rgba(255,255,255,.75);line-height:1.7;}}

/* ── Divider ── */
.val-divider{{border:none;border-top:1px solid rgba(255,255,255,0.07);margin:24px 0;}}
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
def _get_yf_info(ticker: str) -> dict:
    """Fetch full yfinance info dict (price, targets, ratings, estimates)."""
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _get_analyst_estimates(ticker: str) -> dict:
    """
    Pull Wall Street consensus data from yfinance.
    Returns a dict with keys:
        price_targets   – { low, mean, high, current, n_analysts }
        rating          – "Buy" / "Hold" / "Sell" / None
        revenue_est     – DataFrame  (index=period, cols: low/mean/high/growth)
        eps_est         – DataFrame  (index=period, cols: low/mean/high/growth)
        rev_growth_fwd  – float (next year consensus revenue growth)
        eps_growth_fwd  – float (next year consensus EPS growth)
        ebitda_margin   – float (LTM EBITDA / revenue from yfinance)
    """
    result = {
        "price_targets": {},
        "rating": None,
        "revenue_est": pd.DataFrame(),
        "eps_est": pd.DataFrame(),
        "rev_growth_fwd": np.nan,
        "eps_growth_fwd": np.nan,
        "ebitda_margin": np.nan,
        "n_analysts": 0,
    }

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # ── Price targets ──────────────────────────────────────────────────────
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_low    = info.get("targetLowPrice")
        target_mean   = info.get("targetMeanPrice")
        target_high   = info.get("targetHighPrice")
        n_analysts    = info.get("numberOfAnalystOpinions", 0)
        rec           = info.get("recommendationKey", "")

        result["price_targets"] = {
            "current": float(current_price) if current_price else np.nan,
            "low":     float(target_low)    if target_low    else np.nan,
            "mean":    float(target_mean)   if target_mean   else np.nan,
            "high":    float(target_high)   if target_high   else np.nan,
        }
        result["n_analysts"] = int(n_analysts) if n_analysts else 0

        # Normalize recommendation key → human label
        _rec_map = {
            "strong_buy": "Strong Buy", "buy": "Buy",
            "hold": "Hold", "underperform": "Sell", "sell": "Strong Sell",
        }
        result["rating"] = _rec_map.get(rec.lower().replace(" ", "_"), rec.capitalize() if rec else None)

        # ── Revenue estimates ──────────────────────────────────────────────────
        try:
            rev_est = t.revenue_estimate
            if rev_est is not None and not rev_est.empty:
                rows = []
                for period in rev_est.index:
                    row = rev_est.loc[period]
                    rows.append({
                        "Period": str(period),
                        "Low":    _safe_float(row.get("low")),
                        "Mean":   _safe_float(row.get("avg")),
                        "High":   _safe_float(row.get("high")),
                        "Growth": _safe_float(row.get("growth")),
                        "N":      int(row.get("numberOfAnalysts", 0)),
                    })
                result["revenue_est"] = pd.DataFrame(rows).set_index("Period")
                # Forward growth = next year (0y label) mean growth
                if len(rows) >= 2:
                    result["rev_growth_fwd"] = rows[1].get("Growth", np.nan)
                elif rows:
                    result["rev_growth_fwd"] = rows[0].get("Growth", np.nan)
        except Exception:
            pass

        # ── EPS estimates ──────────────────────────────────────────────────────
        try:
            eps_est = t.earnings_estimate
            if eps_est is not None and not eps_est.empty:
                rows_e = []
                for period in eps_est.index:
                    row = eps_est.loc[period]
                    rows_e.append({
                        "Period": str(period),
                        "Low":    _safe_float(row.get("low")),
                        "Mean":   _safe_float(row.get("avg")),
                        "High":   _safe_float(row.get("high")),
                        "Growth": _safe_float(row.get("growth")),
                        "N":      int(row.get("numberOfAnalysts", 0)),
                    })
                result["eps_est"] = pd.DataFrame(rows_e).set_index("Period")
                if len(rows_e) >= 2:
                    result["eps_growth_fwd"] = rows_e[1].get("Growth", np.nan)
                elif rows_e:
                    result["eps_growth_fwd"] = rows_e[0].get("Growth", np.nan)
        except Exception:
            pass

        # ── EBITDA margin proxy from yfinance info ─────────────────────────────
        ebitda   = info.get("ebitda")
        rev      = info.get("totalRevenue")
        if ebitda and rev and rev > 0:
            result["ebitda_margin"] = float(ebitda) / float(rev)

    except Exception as exc:
        _logger.warning("_get_analyst_estimates failed for %s: %s", ticker, exc)

    return result


def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _fmt_pct(v: float, decimals: int = 1, show_plus: bool = False) -> str:
    if np.isnan(v):
        return "—"
    sign = "+" if show_plus and v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


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
    return (
        f'<div class="val-card {css_class}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        + (f'<div class="sub">{sub}</div>' if sub else "")
        + '</div>'
    )


def _badge(text: str, kind: str = "calc") -> str:
    return f'<span class="src-badge {kind}">{text}</span>'


def _beta_badge(br: BetaResult) -> str:
    method_map = {
        "fundamental": ("sec",  "SEC"),
        "regression":  ("reg",  "Regression"),
        "yfinance":    ("yf",   "yfinance"),
        "fallback":    ("calc", "Default"),
    }
    kind, label = method_map.get(br.method, ("calc", br.method))
    return _badge(label, kind)


def _rating_badge(rating: Optional[str]) -> str:
    if not rating:
        return ""
    r = rating.lower()
    if "buy" in r:
        cls = "rating-buy"
    elif "sell" in r or "underperform" in r:
        cls = "rating-sell"
    else:
        cls = "rating-hold"
    return f'<span class="analyst-rating {cls}">{rating}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Analyst Consensus Panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_analyst_panel(ticker: str, estimates: dict, current_price: float) -> None:
    """
    Render the full Wall Street consensus section.
    Includes: rating + price target cards, revenue & EPS estimate tables,
    and an explanation of how each estimate was used to pre-fill the model.
    """
    st.markdown('<div class="consensus-header">Wall Street Consensus</div>', unsafe_allow_html=True)

    pt    = estimates.get("price_targets", {})
    mean  = pt.get("mean", np.nan)
    low   = pt.get("low",  np.nan)
    high  = pt.get("high", np.nan)
    n     = estimates.get("n_analysts", 0)
    rat   = estimates.get("rating")

    # ── Price target + rating row ─────────────────────────────────────────────
    pt_upside = (mean - current_price) / current_price * 100 if (
        not np.isnan(mean) and not np.isnan(current_price) and current_price > 0
    ) else np.nan

    pt_cls = ""
    if not np.isnan(pt_upside):
        pt_cls = "up" if pt_upside >= 0 else "down"

    col_pt1, col_pt2, col_pt3, col_pt4 = st.columns(4)
    with col_pt1:
        rat_html = _rating_badge(rat) if rat else "<span style='color:rgba(255,255,255,.3)'>—</span>"
        st.markdown(
            f'<div class="val-card"><div class="label">Analyst Rating</div>'
            f'<div class="value" style="font-size:17px;padding-top:4px;">{rat_html}</div>'
            f'<div class="sub">{n} analysts</div></div>',
            unsafe_allow_html=True,
        )
    with col_pt2:
        st.markdown(
            _metric_card(
                "Price Target (Mean)",
                _fmt_price(mean),
                f'{_fmt_pct(pt_upside, show_plus=True)} vs current' if not np.isnan(pt_upside) else f"Current: {_fmt_price(current_price)}",
                pt_cls,
            ),
            unsafe_allow_html=True,
        )
    with col_pt3:
        lo_up = (low - current_price) / current_price * 100 if (
            not np.isnan(low) and not np.isnan(current_price) and current_price > 0
        ) else np.nan
        st.markdown(
            _metric_card("Price Target (Low)", _fmt_price(low),
                         _fmt_pct(lo_up, show_plus=True) if not np.isnan(lo_up) else ""),
            unsafe_allow_html=True,
        )
    with col_pt4:
        hi_up = (high - current_price) / current_price * 100 if (
            not np.isnan(high) and not np.isnan(current_price) and current_price > 0
        ) else np.nan
        st.markdown(
            _metric_card("Price Target (High)", _fmt_price(high),
                         _fmt_pct(hi_up, show_plus=True) if not np.isnan(hi_up) else "", "up"),
            unsafe_allow_html=True,
        )

    # ── Estimate tables ───────────────────────────────────────────────────────
    rev_df = estimates.get("revenue_est", pd.DataFrame())
    eps_df = estimates.get("eps_est",     pd.DataFrame())

    if not rev_df.empty or not eps_df.empty:
        est_col1, est_col2 = st.columns(2, gap="large")

        with est_col1:
            st.markdown(
                '<p class="consensus-header" style="margin-top:16px;">Revenue Estimates</p>',
                unsafe_allow_html=True,
            )
            if rev_df.empty:
                st.markdown('<p style="color:rgba(255,255,255,.3);font-size:12px;">Not available</p>', unsafe_allow_html=True)
            else:
                _render_estimate_table(rev_df, is_dollar=True)

        with est_col2:
            st.markdown(
                '<p class="consensus-header" style="margin-top:16px;">EPS Estimates</p>',
                unsafe_allow_html=True,
            )
            if eps_df.empty:
                st.markdown('<p style="color:rgba(255,255,255,.3);font-size:12px;">Not available</p>', unsafe_allow_html=True)
            else:
                _render_estimate_table(eps_df, is_dollar=False)

    # ── Narrative summary ─────────────────────────────────────────────────────
    rev_g  = estimates.get("rev_growth_fwd", np.nan)
    eps_g  = estimates.get("eps_growth_fwd", np.nan)
    eb_m   = estimates.get("ebitda_margin", np.nan)

    notes = []
    if not np.isnan(rev_g):
        notes.append(f"<b>Revenue growth (next year):</b> {rev_g*100:+.1f}% — used as Year 1 growth default.")
    if not np.isnan(eps_g):
        notes.append(f"<b>EPS growth (next year):</b> {eps_g*100:+.1f}% — used as a cross-check.")
    if not np.isnan(eb_m):
        notes.append(f"<b>EBITDA margin (LTM):</b> {eb_m*100:.1f}% — pre-fills the EBITDA margin slider.")
    if n:
        notes.append(f"<b>{n} analysts</b> contributed to this consensus. Use the sliders below to build your own narrative.")

    if notes:
        st.markdown(
            '<div class="fin-note" style="margin-top:12px;border-left:3px solid rgba(10,124,255,0.4);padding-left:14px;">'
            + "<br>".join(notes)
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_estimate_table(df: pd.DataFrame, is_dollar: bool = True) -> None:
    """Render a revenue or EPS estimate DataFrame as a fin-table."""
    hdrs = ""
    for col in ["Low", "Mean", "High", "Growth"]:
        if col in df.columns:
            hdrs += f'<th class="hy">{col}</th>'
    if "N" in df.columns:
        hdrs += '<th class="hy"># Analysts</th>'

    html = (
        f'<div class="fin-wrap"><table class="fin-table">'
        f'<thead><tr><th class="hl">Period</th>{hdrs}</tr></thead><tbody>'
    )

    for period, row in df.iterrows():
        cells = f'<td class="tl">{period}</td>'
        for col in ["Low", "Mean", "High"]:
            if col not in df.columns:
                continue
            v = row.get(col, np.nan)
            if pd.isna(v):
                cells += '<td class="vna">—</td>'
            elif is_dollar:
                cells += f'<td>{_fmt_m(v)}</td>'
            else:
                cells += f'<td>${v:.2f}</td>'
        if "Growth" in df.columns:
            g = row.get("Growth", np.nan)
            if pd.isna(g):
                cells += '<td class="vna">—</td>'
            else:
                cls = "vp" if g >= 0 else "vn"
                cells += f'<td class="{cls}">{g*100:+.1f}%</td>'
        if "N" in df.columns:
            cells += f'<td class="vna">{int(row.get("N", 0))}</td>'
        html += f"<tr>{cells}</tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity table renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_sensitivity(
    df: pd.DataFrame,
    current_price: float,
    base_wacc_str: str = None,
    base_tg_str: str = None,
) -> str:
    if df.empty:
        return "<p style='color:rgba(255,255,255,.3)'>No sensitivity data.</p>"

    flat = [v for col in df.columns for v in df[col] if not pd.isna(v)]
    if not flat:
        return "<p>No data.</p>"
    p10, p90 = np.percentile(flat, 10), np.percentile(flat, 90)

    h = f'<th class="row-label">{df.index.name or ""}</th>'
    for col in df.columns:
        is_base = base_tg_str and col == base_tg_str
        cls = " class='base-col'" if is_base else ""
        h += f"<th{cls}>{col}</th>"

    html = f'<div class="sens-wrap"><table class="sens-table"><thead><tr>{h}</tr></thead><tbody>'

    for idx, row in df.iterrows():
        is_base_row = base_wacc_str and str(idx) == base_wacc_str
        row_open = '<tr style="background:rgba(10,124,255,0.04);">' if is_base_row else "<tr>"
        cells = f'<td class="row-label">{idx}</td>'
        for col in df.columns:
            v = row[col]
            is_base_col = base_tg_str and col == base_tg_str
            base_cls = "base-col " if is_base_col else ""
            if pd.isna(v):
                cells += f'<td class="{base_cls}cell-mid">—</td>'
            else:
                color_cls = "cell-hi" if v >= p90 else ("cell-lo" if v <= p10 else "cell-mid")
                if not np.isnan(current_price) and current_price > 0:
                    upside = (v - current_price) / current_price * 100
                    tip = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
                    label = f"${v:,.2f} <span style='font-size:9.5px;opacity:.5'>({tip})</span>"
                else:
                    label = f"${v:,.2f}"
                cells += f'<td class="{base_cls}{color_cls}">{label}</td>'
        html += f"{row_open}{cells}</tr>"

    return html + "</tbody></table></div>"


# ─────────────────────────────────────────────────────────────────────────────
# Projection table renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_projection_table(
    result,
    assumptions: DCFAssumptions,
    divisor: float = 1e9,
    unit: str = "$B",
) -> str:
    n = assumptions.n_years
    years = [f"Y{i}" for i in range(1, n + 1)]

    # Precompute derived rows
    prev_rev = assumptions.base_revenue
    rev_growths, ebitda_margins, ebits, taxes, nopats = [], [], [], [], []
    das, capexs, dnwcs = [], [], []

    for i, rev in enumerate(result.projected_revenues):
        rev_growths.append((rev / prev_rev - 1) * 100 if prev_rev > 0 else np.nan)
        prev_rev = rev
        ebitda_margins.append(result.projected_ebitda[i] / rev * 100 if rev > 0 else np.nan)
        da   = rev * assumptions.da_pct_revenue
        ebit = result.projected_ebitda[i] - da
        tax  = ebit * assumptions.tax_rate if ebit > 0 else 0.0
        capex = rev * assumptions.capex_pct_revenue
        prev_r = assumptions.base_revenue if i == 0 else result.projected_revenues[i - 1]
        dnwc   = (rev - prev_r) * assumptions.nwc_pct_revenue
        das.append(da); ebits.append(ebit); taxes.append(tax)
        nopats.append(ebit - tax); capexs.append(capex); dnwcs.append(dnwc)

    rows_data = [
        ("REVENUE DRIVERS",      None,                       "rs"),
        ("Revenue",              result.projected_revenues,  "rt"),
        ("Revenue Growth %",     rev_growths,                "pct"),
        ("EBITDA",               result.projected_ebitda,    ""),
        ("EBITDA Margin %",      ebitda_margins,             "pct"),
        ("FREE CASH FLOW BUILD", None,                       "rs"),
        ("EBITDA",               result.projected_ebitda,    ""),
        ("less: D&A (add back)", das,                        ""),
        ("= EBIT",               ebits,                      ""),
        ("less: Taxes on EBIT",  taxes,                      ""),
        ("= NOPAT",              nopats,                     ""),
        ("+ D&A",                das,                        ""),
        ("– CapEx",              capexs,                     ""),
        ("– ΔNWC",               dnwcs,                      ""),
        ("Unlevered FCF",        result.projected_ufcf,      "rt"),
        ("PRESENT VALUE",        None,                       "rs"),
        ("Discount Factor",      result.discount_factors,    "factor"),
        ("PV of UFCF",           result.pv_ufcf,             ""),
    ]

    hdrs = "".join(f"<th>{y}</th>" for y in years)
    html = (
        f'<div class="proj-wrap"><table class="proj-table">'
        f'<thead><tr><th class="hl">{unit}</th>{hdrs}</tr></thead><tbody>'
    )

    for label, data_list, row_cls in rows_data:
        if row_cls == "rs":
            html += f'<tr class="rs"><td class="hl" colspan="{n+1}">{label}</td></tr>'
            continue
        if data_list is None:
            continue

        cells = f'<td class="hl">{label}</td>'
        is_pct    = row_cls == "pct"
        is_factor = row_cls == "factor"

        for v in data_list:
            if isinstance(v, float) and np.isnan(v):
                cells += "<td>—</td>"
            elif is_pct:
                sign = "+" if v >= 0 else ""
                cells += f"<td>{sign}{v:.1f}%</td>"
            elif is_factor:
                cells += f"<td>{v:.4f}x</td>"
            else:
                scaled = v / divisor
                s = f"{abs(scaled):,.0f}" if abs(scaled) >= 100 else f"{abs(scaled):,.2f}"
                neg = ' style="color:#FF3B30"' if v < 0 else ""
                txt = f"({s})" if v < 0 else s
                cells += f"<td{neg}>{txt}</td>"

        tr_cls_map = {"rt": "rt", "pct": "", "": "", "factor": ""}
        tr_c = tr_cls_map.get(row_cls, "")
        html += f'<tr class="{tr_c}">{cells}</tr>'

    return html + "</tbody></table></div>"


# ─────────────────────────────────────────────────────────────────────────────
# Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

def _chart_waterfall(result, assumptions: DCFAssumptions) -> go.Figure:
    n = assumptions.n_years
    labels = [f"Y{i+1}" for i in range(n)] + ["Terminal\nValue"]
    values = list(result.pv_ufcf) + [result.pv_terminal_value]
    bar_colors = [f"rgba(10,124,255,{0.45 + 0.04*i})" for i in range(n)] + ["rgba(255,159,10,0.70)"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[v / 1e9 for v in values],
        marker_color=bar_colors,
        text=[f"${v/1e9:.2f}B" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="rgba(255,255,255,0.65)"),
    ))
    fig.add_annotation(
        x=labels[-1], y=result.pv_terminal_value / 1e9,
        text=f"{result.pv_tv_pct:.0f}% of EV",
        showarrow=False, yanchor="bottom", yshift=22,
        font=dict(size=9.5, color=ORANGE),
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=24, b=0), height=280,
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="$B", gridcolor="rgba(255,255,255,0.05)", tickformat="$.1f"),
        bargap=0.25,
    )
    return fig


def _chart_revenue_fcf(result, assumptions: DCFAssumptions) -> go.Figure:
    n = assumptions.n_years
    years = [f"Y{i+1}" for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Revenue", x=years,
        y=[v / 1e9 for v in result.projected_revenues],
        marker_color="rgba(10,124,255,0.30)",
        marker_line=dict(color="rgba(10,124,255,0.60)", width=1),
    ))
    fig.add_trace(go.Scatter(
        name="Unlevered FCF", x=years,
        y=[v / 1e9 for v in result.projected_ufcf],
        mode="lines+markers",
        line=dict(color=UP, width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        name="EBITDA", x=years,
        y=[v / 1e9 for v in result.projected_ebitda],
        mode="lines+markers",
        line=dict(color=ORANGE, width=1.5, dash="dot"), marker=dict(size=4),
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=24, b=0), height=280,
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="$B", gridcolor="rgba(255,255,255,0.05)", tickformat="$.1f"),
        bargap=0.30, hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_valuation(
    ticker: str,
    summary: Optional[dict] = None,
    data: Optional[dict] = None,
) -> None:
    """
    Render the Valuation page for a given ticker.

    Args:
        ticker  : Stock ticker symbol (e.g. "AAPL")
        summary : Output of build_company_summary() — fetched if None
        data    : Output of fetch_company_data_unified() — fetched if None
    """
    _inject_css()
    ticker = ticker.upper().strip()

    # ── Fetch financial data (reuse session cache if available) ───────────────
    cik = _cik_map().get(ticker, "")

    if data is None:
        cache = st.session_state.get("ticker_data_cache", {})
        data = cache.get(ticker)

    if summary is None:
        scache = st.session_state.get("ticker_summary_cache", {})
        summary = scache.get(ticker)

    if data is None:
        with st.spinner(f"Fetching {ticker} data from SEC…"):
            try:
                from app import fetch_company_data_unified
                data = fetch_company_data_unified(ticker, cik)
            except Exception:
                from sec_engine.sec_fetch import fetch_company_facts
                from sec_engine.ltm import build_ltm_financials
                from sec_engine.normalize import GAAP_MAP
                try:
                    facts_raw = fetch_company_facts(cik) if cik else {}
                    ltm = build_ltm_financials(facts_raw, GAAP_MAP)
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

    # ── SIC, current price, analyst consensus ─────────────────────────────────
    sic_code = ""
    if cik:
        with st.spinner("Loading SIC…"):
            sic_code = _load_sic(cik)

    with st.spinner("Fetching market data & analyst estimates…"):
        yf_info   = _get_yf_info(ticker)
        estimates = _get_analyst_estimates(ticker)

    current_price = _safe(yf_info.get("currentPrice") or yf_info.get("regularMarketPrice"))

    meta = data.get("metadata", {})
    name = meta.get("name") or yf_info.get("longName") or yf_info.get("shortName") or ticker

    # ── Page header (matches financials.py exactly) ───────────────────────────
    st.markdown(
        f'<h1 style="font-size:30px;font-weight:800;color:#fff;margin-bottom:2px;">{ticker}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:13px;color:rgba(255,255,255,.4);margin-top:0;margin-bottom:20px;">'
        f'{name} &nbsp;·&nbsp; Valuation Model</p>',
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A — Analyst Consensus Panel
    # ─────────────────────────────────────────────────────────────────────────
    _render_analyst_panel(ticker, estimates, current_price)

    st.markdown('<hr class="val-divider">', unsafe_allow_html=True)

    # ── Build default assumptions from SEC data ───────────────────────────────
    with st.spinner("Building assumptions from SEC data…"):
        facts_for_beta = _load_facts(cik) if cik else {}
        default_assumptions, beta_result = build_dcf_assumptions(
            ticker=ticker,
            summary=summary,
            data=data,
            sic_code=sic_code,
            facts=facts_for_beta,
        )

    # If analyst consensus has richer forward estimates, blend them in as defaults
    cons_rev_g   = estimates.get("rev_growth_fwd", np.nan)
    cons_ebitda_m = estimates.get("ebitda_margin", np.nan)

    default_rev_g_y1 = (
        cons_rev_g
        if not np.isnan(cons_rev_g)
        else default_assumptions.revenue_growth_y1
    )
    default_ebitda_m = (
        cons_ebitda_m
        if not np.isnan(cons_ebitda_m)
        else default_assumptions.ebitda_margin
    )

    # ── Per-ticker session-state key prefix ───────────────────────────────────
    pfx = f"val_{ticker}_"

    def _state(key: str, default):
        full = pfx + key
        if full not in st.session_state:
            st.session_state[full] = default
        return full

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B — Assumption Sliders
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="asm-header">Your Assumptions</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11.5px;color:rgba(255,255,255,.35);margin:-4px 0 12px 0;">'
        'Pre-filled from SEC EDGAR + Wall Street consensus. Adjust to build your own narrative.</p>',
        unsafe_allow_html=True,
    )

    col_g, col_m, col_w = st.columns(3, gap="large")

    with col_g:
        st.markdown('<div class="asm-header" style="margin-top:4px;">Growth</div>', unsafe_allow_html=True)

        street_g1_note = (
            f"Street consensus: {cons_rev_g*100:+.1f}%"
            if not np.isnan(cons_rev_g) else
            "Pre-filled from 3yr historical CAGR."
        )
        rev_g_y1 = st.slider(
            "Revenue Growth — Year 1",
            min_value=-20.0, max_value=80.0,
            value=float(round(default_rev_g_y1 * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("rev_g_y1", round(default_rev_g_y1 * 100, 1)),
            help=street_g1_note,
        ) / 100

        rev_g_y5 = st.slider(
            "Revenue Growth — Year 5",
            min_value=-10.0, max_value=50.0,
            value=float(round(default_assumptions.revenue_growth_y5 * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("rev_g_y5", round(default_assumptions.revenue_growth_y5 * 100, 1)),
            help="Year 5 revenue growth rate. Growth interpolates linearly Y1→Y5, then fades to terminal.",
        ) / 100

        terminal_g = st.slider(
            "Terminal Growth Rate",
            min_value=0.0, max_value=5.0,
            value=float(round(default_assumptions.terminal_growth_rate * 100, 1)),
            step=0.1, format="%.1f%%",
            key=_state("terminal_g", round(default_assumptions.terminal_growth_rate * 100, 1)),
            help="Perpetuity growth rate. Should not exceed long-run nominal GDP growth (~2.5%).",
        ) / 100

    with col_m:
        st.markdown('<div class="asm-header" style="margin-top:4px;">Profitability & CapEx</div>', unsafe_allow_html=True)

        ebitda_note = (
            f"LTM actual (yfinance): {cons_ebitda_m*100:.1f}%"
            if not np.isnan(cons_ebitda_m) else
            "Pre-filled from LTM EBITDA margin (SEC)."
        )
        ebitda_m = st.slider(
            "EBITDA Margin",
            min_value=0.0, max_value=85.0,
            value=float(round(default_ebitda_m * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("ebitda_m", round(default_ebitda_m * 100, 1)),
            help=ebitda_note,
        ) / 100

        capex_pct = st.slider(
            "CapEx % Revenue",
            min_value=0.0, max_value=40.0,
            value=float(round(default_assumptions.capex_pct_revenue * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("capex_pct", round(default_assumptions.capex_pct_revenue * 100, 1)),
            help="CapEx as % of projected revenue. Pre-filled from LTM actual (SEC).",
        ) / 100

        da_pct = st.slider(
            "D&A % Revenue",
            min_value=0.0, max_value=25.0,
            value=float(round(default_assumptions.da_pct_revenue * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("da_pct", round(default_assumptions.da_pct_revenue * 100, 1)),
            help="Depreciation & Amortization as % of revenue.",
        ) / 100

        nwc_pct = st.slider(
            "NWC Change % Rev Δ",
            min_value=-5.0, max_value=15.0,
            value=float(round(default_assumptions.nwc_pct_revenue * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("nwc_pct", round(default_assumptions.nwc_pct_revenue * 100, 1)),
            help="Incremental change in NWC as % of each year's revenue change. Positive = cash outflow.",
        ) / 100

    with col_w:
        st.markdown('<div class="asm-header" style="margin-top:4px;">WACC Inputs</div>', unsafe_allow_html=True)

        beta_val = st.slider(
            f"Beta {_beta_badge(beta_result)}",
            min_value=0.10, max_value=3.50,
            value=float(round(default_assumptions.beta, 2)),
            step=0.05,
            key=_state("beta", round(default_assumptions.beta, 2)),
            help=f"Equity beta. Source: {beta_result.detail}",
        )

        rf_rate = st.slider(
            "Risk-Free Rate",
            min_value=0.0, max_value=8.0,
            value=float(round(default_assumptions.risk_free_rate * 100, 2)),
            step=0.05, format="%.2f%%",
            key=_state("rf_rate", round(default_assumptions.risk_free_rate * 100, 2)),
            help="10-year US Treasury yield. Update to current market rate.",
        ) / 100

        erp = st.slider(
            "Equity Risk Premium",
            min_value=2.0, max_value=10.0,
            value=float(round(default_assumptions.equity_risk_premium * 100, 2)),
            step=0.25, format="%.2f%%",
            key=_state("erp", round(default_assumptions.equity_risk_premium * 100, 2)),
            help="Market ERP above risk-free rate. Damodaran implied ERP (US, 2025) ≈ 5.5%.",
        ) / 100

        cod = st.slider(
            "Cost of Debt (pre-tax)",
            min_value=0.0, max_value=20.0,
            value=float(round(default_assumptions.cost_of_debt_pretax * 100, 2)),
            step=0.25, format="%.2f%%",
            key=_state("cod", round(default_assumptions.cost_of_debt_pretax * 100, 2)),
            help="Pre-tax cost of debt. Pre-filled as Interest Expense / Total Debt (LTM).",
        ) / 100

        debt_w = st.slider(
            "Debt Weight (% of capital)",
            min_value=0.0, max_value=90.0,
            value=float(round(default_assumptions.debt_weight * 100, 1)),
            step=1.0, format="%.0f%%",
            key=_state("debt_w", round(default_assumptions.debt_weight * 100, 1)),
            help="Debt / (Debt + Market Cap). Pre-filled from current capital structure.",
        ) / 100

        tax_rate = st.slider(
            "Tax Rate",
            min_value=0.0, max_value=50.0,
            value=float(round(default_assumptions.tax_rate * 100, 1)),
            step=0.5, format="%.1f%%",
            key=_state("tax_rate", round(default_assumptions.tax_rate * 100, 1)),
            help="Effective corporate tax rate. Pre-filled from constants.py registry.",
        ) / 100

    # ── Assemble DCFAssumptions from sliders ──────────────────────────────────
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
        st.markdown(f'<div class="val-warn">⚠ {w}</div>', unsafe_allow_html=True)

    if result.warnings and "undefined" in str(result.warnings):
        return

    st.markdown('<hr class="val-divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION C — Key Results Cards
    # ─────────────────────────────────────────────────────────────────────────
    ke, kd, wacc = result.cost_of_equity, result.cost_of_debt_aftertax, result.wacc

    if not np.isnan(result.intrinsic_price) and not np.isnan(current_price) and current_price > 0:
        upside_pct  = (result.intrinsic_price - current_price) / current_price * 100
        upside_str  = f"+{upside_pct:.1f}%" if upside_pct >= 0 else f"{upside_pct:.1f}%"
        upside_cls  = "up" if upside_pct >= 0 else "down"
        upside_lbl  = "Upside to Intrinsic" if upside_pct >= 0 else "Downside to Intrinsic"
    else:
        upside_pct = np.nan
        upside_str = "—"; upside_cls = ""; upside_lbl = "vs. Intrinsic Value"

    # Street vs. DCF comparison card
    street_vs_dcf = ""
    if not np.isnan(estimates.get("price_targets", {}).get("mean", np.nan)) and not np.isnan(result.intrinsic_price):
        street_mean = estimates["price_targets"]["mean"]
        diff = result.intrinsic_price - street_mean
        street_vs_dcf = f"DCF {'above' if diff >= 0 else 'below'} Street by {abs(diff/street_mean)*100:.1f}%"

    cards_html = '<div class="val-cards">'
    cards_html += _metric_card(
        "Intrinsic Price (DCF)", _fmt_price(result.intrinsic_price),
        street_vs_dcf or f"vs. {_fmt_price(current_price)} current",
        "highlight",
    )
    cards_html += _metric_card(upside_lbl, upside_str, "", upside_cls)
    cards_html += _metric_card(
        "Street Target (Mean)", _fmt_price(estimates.get("price_targets", {}).get("mean", np.nan)),
        f"{estimates.get('n_analysts', 0)} analysts · {estimates.get('rating') or '—'}",
        "purple",
    )
    cards_html += _metric_card("Enterprise Value", _fmt_m(result.enterprise_value), "DCF-implied", "")
    cards_html += _metric_card("WACC", f"{wacc:.2%}", f"Ke {ke:.2%} · Kd {kd:.2%}", "")
    cards_html += _metric_card(
        "Terminal Value", f"{result.pv_tv_pct:.0f}% of EV",
        _fmt_m(result.pv_terminal_value), "orange",
    )
    cards_html += _metric_card("Beta", f"{beta_val:.2f}", beta_result.method.capitalize(), "")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION D — Charts
    # ─────────────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown(
            '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
            'color:rgba(255,255,255,.35);margin-bottom:4px;">PV of Cash Flows by Year</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_chart_waterfall(result, assumptions), use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown(
            '<p style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
            'color:rgba(255,255,255,.35);margin-bottom:4px;">Revenue · EBITDA · Unlevered FCF</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_chart_revenue_fcf(result, assumptions), use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION E — Tabs
    # ─────────────────────────────────────────────────────────────────────────
    tab_sens1, tab_sens2, tab_proj, tab_wacc_tab = st.tabs([
        "WACC × Terminal Growth",
        "Revenue Growth × EBITDA Margin",
        "10-Year Model",
        "WACC Bridge",
    ])

    with tab_sens1:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:12px;">'
            'Intrinsic price per share across WACC and terminal growth rate scenarios. '
            'Green = above current price. Red = below. Base case highlighted in blue.</p>',
            unsafe_allow_html=True,
        )
        wacc_range = sorted({
            round(wacc - 0.04, 3), round(wacc - 0.03, 3), round(wacc - 0.02, 3),
            round(wacc - 0.01, 3), round(wacc, 3),
            round(wacc + 0.01, 3), round(wacc + 0.02, 3), round(wacc + 0.03, 3),
        })
        wacc_range = [w for w in wacc_range if 0.03 < w < 0.35]
        tg_range = sorted({
            round(terminal_g - 0.015, 3), round(terminal_g - 0.01, 3),
            round(terminal_g - 0.005, 3), round(terminal_g, 3),
            round(terminal_g + 0.005, 3), round(terminal_g + 0.01, 3),
            round(terminal_g + 0.015, 3),
        })
        tg_range = [t for t in tg_range if 0.0 < t < wacc - 0.01]
        with st.spinner("Computing sensitivity…"):
            sens_df = sensitivity_table(assumptions, wacc_range, tg_range)
        st.markdown(
            _render_sensitivity(sens_df, current_price, f"{wacc:.1%}", f"{terminal_g:.1%}"),
            unsafe_allow_html=True,
        )

    with tab_sens2:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:12px;">'
            'Intrinsic price per share across revenue growth and EBITDA margin scenarios.</p>',
            unsafe_allow_html=True,
        )
        g_range = sorted({
            round(rev_g_y1 - 0.12, 2), round(rev_g_y1 - 0.08, 2), round(rev_g_y1 - 0.04, 2),
            round(rev_g_y1, 2), round(rev_g_y1 + 0.04, 2), round(rev_g_y1 + 0.08, 2),
            round(rev_g_y1 + 0.12, 2),
        })
        g_range = [g for g in g_range if -0.20 < g < 0.80]
        m_range = sorted({
            round(ebitda_m - 0.10, 2), round(ebitda_m - 0.05, 2),
            round(ebitda_m, 2), round(ebitda_m + 0.05, 2), round(ebitda_m + 0.10, 2),
        })
        m_range = [m for m in m_range if 0.0 < m < 0.90]
        with st.spinner("Computing sensitivity…"):
            sens_df2 = sensitivity_table_growth(assumptions, g_range, m_range)
        st.markdown(
            _render_sensitivity(sens_df2, current_price, f"{rev_g_y1:.0%}", f"{ebitda_m:.0%}"),
            unsafe_allow_html=True,
        )

    with tab_proj:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:12px;">'
            '10-year projected income and free cash flow build. All values in $B unless otherwise noted.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _render_projection_table(result, assumptions, divisor=1e9, unit="$B"),
            unsafe_allow_html=True,
        )
        # Terminal value bridge cards
        st.markdown("<br>", unsafe_allow_html=True)
        tv_col1, tv_col2, tv_col3 = st.columns(3)
        with tv_col1:
            st.markdown(
                f'<div class="val-card"><div class="label">Year 10 UFCF</div>'
                f'<div class="value">{_fmt_m(result.projected_ufcf[-1])}</div></div>',
                unsafe_allow_html=True,
            )
        with tv_col2:
            st.markdown(
                f'<div class="val-card orange"><div class="label">Terminal Value (undiscounted)</div>'
                f'<div class="value">{_fmt_m(result.terminal_value)}</div>'
                f'<div class="sub">Y10 UFCF × (1+{terminal_g:.1%}) / ({wacc:.2%} − {terminal_g:.1%})</div></div>',
                unsafe_allow_html=True,
            )
        with tv_col3:
            st.markdown(
                f'<div class="val-card"><div class="label">PV of Terminal Value</div>'
                f'<div class="value">{_fmt_m(result.pv_terminal_value)}</div>'
                f'<div class="sub">{result.pv_tv_pct:.0f}% of Enterprise Value</div></div>',
                unsafe_allow_html=True,
            )

    with tab_wacc_tab:
        st.markdown(
            '<p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:8px;margin-bottom:16px;">'
            'WACC component breakdown and equity value bridge.</p>',
            unsafe_allow_html=True,
        )
        wacc_col1, wacc_col2 = st.columns(2, gap="large")

        with wacc_col1:
            st.markdown('<div class="asm-header">WACC Build</div>', unsafe_allow_html=True)
            wacc_rows = [
                ("Risk-Free Rate",           f"{rf_rate:.2%}",      "10yr UST yield"),
                ("× Beta (β)",               f"{beta_val:.2f}x",    f"{beta_result.method.capitalize()} method"),
                ("× Equity Risk Premium",    f"{erp:.2%}",          "Damodaran implied ERP"),
                ("= Cost of Equity (Ke)",    f"{ke:.2%}",           "Ke = Rf + β × ERP"),
                ("", "", ""),
                ("Cost of Debt (pre-tax)",   f"{cod:.2%}",          "Interest expense / total debt"),
                ("× (1 – Tax Rate)",         f"{1-tax_rate:.2%}",   f"After-tax shield at {tax_rate:.0%}"),
                ("= Cost of Debt (Kd)",      f"{kd:.2%}",           "After-tax cost of debt"),
                ("", "", ""),
                ("Equity Weight",            f"{(1-debt_w):.0%}",   "Market cap / total capital"),
                ("Debt Weight",              f"{debt_w:.0%}",       "Total debt / total capital"),
                ("= WACC",                   f"{wacc:.2%}",         "Ke×We + Kd×Wd"),
            ]
            for label, value, note in wacc_rows:
                if not label:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    continue
                is_result = label.startswith("=")
                weight = "700" if is_result else "400"
                color  = "#fff" if is_result else "rgba(255,255,255,.8)"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                    f'<span style="font-size:12.5px;color:{color};font-weight:{weight};">{label}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:{BLUE if is_result else "#fff"};">{value}'
                    f'<span style="font-size:10px;font-weight:400;color:rgba(255,255,255,.3);margin-left:8px;">{note}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

        with wacc_col2:
            st.markdown('<div class="asm-header">Equity Value Bridge</div>', unsafe_allow_html=True)
            bridge_rows = [
                ("Sum of PV (UFCF, Y1–Y10)",    result.sum_pv_ufcf,        f"{result.pv_ufcf_pct:.0f}% of EV"),
                ("+ PV of Terminal Value",        result.pv_terminal_value,  f"{result.pv_tv_pct:.0f}% of EV"),
                ("= Enterprise Value",            result.enterprise_value,   "DCF-implied"),
                ("", np.nan, ""),
                ("– Net Debt",                    assumptions.net_debt,      "Total Debt – Cash"),
                ("= Equity Value",                result.equity_value,       "EV – Net Debt"),
                ("", np.nan, ""),
                ("÷ Shares Outstanding",          assumptions.shares_outstanding, "Diluted"),
                ("= Intrinsic Price per Share",   result.intrinsic_price,    ""),
                ("Current Market Price",          current_price,             ""),
            ]
            for label, value, note in bridge_rows:
                if not label:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    continue
                is_result = label.startswith("=")
                is_final  = "Intrinsic Price" in label
                color  = BLUE if is_final else ("#fff" if is_result else "rgba(255,255,255,.8)")
                weight = "800" if is_final else ("700" if is_result else "400")
                if isinstance(value, float) and not np.isnan(value):
                    if "÷" in label:
                        val_str = f"{value/1e6:,.1f}M"
                    elif abs(value) < 1000:
                        val_str = f"${value:,.2f}"
                    else:
                        val_str = _fmt_m(value)
                else:
                    val_str = "—"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                    f'<span style="font-size:12.5px;color:{color};font-weight:{weight};">{label}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:{color};">{val_str}'
                    f'<span style="font-size:10px;font-weight:400;color:rgba(255,255,255,.3);margin-left:8px;">{note}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Beta detail expander ──────────────────────────────────────────────────
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
    # Footer — methodology note
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="fin-note">
<b>Data Sources</b> — Financial inputs (Revenue, EBITDA, CapEx, D&A, Debt, Cash) are sourced from
SEC EDGAR via XBRL companyfacts (same pipeline as the Financials page). yfinance supplies current
market price, analyst price targets, and forward revenue/EPS consensus estimates. &nbsp;·&nbsp;
<b>Analyst Consensus</b> — Price targets and estimates are pulled from Yahoo Finance (aggregated from
sell-side research). Coverage and freshness varies by ticker. &nbsp;·&nbsp;
<b>Beta Hierarchy</b> — (1) SEC-derived fundamental beta (Damodaran SIC unlevered β, Hamada re-levering);
(2) 5yr monthly OLS regression vs. SPY with Vasicek shrinkage; (3) Yahoo Finance beta; (4) market β=1.0.
&nbsp;·&nbsp;
<b>Model</b> — Two-stage DCF: Stage 1 (Y1–Y5) explicit growth; Stage 2 (Y6–Y10) linear fade to terminal growth.
Terminal value via Gordon Growth Model. WACC = Ke×We + Kd(1–t)×Wd. &nbsp;·&nbsp;
<b>Limitations</b> — This model is a starting framework. UFCF projections are highly sensitive to growth,
margin, and WACC assumptions. Always adjust assumptions to reflect your own research before drawing
investment conclusions.
</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatibility alias
# Allows existing app.py code using render_dcf() to continue working
# while migrating to the new render_valuation() name.
# ─────────────────────────────────────────────────────────────────────────────
render_dcf = render_valuation
