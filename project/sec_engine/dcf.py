# sec_engine/dcf.py
# ──────────────────────────────────────────────────────────────────────────────
# Discounted Cash Flow (DCF) Engine
# ──────────────────────────────────────────────────────────────────────────────
#
# Architecture
# ------------
# Pure financial logic — zero UI / Streamlit imports.
# All functions are deterministic, side-effect-free, and fully documented.
#
# Beta Calculation Hierarchy
# --------------------------
#  1. SEC-derived fundamental beta (bottom-up, Damodaran methodology):
#       - Unlevered beta estimated from industry SIC via SEC filing data
#       - Re-levered using the company's actual D/E ratio from EDGAR balance sheet
#       - Most stable: not contaminated by short-term price noise
#  2. Regression beta (5yr monthly returns vs. S&P 500 via yfinance):
#       - Standard OLS regression of stock vs. market excess returns
#       - Blended with Vasicek shrinkage toward 1.0 (reduces estimation error)
#  3. yfinance beta field (info["beta"]):
#       - Direct fallback — Yahoo Finance's own 5yr monthly calculation
#  4. Final fallback: β = 1.0 (market beta)
#
# WACC Components
# ---------------
#  - Cost of equity  : CAPM  →  Rf + β × ERP
#  - Cost of debt    : Interest Expense (LTM) / Average Total Debt  ×  (1 – t)
#  - Weights         : Market value of equity / Enterprise value (book debt)
#
# DCF Model
# ---------
#  Base  : LTM UFCF from aggregation.py (SEC primary, yfinance fallback)
#  Stage 1: Years 1–5  — analyst-adjustable growth rate, margins, capex %
#  Stage 2: Years 6–10 — fade to terminal growth (linear interpolation)
#  Terminal: Gordon Growth Model  →  UFCF₁₀ × (1+g) / (WACC – g)
#  Bridge  : Enterprise Value → Equity Value → Price per Share
#              EV – Net Debt = Equity Value
#              Equity Value / Shares Outstanding = Intrinsic Price
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Market constants  (can be overridden by caller)
# ─────────────────────────────────────────────────────────────────────────────
RISK_FREE_RATE_DEFAULT: float = 0.043   # ~10yr UST yield (update periodically)
EQUITY_RISK_PREMIUM_DEFAULT: float = 0.055   # Damodaran implied ERP (US, 2025)

# Vasicek shrinkage weight toward β=1.0.  β_adj = w×β_raw + (1-w)×1.0
# Bloomberg uses 0.67; we use 0.60 to be slightly more conservative.
VASICEK_WEIGHT: float = 0.60

# Industry unlevered beta lookup (SIC 2-digit → median unlevered beta).
# Source: Damodaran Online, "Betas by Sector" (US, Jan 2025 update).
# Used for fundamental beta when regression data is insufficient.
_SIC2_UNLEVERED_BETA: Dict[str, float] = {
    "01": 0.72,  "02": 0.72,  "07": 0.65,  "08": 0.65,  "09": 0.65,
    "10": 0.85,  "11": 0.75,  "12": 0.88,  "13": 0.95,  "14": 0.82,
    "15": 0.95,  "16": 0.90,  "17": 0.88,
    "20": 0.68,  "21": 0.70,  "22": 0.75,  "23": 0.72,  "24": 0.78,
    "25": 0.80,  "26": 0.75,  "27": 0.72,  "28": 0.80,  "29": 0.90,
    "30": 0.78,  "31": 0.76,  "32": 0.76,  "33": 0.88,  "34": 0.85,
    "35": 0.95,  "36": 1.05,  "37": 0.95,  "38": 0.88,  "39": 0.82,
    "40": 0.70,  "41": 0.70,  "42": 0.88,  "43": 0.70,  "44": 0.80,
    "45": 0.90,  "46": 0.78,  "47": 0.90,  "48": 0.75,  "49": 0.50,
    "50": 0.82,  "51": 0.78,  "52": 0.80,  "53": 0.72,  "54": 0.68,
    "55": 0.90,  "56": 0.80,  "57": 0.82,  "58": 0.72,  "59": 0.75,
    "60": 0.42,  "61": 0.55,  "62": 0.95,  "63": 0.55,  "64": 0.60,
    "65": 0.62,  "67": 0.70,
    "70": 0.85,  "72": 0.78,  "73": 1.05,  "75": 0.82,  "76": 0.78,
    "78": 0.90,  "79": 0.95,  "80": 0.75,  "82": 0.72,  "83": 0.68,
    "84": 0.75,  "86": 0.65,  "87": 0.88,
}

_DEFAULT_UNLEVERED_BETA: float = 0.90  # fallback if SIC not in table


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DCFAssumptions:
    """
    All analyst-adjustable inputs to the DCF model.
    Defaults are pre-populated by build_dcf_assumptions() from historical data.
    """
    # ── Projection horizon ──────────────────────────────────────────────────
    n_years: int = 10           # total forecast years (Stage 1 + Stage 2)
    stage1_years: int = 5       # years with explicit growth assumptions

    # ── Stage 1 assumptions (years 1–5) ─────────────────────────────────────
    revenue_growth_y1: float = 0.10   # Year 1 revenue growth rate (decimal)
    revenue_growth_y5: float = 0.08   # Year 5 revenue growth rate (interpolated)
    ebitda_margin: float = 0.25       # EBITDA margin (decimal, applied to projected revenue)
    capex_pct_revenue: float = 0.05   # CapEx as % of projected revenue
    da_pct_revenue: float = 0.04      # D&A as % of projected revenue
    nwc_pct_revenue: float = 0.02     # Change in NWC as % of revenue change

    # ── Stage 2 / terminal assumptions ──────────────────────────────────────
    terminal_growth_rate: float = 0.025   # Perpetuity growth rate (nominal)

    # ── WACC components ──────────────────────────────────────────────────────
    beta: float = 1.0
    risk_free_rate: float = RISK_FREE_RATE_DEFAULT
    equity_risk_premium: float = EQUITY_RISK_PREMIUM_DEFAULT
    cost_of_debt_pretax: float = 0.055    # Interest expense / total debt
    tax_rate: float = 0.21
    debt_weight: float = 0.20             # Debt / (Debt + Equity) at market value
    equity_weight: float = 0.80

    # ── Base period (LTM actuals) ─────────────────────────────────────────────
    base_ufcf: float = np.nan          # LTM Unlevered FCF ($)
    base_revenue: float = np.nan       # LTM Revenue ($)

    # ── Bridge to equity value ────────────────────────────────────────────────
    net_debt: float = 0.0              # Total Debt – Cash ($)
    shares_outstanding: float = np.nan # diluted shares


@dataclass
class DCFResult:
    """Output of run_dcf(). Contains all intermediate values for display."""
    # ── WACC ─────────────────────────────────────────────────────────────────
    cost_of_equity: float = np.nan
    cost_of_debt_aftertax: float = np.nan
    wacc: float = np.nan

    # ── Projected cash flows ──────────────────────────────────────────────────
    projected_revenues: List[float] = field(default_factory=list)
    projected_ebitda: List[float] = field(default_factory=list)
    projected_ufcf: List[float] = field(default_factory=list)
    discount_factors: List[float] = field(default_factory=list)
    pv_ufcf: List[float] = field(default_factory=list)

    # ── Valuation ─────────────────────────────────────────────────────────────
    sum_pv_ufcf: float = np.nan        # PV of Stage 1+2 cash flows
    terminal_value: float = np.nan     # Gordon Growth TV (undiscounted)
    pv_terminal_value: float = np.nan  # PV of terminal value
    enterprise_value: float = np.nan   # sum_pv_ufcf + pv_terminal_value
    equity_value: float = np.nan       # enterprise_value – net_debt
    intrinsic_price: float = np.nan    # equity_value / shares_outstanding

    # ── Composition ──────────────────────────────────────────────────────────
    pv_ufcf_pct: float = np.nan        # % of EV from explicit period CFs
    pv_tv_pct: float = np.nan          # % of EV from terminal value

    # ── Diagnostics ──────────────────────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)


@dataclass
class BetaResult:
    """Beta calculation result with source attribution."""
    beta: float = 1.0
    method: str = "fallback"           # "fundamental", "regression", "yfinance", "fallback"
    unlevered_beta: float = np.nan
    relevered_beta: float = np.nan
    regression_beta_raw: float = np.nan
    regression_r2: float = np.nan
    vasicek_beta: float = np.nan
    detail: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Beta Calculation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v, default: float = np.nan) -> float:
    """Coerce v to float, returning default on failure."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def compute_beta(
    ticker: str,
    sic_code: str = "",
    debt: float = np.nan,
    equity: float = np.nan,
    tax_rate: float = 0.21,
    facts: Optional[dict] = None,
) -> BetaResult:
    """
    Compute beta using a four-level hierarchy:

    1. Fundamental (SEC-derived bottom-up):
       - Look up unlevered beta for the company's SIC code (2-digit)
       - Re-lever using Hamada equation:
             β_levered = β_unlevered × [1 + (1 – t) × (D/E)]
       - Uses EDGAR balance sheet debt/equity so the result is audit-traceable.

    2. Regression (5yr monthly prices vs SPY via yfinance):
       - Minimum 24 monthly observations required
       - Blended with Vasicek shrinkage: β_adj = 0.60×β_raw + 0.40×1.0
       - Falls through to method 3 if insufficient data.

    3. yfinance info["beta"]:
       - Yahoo Finance's published 5yr monthly regression beta.

    4. Fallback β = 1.0

    Args:
        ticker    : Stock ticker (e.g. "AAPL")
        sic_code  : 4-digit SIC string from SEC submissions endpoint
        debt      : Total debt ($) from EDGAR balance sheet
        equity    : Book equity ($) from EDGAR balance sheet
        tax_rate  : Effective tax rate (decimal)
        facts     : SEC companyfacts dict (used for additional balance sheet data)

    Returns:
        BetaResult with .beta, .method, and diagnostic fields.
    """
    result = BetaResult()

    # ── Method 1: Fundamental (SEC SIC → unlevered beta → re-lever) ──────────
    try:
        sic2 = str(sic_code).strip()[:2] if sic_code else ""
        u_beta = _SIC2_UNLEVERED_BETA.get(sic2, _DEFAULT_UNLEVERED_BETA)
        result.unlevered_beta = u_beta

        d = _safe_float(debt, 0.0)
        e = _safe_float(equity)

        if not np.isnan(e) and e > 0 and d >= 0:
            de_ratio = d / e
            # Hamada equation
            relevered = u_beta * (1.0 + (1.0 - tax_rate) * de_ratio)
            result.relevered_beta = relevered
            result.beta = round(relevered, 3)
            result.method = "fundamental"
            result.detail = (
                f"SIC {sic_code or '??'} → β_u={u_beta:.2f}; "
                f"D/E={de_ratio:.2f}; t={tax_rate:.0%}; "
                f"β_L={relevered:.3f} (Hamada)"
            )
            _logger.debug("Beta[fundamental] %s: %s", ticker, result.detail)
        else:
            # Can't re-lever without equity — fall through
            raise ValueError("Equity ≤ 0 or missing — cannot re-lever")

    except Exception as e:
        _logger.debug("Fundamental beta failed for %s: %s", ticker, e)

    # ── Method 2: Regression beta (5yr monthly vs SPY) ───────────────────────
    try:
        import yfinance as yf

        raw = yf.download(
            [ticker, "SPY"],
            period="5y",
            interval="1mo",
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"][[ticker, "SPY"]].dropna()
        else:
            close = raw[["Close"]].dropna()

        if len(close) >= 24 and ticker in close.columns and "SPY" in close.columns:
            ret = close.pct_change().dropna()
            y = ret[ticker].values
            x = ret["SPY"].values

            # OLS regression: y = α + β×x
            cov_xy = np.cov(x, y, ddof=1)
            raw_beta = cov_xy[0, 1] / cov_xy[0, 0]
            r2 = cov_xy[0, 1] ** 2 / (cov_xy[0, 0] * cov_xy[1, 1])

            result.regression_beta_raw = round(raw_beta, 3)
            result.regression_r2 = round(r2, 3)

            # Vasicek shrinkage toward 1.0
            vasicek = VASICEK_WEIGHT * raw_beta + (1.0 - VASICEK_WEIGHT) * 1.0
            result.vasicek_beta = round(vasicek, 3)

            # Only override fundamental if regression R² is meaningful (≥0.10)
            if result.method != "fundamental" or r2 >= 0.10:
                if result.method != "fundamental":
                    result.beta = round(vasicek, 3)
                    result.method = "regression"
                    result.detail = (
                        f"5yr monthly OLS β={raw_beta:.3f}, R²={r2:.2f}; "
                        f"Vasicek({VASICEK_WEIGHT})→β={vasicek:.3f}"
                    )
                # If fundamental succeeded, keep it but store regression data
        else:
            _logger.debug("Insufficient price history for regression beta: %s (%d months)", ticker, len(close))

    except Exception as e:
        _logger.debug("Regression beta failed for %s: %s", ticker, e)

    # ── Method 3: yfinance info["beta"] ──────────────────────────────────────
    if result.method in ("fallback",):
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            yf_beta = _safe_float(info.get("beta"))
            if not np.isnan(yf_beta) and 0.0 < yf_beta < 5.0:
                result.beta = round(yf_beta, 3)
                result.method = "yfinance"
                result.detail = f"Yahoo Finance published beta={yf_beta:.3f}"
        except Exception as e:
            _logger.debug("yfinance beta fallback failed for %s: %s", ticker, e)

    # ── Method 4: Fallback β = 1.0 ────────────────────────────────────────────
    if result.method == "fallback" or np.isnan(result.beta):
        result.beta = 1.0
        result.method = "fallback"
        result.detail = "No beta data available — using market beta β=1.0"

    # Guard: clamp to reasonable range [0.1, 4.0]
    result.beta = float(np.clip(result.beta, 0.1, 4.0))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WACC
# ─────────────────────────────────────────────────────────────────────────────

def compute_wacc(assumptions: DCFAssumptions) -> Tuple[float, float, float]:
    """
    Compute WACC from DCFAssumptions.

    Returns:
        (cost_of_equity, cost_of_debt_after_tax, wacc)  — all as decimals
    """
    ke = assumptions.risk_free_rate + assumptions.beta * assumptions.equity_risk_premium
    kd = assumptions.cost_of_debt_pretax * (1.0 - assumptions.tax_rate)
    we = assumptions.equity_weight
    wd = assumptions.debt_weight
    wacc = ke * we + kd * wd
    return ke, kd, wacc


# ─────────────────────────────────────────────────────────────────────────────
# Projection engine
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_growth(g_start: float, g_end: float, n: int) -> List[float]:
    """Linear interpolation of growth rate over n years."""
    if n <= 1:
        return [g_start]
    return [g_start + (g_end - g_start) * i / (n - 1) for i in range(n)]


def run_dcf(assumptions: DCFAssumptions) -> DCFResult:
    """
    Execute the 10-year two-stage DCF model.

    Stage 1 (years 1–stage1_years):  explicit revenue growth + EBITDA margin
    Stage 2 (years stage1_years+1–n_years):  growth fades linearly to terminal_growth_rate
    Terminal value: Gordon Growth Model on year-n_years UFCF

    Returns:
        DCFResult with all intermediate values populated.
    """
    result = DCFResult()
    warns = []

    # ── Validate inputs ───────────────────────────────────────────────────────
    ke, kd, wacc = compute_wacc(assumptions)
    result.cost_of_equity = ke
    result.cost_of_debt_aftertax = kd
    result.wacc = wacc

    if wacc <= 0:
        warns.append("WACC ≤ 0 — check inputs.")
        result.warnings = warns
        return result

    if assumptions.terminal_growth_rate >= wacc:
        warns.append(
            f"Terminal growth rate ({assumptions.terminal_growth_rate:.1%}) ≥ WACC ({wacc:.1%}). "
            "Gordon Growth Model undefined — reduce terminal growth or raise WACC."
        )
        result.warnings = warns
        return result

    base_rev = _safe_float(assumptions.base_revenue, np.nan)
    base_ufcf = _safe_float(assumptions.base_ufcf, np.nan)

    if np.isnan(base_rev) or base_rev <= 0:
        warns.append("Base revenue is missing or ≤ 0 — cannot project.")
        result.warnings = warns
        return result

    n = assumptions.n_years
    s1 = assumptions.stage1_years
    s2 = n - s1

    # ── Build growth rate schedule ────────────────────────────────────────────
    # Stage 1: analyst-specified, interpolated from y1 to y5
    g_s1 = _interpolate_growth(assumptions.revenue_growth_y1, assumptions.revenue_growth_y5, s1)
    # Stage 2: fade from y5 growth to terminal growth
    g_s2 = _interpolate_growth(assumptions.revenue_growth_y5, assumptions.terminal_growth_rate, s2 + 1)[1:]
    growth_rates = g_s1 + g_s2   # length = n

    # ── Project revenues, EBITDA, UFCF ───────────────────────────────────────
    rev = base_rev
    revs, ebitdas, ufcfs, dfs, pvs = [], [], [], [], []

    for yr in range(1, n + 1):
        g = growth_rates[yr - 1]
        rev = rev * (1.0 + g)
        ebitda = rev * assumptions.ebitda_margin
        da = rev * assumptions.da_pct_revenue
        ebit = ebitda - da
        nopat = ebit * (1.0 - assumptions.tax_rate)
        capex = rev * assumptions.capex_pct_revenue
        # NWC change: positive = cash outflow (investing in working capital)
        if yr == 1:
            prev_rev = base_rev
        else:
            prev_rev = revs[-1]
        delta_nwc = (rev - prev_rev) * assumptions.nwc_pct_revenue
        ufcf = nopat + da - capex - delta_nwc

        df = 1.0 / (1.0 + wacc) ** yr
        pv = ufcf * df

        revs.append(rev)
        ebitdas.append(ebitda)
        ufcfs.append(ufcf)
        dfs.append(df)
        pvs.append(pv)

    # ── Terminal value ────────────────────────────────────────────────────────
    tg = assumptions.terminal_growth_rate
    tv = ufcfs[-1] * (1.0 + tg) / (wacc - tg)
    pv_tv = tv / (1.0 + wacc) ** n

    # ── Enterprise value ──────────────────────────────────────────────────────
    sum_pv = sum(pvs)
    ev = sum_pv + pv_tv

    # ── Equity bridge ─────────────────────────────────────────────────────────
    net_debt = _safe_float(assumptions.net_debt, 0.0)
    eq_val = ev - net_debt
    shares = _safe_float(assumptions.shares_outstanding)
    price = eq_val / shares if not np.isnan(shares) and shares > 0 else np.nan

    # ── Populate result ───────────────────────────────────────────────────────
    result.projected_revenues = revs
    result.projected_ebitda = ebitdas
    result.projected_ufcf = ufcfs
    result.discount_factors = dfs
    result.pv_ufcf = pvs
    result.sum_pv_ufcf = sum_pv
    result.terminal_value = tv
    result.pv_terminal_value = pv_tv
    result.enterprise_value = ev
    result.equity_value = eq_val
    result.intrinsic_price = price
    result.pv_ufcf_pct = sum_pv / ev * 100 if ev != 0 else np.nan
    result.pv_tv_pct = pv_tv / ev * 100 if ev != 0 else np.nan

    if abs(result.pv_tv_pct) > 80:
        warns.append(
            f"Terminal value represents {result.pv_tv_pct:.0f}% of enterprise value. "
            "Results are highly sensitive to terminal growth and WACC assumptions."
        )

    result.warnings = warns
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_table(
    assumptions: DCFAssumptions,
    wacc_range: List[float],
    tg_range: List[float],
) -> pd.DataFrame:
    """
    Build a (WACC × terminal growth) sensitivity table of intrinsic price per share.

    Args:
        assumptions : Base DCFAssumptions (will be copied for each scenario)
        wacc_range  : List of WACC values (decimals, e.g. [0.08, 0.09, 0.10])
        tg_range    : List of terminal growth rates (decimals, e.g. [0.02, 0.025, 0.03])

    Returns:
        DataFrame indexed by WACC (rows) × terminal growth (columns).
        Values are intrinsic price per share, formatted as floats.
    """
    rows = {}
    for w in wacc_range:
        row = {}
        for tg in tg_range:
            a = DCFAssumptions(
                n_years=assumptions.n_years,
                stage1_years=assumptions.stage1_years,
                revenue_growth_y1=assumptions.revenue_growth_y1,
                revenue_growth_y5=assumptions.revenue_growth_y5,
                ebitda_margin=assumptions.ebitda_margin,
                capex_pct_revenue=assumptions.capex_pct_revenue,
                da_pct_revenue=assumptions.da_pct_revenue,
                nwc_pct_revenue=assumptions.nwc_pct_revenue,
                terminal_growth_rate=tg,
                beta=assumptions.beta,
                risk_free_rate=assumptions.risk_free_rate,
                equity_risk_premium=assumptions.equity_risk_premium,
                cost_of_debt_pretax=assumptions.cost_of_debt_pretax,
                tax_rate=assumptions.tax_rate,
                debt_weight=assumptions.debt_weight,
                equity_weight=assumptions.equity_weight,
                base_ufcf=assumptions.base_ufcf,
                base_revenue=assumptions.base_revenue,
                net_debt=assumptions.net_debt,
                shares_outstanding=assumptions.shares_outstanding,
            )
            # Override WACC directly by adjusting equity risk premium
            # to produce the desired WACC given the current weights
            ke_needed = (w - a.cost_of_debt_pretax * (1 - a.tax_rate) * a.debt_weight) / a.equity_weight
            a.equity_risk_premium = max(0.001, (ke_needed - a.risk_free_rate) / max(a.beta, 0.01))
            r = run_dcf(a)
            row[f"{tg:.1%}"] = round(r.intrinsic_price, 2) if not np.isnan(r.intrinsic_price) else np.nan
        rows[f"{w:.1%}"] = row

    df = pd.DataFrame(rows).T
    df.index.name = "WACC"
    df.columns.name = "Terminal Growth"
    return df


def sensitivity_table_growth(
    assumptions: DCFAssumptions,
    growth_range: List[float],
    margin_range: List[float],
) -> pd.DataFrame:
    """
    Build a (Revenue Growth Y1 × EBITDA Margin) sensitivity table.

    Returns:
        DataFrame indexed by revenue_growth_y1 × ebitda_margin.
        Values are intrinsic price per share.
    """
    rows = {}
    for g in growth_range:
        row = {}
        for m in margin_range:
            a = DCFAssumptions(
                n_years=assumptions.n_years,
                stage1_years=assumptions.stage1_years,
                revenue_growth_y1=g,
                revenue_growth_y5=max(assumptions.terminal_growth_rate + 0.01, g * 0.60),
                ebitda_margin=m,
                capex_pct_revenue=assumptions.capex_pct_revenue,
                da_pct_revenue=assumptions.da_pct_revenue,
                nwc_pct_revenue=assumptions.nwc_pct_revenue,
                terminal_growth_rate=assumptions.terminal_growth_rate,
                beta=assumptions.beta,
                risk_free_rate=assumptions.risk_free_rate,
                equity_risk_premium=assumptions.equity_risk_premium,
                cost_of_debt_pretax=assumptions.cost_of_debt_pretax,
                tax_rate=assumptions.tax_rate,
                debt_weight=assumptions.debt_weight,
                equity_weight=assumptions.equity_weight,
                base_ufcf=assumptions.base_ufcf,
                base_revenue=assumptions.base_revenue,
                net_debt=assumptions.net_debt,
                shares_outstanding=assumptions.shares_outstanding,
            )
            r = run_dcf(a)
            row[f"{m:.0%}"] = round(r.intrinsic_price, 2) if not np.isnan(r.intrinsic_price) else np.nan
        rows[f"{g:.0%}"] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Rev Growth Y1"
    df.columns.name = "EBITDA Margin"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Assumption builder — populates DCFAssumptions from live company data
# ─────────────────────────────────────────────────────────────────────────────

def build_dcf_assumptions(
    ticker: str,
    summary: dict,
    data: dict,
    sic_code: str = "",
    facts: Optional[dict] = None,
) -> Tuple[DCFAssumptions, BetaResult]:
    """
    Auto-populate DCFAssumptions from the company's historical data.

    This is the connection point between the Financials/aggregation layer
    and the DCF engine. It reads from the same summary dict that powers
    the Screener, Tearsheet, Ratios, and Multiples pages.

    Data hierarchy for each assumption:
      - Revenue growth  : 3yr historical CAGR (fallback: 2yr, then YoY, then 8%)
      - EBITDA margin   : LTM EBITDA margin from summary (fallback: 20%)
      - CapEx % revenue : LTM capex/revenue from summary (fallback: 5%)
      - D&A % revenue   : LTM D&A/revenue (fallback: 4%)
      - Cost of debt    : Interest expense / average total debt (fallback: 5.5%)
      - Debt weight     : Total debt / (Total debt + Market cap equity)
      - Net debt        : Total debt – Cash (from balance_data)
      - Shares          : Market cap / current price, or yfinance sharesOutstanding
      - Beta            : compute_beta() hierarchy (see above)

    Args:
        ticker   : Stock ticker
        summary  : Output of build_company_summary()
        data     : Output of fetch_company_data_unified() (contains ltm_data, balance_data, metadata)
        sic_code : 4-digit SIC from SEC submissions
        facts    : Raw SEC companyfacts dict (for fundamental beta)

    Returns:
        (DCFAssumptions, BetaResult)
    """
    a = DCFAssumptions()
    ltm = data.get("ltm_data", {})
    bal = data.get("balance_data", {})
    meta = data.get("metadata", {})

    def _g(d: dict, key: str, default=np.nan):
        v = d.get(key, default)
        return _safe_float(v, default)

    # ── Base revenue & UFCF ──────────────────────────────────────────────────
    a.base_revenue = _g(ltm, "revenue")
    a.base_ufcf = _safe_float(summary.get("UFCF"), np.nan)

    # If UFCF is not in summary, compute from LFCF
    if np.isnan(a.base_ufcf):
        lfcf = _safe_float(summary.get("LFCF"), np.nan)
        if not np.isnan(lfcf):
            a.base_ufcf = lfcf

    # ── Revenue growth: 3yr CAGR → 2yr → YoY → 8% default ──────────────────
    rev3 = _safe_float(summary.get("Revenue 3yr CAGR %")) / 100
    rev2 = _safe_float(summary.get("Revenue 2yr CAGR %")) / 100
    rev_yoy = _safe_float(summary.get("Revenue YoY %")) / 100

    if not np.isnan(rev3) and -0.5 < rev3 < 2.0:
        base_growth = rev3
    elif not np.isnan(rev2) and -0.5 < rev2 < 2.0:
        base_growth = rev2
    elif not np.isnan(rev_yoy) and -0.5 < rev_yoy < 2.0:
        base_growth = rev_yoy
    else:
        base_growth = 0.08

    # Y1 = historical growth; Y5 = blend toward terminal (fade)
    a.revenue_growth_y1 = round(float(np.clip(base_growth, -0.20, 0.60)), 4)
    a.revenue_growth_y5 = round(float(np.clip(base_growth * 0.65, 0.02, 0.40)), 4)

    # ── EBITDA margin (LTM) ───────────────────────────────────────────────────
    ebitda_m = _safe_float(summary.get("EBITDA Margin %")) / 100
    if np.isnan(ebitda_m) or ebitda_m <= 0:
        ebitda_m = 0.20
    a.ebitda_margin = round(float(np.clip(ebitda_m, 0.01, 0.85)), 4)

    # ── CapEx % revenue (LTM) ─────────────────────────────────────────────────
    capex_pct = _safe_float(summary.get("CapEx % Revenue")) / 100
    if np.isnan(capex_pct) or capex_pct <= 0:
        capex_pct = 0.05
    a.capex_pct_revenue = round(float(np.clip(capex_pct, 0.005, 0.40)), 4)

    # ── D&A % revenue ─────────────────────────────────────────────────────────
    rev = _safe_float(ltm.get("revenue"), np.nan)
    da = _safe_float(ltm.get("depreciation"), np.nan)
    if not np.isnan(rev) and rev > 0 and not np.isnan(da) and da > 0:
        a.da_pct_revenue = round(float(np.clip(da / rev, 0.005, 0.25)), 4)
    else:
        a.da_pct_revenue = 0.04

    # ── Tax rate ──────────────────────────────────────────────────────────────
    from sec_engine.constants import get_effective_tax_rate
    a.tax_rate = get_effective_tax_rate(ticker)

    # ── Cost of debt ──────────────────────────────────────────────────────────
    int_exp = abs(_safe_float(ltm.get("interest_expense"), np.nan))
    total_debt = _safe_float(bal.get("debt"), np.nan)
    if not np.isnan(int_exp) and not np.isnan(total_debt) and total_debt > 0:
        cod = int_exp / total_debt
        a.cost_of_debt_pretax = round(float(np.clip(cod, 0.01, 0.20)), 4)
    else:
        a.cost_of_debt_pretax = 0.055

    # ── Capital structure weights ─────────────────────────────────────────────
    mkt_cap = _safe_float(meta.get("market_cap"), np.nan)
    if not np.isnan(total_debt) and not np.isnan(mkt_cap) and (total_debt + mkt_cap) > 0:
        wd = total_debt / (total_debt + mkt_cap)
        a.debt_weight = round(float(np.clip(wd, 0.0, 0.95)), 4)
        a.equity_weight = round(1.0 - a.debt_weight, 4)
    else:
        a.debt_weight = 0.20
        a.equity_weight = 0.80

    # ── Net debt ──────────────────────────────────────────────────────────────
    cash = _safe_float(bal.get("cash"), 0.0)
    debt = _safe_float(bal.get("debt"), 0.0)
    a.net_debt = debt - cash

    # ── Shares outstanding ────────────────────────────────────────────────────
    # Market cap / current price is the most reliable proxy for diluted shares
    shares = np.nan
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        shares = _safe_float(info.get("sharesOutstanding"))
        if np.isnan(shares):
            price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
            if not np.isnan(mkt_cap) and not np.isnan(price) and price > 0:
                shares = mkt_cap / price
    except Exception:
        pass

    # Fallback: derive from market cap and a rough price estimate
    if np.isnan(shares) and not np.isnan(mkt_cap):
        # If we can get price from summary context, use it; else leave as nan
        pass

    a.shares_outstanding = shares

    # ── Beta (SEC-primary hierarchy) ─────────────────────────────────────────
    equity_book = _safe_float(bal.get("equity"), np.nan)
    beta_result = compute_beta(
        ticker=ticker,
        sic_code=sic_code,
        debt=debt,
        equity=equity_book,
        tax_rate=a.tax_rate,
        facts=facts,
    )
    a.beta = beta_result.beta

    return a, beta_result
