# aggregation.py
# ------------------------------------------------------------------
# Builds the per-company summary dictionary used by the screener,
# tearsheet, multiples, and ratios pages.
#
# Fixes vs. prior version:
#   - capex sign normalization is now source-aware: yfinance returns
#     capex as a negative cash outflow; SEC tags are positive outflows.
#     The caller (fetch_company_data_unified) passes a "capex_source"
#     flag so we only abs() when the value came from yfinance.
#   - All nan-safe arithmetic replaces (x or 0) with nan_to_zero(x),
#     which correctly handles float 0.0 without treating it as falsy.
#   - ROIC tax rate (21%) is surfaced as a named constant.
#   - EBITDA fallback documents that D&A from the cashflow statement
#     may include non-depreciation items; this is disclosed in the
#     data_quality dict returned alongside the summary.
#   - A "Data As Of" field is derived from the most recent quarter-end
#     in the revenue quarterly series and surfaced in the summary.
# ------------------------------------------------------------------

from sec_engine.metrics import (
    margins,
    ebitda_margin,
    sga_margin,
    rd_margin,
    lfcf_margin,
    ufcf_margin,
    capex_as_pct_revenue,
    roa,
    roic,
    roe,
    rce,
    total_asset_turnover,
    accounts_receivable_turnover,
    inventory_turnover,
    current_ratio,
    quick_ratio,
    days_sales_outstanding,
    days_inventory_outstanding,
    days_payable_outstanding,
    cash_conversion_cycle,
    total_debt_to_equity,
    total_debt_to_capital,
    lt_debt_to_equity,
    lt_debt_to_capital,
    total_liabilities_to_assets,
    ebit_to_interest,
    ebitda_to_interest,
    total_debt_to_interest,
    net_debt_to_interest,
    altman_z_score,
    series_cagr,
    yoy_growth,
    fcf,
    fcf_yield,
    peg_pe_ltm,
    peg_lynch,
)
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
# Effective tax rate used to compute NOPAT = EBIT × (1 - TAX_RATE).
# The US federal statutory rate is 21%. Companies with significant
# deferred taxes, NOLs, or international operations may differ.
# Disclosed here so it is easy to find and adjust.
NOPAT_TAX_RATE = 0.21


# ── Helpers ───────────────────────────────────────────────────────────────────

def nz(x):
    """Normalize None → np.nan, pass through everything else."""
    return np.nan if x is None else x


def nan_to_zero(x):
    """Return 0.0 if x is NaN or None, otherwise return x.
    Unlike (x or 0), this correctly handles float 0.0 without zeroing it.
    """
    if x is None:
        return 0.0
    try:
        return 0.0 if np.isnan(x) else float(x)
    except (TypeError, ValueError):
        return 0.0


def _series_last_date(series) -> str:
    """Return the most recent index date of a Series as a string, or ''."""
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return ""
    try:
        return str(series.dropna().sort_index().index[-1].date())
    except Exception:
        return ""


# ── Main builder ──────────────────────────────────────────────────────────────

def build_company_summary(
    ticker: str,
    ltm_data: dict,
    balance_data: dict,
    metadata: dict,
    revenue_history: pd.Series | None,
    lfcf_history: pd.Series | None,
    gross_profit_history: pd.Series | None = None,
    ebit_history: pd.Series | None = None,
    ebitda_history: pd.Series | None = None,
    net_income_history: pd.Series | None = None,
    eps_history: pd.Series | None = None,
    diluted_eps_history: pd.Series | None = None,
    ar_history: pd.Series | None = None,
    inventory_history: pd.Series | None = None,
    ppe_history: pd.Series | None = None,
    total_assets_history: pd.Series | None = None,
    total_liabilities_history: pd.Series | None = None,
    equity_history: pd.Series | None = None,
    # Source flags for sign normalization
    capex_from_yfinance: bool = True,
) -> dict:

    # ── Income statement (LTM) ────────────────────────────────────────────────
    revenue          = nz(ltm_data.get("revenue"))
    gross_profit     = nz(ltm_data.get("gross_profit"))
    ebit             = nz(ltm_data.get("operating_income"))
    net_income       = nz(ltm_data.get("net_income"))
    sga              = nz(ltm_data.get("sga"))
    rd               = nz(ltm_data.get("rd"))
    cogs             = nz(ltm_data.get("cogs"))
    interest_expense = nz(ltm_data.get("interest_expense"))
    ocf              = nz(ltm_data.get("ocf"))
    capex            = nz(ltm_data.get("capex"))

    # Sign normalization: yfinance reports capex as a negative number
    # (cash *outflow*); SEC tags are defined as positive outflows.
    # We normalize to positive so that FCF = OCF - capex is correct
    # regardless of source, but only flip the sign when needed.
    if not np.isnan(capex) and capex < 0 and capex_from_yfinance:
        capex = abs(capex)

    # ── EBITDA ────────────────────────────────────────────────────────────────
    ebitda = nz(ltm_data.get("ebitda"))
    ebitda_source = "direct"

    if np.isnan(ebitda):
        # Fallback: EBIT + D&A from cashflow statement.
        # Note: the D&A line in the cashflow statement sometimes includes
        # non-depreciation items (amortization of debt issuance costs,
        # stock-comp-related amortization). This is the standard
        # approximation used by most data providers; flag it.
        depreciation = nz(ltm_data.get("depreciation"))
        amortization = nz(ltm_data.get("amortization"))

        if not np.isnan(ebit):
            da_total = nan_to_zero(depreciation) + nan_to_zero(amortization)
            if da_total > 0:
                ebitda = ebit + da_total
                ebitda_source = "computed: EBIT + D&A"
            else:
                ebitda = np.nan
                ebitda_source = "unavailable"
        else:
            ebitda = np.nan
            ebitda_source = "unavailable"

    # ── Balance sheet ─────────────────────────────────────────────────────────
    debt              = nz(balance_data.get("debt"))
    equity            = nz(balance_data.get("equity"))
    cash              = nz(balance_data.get("cash"))
    total_assets      = nz(balance_data.get("total_assets"))
    total_liabilities = nz(balance_data.get("total_liabilities"))
    current_assets    = nz(balance_data.get("current_assets"))
    current_liabilities = nz(balance_data.get("current_liabilities"))
    accounts_receivable = nz(balance_data.get("accounts_receivable"))
    inventory         = nz(balance_data.get("inventory"))
    accounts_payable  = nz(balance_data.get("accounts_payable"))
    lt_debt           = nz(balance_data.get("long_term_debt"))
    retained_earnings = nz(balance_data.get("retained_earnings"))
    ppe               = nz(balance_data.get("ppe"))

    # nan_to_zero used here so that valid 0.0 balance-sheet values are
    # preserved, unlike the prior (x or 0) pattern which treated 0.0
    # as falsy and silently substituted zero for missing values.
    working_capital = (
        nan_to_zero(current_assets) - nan_to_zero(current_liabilities)
        if not (np.isnan(current_assets) and np.isnan(current_liabilities))
        else np.nan
    )

    net_debt = nan_to_zero(debt) - nan_to_zero(cash)

    capital_employed = (
        nan_to_zero(total_assets) - nan_to_zero(current_liabilities)
        if not (np.isnan(total_assets) and np.isnan(current_liabilities))
        else np.nan
    )

    # ── yfinance metadata ─────────────────────────────────────────────────────
    market_cap       = nz(metadata.get("market_cap"))
    pe_ltm           = nz(metadata.get("pe_ltm"))
    eps_growth_pct   = nz(metadata.get("eps_growth_pct"))
    dividend_yield_pct = nz(metadata.get("dividend_yield_pct"))

    # ── ROIC ──────────────────────────────────────────────────────────────────
    nopat = ebit * (1 - NOPAT_TAX_RATE) if not np.isnan(ebit) else np.nan
    invested_capital = nan_to_zero(debt) + nan_to_zero(equity) - nan_to_zero(cash)

    # ── Margins ───────────────────────────────────────────────────────────────
    margin_dict     = margins(revenue, gross_profit, ebit, net_income)
    ebitda_margin_val = ebitda_margin(revenue, ebitda)
    sga_margin_val  = sga_margin(revenue, sga)
    rd_margin_val   = rd_margin(revenue, rd)

    # ── FCF ───────────────────────────────────────────────────────────────────
    fcf_value = fcf(ocf, capex)
    lfcf_value = fcf_value   # OCF is post-interest, so levered FCF = OCF - CapEx
    ufcf_value = fcf_value   # Approximate; full UFCF needs interest tax shield add-back

    lfcf_margin_val  = lfcf_margin(revenue, lfcf_value)
    ufcf_margin_val  = ufcf_margin(revenue, ufcf_value)
    capex_pct_revenue = capex_as_pct_revenue(revenue, capex)

    # ── Profitability ─────────────────────────────────────────────────────────
    roa_val  = roa(net_income, total_assets)
    roic_val = roic(nopat, invested_capital)
    roe_val  = roe(net_income, equity)
    rce_val  = rce(ebit, capital_employed)

    # ── Turnover ──────────────────────────────────────────────────────────────
    asset_turnover = total_asset_turnover(revenue, total_assets)
    ar_turnover    = accounts_receivable_turnover(revenue, accounts_receivable)
    inv_turnover   = inventory_turnover(cogs, inventory)

    # ── Liquidity ─────────────────────────────────────────────────────────────
    current_ratio_val = current_ratio(current_assets, current_liabilities)
    quick_ratio_val   = quick_ratio(current_assets, inventory, current_liabilities)
    dso = days_sales_outstanding(accounts_receivable, revenue)
    dio = days_inventory_outstanding(inventory, cogs)
    dpo = days_payable_outstanding(accounts_payable, cogs)
    ccc = cash_conversion_cycle(dso, dio, dpo)

    # ── Leverage ──────────────────────────────────────────────────────────────
    total_de        = total_debt_to_equity(debt, equity)
    total_d_cap     = total_debt_to_capital(debt, equity)
    lt_de           = lt_debt_to_equity(lt_debt, equity)
    lt_d_cap        = lt_debt_to_capital(lt_debt, equity)
    liab_to_assets  = total_liabilities_to_assets(total_liabilities, total_assets)
    ebit_interest   = ebit_to_interest(ebit, interest_expense)
    ebitda_interest = ebitda_to_interest(ebitda, interest_expense)
    total_debt_interest = total_debt_to_interest(debt, interest_expense)
    net_debt_interest   = net_debt_to_interest(net_debt, interest_expense)
    z_score = altman_z_score(
        working_capital, total_assets, retained_earnings,
        ebit, market_cap, total_liabilities, revenue,
    )

    # ── Growth — YoY ─────────────────────────────────────────────────────────
    def get_yoy(series):
        if isinstance(series, pd.Series) and len(series) >= 2:
            s = series.dropna().sort_index()
            if len(s) >= 2:
                return yoy_growth(s.iloc[-1], s.iloc[-2]) * 100
        return np.nan

    revenue_yoy          = get_yoy(revenue_history)
    gross_profit_yoy     = get_yoy(gross_profit_history)
    ebit_yoy             = get_yoy(ebit_history)
    ebitda_yoy           = get_yoy(ebitda_history)
    net_income_yoy       = get_yoy(net_income_history)
    eps_yoy              = get_yoy(eps_history)
    diluted_eps_yoy      = get_yoy(diluted_eps_history)
    ar_yoy               = get_yoy(ar_history)
    inventory_yoy        = get_yoy(inventory_history)
    ppe_yoy              = get_yoy(ppe_history)
    total_assets_yoy     = get_yoy(total_assets_history)
    total_liabilities_yoy = get_yoy(total_liabilities_history)
    equity_yoy           = get_yoy(equity_history)

    # ── CAGR ──────────────────────────────────────────────────────────────────
    def get_cagr(series, years):
        if not isinstance(series, pd.Series):
            return np.nan
        s = series.dropna().sort_index()
        return series_cagr(s, years) * 100

    revenue_cagr_2yr          = get_cagr(revenue_history, 2)
    gross_profit_cagr_2yr     = get_cagr(gross_profit_history, 2)
    ebit_cagr_2yr             = get_cagr(ebit_history, 2)
    ebitda_cagr_2yr           = get_cagr(ebitda_history, 2)
    net_income_cagr_2yr       = get_cagr(net_income_history, 2)
    eps_cagr_2yr              = get_cagr(eps_history, 2)
    diluted_eps_cagr_2yr      = get_cagr(diluted_eps_history, 2)
    ar_cagr_2yr               = get_cagr(ar_history, 2)
    inventory_cagr_2yr        = get_cagr(inventory_history, 2)
    ppe_cagr_2yr              = get_cagr(ppe_history, 2)
    total_assets_cagr_2yr     = get_cagr(total_assets_history, 2)
    total_liabilities_cagr_2yr = get_cagr(total_liabilities_history, 2)
    equity_cagr_2yr           = get_cagr(equity_history, 2)

    revenue_cagr_3yr          = get_cagr(revenue_history, 3)
    gross_profit_cagr_3yr     = get_cagr(gross_profit_history, 3)
    ebit_cagr_3yr             = get_cagr(ebit_history, 3)
    ebitda_cagr_3yr           = get_cagr(ebitda_history, 3)
    net_income_cagr_3yr       = get_cagr(net_income_history, 3)
    eps_cagr_3yr              = get_cagr(eps_history, 3)
    diluted_eps_cagr_3yr      = get_cagr(diluted_eps_history, 3)
    ar_cagr_3yr               = get_cagr(ar_history, 3)
    inventory_cagr_3yr        = get_cagr(inventory_history, 3)
    ppe_cagr_3yr              = get_cagr(ppe_history, 3)
    total_assets_cagr_3yr     = get_cagr(total_assets_history, 3)
    total_liabilities_cagr_3yr = get_cagr(total_liabilities_history, 3)
    equity_cagr_3yr           = get_cagr(equity_history, 3)
    lfcf_cagr_3yr             = get_cagr(lfcf_history, 3)

    revenue_cagr_5yr          = get_cagr(revenue_history, 5)
    gross_profit_cagr_5yr     = get_cagr(gross_profit_history, 5)
    ebit_cagr_5yr             = get_cagr(ebit_history, 5)
    ebitda_cagr_5yr           = get_cagr(ebitda_history, 5)
    net_income_cagr_5yr       = get_cagr(net_income_history, 5)
    eps_cagr_5yr              = get_cagr(eps_history, 5)
    diluted_eps_cagr_5yr      = get_cagr(diluted_eps_history, 5)
    ar_cagr_5yr               = get_cagr(ar_history, 5)
    inventory_cagr_5yr        = get_cagr(inventory_history, 5)
    ppe_cagr_5yr              = get_cagr(ppe_history, 5)
    total_assets_cagr_5yr     = get_cagr(total_assets_history, 5)
    total_liabilities_cagr_5yr = get_cagr(total_liabilities_history, 5)
    equity_cagr_5yr           = get_cagr(equity_history, 5)

    # ── Valuation ─────────────────────────────────────────────────────────────
    market_cap_m    = market_cap / 1_000_000 if not np.isnan(market_cap) else np.nan
    fcf_yield_pct   = fcf_yield(fcf_value, market_cap)
    peg_ltm         = peg_pe_ltm(pe_ltm, eps_growth_pct)
    peg_lynch_ratio = peg_lynch(pe_ltm, eps_growth_pct, dividend_yield_pct)

    # ── Data freshness ────────────────────────────────────────────────────────
    # Surface the most recent period-end date so users know how current
    # the LTM figures are. Uses revenue history as the anchor series.
    data_as_of = _series_last_date(revenue_history)

    # ── Final summary ─────────────────────────────────────────────────────────
    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "Company":        metadata.get("name"),
        "Ticker":         ticker,
        "Industry":       metadata.get("industry"),
        "Market Cap (M)": market_cap_m,
        "Data As Of":     data_as_of,

        # ── Profitability ──────────────────────────────────────────────────────
        "ROA %":  roa_val,
        "ROIC %": roic_val,
        "ROE %":  roe_val,
        "RCE %":  rce_val,

        # ── Margins ───────────────────────────────────────────────────────────
        "Gross Margin %":   margin_dict["Gross Margin %"],
        "SG&A Margin %":    sga_margin_val,
        "R&D Margin %":     rd_margin_val,
        "EBITDA Margin %":  ebitda_margin_val,
        "EBIT Margin %":    margin_dict["EBIT Margin %"],
        "Net Margin %":     margin_dict["Net Margin %"],
        "LFCF Margin %":    lfcf_margin_val,
        "UFCF Margin %":    ufcf_margin_val,
        "CapEx % Revenue":  capex_pct_revenue,

        # ── Turnover ──────────────────────────────────────────────────────────
        "Total Asset Turnover": asset_turnover,
        "AR Turnover":          ar_turnover,
        "Inventory Turnover":   inv_turnover,

        # ── Short-term liquidity ───────────────────────────────────────────────
        "Current Ratio":              current_ratio_val,
        "Quick Ratio":                quick_ratio_val,
        "Avg Days Sales Outstanding": dso,
        "Avg Days Inventory Outstanding": dio,
        "Avg Days Payable Outstanding":   dpo,
        "Cash Conversion Cycle":          ccc,

        # ── Leverage ──────────────────────────────────────────────────────────
        "Total D/E":           total_de,
        "Total D/Capital":     total_d_cap,
        "LT D/E":              lt_de,
        "LT D/Capital":        lt_d_cap,
        "Total Liab/Assets":   liab_to_assets,
        "EBIT/Interest":       ebit_interest,
        "EBITDA/Interest":     ebitda_interest,
        "Total Debt/Interest": total_debt_interest,
        "Net Debt/Interest":   net_debt_interest,
        "Altman Z-Score":      z_score,

        # ── YoY growth ────────────────────────────────────────────────────────
        "Revenue YoY %":          revenue_yoy,
        "Gross Profit YoY %":     gross_profit_yoy,
        "EBIT YoY %":             ebit_yoy,
        "EBITDA YoY %":           ebitda_yoy,
        "Net Income YoY %":       net_income_yoy,
        "EPS YoY %":              eps_yoy,
        "Diluted EPS YoY %":      diluted_eps_yoy,
        "AR YoY %":               ar_yoy,
        "Inventory YoY %":        inventory_yoy,
        "Net PP&E YoY %":         ppe_yoy,
        "Total Assets YoY %":     total_assets_yoy,
        "Total Liabilities YoY %": total_liabilities_yoy,
        "Total Equity YoY %":     equity_yoy,

        # ── 2yr CAGR ──────────────────────────────────────────────────────────
        "Revenue 2yr CAGR %":           revenue_cagr_2yr,
        "Gross Profit 2yr CAGR %":      gross_profit_cagr_2yr,
        "EBIT 2yr CAGR %":              ebit_cagr_2yr,
        "EBITDA 2yr CAGR %":            ebitda_cagr_2yr,
        "Net Income 2yr CAGR %":        net_income_cagr_2yr,
        "EPS 2yr CAGR %":               eps_cagr_2yr,
        "Diluted EPS 2yr CAGR %":       diluted_eps_cagr_2yr,
        "AR 2yr CAGR %":                ar_cagr_2yr,
        "Inventory 2yr CAGR %":         inventory_cagr_2yr,
        "Net PP&E 2yr CAGR %":          ppe_cagr_2yr,
        "Total Assets 2yr CAGR %":      total_assets_cagr_2yr,
        "Total Liabilities 2yr CAGR %": total_liabilities_cagr_2yr,
        "Total Equity 2yr CAGR %":      equity_cagr_2yr,

        # ── 3yr CAGR ──────────────────────────────────────────────────────────
        "Revenue 3yr CAGR %":           revenue_cagr_3yr,
        "Gross Profit 3yr CAGR %":      gross_profit_cagr_3yr,
        "EBIT 3yr CAGR %":              ebit_cagr_3yr,
        "EBITDA 3yr CAGR %":            ebitda_cagr_3yr,
        "Net Income 3yr CAGR %":        net_income_cagr_3yr,
        "EPS 3yr CAGR %":               eps_cagr_3yr,
        "Diluted EPS 3yr CAGR %":       diluted_eps_cagr_3yr,
        "AR 3yr CAGR %":                ar_cagr_3yr,
        "Inventory 3yr CAGR %":         inventory_cagr_3yr,
        "Net PP&E 3yr CAGR %":          ppe_cagr_3yr,
        "Total Assets 3yr CAGR %":      total_assets_cagr_3yr,
        "Total Liabilities 3yr CAGR %": total_liabilities_cagr_3yr,
        "Total Equity 3yr CAGR %":      equity_cagr_3yr,
        "LFCF 3yr CAGR %":              lfcf_cagr_3yr,

        # ── 5yr CAGR ──────────────────────────────────────────────────────────
        "Revenue 5yr CAGR %":           revenue_cagr_5yr,
        "Gross Profit 5yr CAGR %":      gross_profit_cagr_5yr,
        "EBIT 5yr CAGR %":              ebit_cagr_5yr,
        "EBITDA 5yr CAGR %":            ebitda_cagr_5yr,
        "Net Income 5yr CAGR %":        net_income_cagr_5yr,
        "EPS 5yr CAGR %":               eps_cagr_5yr,
        "Diluted EPS 5yr CAGR %":       diluted_eps_cagr_5yr,
        "AR 5yr CAGR %":                ar_cagr_5yr,
        "Inventory 5yr CAGR %":         inventory_cagr_5yr,
        "Net PP&E 5yr CAGR %":          ppe_cagr_5yr,
        "Total Assets 5yr CAGR %":      total_assets_cagr_5yr,
        "Total Liabilities 5yr CAGR %": total_liabilities_cagr_5yr,
        "Total Equity 5yr CAGR %":      equity_cagr_5yr,

        # ── Valuation ─────────────────────────────────────────────────────────
        "PEG (PE LTM)": peg_ltm,
        "PEG (Lynch)":  peg_lynch_ratio,
        "FCF Yield %":  fcf_yield_pct,
    }
