import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sec_engine.sec_fetch import fetch_company_submissions
from sec_engine.metrics import (
    safe_divide,
    margins, ebitda_margin, sga_margin, rd_margin, lfcf_margin, ufcf_margin,
    capex_as_pct_revenue,
    roa, roic, roe, rce,
    total_asset_turnover, accounts_receivable_turnover, inventory_turnover,
    current_ratio, quick_ratio,
    days_sales_outstanding, days_inventory_outstanding, days_payable_outstanding,
    cash_conversion_cycle,
    total_debt_to_equity, total_debt_to_capital,
    lt_debt_to_equity, lt_debt_to_capital,
    total_liabilities_to_assets,
    ebit_to_interest, ebitda_to_interest, total_debt_to_interest, net_debt_to_interest,
    altman_z_score,
    fcf, fcf_yield,
    yoy_growth, series_cagr, cagr,
    peg_pe_ltm, peg_lynch,
)
from sec_engine.constants import get_effective_tax_rate

# ── Peer override registry lives in peer_finder to avoid circular imports.
# capital_iq_style_peer_finder imports get_peer_override from peer_finder,
# and peer_finder must not import from aggregation.  Re-export here so that
# callers who used to do `from sec_engine.aggregation import set_peer_override`
# continue to work unchanged.
from sec_engine.peer_finder import (
    PEER_OVERRIDES,
    set_peer_override,
    clear_peer_override,
    list_peer_overrides,
    get_peer_override,
    _parse_market_cap,
    get_company_sic,
    build_sic_map,
    find_peers_by_sic,
)


# ─────────────────────────────────────────────────────────────────────────────
# build_company_summary
# ─────────────────────────────────────────────────────────────────────────────

def _safe_series_cagr(series: Optional[pd.Series], years: int) -> float:
    """Return CAGR for a series, or NaN if the series is None/too short."""
    if series is None or not isinstance(series, pd.Series) or len(series) < years + 1:
        return np.nan
    return series_cagr(series, years)


def _safe_yoy(series: Optional[pd.Series]) -> float:
    """Return most-recent YoY growth for an annual series, or NaN."""
    if series is None or not isinstance(series, pd.Series) or len(series) < 2:
        return np.nan
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan
    return yoy_growth(float(clean.iloc[-1]), float(clean.iloc[-2])) * 100


def _pct(value: float) -> float:
    """Multiply by 100 only if the value is not NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return value * 100


def build_company_summary(
    ticker: str,
    ltm_data: dict,
    balance_data: dict,
    metadata: dict,
    revenue_history: Optional[pd.Series] = None,
    lfcf_history: Optional[pd.Series] = None,
    gross_profit_history: Optional[pd.Series] = None,
    ebit_history: Optional[pd.Series] = None,
    ebitda_history: Optional[pd.Series] = None,
    net_income_history: Optional[pd.Series] = None,
    eps_history: Optional[pd.Series] = None,
    diluted_eps_history: Optional[pd.Series] = None,
    ar_history: Optional[pd.Series] = None,
    inventory_history: Optional[pd.Series] = None,
    ppe_history: Optional[pd.Series] = None,
    total_assets_history: Optional[pd.Series] = None,
    total_liabilities_history: Optional[pd.Series] = None,
    equity_history: Optional[pd.Series] = None,
    capex_from_yfinance: bool = True,
) -> dict:
    """
    Compute a flat dict of screener / tearsheet metrics for a single company.

    Parameters
    ----------
    ticker               : Ticker symbol (used for tax-rate lookup).
    ltm_data             : LTM flow metrics keyed by snake_case name.
    balance_data         : Latest-quarter balance sheet keyed by snake_case name.
    metadata             : Company metadata (name, market_cap, pe_ltm, etc.).
    *_history            : Optional annual pd.Series (index = fiscal year-end date)
                           for CAGR and YoY growth calculations.
    capex_from_yfinance  : When True, capex in ltm_data is a negative outflow
                           (yfinance convention) and will be abs()'d here.
                           When False (SEC source), capex is already a positive
                           outflow and no sign flip is performed.

    Returns
    -------
    dict with 100+ keys used by app.py to build the screener DataFrame and
    feed the Tearsheet, Multiples, and Ratios pages.
    """

    # ── Unpack LTM data ───────────────────────────────────────────────────────
    revenue          = ltm_data.get("revenue", np.nan)
    gross_profit     = ltm_data.get("gross_profit", np.nan)
    operating_income = ltm_data.get("operating_income", np.nan)
    ebit             = operating_income   # alias
    net_income       = ltm_data.get("net_income", np.nan)
    ebitda           = ltm_data.get("ebitda", np.nan)
    ocf              = ltm_data.get("ocf", np.nan)
    capex_raw        = ltm_data.get("capex", np.nan)
    depreciation     = ltm_data.get("depreciation", np.nan)
    amortization     = ltm_data.get("amortization", np.nan)
    sga              = ltm_data.get("sga", np.nan)
    rd               = ltm_data.get("rd", np.nan)
    cogs             = ltm_data.get("cogs", np.nan)
    interest_expense = ltm_data.get("interest_expense", np.nan)

    # Normalise capex to a positive number regardless of source sign convention.
    # yfinance reports it as a negative cash outflow; SEC tags report it positive.
    if not pd.isna(capex_raw):
        capex = abs(float(capex_raw)) if capex_from_yfinance else float(capex_raw)
    else:
        capex = np.nan

    # Derived EBITDA fallback (operating_income + D&A)
    if pd.isna(ebitda) and not pd.isna(ebit):
        da = 0.0
        if not pd.isna(depreciation):
            da += float(depreciation)
        if not pd.isna(amortization):
            da += float(amortization)
        ebitda = float(ebit) + da if da > 0 else np.nan

    # ── Unpack balance sheet data ─────────────────────────────────────────────
    total_assets       = balance_data.get("total_assets", np.nan)
    current_assets     = balance_data.get("current_assets", np.nan)
    cash               = balance_data.get("cash", np.nan)
    accounts_receivable = balance_data.get("accounts_receivable", np.nan)
    inventory_val      = balance_data.get("inventory", np.nan)
    ppe                = balance_data.get("ppe", np.nan)
    total_liabilities  = balance_data.get("total_liabilities", np.nan)
    current_liabilities = balance_data.get("current_liabilities", np.nan)
    accounts_payable   = balance_data.get("accounts_payable", np.nan)
    long_term_debt     = balance_data.get("long_term_debt", np.nan)
    total_debt         = balance_data.get("debt", np.nan)
    equity             = balance_data.get("equity", np.nan)
    retained_earnings  = balance_data.get("retained_earnings", np.nan)

    # ── Derived balance sheet items ───────────────────────────────────────────
    net_debt = (
        float(total_debt) - float(cash)
        if not pd.isna(total_debt) and not pd.isna(cash)
        else np.nan
    )

    working_capital = (
        float(current_assets) - float(current_liabilities)
        if not pd.isna(current_assets) and not pd.isna(current_liabilities)
        else np.nan
    )

    # Invested capital = equity + total debt - cash
    if not pd.isna(equity) and not pd.isna(total_debt) and not pd.isna(cash):
        invested_capital = float(equity) + float(total_debt) - float(cash)
    elif not pd.isna(equity) and not pd.isna(total_debt):
        # Cash not available — omit rather than substitute long_term_debt,
        # which would understate IC by ignoring short-term borrowings.
        invested_capital = float(equity) + float(total_debt)
    else:
        invested_capital = np.nan

    # Capital employed = total assets - current liabilities
    capital_employed = (
        float(total_assets) - float(current_liabilities)
        if not pd.isna(total_assets) and not pd.isna(current_liabilities)
        else np.nan
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    name         = metadata.get("name", ticker)
    industry     = metadata.get("industry", "Unknown")
    market_cap   = metadata.get("market_cap", np.nan)
    pe_ltm       = metadata.get("pe_ltm", np.nan)
    eps_growth_pct = metadata.get("eps_growth_pct", np.nan)   # already *100
    dividend_yield_pct = metadata.get("dividend_yield_pct", np.nan)  # already *100

    market_cap_m = float(market_cap) / 1e6 if not pd.isna(market_cap) and market_cap else np.nan

    # Enterprise value = market cap + total debt - cash
    if not pd.isna(market_cap) and not pd.isna(total_debt) and not pd.isna(cash):
        enterprise_value = float(market_cap) + float(total_debt) - float(cash)
    elif not pd.isna(market_cap):
        enterprise_value = float(market_cap)
    else:
        enterprise_value = np.nan

    # ── Tax rate ─────────────────────────────────────────────────────────────
    tax_rate = get_effective_tax_rate(ticker)

    # ── Free cash flow ───────────────────────────────────────────────────────
    lfcf_val = fcf(ocf, capex)  # levered FCF = OCF - CapEx

    # Unlevered FCF = LFCF + interest_expense × (1 − tax_rate)
    # interest_expense must be a *positive* expense value here.
    # The SEC path already stores it positive (normalize.py tags report the
    # absolute expense).  The yfinance path abs()'s it at ingestion in app.py.
    # We abs() once more as a belt-and-suspenders guard so sign differences in
    # either upstream path never silently subtract interest instead of adding it.
    if not pd.isna(lfcf_val) and not pd.isna(interest_expense):
        ufcf_val = lfcf_val + abs(float(interest_expense)) * (1 - tax_rate)
    else:
        ufcf_val = lfcf_val

    # ── NOPAT & ROIC ─────────────────────────────────────────────────────────
    nopat = float(ebit) * (1 - tax_rate) if not pd.isna(ebit) else np.nan

    # ── Margins ──────────────────────────────────────────────────────────────
    margin_dict     = margins(revenue, gross_profit, ebit, net_income)
    gross_margin    = margin_dict["Gross Margin %"]
    ebit_margin     = margin_dict["EBIT Margin %"]
    net_margin      = margin_dict["Net Margin %"]
    ebitda_margin_v = ebitda_margin(revenue, ebitda)
    sga_margin_v    = sga_margin(revenue, sga)
    rd_margin_v     = rd_margin(revenue, rd)
    lfcf_margin_v   = lfcf_margin(revenue, lfcf_val)
    ufcf_margin_v   = ufcf_margin(revenue, ufcf_val)
    capex_pct_rev   = capex_as_pct_revenue(revenue, capex)

    # ── Profitability ratios ──────────────────────────────────────────────────
    roa_v  = roa(net_income, total_assets)
    roic_v = roic(nopat, invested_capital)
    roe_v  = roe(net_income, equity)
    rce_v  = rce(ebit, capital_employed)

    # ── Asset turnover ───────────────────────────────────────────────────────
    tat_v  = total_asset_turnover(revenue, total_assets)
    art_v  = accounts_receivable_turnover(revenue, accounts_receivable)
    invt_v = inventory_turnover(cogs, inventory_val)

    # ── Liquidity ────────────────────────────────────────────────────────────
    cr_v  = current_ratio(current_assets, current_liabilities)
    qr_v  = quick_ratio(current_assets, inventory_val, current_liabilities)
    dso_v = days_sales_outstanding(accounts_receivable, revenue)
    dio_v = days_inventory_outstanding(inventory_val, cogs)
    dpo_v = days_payable_outstanding(accounts_payable, cogs)
    ccc_v = cash_conversion_cycle(dso_v, dio_v, dpo_v)

    # ── Leverage ─────────────────────────────────────────────────────────────
    tde_v  = total_debt_to_equity(total_debt, equity)
    tdc_v  = total_debt_to_capital(total_debt, equity)
    ltde_v = lt_debt_to_equity(long_term_debt, equity)
    ltdc_v = lt_debt_to_capital(long_term_debt, equity)
    tla_v  = total_liabilities_to_assets(total_liabilities, total_assets)
    ebit_int_v   = ebit_to_interest(ebit, interest_expense)
    ebitda_int_v = ebitda_to_interest(ebitda, interest_expense)
    td_int_v     = total_debt_to_interest(total_debt, interest_expense)
    nd_int_v     = net_debt_to_interest(net_debt, interest_expense)

    # Altman Z-Score (uses market-cap for X4, not book equity)
    z_score = altman_z_score(
        working_capital, total_assets, retained_earnings, ebit,
        market_cap if not pd.isna(market_cap) else np.nan,
        total_liabilities, revenue
    )

    # ── Valuation multiples ───────────────────────────────────────────────────
    ev_revenue  = safe_divide(enterprise_value, revenue)
    ev_ebitda   = safe_divide(enterprise_value, ebitda)
    ev_ebit     = safe_divide(enterprise_value, ebit)
    fcf_yield_v = fcf_yield(lfcf_val, market_cap)

    peg_v      = peg_pe_ltm(pe_ltm, eps_growth_pct)
    peg_lynch_v = peg_lynch(pe_ltm, eps_growth_pct, dividend_yield_pct)

    # ── Historical growth (YoY) ───────────────────────────────────────────────
    rev_yoy       = _safe_yoy(revenue_history)
    gp_yoy        = _safe_yoy(gross_profit_history)
    ebit_yoy      = _safe_yoy(ebit_history)
    ebitda_yoy    = _safe_yoy(ebitda_history)
    ni_yoy        = _safe_yoy(net_income_history)
    eps_yoy       = _safe_yoy(diluted_eps_history)
    dileps_yoy    = _safe_yoy(diluted_eps_history)
    ar_yoy        = _safe_yoy(ar_history)
    inv_yoy       = _safe_yoy(inventory_history)
    ppe_yoy       = _safe_yoy(ppe_history)
    ta_yoy        = _safe_yoy(total_assets_history)
    tl_yoy        = _safe_yoy(total_liabilities_history)
    eq_yoy        = _safe_yoy(equity_history)

    # ── CAGR helper ──────────────────────────────────────────────────────────
    def _cagr_pct(series, years):
        v = _safe_series_cagr(series, years)
        return v * 100 if not np.isnan(v) else np.nan

    # Revenue 2/3yr CAGRs
    rev_2y  = _cagr_pct(revenue_history, 2)
    rev_3y  = _cagr_pct(revenue_history, 3)
    rev_5y  = _cagr_pct(revenue_history, 5)

    # LFCF 3yr CAGR
    lfcf_3y = _cagr_pct(lfcf_history, 3)
    lfcf_2y = _cagr_pct(lfcf_history, 2)
    lfcf_5y = _cagr_pct(lfcf_history, 5)

    # Other 2yr CAGRs
    gp_2y    = _cagr_pct(gross_profit_history, 2)
    ebit_2y  = _cagr_pct(ebit_history, 2)
    ebd_2y   = _cagr_pct(ebitda_history, 2)
    ni_2y    = _cagr_pct(net_income_history, 2)
    eps_2y   = _cagr_pct(eps_history, 2)
    dil_2y   = _cagr_pct(diluted_eps_history, 2)
    ar_2y    = _cagr_pct(ar_history, 2)
    inv_2y   = _cagr_pct(inventory_history, 2)
    ppe_2y   = _cagr_pct(ppe_history, 2)
    ta_2y    = _cagr_pct(total_assets_history, 2)
    tl_2y    = _cagr_pct(total_liabilities_history, 2)
    eq_2y    = _cagr_pct(equity_history, 2)

    # Other 3yr CAGRs
    gp_3y    = _cagr_pct(gross_profit_history, 3)
    ebit_3y  = _cagr_pct(ebit_history, 3)
    ebd_3y   = _cagr_pct(ebitda_history, 3)
    ni_3y    = _cagr_pct(net_income_history, 3)
    eps_3y   = _cagr_pct(eps_history, 3)
    dil_3y   = _cagr_pct(diluted_eps_history, 3)
    ar_3y    = _cagr_pct(ar_history, 3)
    inv_3y   = _cagr_pct(inventory_history, 3)
    ppe_3y   = _cagr_pct(ppe_history, 3)
    ta_3y    = _cagr_pct(total_assets_history, 3)
    tl_3y    = _cagr_pct(total_liabilities_history, 3)
    eq_3y    = _cagr_pct(equity_history, 3)

    # Other 5yr CAGRs
    gp_5y    = _cagr_pct(gross_profit_history, 5)
    ebit_5y  = _cagr_pct(ebit_history, 5)
    ebd_5y   = _cagr_pct(ebitda_history, 5)
    ni_5y    = _cagr_pct(net_income_history, 5)
    eps_5y   = _cagr_pct(eps_history, 5)
    dil_5y   = _cagr_pct(diluted_eps_history, 5)
    ar_5y    = _cagr_pct(ar_history, 5)
    inv_5y   = _cagr_pct(inventory_history, 5)
    ppe_5y   = _cagr_pct(ppe_history, 5)
    ta_5y    = _cagr_pct(total_assets_history, 5)
    tl_5y    = _cagr_pct(total_liabilities_history, 5)
    eq_5y    = _cagr_pct(equity_history, 5)

    # Inventory accelerating CAGR (5yr vs 3yr — positive means acceleration)
    inv_5y_ac = np.nan
    if not np.isnan(inv_5y) and not np.isnan(inv_3y):
        inv_5y_ac = inv_5y - inv_3y

    # ── Assemble output dict ──────────────────────────────────────────────────
    summary = {
        # ── Identity ─────────────────────────────────────────────────────────
        "Ticker":          ticker,
        "Company":         name,
        "Industry":        industry,
        "Market Cap (M)":  market_cap_m,
        "Data As Of":      pd.Timestamp.now().strftime("%Y-%m-%d"),

        # ── LTM financials (raw, in $) ───────────────────────────────────────
        "Revenue":          revenue,
        "Gross Profit":     gross_profit,
        "EBIT":             ebit,
        "EBITDA":           ebitda,
        "Net Income":       net_income,
        "OCF":              ocf,
        "CapEx":            capex,
        "LFCF":             lfcf_val,
        "UFCF":             ufcf_val,
        "Interest Expense": interest_expense,
        "Total Debt":       total_debt,
        "Net Debt":         net_debt,
        "Cash":             cash,

        # ── Margins ──────────────────────────────────────────────────────────
        "Gross Margin %":    gross_margin,
        "EBIT Margin %":     ebit_margin,
        "EBITDA Margin %":   ebitda_margin_v,
        "Net Margin %":      net_margin,
        "SG&A Margin %":     sga_margin_v,
        "R&D Margin %":      rd_margin_v,
        "LFCF Margin %":     lfcf_margin_v,
        "UFCF Margin %":     ufcf_margin_v,
        "CapEx % Revenue":   capex_pct_rev,

        # ── Profitability ─────────────────────────────────────────────────────
        "ROA %":   roa_v,
        "ROIC %":  roic_v,
        "ROE %":   roe_v,
        "RCE %":   rce_v,

        # ── Turnover ─────────────────────────────────────────────────────────
        "Total Asset Turnover":   tat_v,
        "AR Turnover":            art_v,
        "Inventory Turnover":     invt_v,

        # ── Liquidity ─────────────────────────────────────────────────────────
        "Current Ratio":                  cr_v,
        "Quick Ratio":                    qr_v,
        "Avg Days Sales Outstanding":     dso_v,
        "Avg Days Inventory Outstanding": dio_v,
        "Avg Days Payable Outstanding":   dpo_v,
        "Cash Conversion Cycle":          ccc_v,

        # ── Leverage ─────────────────────────────────────────────────────────
        "Total D/E":         tde_v,
        "Total D/Capital":   tdc_v,
        "LT D/E":            ltde_v,
        "LT D/Capital":      ltdc_v,
        "Total Liab/Assets": tla_v,
        "EBIT/Interest":     ebit_int_v,
        "EBITDA/Interest":   ebitda_int_v,
        "Total Debt/Interest": td_int_v,
        "Net Debt/Interest":   nd_int_v,
        "Altman Z-Score":    z_score,

        # ── Valuation ────────────────────────────────────────────────────────
        "PE LTM":        pe_ltm,
        "EV/Revenue":    ev_revenue,
        "EV/EBITDA":     ev_ebitda,
        "EV/EBIT":       ev_ebit,
        "FCF Yield %":   fcf_yield_v,
        "PEG (PE LTM)":  peg_v,
        "PEG (Lynch)":   peg_lynch_v,

        # ── YoY growth ───────────────────────────────────────────────────────
        "Revenue YoY %":           rev_yoy,
        "Gross Profit YoY %":      gp_yoy,
        "EBIT YoY %":              ebit_yoy,
        "EBITDA YoY %":            ebitda_yoy,
        "Net Income YoY %":        ni_yoy,
        "EPS YoY %":               eps_yoy,
        "Diluted EPS YoY %":       dileps_yoy,
        "AR YoY %":                ar_yoy,
        "Inventory YoY %":         inv_yoy,
        "Net PP&E YoY %":          ppe_yoy,
        "Total Assets YoY %":      ta_yoy,
        "Total Liabilities YoY %": tl_yoy,
        "Total Equity YoY %":      eq_yoy,

        # ── 2yr CAGRs ─────────────────────────────────────────────────────────
        "Revenue 2yr CAGR %":           rev_2y,
        "Gross Profit 2yr CAGR %":      gp_2y,
        "EBIT 2yr CAGR %":              ebit_2y,
        "EBITDA 2yr CAGR %":            ebd_2y,
        "Net Income 2yr CAGR %":        ni_2y,
        "EPS 2yr CAGR %":               eps_2y,
        "Diluted EPS 2yr CAGR %":       dil_2y,
        "AR 2yr CAGR %":                ar_2y,
        "Inventory 2yr CAGR %":         inv_2y,
        "Net PP&E 2yr CAGR %":          ppe_2y,
        "Total Assets 2yr CAGR %":      ta_2y,
        "Total Liabilities 2yr CAGR %": tl_2y,
        "Total Equity 2yr CAGR %":      eq_2y,
        "LFCF 2yr CAGR %":              lfcf_2y,

        # ── 3yr CAGRs ─────────────────────────────────────────────────────────
        "Revenue 3yr CAGR %":           rev_3y,
        "Gross Profit 3yr CAGR %":      gp_3y,
        "EBIT 3yr CAGR %":              ebit_3y,
        "EBITDA 3yr CAGR %":            ebd_3y,
        "Net Income 3yr CAGR %":        ni_3y,
        "EPS 3yr CAGR %":               eps_3y,
        "Diluted EPS 3yr CAGR %":       dil_3y,
        "AR 3yr CAGR %":                ar_3y,
        "Inventory 3yr CAGR %":         inv_3y,
        "Net PP&E 3yr CAGR %":          ppe_3y,
        "Total Assets 3yr CAGR %":      ta_3y,
        "Total Liabilities 3yr CAGR %": tl_3y,
        "Total Equity 3yr CAGR %":      eq_3y,
        "LFCF 3yr CAGR %":              lfcf_3y,

        # ── 5yr CAGRs ─────────────────────────────────────────────────────────
        "Revenue 5yr CAGR %":           rev_5y,
        "Gross Profit 5yr CAGR %":      gp_5y,
        "EBIT 5yr CAGR %":              ebit_5y,
        "EBITDA 5yr CAGR %":            ebd_5y,
        "Net Income 5yr CAGR %":        ni_5y,
        "EPS 5yr CAGR %":               eps_5y,
        "Diluted EPS 5yr CAGR %":       dil_5y,
        "AR 5yr CAGR %":                ar_5y,
        "Inventory 5yr CAGR %":         inv_5y,
        "Inventory 5yr acCAGR %":        inv_5y_ac,
        "Net PP&E 5yr CAGR %":          ppe_5y,
        "Total Assets 5yr CAGR %":      ta_5y,
        "Total Liabilities 5yr CAGR %": tl_5y,
        "Total Equity 5yr CAGR %":      eq_5y,
        "LFCF 5yr CAGR %":              lfcf_5y,
    }

    return summary
