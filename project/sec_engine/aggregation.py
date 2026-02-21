from sec_engine.metrics import (
    # Margins
    margins,
    ebitda_margin,
    sga_margin,
    rd_margin,
    lfcf_margin,
    ufcf_margin,
    capex_as_pct_revenue,
    # Profitability
    roa,
    roic,
    roe,
    rce,
    # Asset Turnover
    total_asset_turnover,
    accounts_receivable_turnover,
    inventory_turnover,
    # Liquidity
    current_ratio,
    quick_ratio,
    days_sales_outstanding,
    days_inventory_outstanding,
    days_payable_outstanding,
    cash_conversion_cycle,
    # Leverage
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
    # Growth
    series_cagr,
    yoy_growth,
    # Valuation
    fcf,
    fcf_yield,
    peg_pe_ltm,
    peg_lynch,
)
import numpy as np
import pandas as pd

def nz(x):
    """Normalize None → np.nan."""
    return np.nan if x is None else x

def build_company_summary(
    ticker: str,
    ltm_data: dict,
    balance_data: dict,
    metadata: dict,
    revenue_history: pd.Series | None,
    lfcf_history: pd.Series | None,
    # NEW: Additional historical series for CAGR calculations
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
) -> dict:

    # -----------------------------
    # Normalize SEC LTM values - Income Statement
    # -----------------------------
    revenue = nz(ltm_data.get("revenue"))
    gross_profit = nz(ltm_data.get("gross_profit"))
    ebit = nz(ltm_data.get("operating_income"))
    net_income = nz(ltm_data.get("net_income"))
    sga = nz(ltm_data.get("sga"))
    rd = nz(ltm_data.get("rd"))
    cogs = nz(ltm_data.get("cogs"))
    interest_expense = nz(ltm_data.get("interest_expense"))

    ocf = nz(ltm_data.get("ocf"))
    capex = nz(ltm_data.get("capex"))
    # yfinance returns capex as a negative number (cash outflow). Normalize to positive
    # so that fcf = ocf - capex works correctly regardless of source.
    if not np.isnan(capex) and capex < 0:
        capex = abs(capex)
    
    # Calculate EBITDA (EBIT + Depreciation & Amortization)
    ebitda = nz(ltm_data.get("ebitda"))
    if np.isnan(ebitda):
        depreciation = nz(ltm_data.get("depreciation"))
        amortization = nz(ltm_data.get("amortization"))
        
        if not np.isnan(ebit):
            da_total = 0
            if not np.isnan(depreciation):
                da_total += depreciation
            if not np.isnan(amortization):
                da_total += amortization
            
            if da_total > 0:
                ebitda = ebit + da_total
            else:
                ebitda = np.nan
        else:
            ebitda = np.nan

    # -----------------------------
    # Balance sheet - Current Year
    # -----------------------------
    debt = nz(balance_data.get("debt"))
    equity = nz(balance_data.get("equity"))
    cash = nz(balance_data.get("cash"))
    total_assets = nz(balance_data.get("total_assets"))
    total_liabilities = nz(balance_data.get("total_liabilities"))
    current_assets = nz(balance_data.get("current_assets"))
    current_liabilities = nz(balance_data.get("current_liabilities"))
    accounts_receivable = nz(balance_data.get("accounts_receivable"))
    inventory = nz(balance_data.get("inventory"))
    accounts_payable = nz(balance_data.get("accounts_payable"))
    lt_debt = nz(balance_data.get("long_term_debt"))
    retained_earnings = nz(balance_data.get("retained_earnings"))
    ppe = nz(balance_data.get("ppe"))  # Net PP&E

    # Working capital
    working_capital = (current_assets or 0) - (current_liabilities or 0) if current_assets and current_liabilities else np.nan

    # Net debt
    net_debt = (debt or 0) - (cash or 0)

    # -----------------------------
    # yfinance metadata
    # -----------------------------
    market_cap = nz(metadata.get("market_cap"))
    pe_ltm = nz(metadata.get("pe_ltm"))
    eps_growth_pct = nz(metadata.get("eps_growth_pct"))
    dividend_yield_pct = nz(metadata.get("dividend_yield_pct"))

    # -----------------------------
    # ROIC components
    # -----------------------------
    nopat = ebit * (1 - 0.21) if not np.isnan(ebit) else np.nan
    invested_capital = (debt or 0) + (equity or 0) - (cash or 0)
    
    # Capital Employed (Total Assets - Current Liabilities)
    capital_employed = (total_assets or 0) - (current_liabilities or 0) if total_assets and current_liabilities else np.nan

    # -----------------------------
    # Basic Margins
    # -----------------------------
    margin_dict = margins(revenue, gross_profit, ebit, net_income)
    
    # EBITDA Margin
    ebitda_margin_val = ebitda_margin(revenue, ebitda)
    
    # Additional margins
    sga_margin_val = sga_margin(revenue, sga)
    rd_margin_val = rd_margin(revenue, rd)
    
    # Free Cash Flow
    fcf_value = fcf(ocf, capex)
    ufcf_value = fcf_value  # Unlevered FCF approximation (before interest tax shield)
    # yfinance OCF is already after interest payments, so levered FCF = OCF - CapEx.
    # Do NOT subtract interest_expense again — that would double-count it.
    lfcf_value = fcf_value
    
    lfcf_margin_val = lfcf_margin(revenue, lfcf_value)
    ufcf_margin_val = ufcf_margin(revenue, ufcf_value)
    capex_pct_revenue = capex_as_pct_revenue(revenue, capex)

    # -----------------------------
    # Profitability Ratios
    # -----------------------------
    roa_val = roa(net_income, total_assets)
    roic_val = roic(nopat, invested_capital)
    roe_val = roe(net_income, equity)
    rce_val = rce(ebit, capital_employed)

    # -----------------------------
    # Asset Turnover Ratios
    # -----------------------------
    asset_turnover = total_asset_turnover(revenue, total_assets)
    ar_turnover = accounts_receivable_turnover(revenue, accounts_receivable)
    inv_turnover = inventory_turnover(cogs, inventory)

    # -----------------------------
    # Short-Term Liquidity
    # -----------------------------
    current_ratio_val = current_ratio(current_assets, current_liabilities)
    quick_ratio_val = quick_ratio(current_assets, inventory, current_liabilities)
    dso = days_sales_outstanding(accounts_receivable, revenue)
    dio = days_inventory_outstanding(inventory, cogs)
    dpo = days_payable_outstanding(accounts_payable, cogs)
    ccc = cash_conversion_cycle(dso, dio, dpo)

    # -----------------------------
    # Long-Term Liquidity (Leverage)
    # -----------------------------
    total_de = total_debt_to_equity(debt, equity)
    total_d_cap = total_debt_to_capital(debt, equity)
    lt_de = lt_debt_to_equity(lt_debt, equity)
    lt_d_cap = lt_debt_to_capital(lt_debt, equity)
    liab_to_assets = total_liabilities_to_assets(total_liabilities, total_assets)
    ebit_interest = ebit_to_interest(ebit, interest_expense)
    ebitda_interest = ebitda_to_interest(ebitda, interest_expense)
    total_debt_interest = total_debt_to_interest(debt, interest_expense)
    net_debt_interest = net_debt_to_interest(net_debt, interest_expense)
    z_score = altman_z_score(working_capital, total_assets, retained_earnings, ebit, market_cap, total_liabilities, revenue)

    # -----------------------------
    # Growth - YoY (from historical series)
    # -----------------------------
    def get_yoy(series):
        if isinstance(series, pd.Series) and len(series) >= 2:
            return yoy_growth(series.iloc[-1], series.iloc[-2]) * 100
        return np.nan

    revenue_yoy = get_yoy(revenue_history)
    gross_profit_yoy = get_yoy(gross_profit_history)
    ebit_yoy = get_yoy(ebit_history)
    ebitda_yoy = get_yoy(ebitda_history)
    net_income_yoy = get_yoy(net_income_history)
    eps_yoy = get_yoy(eps_history)
    diluted_eps_yoy = get_yoy(diluted_eps_history)
    ar_yoy = get_yoy(ar_history)
    inventory_yoy = get_yoy(inventory_history)
    ppe_yoy = get_yoy(ppe_history)
    total_assets_yoy = get_yoy(total_assets_history)
    total_liabilities_yoy = get_yoy(total_liabilities_history)
    equity_yoy = get_yoy(equity_history)

    # -----------------------------
    # CAGR - 2 Year
    # -----------------------------
    def get_cagr(series, years):
        return series_cagr(series, years) * 100 if isinstance(series, pd.Series) else np.nan

    revenue_cagr_2yr = get_cagr(revenue_history, 2)
    gross_profit_cagr_2yr = get_cagr(gross_profit_history, 2)
    ebit_cagr_2yr = get_cagr(ebit_history, 2)
    ebitda_cagr_2yr = get_cagr(ebitda_history, 2)
    net_income_cagr_2yr = get_cagr(net_income_history, 2)
    eps_cagr_2yr = get_cagr(eps_history, 2)
    diluted_eps_cagr_2yr = get_cagr(diluted_eps_history, 2)
    ar_cagr_2yr = get_cagr(ar_history, 2)
    inventory_cagr_2yr = get_cagr(inventory_history, 2)
    ppe_cagr_2yr = get_cagr(ppe_history, 2)
    total_assets_cagr_2yr = get_cagr(total_assets_history, 2)
    total_liabilities_cagr_2yr = get_cagr(total_liabilities_history, 2)
    equity_cagr_2yr = get_cagr(equity_history, 2)

    # -----------------------------
    # CAGR - 3 Year
    # -----------------------------
    revenue_cagr_3yr = get_cagr(revenue_history, 3)
    gross_profit_cagr_3yr = get_cagr(gross_profit_history, 3)
    ebit_cagr_3yr = get_cagr(ebit_history, 3)
    ebitda_cagr_3yr = get_cagr(ebitda_history, 3)
    net_income_cagr_3yr = get_cagr(net_income_history, 3)
    eps_cagr_3yr = get_cagr(eps_history, 3)
    diluted_eps_cagr_3yr = get_cagr(diluted_eps_history, 3)
    ar_cagr_3yr = get_cagr(ar_history, 3)
    inventory_cagr_3yr = get_cagr(inventory_history, 3)
    ppe_cagr_3yr = get_cagr(ppe_history, 3)
    total_assets_cagr_3yr = get_cagr(total_assets_history, 3)
    total_liabilities_cagr_3yr = get_cagr(total_liabilities_history, 3)
    equity_cagr_3yr = get_cagr(equity_history, 3)

    lfcf_cagr_3yr = get_cagr(lfcf_history, 3)

    # -----------------------------
    # CAGR - 5 Year
    # -----------------------------
    revenue_cagr_5yr = get_cagr(revenue_history, 5)
    gross_profit_cagr_5yr = get_cagr(gross_profit_history, 5)
    ebit_cagr_5yr = get_cagr(ebit_history, 5)
    ebitda_cagr_5yr = get_cagr(ebitda_history, 5)
    net_income_cagr_5yr = get_cagr(net_income_history, 5)
    eps_cagr_5yr = get_cagr(eps_history, 5)
    diluted_eps_cagr_5yr = get_cagr(diluted_eps_history, 5)
    ar_cagr_5yr = get_cagr(ar_history, 5)
    inventory_cagr_5yr = get_cagr(inventory_history, 5)
    ppe_cagr_5yr = get_cagr(ppe_history, 5)
    total_assets_cagr_5yr = get_cagr(total_assets_history, 5)
    total_liabilities_cagr_5yr = get_cagr(total_liabilities_history, 5)
    equity_cagr_5yr = get_cagr(equity_history, 5)

    # -----------------------------
    # FCF Yield & PEG
    # -----------------------------
    fcf_yield_pct = fcf_yield(fcf_value, market_cap)
    peg_ltm = peg_pe_ltm(pe_ltm, eps_growth_pct)
    peg_lynch_ratio = peg_lynch(pe_ltm, eps_growth_pct, dividend_yield_pct)

    # -----------------------------
    # Market cap in millions
    # -----------------------------
    market_cap_m = market_cap / 1_000_000 if not np.isnan(market_cap) else np.nan

    # -----------------------------
    # Final summary dictionary
    # -----------------------------
    return {
        # ========== BASIC INFO ==========
        "Company": metadata.get("name"),
        "Ticker": ticker,
        "Industry": metadata.get("industry"),
        "Market Cap (M)": market_cap_m,

        # ========== PROFITABILITY RATIOS ==========
        "ROA %": roa_val,
        "ROIC %": roic_val,
        "ROE %": roe_val,
        "RCE %": rce_val,

        # ========== MARGIN ANALYSIS ==========
        "Gross Margin %": margin_dict["Gross Margin %"],
        "SG&A Margin %": sga_margin_val,
        "R&D Margin %": rd_margin_val,
        "EBITDA Margin %": ebitda_margin_val,
        "EBIT Margin %": margin_dict["EBIT Margin %"],
        "Net Margin %": margin_dict["Net Margin %"],
        "LFCF Margin %": lfcf_margin_val,
        "UFCF Margin %": ufcf_margin_val,
        "CapEx % Revenue": capex_pct_revenue,

        # ========== ASSET TURNOVER ==========
        "Total Asset Turnover": asset_turnover,
        "AR Turnover": ar_turnover,
        "Inventory Turnover": inv_turnover,

        # ========== SHORT-TERM LIQUIDITY ==========
        "Current Ratio": current_ratio_val,
        "Quick Ratio": quick_ratio_val,
        "Avg Days Sales Outstanding": dso,
        "Avg Days Inventory Outstanding": dio,
        "Avg Days Payable Outstanding": dpo,
        "Cash Conversion Cycle": ccc,

        # ========== LONG-TERM LIQUIDITY ==========
        "Total D/E": total_de,
        "Total D/Capital": total_d_cap,
        "LT D/E": lt_de,
        "LT D/Capital": lt_d_cap,
        "Total Liab/Assets": liab_to_assets,
        "EBIT/Interest": ebit_interest,
        "EBITDA/Interest": ebitda_interest,
        "Total Debt/Interest": total_debt_interest,
        "Net Debt/Interest": net_debt_interest,
        "Altman Z-Score": z_score,

        # ========== GROWTH OVER PRIOR YEAR (YoY %) ==========
        "Revenue YoY %": revenue_yoy,
        "Gross Profit YoY %": gross_profit_yoy,
        "EBIT YoY %": ebit_yoy,
        "EBITDA YoY %": ebitda_yoy,
        "Net Income YoY %": net_income_yoy,
        "EPS YoY %": eps_yoy,
        "Diluted EPS YoY %": diluted_eps_yoy,
        "AR YoY %": ar_yoy,
        "Inventory YoY %": inventory_yoy,
        "Net PP&E YoY %": ppe_yoy,
        "Total Assets YoY %": total_assets_yoy,
        "Total Liabilities YoY %": total_liabilities_yoy,
        "Total Equity YoY %": equity_yoy,

        # ========== CAGR - 2 YEAR (%) ==========
        "Revenue 2yr CAGR %": revenue_cagr_2yr,
        "Gross Profit 2yr CAGR %": gross_profit_cagr_2yr,
        "EBIT 2yr CAGR %": ebit_cagr_2yr,
        "EBITDA 2yr CAGR %": ebitda_cagr_2yr,
        "Net Income 2yr CAGR %": net_income_cagr_2yr,
        "EPS 2yr CAGR %": eps_cagr_2yr,
        "Diluted EPS 2yr CAGR %": diluted_eps_cagr_2yr,
        "AR 2yr CAGR %": ar_cagr_2yr,
        "Inventory 2yr CAGR %": inventory_cagr_2yr,
        "Net PP&E 2yr CAGR %": ppe_cagr_2yr,
        "Total Assets 2yr CAGR %": total_assets_cagr_2yr,
        "Total Liabilities 2yr CAGR %": total_liabilities_cagr_2yr,
        "Total Equity 2yr CAGR %": equity_cagr_2yr,

        # ========== CAGR - 3 YEAR (%) ==========
        "Revenue 3yr CAGR %": revenue_cagr_3yr,
        "Gross Profit 3yr CAGR %": gross_profit_cagr_3yr,
        "EBIT 3yr CAGR %": ebit_cagr_3yr,
        "EBITDA 3yr CAGR %": ebitda_cagr_3yr,
        "Net Income 3yr CAGR %": net_income_cagr_3yr,
        "EPS 3yr CAGR %": eps_cagr_3yr,
        "Diluted EPS 3yr CAGR %": diluted_eps_cagr_3yr,
        "AR 3yr CAGR %": ar_cagr_3yr,
        "Inventory 3yr CAGR %": inventory_cagr_3yr,
        "Net PP&E 3yr CAGR %": ppe_cagr_3yr,
        "Total Assets 3yr CAGR %": total_assets_cagr_3yr,
        "Total Liabilities 3yr CAGR %": total_liabilities_cagr_3yr,
        "Total Equity 3yr CAGR %": equity_cagr_3yr,
        "LFCF 3yr CAGR %": lfcf_cagr_3yr,

        # ========== CAGR - 5 YEAR (%) ==========
        "Revenue 5yr CAGR %": revenue_cagr_5yr,
        "Gross Profit 5yr CAGR %": gross_profit_cagr_5yr,
        "EBIT 5yr CAGR %": ebit_cagr_5yr,
        "EBITDA 5yr CAGR %": ebitda_cagr_5yr,
        "Net Income 5yr CAGR %": net_income_cagr_5yr,
        "EPS 5yr CAGR %": eps_cagr_5yr,
        "Diluted EPS 5yr CAGR %": diluted_eps_cagr_5yr,
        "AR 5yr CAGR %": ar_cagr_5yr,
        "Inventory 5yr CAGR %": inventory_cagr_5yr,
        "Net PP&E 5yr CAGR %": ppe_cagr_5yr,
        "Total Assets 5yr CAGR %": total_assets_cagr_5yr,
        "Total Liabilities 5yr CAGR %": total_liabilities_cagr_5yr,
        "Total Equity 5yr CAGR %": equity_cagr_5yr,

        # ========== VALUATION ==========
        "PEG (PE LTM)": peg_ltm,
        "PEG (Lynch)": peg_lynch_ratio,
        "FCF Yield %": fcf_yield_pct,
    }