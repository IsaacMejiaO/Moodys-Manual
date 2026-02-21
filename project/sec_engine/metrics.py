# metrics.py
# ---------------------------------------------
# Core financial metric calculations.
# All functions are pure and safe for production use.

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Safe division helper
# ---------------------------------------------------------
def safe_divide(a, b, default=np.nan):
    if b is None or b == 0 or np.isnan(b):
        return default
    if a is None or np.isnan(a):
        return default
    return a / b

# ---------------------------------------------------------
# Margins
# ---------------------------------------------------------
def margins(revenue, gross_profit, ebit, net_income):
    return {
        "Gross Margin %": safe_divide(gross_profit, revenue) * 100,
        "EBIT Margin %": safe_divide(ebit, revenue) * 100,
        "Net Margin %": safe_divide(net_income, revenue) * 100,
    }

def ebitda_margin(revenue, ebitda):
    return safe_divide(ebitda, revenue) * 100

def sga_margin(revenue, sga):
    """SG&A as % of revenue"""
    return safe_divide(sga, revenue) * 100

def rd_margin(revenue, rd):
    """R&D as % of revenue"""
    return safe_divide(rd, revenue) * 100

def lfcf_margin(revenue, lfcf):
    """Levered FCF as % of revenue"""
    return safe_divide(lfcf, revenue) * 100

def ufcf_margin(revenue, ufcf):
    """Unlevered FCF as % of revenue"""
    return safe_divide(ufcf, revenue) * 100

def capex_as_pct_revenue(revenue, capex):
    """CapEx as % of revenue"""
    return safe_divide(abs(capex), revenue) * 100

# ---------------------------------------------------------
# Profitability Ratios
# ---------------------------------------------------------
def roa(net_income, total_assets):
    """Return on Assets"""
    return safe_divide(net_income, total_assets) * 100

def roic(nopat, invested_capital):
    """Return on Invested Capital"""
    return safe_divide(nopat, invested_capital) * 100

def roe(net_income, equity):
    """Return on Equity"""
    return safe_divide(net_income, equity) * 100

def rce(ebit, capital_employed):
    """Return on Capital Employed (EBIT / Capital Employed)"""
    return safe_divide(ebit, capital_employed) * 100

# ---------------------------------------------------------
# Asset Turnover Ratios
# ---------------------------------------------------------
def total_asset_turnover(revenue, total_assets):
    """Revenue / Total Assets"""
    return safe_divide(revenue, total_assets)

def accounts_receivable_turnover(revenue, accounts_receivable):
    """Revenue / Accounts Receivable"""
    return safe_divide(revenue, accounts_receivable)

def inventory_turnover(cogs, inventory):
    """COGS / Inventory"""
    return safe_divide(cogs, inventory)

# ---------------------------------------------------------
# Liquidity Ratios
# ---------------------------------------------------------
def current_ratio(current_assets, current_liabilities):
    """Current Assets / Current Liabilities"""
    return safe_divide(current_assets, current_liabilities)

def quick_ratio(current_assets, inventory, current_liabilities):
    """(Current Assets - Inventory) / Current Liabilities"""
    quick_assets = (current_assets or 0) - (inventory or 0)
    return safe_divide(quick_assets, current_liabilities)

def days_sales_outstanding(accounts_receivable, revenue):
    """(Accounts Receivable / Revenue) * 365"""
    return safe_divide(accounts_receivable, revenue) * 365

def days_inventory_outstanding(inventory, cogs):
    """(Inventory / COGS) * 365"""
    return safe_divide(inventory, cogs) * 365

def days_payable_outstanding(accounts_payable, cogs):
    """(Accounts Payable / COGS) * 365"""
    return safe_divide(accounts_payable, cogs) * 365

def cash_conversion_cycle(dso, dio, dpo):
    """DSO + DIO - DPO"""
    if np.isnan(dso) or np.isnan(dio) or np.isnan(dpo):
        return np.nan
    return dso + dio - dpo

# ---------------------------------------------------------
# Leverage Ratios
# ---------------------------------------------------------
def total_debt_to_equity(total_debt, equity):
    """Total Debt / Equity"""
    return safe_divide(total_debt, equity)

def total_debt_to_capital(total_debt, equity):
    """Total Debt / (Total Debt + Equity)"""
    total_capital = (total_debt or 0) + (equity or 0)
    return safe_divide(total_debt, total_capital)

def lt_debt_to_equity(lt_debt, equity):
    """Long-term Debt / Equity"""
    return safe_divide(lt_debt, equity)

def lt_debt_to_capital(lt_debt, equity):
    """Long-term Debt / (Long-term Debt + Equity)"""
    total_capital = (lt_debt or 0) + (equity or 0)
    return safe_divide(lt_debt, total_capital)

def total_liabilities_to_assets(total_liabilities, total_assets):
    """Total Liabilities / Total Assets"""
    return safe_divide(total_liabilities, total_assets)

def ebit_to_interest(ebit, interest_expense):
    """EBIT / Interest Expense (Interest Coverage)"""
    return safe_divide(ebit, abs(interest_expense))

def ebitda_to_interest(ebitda, interest_expense):
    """EBITDA / Interest Expense"""
    return safe_divide(ebitda, abs(interest_expense))

def total_debt_to_interest(total_debt, interest_expense):
    """Total Debt / Interest Expense"""
    return safe_divide(total_debt, abs(interest_expense))

def net_debt_to_interest(net_debt, interest_expense):
    """Net Debt / Interest Expense"""
    return safe_divide(net_debt, abs(interest_expense))

def altman_z_score(working_capital, total_assets, retained_earnings, ebit, equity, total_liabilities, revenue):
    """
    Altman Z-Score for public companies:
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets
    """
    x1 = safe_divide(working_capital, total_assets, 0)
    x2 = safe_divide(retained_earnings, total_assets, 0)
    x3 = safe_divide(ebit, total_assets, 0)
    x4 = safe_divide(equity, total_liabilities, 0)
    x5 = safe_divide(revenue, total_assets, 0)
    
    z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    
    if np.isnan(z_score):
        return np.nan
    return z_score

# ---------------------------------------------------------
# CAGR
# ---------------------------------------------------------
def cagr(start_value: float, end_value: float, years: int) -> float:
    if start_value is None or end_value is None:
        return np.nan
    if np.isnan(start_value) or np.isnan(end_value):
        return np.nan
    if start_value == 0:
        return np.nan
    if years <= 0:
        return np.nan
    # If start is negative and end is positive (or vice versa), CAGR is not
    # mathematically meaningful via the power formula. Return nan for sign flips.
    if start_value < 0 and end_value > 0:
        return np.nan
    if start_value < 0 and end_value < 0:
        # Both negative: treat as growth in absolute value but flip sign
        return (abs(end_value) / abs(start_value)) ** (1 / years) - 1
    return (end_value / start_value) ** (1 / years) - 1

def series_cagr(series: pd.Series, years: int = 3) -> float:
    """Calculate CAGR from a pandas Series"""
    if series is None or len(series) < years + 1:
        return np.nan
    start = series.iloc[-(years + 1)]
    end = series.iloc[-1]
    return cagr(start, end, years)

def yoy_growth(current_value, prior_value):
    """Year-over-year growth rate"""
    if prior_value is None or prior_value == 0:
        return np.nan
    if current_value is None:
        return np.nan
    if np.isnan(current_value) or np.isnan(prior_value):
        return np.nan
    return (current_value - prior_value) / abs(prior_value)

# ---------------------------------------------------------
# PEG ratios
# ---------------------------------------------------------
def peg_pe_ltm(pe_ltm: float, eps_growth_pct: float) -> float:
    if pe_ltm is None or eps_growth_pct is None:
        return np.nan
    if np.isnan(pe_ltm) or np.isnan(eps_growth_pct):
        return np.nan
    # Conventional PEG is only meaningful for positive P/E and growth.
    if pe_ltm <= 0 or eps_growth_pct <= 0:
        return np.nan
    return safe_divide(pe_ltm, eps_growth_pct)

def peg_lynch(pe_ltm: float, eps_growth_pct: float, dividend_yield_pct: float = 0.0) -> float:
    if pe_ltm is None or np.isnan(pe_ltm) or pe_ltm <= 0:
        return np.nan
    growth_plus_yield = (eps_growth_pct or 0) + (dividend_yield_pct or 0)
    if growth_plus_yield <= 0:
        return np.nan
    return safe_divide(pe_ltm, growth_plus_yield)

# ---------------------------------------------------------
# Free Cash Flow
# ---------------------------------------------------------
def fcf(ocf, capex):
    if ocf is None or capex is None:
        return np.nan
    return ocf - capex

def fcf_yield(fcf_value, market_cap):
    return safe_divide(fcf_value, market_cap) * 100