# calculations.py
# -----------------------------
import pandas as pd
import numpy as np

def safe_divide(a, b):
    return np.nan if b in [0, None] else a / b

def margins(income):
    """
    Computes margin metrics from income statement
    """
    revenue = income.loc["Total Revenue"].iloc[0]
    gross_profit = income.loc["Gross Profit"].iloc[0]
    ebit = income.loc["Ebit"].iloc[0]
    net_income = income.loc["Net Income"].iloc[0]

    return {
        "Gross Margin %": safe_divide(gross_profit, revenue) * 100,
        "EBIT Margin %": safe_divide(ebit, revenue) * 100,
        "Net Margin %": safe_divide(net_income, revenue) * 100
    }

def ebitda_margin(income):
    """
    Approximates EBITDA margin
    """
    revenue = income.loc["Total Revenue"].iloc[0]
    ebitda = income.loc["Ebitda"].iloc[0]
    return safe_divide(ebitda, revenue) * 100

def revenue_cagr(history, years=3):
    """
    Revenue CAGR using price history proxy (yfinance limitation)
    Replace with SEC revenue history later
    """
    prices = history["Close"]
    start = prices.iloc[-(252 * years)]
    end = prices.iloc[-1]

    return (end / start) ** (1 / years) - 1

def roic(income, balance):
    """
    ROIC = NOPAT / Invested Capital
    """
    ebit = income.loc["Ebit"].iloc[0]
    tax_rate = 0.21
    nopat = ebit * (1 - tax_rate)

    debt = balance.loc["Total Debt"].iloc[0]
    equity = balance.loc["Total Stockholder Equity"].iloc[0]

    invested_capital = debt + equity

    return safe_divide(nopat, invested_capital) * 100

def peg_ratios(info):
    """
    PEG calculations
    """
    pe = info.get("trailingPE")
    growth = info.get("earningsQuarterlyGrowth")

    peg_ltm = safe_divide(pe, growth * 100) if growth else np.nan
    peg_lynch = safe_divide(pe, info.get("revenueGrowth", 0) * 100)

    return peg_ltm, peg_lynch

def fcf_metrics(cashflow, info):
    """
    FCF Yield & LFCF growth
    """
    fcf = cashflow.loc["Free Cash Flow"].iloc[0]
    market_cap = info.get("marketCap")

    fcf_yield = safe_divide(fcf, market_cap) * 100

    return fcf_yield
