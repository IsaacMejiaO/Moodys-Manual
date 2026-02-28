# app.py
"""
SEC-Based Equity Analyzer - Main Application
============================================

A comprehensive equity analysis platform with:
- Screener with financial metrics
- Company tearsheets
- Multiples analysis with upgraded Capital IQ peer finder
- Financial ratios analysis
- Valuation (DCF — SEC-primary beta, two-stage model, sensitivity tables)
- Portfolio optimization (Monte Carlo)
- Portfolio performance tracking (XIRR/Dollar-Weighted Returns)

Fixes vs. prior version:
  - Bare `except: pass` on yfinance block replaced with
    `except Exception as e: logging.warning(...)` so real errors are
    visible in logs rather than silently swallowed.
  - Bare `except:` on EPS history replaced with `except Exception`.
  - `except Exception:` on _try_autoload_screener_csv() loop now logs
    the failure at DEBUG level.
  - Interest expense sign normalization: yfinance reports Interest Expense
    as a NEGATIVE value (cash outflow). We now abs() it at the point of
    ingestion so that all downstream consumers (ebit_to_interest, UFCF,
    etc.) receive a positive expense value, consistent with the SEC path.
  - Session-state ticker caches (ticker_data_cache, ticker_summary_cache)
    are now capped at MAX_TICKER_CACHE_SIZE entries using an LRU-style
    eviction policy so long-running sessions do not grow memory unboundedly.
  - DCF page renamed to "Valuation" in the sidebar.
"""
import sys
import os
import logging

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Maximum number of per-ticker entries held in session-state caches.
# When the limit is reached the oldest entry is evicted (LRU-style).
# At ~2–5 MB per ticker this caps in-session memory at ~100–250 MB.
MAX_TICKER_CACHE_SIZE = 50

_logger = logging.getLogger(__name__)

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from sec_engine.universe import UNIVERSE
from sec_engine.cik_loader import load_full_cik_map
from sec_engine.sec_fetch import fetch_company_facts, fetch_company_submissions
from sec_engine.normalize import GAAP_MAP
from sec_engine.ltm import extract_quarterly_series, extract_annual_series
from sec_engine.aggregation import build_company_summary

from ui.tearsheet import render_tearsheet
from ui.financials import render_financials
from ui.multiples import render_multiples
from ui.ratios import render_ratios
from ui.dcf import render_dcf
from ui.performance import render_performance, load_transactions_from_csv

# ---------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------
st.set_page_config(page_title="SEC-Based Equity Analyzer", layout="wide")

# ---------------------------------------------------------
# PERFORMANCE OPTIMIZATION: Initialize Cache
# ---------------------------------------------------------
def initialize_performance_cache():
    """Initialize all caching structures for performance"""
    if "ticker_data_cache" not in st.session_state:
        st.session_state["ticker_data_cache"] = {}
    if "ticker_summary_cache" not in st.session_state:
        st.session_state["ticker_summary_cache"] = {}
    if "last_selected_ticker" not in st.session_state:
        st.session_state["last_selected_ticker"] = None
    if "lazy_load_enabled" not in st.session_state:
        st.session_state["lazy_load_enabled"] = True

# Initialize caches
initialize_performance_cache()

# ---------------------------------------------------------
# Router + persistent universe state
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

# Redirect stale session state from the old standalone Portfolio page
if st.session_state.get("page") == "portfolio_monte_carlo":
    st.session_state["page"] = "performance"

if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = None

if "uploaded_universe" not in st.session_state:
    st.session_state["uploaded_universe"] = None
if "loaded_universe_key" not in st.session_state:
    st.session_state["loaded_universe_key"] = None
if "loaded_df" not in st.session_state:
    st.session_state["loaded_df"] = pd.DataFrame()
if "loaded_error_df" not in st.session_state:
    st.session_state["loaded_error_df"] = pd.DataFrame()
if "loaded_sic_map" not in st.session_state:
    st.session_state["loaded_sic_map"] = {}

# ---------------------------------------------------------
# AUTO-LOAD: Custom Screener Universe
# ---------------------------------------------------------

_DEFAULT_SCREENER_TICKERS = [
    "NVDA", "GOOG", "GOOGL", "MSFT", "AMZN", "AVGO", "META", "LLY", "V", "JNJ",
    "ORCL", "MA", "MU", "NFLX", "PLTR", "PM", "CRM", "APP", "KLAC", "ISRG",
    "APH", "AMGN", "UBER", "ANET", "BKNG", "SPGI", "SCCO", "INTU", "BSX", "SYK",
    "NEM", "HON", "NOW", "PGR", "ADBE", "PH", "VRTX", "ADP", "MCO",
    "HOOD", "HWM", "MMC", "WM", "BAM", "CDNS", "TDG", "MAR", "ORLY",
    "ABNB", "EQIX", "CTAS", "MNST", "RCL", "SLB", "MRVL", "VRT", "RSG", "HLT",
    "MSI", "SPG", "AZO", "NDAQ", "COIN", "IDXX", "URI", "ADSK", "FTNT", "ZTS",
    "PSA", "CMG", "BKR", "FAST", "ALL", "PYPL", "AME", "MPWR", "OKE", "WDAY",
    "MSCI", "ROP", "YUM", "FANG", "HEI", "RDDT", "FIX",
    "PAYX", "CPRT", "WAB", "LVS", "RMD", "FICO", "VEEV", "CCL", "XYL", "FISV",
    "UI", "EXR", "VICI", "ROL", "VRSK", "FOXA", "FOX", "CBOE", "FTAI",
    "DXCM", "HUBB", "TW", "FSLR", "BR", "CINF", "CW", "TPL", "VRSN", "RGLD",
    "RL", "CPAY", "INCY", "SSNC", "BWXT", "PTC", "UTHR", "WWD", "TYL", "LII",
    "HL", "PINS", "ZBH", "MEDP", "WES", "IT", "CRS", "GEN", "TTD", "ITT", "RBC",
    "CDE", "EVR", "DECK", "NXT", "PEN", "GDDY", "JKHY", "OHI", "ERIE",
    "GLPI", "NBIX", "HST", "GMED", "RMBS", "DT", "PR", "BSY", "EXEL", "NYT",
    "SPXC", "STRL", "FDS", "BMRN", "MANH", "CART", "EGP", "WTS",
    "KNSL", "PJT", "FRT", "AM", "MORN", "PEGA", "HLNE", "AWI", "CTRE", "HALO",
    "PAYC", "CAVA", "FR", "NNN", "PLNT", "IDCC", "APPF", "DOCS", "WING", "ZWS",
    "PCTY", "CHDN", "HESM", "MSA", "HQY", "FSS", "BYD", "EXLS", "DUOL",
]

_DEFAULT_SCREENER_CSV_PATHS = [
    "CustomScreener.csv",
    "ui/CustomScreener.csv",
    "data/CustomScreener.csv",
]

def _try_autoload_screener_csv() -> bool:
    """
    Silently load CustomScreener.csv from well-known paths on first run.
    Returns True if a file was successfully loaded.
    """
    for path in _DEFAULT_SCREENER_CSV_PATHS:
        if not os.path.exists(path):
            continue
        try:
            df_csv = pd.read_csv(path)
            df_csv.columns = [c.lower().strip().lstrip("\ufeff") for c in df_csv.columns]
            if "ticker" not in df_csv.columns:
                continue
            tickers = (
                df_csv["ticker"]
                .astype(str)
                .str.upper()
                .str.strip()
                .dropna()
                .unique()
                .tolist()
            )
            tickers = [t for t in tickers if t and t != "NAN"]
            if not tickers:
                continue
            st.session_state["uploaded_universe"] = tickers
            st.session_state["_screener_source"] = f"file:{path}"
            return True
        except Exception as _e:
            _logger.debug("Failed to load screener CSV from %s: %s", path, _e)
            continue
    return False

def _init_screener_universe() -> None:
    """
    Called once on startup.
    Priority: (1) already set in session_state, (2) disk CSV, (3) hardcoded default.
    """
    if "uploaded_universe" in st.session_state and st.session_state["uploaded_universe"] is not None:
        return

    if _try_autoload_screener_csv():
        return

    st.session_state["uploaded_universe"] = _DEFAULT_SCREENER_TICKERS
    st.session_state["_screener_source"] = "default"

_init_screener_universe()

# ---------------------------------------------------------
# Load full CIK map (cached)
# ---------------------------------------------------------
@st.cache_data
def get_cik_map():
    return load_full_cik_map()

CIK_MAP = get_cik_map()

# ---------------------------------------------------------
# UNIFIED DATA FETCHER (yfinance primary, SEC backup)
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_data_unified(ticker: str, cik: str = None):
    """
    Fetch company data using yfinance as primary source, SEC as backup.
    Returns dict with ltm_data, balance_data, metadata, and historical series.
    """
    ltm_data = {}
    balance_data = {}
    metadata = {}
    historical_data = {}
    _capex_source = "yfinance"

    # =========================================================
    # PRIMARY SOURCE: YFINANCE
    # =========================================================
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        income_stmt      = stock.income_stmt      if hasattr(stock, 'income_stmt')              else pd.DataFrame()
        balance_sheet    = stock.balance_sheet     if hasattr(stock, 'balance_sheet')            else pd.DataFrame()
        cashflow         = stock.cashflow          if hasattr(stock, 'cashflow')                 else pd.DataFrame()
        quarterly_income = stock.quarterly_income_stmt  if hasattr(stock, 'quarterly_income_stmt')  else pd.DataFrame()
        quarterly_balance= stock.quarterly_balance_sheet if hasattr(stock, 'quarterly_balance_sheet') else pd.DataFrame()
        quarterly_cashflow=stock.quarterly_cashflow if hasattr(stock, 'quarterly_cashflow')      else pd.DataFrame()

        earnings_growth = info.get("earningsGrowth")
        if earnings_growth is None:
            earnings_growth = info.get("earningsQuarterlyGrowth")
        dividend_yield = info.get("dividendYield")

        metadata = {
            "name":              info.get("longName", info.get("shortName", ticker)),
            "industry":          info.get("industry", "Unknown"),
            "sector":            info.get("sector", "Unknown"),
            "market_cap":        info.get("marketCap", np.nan),
            "pe_ltm":            info.get("trailingPE", np.nan),
            "eps_growth_pct":    (earnings_growth * 100) if earnings_growth is not None else np.nan,
            "dividend_yield_pct":(dividend_yield  * 100) if dividend_yield  is not None else np.nan,
        }

        def get_ltm_from_quarterly(df, field_name):
            """
            Sum last 4 quarterly periods to produce LTM.
            Returns NaN if fewer than 4 periods or a gap > 120 days exists.
            """
            if df is None or df.empty or field_name not in df.index:
                return np.nan
            row = df.loc[field_name].dropna().head(4)
            if len(row) < 4:
                return np.nan
            dates = sorted(row.index)
            for i in range(len(dates) - 1):
                if (dates[i + 1] - dates[i]).days > 120:
                    return np.nan
            return float(row.sum())

        def get_latest_value(df, field_name):
            if df is None or df.empty or field_name not in df.index:
                return np.nan
            return float(df.loc[field_name].iloc[0])

        ltm_data = {
            "revenue":          get_ltm_from_quarterly(quarterly_income,   "Total Revenue"),
            "gross_profit":     get_ltm_from_quarterly(quarterly_income,   "Gross Profit"),
            "operating_income": get_ltm_from_quarterly(quarterly_income,   "Operating Income"),
            "net_income":       get_ltm_from_quarterly(quarterly_income,   "Net Income"),
            "sga":              get_ltm_from_quarterly(quarterly_income,   "Selling General And Administration"),
            "rd":               get_ltm_from_quarterly(quarterly_income,   "Research And Development"),
            "cogs":             get_ltm_from_quarterly(quarterly_income,   "Cost Of Revenue"),
            "interest_expense": get_ltm_from_quarterly(quarterly_income,   "Interest Expense"),
            "ebitda":           get_ltm_from_quarterly(quarterly_income,   "EBITDA"),
            "depreciation":     get_ltm_from_quarterly(quarterly_cashflow, "Depreciation And Amortization"),
            "amortization":     np.nan,
            "ocf":              get_ltm_from_quarterly(quarterly_cashflow, "Operating Cash Flow"),
            "capex":            get_ltm_from_quarterly(quarterly_cashflow, "Capital Expenditure"),
        }

        # yfinance reports Interest Expense as negative — normalize to positive
        ie = ltm_data.get("interest_expense", np.nan)
        if not pd.isna(ie) and ie < 0:
            ltm_data["interest_expense"] = abs(ie)

        balance_data = {
            "total_assets":       get_latest_value(quarterly_balance, "Total Assets"),
            "current_assets":     get_latest_value(quarterly_balance, "Current Assets"),
            "cash":               get_latest_value(quarterly_balance, "Cash And Cash Equivalents"),
            "accounts_receivable":get_latest_value(quarterly_balance, "Accounts Receivable"),
            "inventory":          get_latest_value(quarterly_balance, "Inventory"),
            "ppe":                get_latest_value(quarterly_balance, "Net PPE"),
            "total_liabilities":  get_latest_value(quarterly_balance, "Total Liabilities Net Minority Interest"),
            "current_liabilities":get_latest_value(quarterly_balance, "Current Liabilities"),
            "accounts_payable":   get_latest_value(quarterly_balance, "Accounts Payable"),
            "long_term_debt":     get_latest_value(quarterly_balance, "Long Term Debt"),
            "debt":               get_latest_value(quarterly_balance, "Total Debt"),
            "equity":             get_latest_value(quarterly_balance, "Stockholders Equity"),
            "retained_earnings":  get_latest_value(quarterly_balance, "Retained Earnings"),
        }

        def get_annual_series(df, field_name):
            if df is None or df.empty or field_name not in df.index:
                return None
            series = df.loc[field_name].sort_index()
            return series if not series.empty else None

        historical_data = {
            "revenue_history":           get_annual_series(income_stmt,   "Total Revenue"),
            "gross_profit_history":      get_annual_series(income_stmt,   "Gross Profit"),
            "ebit_history":              get_annual_series(income_stmt,   "Operating Income"),
            "ebitda_history":            get_annual_series(income_stmt,   "EBITDA"),
            "net_income_history":        get_annual_series(income_stmt,   "Net Income"),
            "ar_history":                get_annual_series(balance_sheet, "Accounts Receivable"),
            "inventory_history":         get_annual_series(balance_sheet, "Inventory"),
            "ppe_history":               get_annual_series(balance_sheet, "Net PPE"),
            "total_assets_history":      get_annual_series(balance_sheet, "Total Assets"),
            "total_liabilities_history": get_annual_series(balance_sheet, "Total Liabilities Net Minority Interest"),
            "equity_history":            get_annual_series(balance_sheet, "Stockholders Equity"),
        }

        ocf_history   = get_annual_series(cashflow, "Operating Cash Flow")
        capex_history = get_annual_series(cashflow, "Capital Expenditure")

        if ocf_history is not None and capex_history is not None:
            combined = pd.concat([ocf_history, capex_history], axis=1, keys=['ocf', 'capex']).dropna()
            historical_data["lfcf_history"] = combined['ocf'] - combined['capex'] if not combined.empty else None
        else:
            historical_data["lfcf_history"] = None

        try:
            historical_data["eps_history"]         = None
            historical_data["diluted_eps_history"] = get_annual_series(income_stmt, "Diluted EPS")
        except Exception as _e:
            _logger.debug("Could not extract EPS history for %s: %s", ticker, _e)
            historical_data["eps_history"]         = None
            historical_data["diluted_eps_history"] = None

    except Exception as e:
        _logger.warning("yfinance fetch failed for %s: %s", ticker, e)

    # =========================================================
    # BACKUP SOURCE: SEC FILINGS
    # =========================================================
    if cik:
        try:
            facts = fetch_company_facts(cik)

            def ltm_value_sec(tags):
                from sec_engine.ltm import extract_quarterly_series, build_ltm
                return build_ltm(extract_quarterly_series(facts, tags))

            def latest_balance_sec(tags):
                from sec_engine.ltm import extract_quarterly_series, latest_balance
                return latest_balance(extract_quarterly_series(facts, tags))

            def annual_series_sec(tags):
                from sec_engine.ltm import extract_annual_series
                s = extract_annual_series(facts, tags)
                return s if not s.empty else None

            sec_backup_ltm = {
                "revenue":          ltm_value_sec(GAAP_MAP.get("revenue", [])),
                "gross_profit":     ltm_value_sec(GAAP_MAP.get("gross_profit", [])),
                "operating_income": ltm_value_sec(GAAP_MAP.get("operating_income", [])),
                "net_income":       ltm_value_sec(GAAP_MAP.get("net_income", [])),
                "sga":              ltm_value_sec(GAAP_MAP.get("sga", [])),
                "rd":               ltm_value_sec(GAAP_MAP.get("rd", [])),
                "cogs":             ltm_value_sec(GAAP_MAP.get("cogs", [])),
                "interest_expense": ltm_value_sec(GAAP_MAP.get("interest_expense", [])),
                "ocf":              ltm_value_sec(GAAP_MAP.get("ocf", [])),
                "capex":            ltm_value_sec(GAAP_MAP.get("capex", [])),
                "ebitda":           ltm_value_sec(GAAP_MAP.get("ebitda", [])),
                "depreciation":     ltm_value_sec(GAAP_MAP.get("depreciation", [])),
            }

            for key, value in sec_backup_ltm.items():
                if pd.isna(ltm_data.get(key, np.nan)) and not pd.isna(value):
                    ltm_data[key] = value
                    if key == "capex":
                        _capex_source = "sec"

            sec_backup_balance = {
                "total_assets":        latest_balance_sec(GAAP_MAP.get("total_assets", [])),
                "current_assets":      latest_balance_sec(GAAP_MAP.get("current_assets", [])),
                "cash":                latest_balance_sec(GAAP_MAP.get("cash", [])),
                "accounts_receivable": latest_balance_sec(GAAP_MAP.get("accounts_receivable", [])),
                "inventory":           latest_balance_sec(GAAP_MAP.get("inventory", [])),
                "ppe":                 latest_balance_sec(GAAP_MAP.get("ppe", [])),
                "total_liabilities":   latest_balance_sec(GAAP_MAP.get("total_liabilities", [])),
                "current_liabilities": latest_balance_sec(GAAP_MAP.get("current_liabilities", [])),
                "accounts_payable":    latest_balance_sec(GAAP_MAP.get("accounts_payable", [])),
                "long_term_debt":      latest_balance_sec(GAAP_MAP.get("long_term_debt", [])),
                "debt":                latest_balance_sec(GAAP_MAP.get("debt", [])),
                "equity":              latest_balance_sec(GAAP_MAP.get("equity", [])),
                "retained_earnings":   latest_balance_sec(GAAP_MAP.get("retained_earnings", [])),
            }

            for key, value in sec_backup_balance.items():
                if pd.isna(balance_data.get(key, np.nan)) and not pd.isna(value):
                    balance_data[key] = value

            sec_historical_backup = {
                "revenue_history":           annual_series_sec(GAAP_MAP.get("revenue", [])),
                "gross_profit_history":      annual_series_sec(GAAP_MAP.get("gross_profit", [])),
                "ebit_history":              annual_series_sec(GAAP_MAP.get("operating_income", [])),
                "ebitda_history":            annual_series_sec(GAAP_MAP.get("ebitda", [])),
                "net_income_history":        annual_series_sec(GAAP_MAP.get("net_income", [])),
                "ar_history":                annual_series_sec(GAAP_MAP.get("accounts_receivable", [])),
                "inventory_history":         annual_series_sec(GAAP_MAP.get("inventory", [])),
                "ppe_history":               annual_series_sec(GAAP_MAP.get("ppe", [])),
                "total_assets_history":      annual_series_sec(GAAP_MAP.get("total_assets", [])),
                "total_liabilities_history": annual_series_sec(GAAP_MAP.get("total_liabilities", [])),
                "equity_history":            annual_series_sec(GAAP_MAP.get("equity", [])),
            }

            for key, value in sec_historical_backup.items():
                if historical_data.get(key) is None and value is not None:
                    historical_data[key] = value

            if historical_data.get("lfcf_history") is None:
                ocf_sec   = annual_series_sec(GAAP_MAP.get("ocf", []))
                capex_sec = annual_series_sec(GAAP_MAP.get("capex", []))
                if ocf_sec is not None and capex_sec is not None:
                    combined = pd.concat([ocf_sec, capex_sec], axis=1, keys=["ocf", "capex"]).dropna()
                    if not combined.empty:
                        historical_data["lfcf_history"] = combined["ocf"] - combined["capex"]

        except Exception as exc:
            _logger.warning("SEC backup failed for %s (CIK %s): %s", ticker, cik, exc)

    return {
        "ltm_data":    ltm_data,
        "balance_data":balance_data,
        "metadata":    metadata,
        "capex_source":_capex_source,
        **historical_data,
    }

# ---------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------
def get_or_load_ticker_data(ticker: str):
    """Get ticker data from cache or load it (LRU eviction)."""
    cache = st.session_state["ticker_data_cache"]

    if ticker in cache:
        cache[ticker] = cache.pop(ticker)
        return cache[ticker]

    cik  = CIK_MAP.get(ticker)
    data = fetch_company_data_unified(ticker, cik)

    if len(cache) >= MAX_TICKER_CACHE_SIZE:
        del cache[next(iter(cache))]

    cache[ticker] = data
    return data


def get_or_compute_summary(ticker: str):
    """Get summary from cache or compute it (LRU eviction)."""
    summary_cache = st.session_state["ticker_summary_cache"]

    if ticker in summary_cache:
        summary_cache[ticker] = summary_cache.pop(ticker)
        return summary_cache[ticker]

    data    = get_or_load_ticker_data(ticker)
    summary = build_company_summary(
        ticker=ticker,
        ltm_data=data.get('ltm_data', {}),
        balance_data=data.get('balance_data', {}),
        metadata=data.get('metadata', {}),
        revenue_history=data.get('revenue_history'),
        lfcf_history=data.get('lfcf_history'),
        gross_profit_history=data.get('gross_profit_history'),
        ebit_history=data.get('ebit_history'),
        ebitda_history=data.get('ebitda_history'),
        net_income_history=data.get('net_income_history'),
        eps_history=data.get('eps_history'),
        diluted_eps_history=data.get('diluted_eps_history'),
        ar_history=data.get('ar_history'),
        inventory_history=data.get('inventory_history'),
        ppe_history=data.get('ppe_history'),
        total_assets_history=data.get('total_assets_history'),
        total_liabilities_history=data.get('total_liabilities_history'),
        equity_history=data.get('equity_history'),
        capex_from_yfinance=data.get("capex_source") == "yfinance",
    )

    if len(summary_cache) >= MAX_TICKER_CACHE_SIZE:
        del summary_cache[next(iter(summary_cache))]

    summary_cache[ticker] = summary
    return summary


# ---------------------------------------------------------
# Parallel loader (screener universe)
# ---------------------------------------------------------
def load_single_ticker_summary(ticker: str) -> dict:
    try:
        cik = CIK_MAP.get(ticker)
        if cik is None:
            return None

        sic_code = ''
        try:
            submissions = fetch_company_submissions(cik)
            sic_code    = submissions.get('sic', '')
        except Exception:
            pass

        data    = fetch_company_data_unified(ticker, cik)
        summary = build_company_summary(
            ticker=ticker,
            ltm_data=data["ltm_data"],
            balance_data=data["balance_data"],
            metadata=data["metadata"],
            revenue_history=data.get("revenue_history"),
            lfcf_history=data.get("lfcf_history"),
            gross_profit_history=data.get("gross_profit_history"),
            ebit_history=data.get("ebit_history"),
            ebitda_history=data.get("ebitda_history"),
            net_income_history=data.get("net_income_history"),
            eps_history=data.get("eps_history"),
            diluted_eps_history=data.get("diluted_eps_history"),
            ar_history=data.get("ar_history"),
            inventory_history=data.get("inventory_history"),
            ppe_history=data.get("ppe_history"),
            total_assets_history=data.get("total_assets_history"),
            total_liabilities_history=data.get("total_liabilities_history"),
            equity_history=data.get("equity_history"),
            capex_from_yfinance=data.get("capex_source") == "yfinance",
        )
        return {"summary": summary, "sic": sic_code, "ticker": ticker}

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@st.cache_data(show_spinner=False)
def load_data_parallel(universe, max_workers=10):
    rows, errors, sic_map = [], [], {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(load_single_ticker_summary, t): t for t in universe
        }
        for future in as_completed(future_to_ticker):
            result = future.result()
            if result is None:
                continue
            if "error" in result:
                errors.append({"Ticker": result["ticker"], "Error": result["error"]})
            else:
                rows.append(result["summary"])
                if result.get("sic"):
                    sic_map[result["ticker"]] = result["sic"]
    df       = pd.DataFrame(rows)  if rows   else pd.DataFrame()
    error_df = pd.DataFrame(errors) if errors else pd.DataFrame()
    return df, error_df, sic_map


def load_data_with_progress_parallel(universe, max_workers=10):
    rows, errors, sic_map = [], [], {}
    progress_bar = st.progress(0)
    status_text  = st.empty()
    completed, total = 0, len(universe)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(load_single_ticker_summary, t): t for t in universe
        }
        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            status_text.text(f"Loading {ticker}... ({completed}/{total})")
            progress_bar.progress(completed / total)
            result = future.result()
            if result is None:
                continue
            if "error" in result:
                errors.append({"Ticker": result["ticker"], "Error": result["error"]})
            else:
                rows.append(result["summary"])
                if result.get("sic"):
                    sic_map[result["ticker"]] = result["sic"]

    progress_bar.empty()
    status_text.empty()
    df       = pd.DataFrame(rows)  if rows   else pd.DataFrame()
    error_df = pd.DataFrame(errors) if errors else pd.DataFrame()
    return df, error_df, sic_map


# =================================================================
# SIDEBAR
# =================================================================
current_page = st.session_state.get("page", "dashboard")

st.sidebar.markdown(
    """
    <div style="text-align:center;font-weight:900;font-size:46px;line-height:1.05;margin:0.2rem 0 0.9rem 0;">
        Moody's Manual
    </div>
    """,
    unsafe_allow_html=True,
)

universe = st.session_state.get("uploaded_universe") or UNIVERSE

# Load screener universe data
cache_key   = str(sorted(universe))
needs_reload = (
    st.session_state.get("loaded_universe_key") != cache_key
    or st.session_state.get("loaded_df", pd.DataFrame()).empty
)

if needs_reload:
    st.session_state["ticker_data_cache"]    = {}
    st.session_state["ticker_summary_cache"] = {}
    df, error_df, sic_map = load_data_with_progress_parallel(universe, max_workers=8)
    st.session_state["loaded_universe_key"] = cache_key
    st.session_state["loaded_df"]           = df
    st.session_state["loaded_error_df"]     = error_df
    st.session_state["loaded_sic_map"]      = sic_map
else:
    df       = st.session_state.get("loaded_df",       pd.DataFrame())
    error_df = st.session_state.get("loaded_error_df", pd.DataFrame())
    sic_map  = st.session_state.get("loaded_sic_map",  {})

st.session_state["sic_map"]    = sic_map
st.session_state["company_df"] = df

# ── Navigation ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")

selected_ticker = st.session_state.get("selected_ticker")
show_ticker_pages = current_page in {
    "dashboard", "tearsheet", "financials", "multiples", "ratios", "dcf"
}

if st.sidebar.button("Data", width="stretch",
                     type="primary" if current_page == "data_controls" else "secondary"):
    st.session_state["page"] = "data_controls"
    st.rerun()

if st.sidebar.button("Screener", width="stretch",
                     type="primary" if current_page == "dashboard" else "secondary"):
    st.session_state["page"] = "dashboard"
    st.rerun()

if st.sidebar.button("Tearsheet", width="stretch",
                     type="primary" if current_page == "tearsheet" else "secondary"):
    if selected_ticker:
        st.session_state["page"] = "tearsheet"
        st.rerun()
    else:
        st.sidebar.warning("Click a ticker row in the Screener first.")

if st.sidebar.button("Financials", width="stretch",
                     type="primary" if current_page == "financials" else "secondary"):
    if selected_ticker:
        st.session_state["page"] = "financials"
        st.rerun()
    else:
        st.sidebar.warning("Click a ticker row in the Screener first.")

if st.sidebar.button("Multiples", width="stretch",
                     type="primary" if current_page == "multiples" else "secondary"):
    if selected_ticker:
        st.session_state["page"] = "multiples"
        st.rerun()
    else:
        st.sidebar.warning("Click a ticker row in the Screener first.")

if st.sidebar.button("Ratios", width="stretch",
                     type="primary" if current_page == "ratios" else "secondary"):
    if selected_ticker:
        st.session_state["page"] = "ratios"
        st.rerun()
    else:
        st.sidebar.warning("Click a ticker row in the Screener first.")

# ── "Valuation" in sidebar (page key stays "dcf" internally) ─────────────────
if st.sidebar.button("Valuation", width="stretch",
                     type="primary" if current_page == "dcf" else "secondary"):
    if selected_ticker:
        st.session_state["page"] = "dcf"
        st.rerun()
    else:
        st.sidebar.warning("Click a ticker row in the Screener first.")

if st.sidebar.button("Performance", width="stretch",
                     type="primary" if current_page == "performance" else "secondary"):
    st.session_state["page"] = "performance"
    st.rerun()


# =================================================================
# DATA CONTROLS PAGE
# =================================================================
if st.session_state["page"] == "data_controls":

    st.markdown(
        '<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:2px;">Data</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .col-divider {
            border: none;
            border-top: 1px solid #2a2a3a;
            margin: 14px 0 16px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT — SCREENER DATA ──────────────────────────────────────────────────
    with col_left:
        st.markdown("### Screener Data")

        with st.expander("Example layout", expanded=False):
            st.dataframe(
                pd.DataFrame({
                    "ticker":  ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
                    "company": ["Apple Inc.", "Microsoft Corp.", "NVIDIA Corp.",
                                "Alphabet Inc.", "Meta Platforms"],
                }),
                hide_index=True,
                use_container_width=True,
                height=212,
                column_config={
                    "ticker":  st.column_config.TextColumn("ticker",             width="small"),
                    "company": st.column_config.TextColumn("company (optional)", width="medium"),
                },
            )

        st.markdown('<hr class="col-divider">', unsafe_allow_html=True)

        universe_file = st.file_uploader(
            "Universe file",
            type=["csv", "xlsx"],
            key="data_controls_universe_file",
            label_visibility="collapsed",
        )

        screener_sample_csv = (
            "ticker,company\n"
            "AAPL,Apple Inc.\n"
            "MSFT,Microsoft Corp.\n"
            "NVDA,NVIDIA Corp.\n"
            "GOOGL,Alphabet Inc.\n"
            "META,Meta Platforms Inc.\n"
        )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1.4])
        with btn_col1:
            if st.button("Apply Screener Data", key="apply_screener_data",
                         type="primary", use_container_width=True):
                if universe_file is None:
                    st.warning("Upload a file first.")
                else:
                    df_up = (pd.read_csv(universe_file) if universe_file.name.endswith(".csv")
                             else pd.read_excel(universe_file))
                    df_up.columns = [c.lower().strip() for c in df_up.columns]
                    if "ticker" not in df_up.columns:
                        st.error("File must contain a 'ticker' column.")
                    else:
                        new_universe = (
                            df_up["ticker"].astype(str).str.upper().str.strip()
                            .unique().tolist()
                        )
                        st.session_state["uploaded_universe"]   = new_universe
                        st.session_state["loaded_universe_key"] = None
                        st.success(f"Screener universe updated: {len(new_universe)} tickers.")

        with btn_col2:
            if st.button("Reset to Default", key="reset_screener_data", use_container_width=True):
                st.session_state["uploaded_universe"]   = _DEFAULT_SCREENER_TICKERS
                st.session_state["loaded_universe_key"] = None
                st.success(f"Reset to default ({len(_DEFAULT_SCREENER_TICKERS)} tickers).")

        with btn_col3:
            st.download_button(
                "⬇ Sample",
                data=screener_sample_csv,
                file_name="screener_universe_example.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── RIGHT — PORTFOLIO CONTROLS ────────────────────────────────────────────
    with col_right:
        st.markdown("### Portfolio Controls")

        with st.expander("Example layout", expanded=False):
            st.dataframe(
                pd.DataFrame({
                    "date":   ["2024-01-15","2024-01-15","2024-02-01","2024-02-10",
                               "2024-03-01","2024-03-15","2024-04-01","2024-04-10"],
                    "ticker": ["","AAPL","MSFT","AAPL","NVDA","","AAPL",""],
                    "action": ["deposit","buy","buy","sell","buy","deposit","dividend","withdrawal"],
                    "shares": [0,10,5,3,8,0,0,0],
                    "price":  [5000.00,185.50,375.20,192.30,620.10,1000.00,0.25,2500.00],
                }),
                hide_index=True,
                use_container_width=True,
                height=315,
                column_config={
                    "date":   st.column_config.TextColumn("date",      width="medium"),
                    "ticker": st.column_config.TextColumn("ticker",    width="small"),
                    "action": st.column_config.TextColumn("action",    width="small"),
                    "shares": st.column_config.NumberColumn("shares",  width="small", format="%g"),
                    "price":  st.column_config.NumberColumn("price ($)",width="small", format="%.2f"),
                },
            )

        st.markdown('<hr class="col-divider">', unsafe_allow_html=True)

        tx_file = st.file_uploader(
            "Transactions CSV",
            type=["csv", "xlsx"],
            key="data_controls_transactions_file",
            label_visibility="collapsed",
        )

        portfolio_sample_csv = (
            "date,ticker,action,shares,price\n"
            "2024-01-15,,deposit,0,5000\n"
            "2024-01-15,AAPL,buy,10,185.50\n"
            "2024-01-15,MSFT,buy,5,375.20\n"
            "2024-02-01,NVDA,buy,8,620.10\n"
            "2024-02-10,AAPL,sell,3,192.30\n"
            "2024-03-01,,deposit,0,2000\n"
            "2024-03-15,META,buy,6,505.00\n"
            "2024-04-01,AAPL,dividend,0,0.25\n"
            "2024-04-10,NVDA,sell,2,850.00\n"
            "2024-05-01,,withdrawal,0,2500\n"
        )

        btn_col4, btn_col5, btn_col6 = st.columns([2, 1.6, 1.4])
        with btn_col4:
            if st.button("Apply Portfolio File", key="apply_portfolio_data",
                         type="primary", use_container_width=True):
                if tx_file is None:
                    st.warning("Upload a file first.")
                else:
                    if tx_file.name.endswith(".xlsx"):
                        import io
                        csv_buf = io.StringIO()
                        pd.read_excel(tx_file).to_csv(csv_buf, index=False)
                        csv_buf.seek(0)
                        ok = load_transactions_from_csv(csv_buf)
                    else:
                        ok = load_transactions_from_csv(tx_file)
                    st.success("Portfolio transactions loaded.") if ok else st.error("Could not parse the file.")

        with btn_col5:
            if st.button("Go to Performance →", key="go_to_performance", use_container_width=True):
                st.session_state["page"] = "performance"
                st.rerun()

        with btn_col6:
            st.download_button(
                "⬇ Sample",
                data=portfolio_sample_csv,
                file_name="portfolio_transactions_example.csv",
                mime="text/csv",
                use_container_width=True,
            )


# =================================================================
# SCREENER (DASHBOARD) PAGE
# =================================================================
elif st.session_state["page"] == "dashboard":

    if df.empty:
        st.warning("No data available.")
    else:
        df_adj = df.copy()

        # Remove columns not shown in the screener
        columns_to_remove = [
            "Data As Of",
            "Revenue", "Gross Profit", "EBIT", "EBITDA", "Net Income",
            "OCF", "CapEx", "LFCF", "UFCF", "Interest Expense",
            "Total Debt", "Net Debt", "Cash",
            "ROA %", "ROE %", "RCE %", "SG&A Margin %", "R&D Margin %",
            "LFCF Margin %", "UFCF Margin %", "CapEx % Revenue",
            "Total Asset Turnover", "AR Turnover", "Inventory Turnover",
            "Current Ratio", "Quick Ratio", "Avg Days Sales Outstanding",
            "Avg Days Inventory Outstanding", "Avg Days Payable Outstanding",
            "Cash Conversion Cycle",
            "Total D/E", "Total D/Capital", "LT D/E", "LT D/Capital",
            "Total Liab/Assets", "EBIT/Interest", "EBITDA/Interest",
            "Total Debt/Interest", "Net Debt/Interest", "Altman Z-Score",
            "PE LTM", "EV/Revenue", "EV/EBITDA", "EV/EBIT",
            "Revenue YoY %", "Gross Profit YoY %", "EBIT YoY %", "EBITDA YoY %",
            "Net Income YoY %", "EPS YoY %", "Diluted EPS YoY %", "AR YoY %",
            "Inventory YoY %", "Net PP&E YoY %", "Total Assets YoY %",
            "Total Liabilities YoY %", "Total Equity YoY %",
            "Gross Profit 2yr CAGR %", "EBIT 2yr CAGR %", "EBITDA 2yr CAGR %",
            "Net Income 2yr CAGR %", "EPS 2yr CAGR %", "Diluted EPS 2yr CAGR %",
            "AR 2yr CAGR %", "Inventory 2yr CAGR %", "Net PP&E 2yr CAGR %",
            "Total Assets 2yr CAGR %", "Total Liabilities 2yr CAGR %",
            "Total Equity 2yr CAGR %", "LFCF 2yr CAGR %",
            "Gross Profit 3yr CAGR %", "EBIT 3yr CAGR %", "EBITDA 3yr CAGR %",
            "Net Income 3yr CAGR %", "EPS 3yr CAGR %", "Diluted EPS 3yr CAGR %",
            "AR 3yr CAGR %", "Inventory 3yr CAGR %", "Net PP&E 3yr CAGR %",
            "Total Assets 3yr CAGR %", "Total Liabilities 3yr CAGR %",
            "Total Equity 3yr CAGR %",
            "Gross Profit 5yr CAGR %", "EBIT 5yr CAGR %", "EBITDA 5yr CAGR %",
            "Net Income 5yr CAGR %", "EPS 5yr CAGR %", "Diluted EPS 5yr CAGR %",
            "AR 5yr CAGR %", "Inventory 5yr CAGR %", "Inventory 5yr acCAGR %",
            "Net PP&E 5yr CAGR %", "Total Assets 5yr CAGR %",
            "Total Liabilities 5yr CAGR %", "Total Equity 5yr CAGR %",
            "LFCF 5yr CAGR %", "Revenue 5yr CAGR %",
        ]
        for col in columns_to_remove:
            if col in df_adj.columns:
                df_adj = df_adj.drop(columns=[col])

        # Swap EBIT Margin % ↔ EBITDA Margin %
        if "EBIT Margin %" in df_adj.columns and "EBITDA Margin %" in df_adj.columns:
            cols = list(df_adj.columns)
            ei, edi = cols.index("EBIT Margin %"), cols.index("EBITDA Margin %")
            cols[ei], cols[edi] = cols[edi], cols[ei]
            df_adj = df_adj[cols]

        # Move FCF Yield % after LFCF 3yr CAGR %
        if "FCF Yield %" in df_adj.columns and "LFCF 3yr CAGR %" in df_adj.columns:
            cols = list(df_adj.columns)
            cols.remove("FCF Yield %")
            cols.insert(cols.index("LFCF 3yr CAGR %") + 1, "FCF Yield %")
            df_adj = df_adj[cols]

        st.markdown(
            '<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:4px;">Screener</h1>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
            [data-testid="stDataFrame"] th div[data-testid="column-header-cell-text"],
            [data-testid="stDataFrame"] th span {
                white-space: normal !important;
                word-break: break-word !important;
                line-height: 1.3 !important;
            }
            [data-testid="stDataFrame"] thead th {
                vertical-align: top !important;
                min-height: 52px !important;
            }
            [data-testid="stDataFrame"] td,
            [data-testid="stDataFrame"] th {
                padding: 4px 6px !important;
                font-size: 13px !important;
            }
            [data-testid="stDataFrame"] .ag-row-odd  { background-color: #1e1e2e !important; }
            [data-testid="stDataFrame"] .ag-row-even { background-color: #16161f !important; }
            [data-testid="stDataFrame"] .ag-row-hover    { background-color: #2a2a3e !important; }
            [data-testid="stDataFrame"] .ag-row-selected { background-color: #1a3a5c !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        id_cols     = [c for c in ["Ticker", "Company", "Name", "Industry"] if c in df_adj.columns]
        metric_cols = [c for c in df_adj.columns if c not in id_cols]

        if metric_cols:
            has_data = df_adj[metric_cols].apply(
                lambda row: any(pd.notna(v) and str(v).strip() not in ("", "nan") for v in row),
                axis=1,
            )
            df_display = df_adj.loc[has_data].copy()
        else:
            df_display = df_adj.dropna(how="all").copy()

        for col in metric_cols:
            if col in df_display.columns and col != "Market Cap (M)":
                df_display[col] = pd.to_numeric(df_display[col], errors="coerce")

        pct_cols = {
            "Gross Margin %", "EBIT Margin %", "EBITDA Margin %",
            "Net Margin %", "ROIC %", "FCF Yield %",
            "Revenue 2yr CAGR %", "Revenue 3yr CAGR %", "LFCF 3yr CAGR %",
        }
        for peg_col in ("PEG (PE LTM)", "PEG (Lynch)"):
            if peg_col in df_display.columns:
                df_display[peg_col] = pd.to_numeric(df_display[peg_col], errors="coerce") * 100

        ratio_cols  = {"PEG (PE LTM)", "PEG (Lynch)"}
        mktcap_cols = {"Market Cap (M)"}

        if "Market Cap (M)" in df_display.columns:
            df_display["Market Cap (M)"] = pd.to_numeric(
                df_display["Market Cap (M)"], errors="coerce"
            ).apply(lambda x: f"{x:>15,.0f}" if pd.notna(x) else "")

        column_config = {}
        for col in df_display.columns:
            if col == "Ticker":
                column_config[col] = st.column_config.TextColumn(col, width="small")
            elif col in ("Company", "Name"):
                column_config[col] = st.column_config.TextColumn(col, width="medium")
            elif col == "Industry":
                column_config[col] = st.column_config.TextColumn(col, width="medium")
            elif col in mktcap_cols:
                column_config[col] = st.column_config.TextColumn(col, width="small")
            elif col in pct_cols:
                column_config[col] = st.column_config.NumberColumn(col, format="%.0f%%", width="small")
            elif col in ratio_cols:
                column_config[col] = st.column_config.NumberColumn(col, format="%.0f%%", width="small")
            else:
                column_config[col] = st.column_config.NumberColumn(col, format="%.0f", width="small")

        event = st.dataframe(
            df_display,
            column_config=column_config,
            use_container_width=True,
            height=620,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Center Gross Margin % → FCF Yield % columns
        _cols = list(df_display.columns)
        if "Gross Margin %" in _cols and "FCF Yield %" in _cols:
            _start = _cols.index("Gross Margin %") + 1
            _end   = _cols.index("FCF Yield %")    + 1
            _cs = ", ".join(
                f'[data-testid="stDataFrame"] .ag-cell[aria-colindex="{i}"]'
                for i in range(_start, _end + 1)
            )
            _hs = ", ".join(
                f'[data-testid="stDataFrame"] .ag-header-cell[aria-colindex="{i}"] .ag-header-cell-label'
                for i in range(_start, _end + 1)
            )
            st.markdown(
                f"<style>{_cs}{{text-align:center!important;justify-content:center!important;}}"
                f"{_hs}{{justify-content:center!important;}}</style>",
                unsafe_allow_html=True,
            )

        selected_rows = event.selection.rows
        if selected_rows:
            clicked_ticker = df_display.iloc[selected_rows[0]]["Ticker"]
            if clicked_ticker != st.session_state.get("selected_ticker"):
                st.session_state["selected_ticker"] = clicked_ticker
                with st.spinner(f"Loading {clicked_ticker}..."):
                    get_or_load_ticker_data(clicked_ticker)
                st.rerun()

        if not error_df.empty:
            with st.expander(f"⚠️ Errors ({len(error_df)} tickers failed)"):
                st.dataframe(error_df, use_container_width=True)


# =================================================================
# TEARSHEET PAGE
# =================================================================
elif st.session_state["page"] == "tearsheet":
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        st.warning("No ticker selected.")
    else:
        render_tearsheet(ticker=ticker)


# =================================================================
# FINANCIALS PAGE
# =================================================================
elif st.session_state["page"] == "financials":
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        st.warning("No ticker selected.")
    else:
        render_financials(ticker=ticker)


# =================================================================
# MULTIPLES PAGE
# =================================================================
elif st.session_state["page"] == "multiples":
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        st.warning("No ticker selected.")
    else:
        render_multiples(ticker)


# =================================================================
# RATIOS PAGE
# =================================================================
elif st.session_state["page"] == "ratios":
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        st.warning("No ticker selected.")
    else:
        render_ratios(ticker=ticker)


# =================================================================
# VALUATION PAGE  (internal key: "dcf")
# =================================================================
elif st.session_state["page"] == "dcf":
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        st.warning("No ticker selected.")
    else:
        data    = st.session_state.get("ticker_data_cache",    {}).get(ticker)
        summary = st.session_state.get("ticker_summary_cache", {}).get(ticker)
        render_dcf(ticker=ticker, summary=summary, data=data)


# =================================================================
# PERFORMANCE PAGE
# =================================================================
elif st.session_state["page"] == "performance":
    render_performance()


# =================================================================
# FOOTER
# =================================================================
st.sidebar.markdown("---")
