# ui/performance.py
"""
Portfolio Performance Tracking — Plain-English Edition
=======================================================
Design philosophy: "Your money's story, told simply."
"""

from __future__ import annotations

import hashlib
import os
import traceback
import warnings
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf
from scipy import stats
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=FutureWarning)

UP      = "#00C805"
DOWN    = "#FF3B30"
BLUE    = "#0A7CFF"
ORANGE  = "#FF9F0A"
GREY    = "#6E6E73"
BORDER  = "#E5E5EA"
CARD_BG = "transparent"

SECTOR_MAP: Dict[str, str] = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology",
    "GOOG":"Technology","NVDA":"Technology","META":"Technology",
    "AVGO":"Technology","CSCO":"Technology","ACN":"Technology",
    "ADBE":"Technology","CRM":"Technology","ORCL":"Technology",
    "INTC":"Technology","AMD":"Technology","QCOM":"Technology",
    "TSM":"Technology","ASML":"Technology",
    "AMZN":"Consumer Cyclical","TSLA":"Consumer Cyclical",
    "HD":"Consumer Cyclical","NKE":"Consumer Cyclical",
    "WMT":"Consumer Defensive","PG":"Consumer Defensive",
    "KO":"Consumer Defensive","COST":"Consumer Defensive",
    "PEP":"Consumer Defensive",
    "BRK.B":"Financial Services","V":"Financial Services",
    "JPM":"Financial Services","MA":"Financial Services",
    "BAC":"Financial Services",
    "UNH":"Healthcare","JNJ":"Healthcare","PFE":"Healthcare",
    "ABBV":"Healthcare","TMO":"Healthcare","MRK":"Healthcare",
    "LRN":"Education","STRL":"Industrials",
    "WM":"Industrials","UNP":"Industrials",
    "XOM":"Energy","CVX":"Energy",
    "DIS":"Communication Services","NFLX":"Communication Services",
    "VZ":"Communication Services","CENTA":"Consumer Defensive",
}

METRIC_LABELS = {
    "net_invested":         ("Money I Put In",
                             "Total cash I've deposited into my portfolio over time."),
    "current_value":        ("What It's Worth Today",
                             "Current market value of all my stocks + any uninvested cash."),
    "total_return_dollars": ("Total Profit or Loss",
                             "Dollars I've made compared to what I put in."),
    "total_return_pct":     ("Simple Total Return",
                             "Total profit as a percentage of what I originally invested."),
    "period_return":        ("Return for This Period",
                             "How much my portfolio grew (or shrank) during the time window I selected."),
    "xirr":                 ("Annualized Return (My Timing)",
                             "Accounting for exactly when I deposited money."),
    "cagr":                 ("Return Over the Last Year",
                             "Annual rate if portfolio had grown at a perfectly steady rate."),
    "ann_vol":              ("How Much It Bounces Around",
                             "A measure of how wildly my portfolio swings day-to-day. Lower is calmer."),
    "sharpe":               ("Quality of Return",
                             "How much return I'm getting per unit of risk. Above 1 is good. Above 2 is excellent."),
    "sortino":              ("Downside Risk Score",
                             "Like the Quality of Return score, but only penalizes for bad days — not good ones."),
    "max_drawdown":         ("Biggest Drop Ever",
                             "The largest percentage fall from a peak to a trough my portfolio has experienced."),
    "calmar":               ("Reward vs Worst-Case",
                             "Compares my annual growth to my worst-ever drop. Higher means I recovered well."),
    "beta":                 ("How I Move With the Market",
                             "If the stock market rises 10%, a beta of 1.2 means my portfolio tends to rise ~12%. Below 1 means calmer than the market."),
    "alpha":                ("My Edge Over the Market",
                             "Extra return I've earned above what the market alone would have given me. Positive alpha = I beat the market."),
    "tracking_error":       ("How Different I Am From the Index",
                             "How much my returns diverge from the S&P 500 on a typical day."),
    "info_ratio":           ("Consistency of Beating the Market",
                             "How reliably I outperform the S&P 500. Above 0.5 is good; above 1 is excellent."),
    "excess_return_ann":    ("Extra Return vs S&P 500",
                             "How many percentage points per year I earned above (or below) the S&P 500."),
}

def _label(key: str) -> str:
    return METRIC_LABELS.get(key, (key, ""))[0]

def _tooltip(key: str) -> str:
    return METRIC_LABELS.get(key, ("", key))[1]

def _metric_card(key, value_str, color="#ffffff", emoji="",
                 override_label="", override_tooltip="") -> str:
    label   = override_label   or _label(key)
    tooltip = override_tooltip or _tooltip(key)
    return f"""
<div title="{tooltip}" style="background:{CARD_BG};border-radius:12px;padding:16px 18px;
     cursor:help;border:1px solid {BORDER};height:100%;">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.05em;color:#ffffff;
              text-transform:uppercase;margin-bottom:6px;">{(emoji+' ') if emoji else ''}{label}</div>
  <div style="font-size:26px;font-weight:700;line-height:1.1;color:{color};">{value_str}</div>
  <div style="font-size:11px;color:#ffffff;margin-top:6px;line-height:1.4;opacity:0.7;">{tooltip}</div>
</div>"""

def _verdict_card(title, verdict, verdict_color, body, fixed_height: Optional[int] = None) -> str:
    height_style = f"height:{fixed_height}px;" if fixed_height else ""
    return f"""
<div style="background:{CARD_BG};border-radius:12px;padding:18px 20px;border:1px solid {BORDER};
     border-left:4px solid {verdict_color};margin-bottom:12px;{height_style}overflow:hidden;">
  <div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:4px;">{title}</div>
  <div style="font-size:15px;font-weight:700;color:{verdict_color};margin-bottom:6px;">{verdict}</div>
  <div style="font-size:13px;color:#ffffff;opacity:0.85;line-height:1.6;">{body}</div>
</div>"""

def _section_header(title, subtitle="") -> None:
    sub = f'<p style="color:#ffffff;opacity:0.7;font-size:14px;margin:2px 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div style="margin:28px 0 12px 0;"><h3 style="font-size:20px;font-weight:700;color:#ffffff;margin:0;">'
        f'{title}</h3>{sub}</div>', unsafe_allow_html=True)

def _divider() -> None:
    st.markdown('<hr style="border:none;border-top:1px solid #E5E5EA;margin:20px 0;">', unsafe_allow_html=True)


# ── Narrative ────────────────────────────────────────────────────────────────

def _generate_narrative(returns, risk, start_date, transactions) -> str:
    parts = []
    days = (datetime.now() - start_date).days
    if days < 60:        tenure = "just started investing"
    elif days < 365:     tenure = f"been investing for about {days//30} month{'s' if days//30>1 else ''}"
    elif days < 730:     tenure = "been investing for about a year"
    else:                tenure = f"been investing for {round(days/365.25,1)} years"
    parts.append(f"I've {tenure}.")

    invested = returns.get("net_invested", 0)
    current  = returns.get("current_value", 0)
    gain     = returns.get("total_return_dollars", 0)
    pct      = returns.get("total_return_pct", 0)
    if invested > 0:
        if gain > 0:
            parts.append(f"I put in **${invested:,.0f}** and it's now worth **${current:,.0f}** — a gain of **${gain:,.0f} ({pct:+.1f}%)**.")
        elif gain < 0:
            parts.append(f"I put in **${invested:,.0f}** and it's currently worth **${current:,.0f}** — that's **${abs(gain):,.0f} ({pct:+.1f}%)** below what I invested.")
        else:
            parts.append(f"I put in **${invested:,.0f}** and it's currently worth about the same.")

    cagr = risk.get("cagr", np.nan)
    if not np.isnan(cagr):
        c = cagr*100
        comment = ("" if c>15 else
                   "" if c>8 else
                   "" if c>3 else
                   "" if c>0 else
                   "")
        parts.append(f"My portfolio has grown at roughly {c:.1f}% per year. {comment}")

    vol = risk.get("ann_vol", np.nan)
    if not np.isnan(vol):
        v = vol*100
        parts.append("" if v<8 else
                      "" if v<15 else
                      "" if v<25 else
                      "")

    mdd = risk.get("max_drawdown", np.nan)
    if not np.isnan(mdd) and mdd < -0.10:
        parts.append(f"The biggest drop I've experienced was {mdd*100:.1f}% from a peak.")

    alpha = risk.get("alpha", np.nan)
    if not np.isnan(alpha):
        ap = alpha*100
        if ap > 2:    parts.append(f"I've been beating the S&P 500 by about {ap:.1f}% per year.")
        elif ap < -2: parts.append(f"The S&P 500 has been outperforming my portfolio by about {abs(ap):.1f} percentage points per year.")

    return " ".join(parts)


# ── XIRR ─────────────────────────────────────────────────────────────────────

def xirr(dates, cash_flows) -> float:
    if len(dates) < 2: return np.nan
    t0  = min(dates)
    yrs = np.array([(d - t0).days / 365.25 for d in dates])
    cfs = np.array(cash_flows, dtype=float)
    def npv(r): return float(np.sum(cfs / (1.0 + r) ** yrs))
    try:    return brentq(npv, -0.9999, 100.0, maxiter=500, xtol=1e-8)
    except: return np.nan


# ── Risk-free rate ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def _risk_free_rate() -> float:
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
        r   = requests.get(url, timeout=5)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text)); df.columns = ["date","rate"]
            df = df[df["rate"]!="."].copy()
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce").dropna()
            if not df.empty: return float(df["rate"].iloc[-1]) / 100.0
    except Exception: pass
    return 0.0


# ── Hardcoded default transactions ───────────────────────────────────────────
# These are the owner's transactions, loaded automatically on startup.
# Any user can override them by uploading their own CSV via the sidebar.

_DEFAULT_TRANSACTIONS_CSV = """date,ticker,action,shares,price
5/27/2025,MSFT,buy,2,457.17
5/27/2025,CENTA,buy,20,31.83
5/27/2025,AMZN,buy,5,202.86
5/27/2025,,deposit,0,3000
5/30/2025,LRN,buy,3,150.61
6/2/2025,CENTA,sell,5,31.71
6/2/2025,LRN,buy,1,151.82
6/11/2025,,deposit,0,1000
6/13/2025,WM,buy,3,237.39
6/20/2025,NVDA,buy,2,145.48
6/25/2025,CENTA,sell,10,31.39
6/25/2025,NVDA,buy,2,152.54
6/27/2025,CENTA,sell,5,31.17
6/27/2025,GOOGL,buy,1,173.74
6/30/2025,42809H107,dividend,0,1.5
6/30/2025,UNP,dividend,0,1.34
7/21/2025,CVX,buy,3,149.68
7/21/2025,42809H107,sell,3,160.26
7/25/2025,CVX,sell,2,155.37
7/31/2025,STRL,buy,3,266.99
7/31/2025,WM,sell,2,232.47
9/8/2025,AMZN,sell,1,234.16
9/8/2025,AMZN,sell,1,234.29
9/8/2025,,deposit,0,60
9/8/2025,TSM,buy,4,245.17
9/8/2025,UNP,sell,1,217.65
9/8/2025,WM,sell,1,218.69
9/10/2025,CVX,dividend,0,1.71
9/11/2025,MSFT,dividend,0,1.66
9/15/2025,GOOGL,dividend,0,0.21
9/15/2025,LRN,sell,4,138.6
9/15/2025,AMZN,buy,1,231.47
9/15/2025,STRL,buy,1,323.45
9/18/2025,,deposit,0,950
9/18/2025,ASML,buy,1,921.06
9/30/2025,UNP,dividend,0,1.38
10/2/2025,NVDA,dividend,0,0.04
10/9/2025,TSM,dividend,0,3.29
10/16/2025,AMZN,sell,4,215.59
10/20/2025,,withdrawal,0,909.03
11/6/2025,ASML,dividend,0,1.86
11/11/2025,MSFT,sell,1,509.19
11/11/2025,NVDA,sell,4,195.19
11/12/2025,,withdrawal,0,1291.78
12/10/2025,CVX,dividend,0,1.71
12/11/2025,MSFT,dividend,0,0.91
12/15/2025,GOOGL,dividend,0,0.21
1/2/2026,TSM,sell,4,321.56
1/2/2026,STRL,sell,3,316.91
1/2/2026,MSFT,sell,1,472.47
1/2/2026,GOOGL,sell,1,315
1/2/2026,ASML,sell,1,1169.22
1/5/2026,,withdrawal,0,4196.58
1/8/2026,TSM,dividend,0,3.18
"""


def _parse_transactions_csv(csv_text: str) -> Optional[pd.DataFrame]:
    """Parse and validate a CSV string into a transactions DataFrame. Returns None on failure."""
    try:
        df = pd.read_csv(StringIO(csv_text))
        required = {"date", "ticker", "action", "shares", "price"}
        if not required.issubset(df.columns):
            return None
        df["date"]   = pd.to_datetime(df["date"])
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
        df["price"]  = pd.to_numeric(df["price"],  errors="coerce").fillna(0)
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        cash_actions = {"deposit", "withdrawal", "dividend"}
        is_cash      = df["action"].isin(cash_actions) & (df["shares"] == 0)
        df["total"]  = np.where(is_cash, df["price"], df["shares"] * df["price"])
        valid_actions = {"buy", "sell", "dividend", "deposit", "withdrawal"}
        bad = set(df["action"].unique()) - valid_actions
        if bad:
            return None
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


# ── Session state + auto-load ─────────────────────────────────────────────────

def _init_state() -> None:
    if "transactions" not in st.session_state:
        # 1. Try loading from disk (legacy / other environments)
        _try_autoload_csv()

    if "transactions" not in st.session_state:
        # 2. Fall back to hardcoded default transactions
        df = _parse_transactions_csv(_DEFAULT_TRANSACTIONS_CSV)
        if df is not None:
            st.session_state.transactions    = df
            st.session_state.price_cache     = {}
            st.session_state._source         = "default"
        else:
            # 3. Truly empty — should never happen unless CSV above is malformed
            st.session_state.transactions = pd.DataFrame(
                columns=["date", "ticker", "action", "shares", "price", "total"])
            st.session_state._source = "empty"

    if "price_cache" not in st.session_state:
        st.session_state.price_cache = {}


_DEFAULT_CSV_PATHS = [
    "portfolio_transactions.csv",
    "ui/portfolio_transactions.csv",
    "data/portfolio_transactions.csv",
]

def _try_autoload_csv() -> bool:
    """Silently load portfolio_transactions.csv from well-known paths on first run."""
    for path in _DEFAULT_CSV_PATHS:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            required = {"date","ticker","action","shares","price"}
            if not required.issubset(df.columns): continue
            df["date"]   = pd.to_datetime(df["date"])
            df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
            df["price"]  = pd.to_numeric(df["price"],  errors="coerce").fillna(0)
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            df["action"] = df["action"].astype(str).str.strip().str.lower()
            cash_actions = {"deposit","withdrawal","dividend"}
            is_cash      = df["action"].isin(cash_actions) & (df["shares"]==0)
            df["total"]  = np.where(is_cash, df["price"], df["shares"]*df["price"])
            valid_actions = {"buy","sell","dividend","deposit","withdrawal"}
            if not set(df["action"].unique()).issubset(valid_actions): continue
            st.session_state.transactions    = df.sort_values("date").reset_index(drop=True)
            st.session_state.price_cache     = {}
            st.session_state._source         = f"file:{path}"
            return True
        except Exception:
            continue
    return False


# ── Ticker validation ─────────────────────────────────────────────────────────

def _is_valid_ticker(t) -> bool:
    if not isinstance(t, str): return False
    t = t.strip()
    if not t or t.lower()=="nan": return False
    if len(t)==9 and t.isalnum() and sum(c.isdigit() for c in t)>=3: return False
    return True

def _stock_tickers_from(df) -> List[str]:
    return [t for t in df["ticker"].unique() if _is_valid_ticker(t)]


# ── Transaction helpers ───────────────────────────────────────────────────────

def load_transactions_from_csv(file) -> bool:
    try:
        df = pd.read_csv(file)
        required = {"date","ticker","action","shares","price"}
        missing  = required - set(df.columns)
        if missing: st.error(f"Missing columns: {', '.join(missing)}"); return False
        df["date"]   = pd.to_datetime(df["date"])
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
        df["price"]  = pd.to_numeric(df["price"],  errors="coerce").fillna(0)
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        cash_actions = {"deposit","withdrawal","dividend"}
        is_cash_row  = df["action"].isin(cash_actions) & (df["shares"]==0)
        df["total"]  = np.where(is_cash_row, df["price"], df["shares"]*df["price"])
        valid_actions = {"buy","sell","dividend","deposit","withdrawal"}
        bad = set(df["action"].unique()) - valid_actions
        if bad: st.error(f"Unknown action(s): {bad}. Allowed: {valid_actions}"); return False
        st.session_state.transactions = df.sort_values("date").reset_index(drop=True)
        st.session_state.price_cache  = {}
        st.session_state._source      = "uploaded"
        all_tickers = df[df["action"].isin({"buy","sell"})]["ticker"].unique()
        cusips = [t for t in all_tickers
                  if not _is_valid_ticker(t) and isinstance(t,str) and t.strip() and t.lower()!="nan"]
        if cusips:
            st.warning(f"⚠️ These tickers look like internal ID numbers and will be skipped: **{', '.join(cusips)}**. Replace them with actual symbols like AAPL or MSFT.")
        return True
    except Exception as exc:
        st.error(f"Couldn't read the file: {exc}"); return False


# ── Price data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _current_price(ticker: str) -> float:
    try:
        info  = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price: return float(price)
        hist  = yf.Ticker(ticker).history(period="2d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    except: return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def _ticker_meta(ticker: str) -> Dict[str, str]:
    try:
        info = yf.Ticker(ticker).info
        return {"name": info.get("longName", ticker), "sector": info.get("sector", SECTOR_MAP.get(ticker,"Other"))}
    except: return {"name": ticker, "sector": SECTOR_MAP.get(ticker,"Other")}

@st.cache_data(ttl=3600, show_spinner=False)
def _benchmark_history(start, end, symbol="SPY") -> pd.DataFrame:
    try:
        hist = yf.Ticker(symbol).history(start=start-timedelta(days=5), end=end+timedelta(days=2), auto_adjust=True)
        if hist.empty: return pd.DataFrame(columns=["date","close"])
        df = hist.reset_index()[["Date","Close"]].copy(); df.columns = ["date","close"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["date","close"]); df = df[df["close"]>0]
        return df.sort_values("date").reset_index(drop=True)
    except: return pd.DataFrame(columns=["date","close"])


# ── Portfolio calculations ────────────────────────────────────────────────────

def _current_holdings(transactions, as_of=None) -> Dict:
    if as_of is None: as_of = datetime.now()
    df = transactions[transactions["date"] <= as_of].copy()
    if df.empty: return {}
    portfolio = {}
    for ticker in _stock_tickers_from(df):
        t = df[df["ticker"]==ticker]; shares, cost = 0.0, 0.0
        for _, row in t.iterrows():
            act = row["action"].lower()
            if act=="buy":                      shares += float(row["shares"]); cost += float(row["total"])
            elif act=="sell" and shares > 0:
                cost_per = cost/shares; shares -= float(row["shares"]); cost -= float(row["shares"])*cost_per
        if shares > 1e-6:
            price = _current_price(ticker); meta = _ticker_meta(ticker)
            portfolio[ticker] = {"shares":shares,"cost_basis":max(cost,0),"current_price":price,
                                 "current_value":shares*price,"avg_cost":cost/shares,
                                 "sector":meta["sector"],"name":meta["name"]}
    return portfolio

def _build_cash_ledger(tx, date_range) -> pd.Series:
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.tz_localize(None).dt.normalize()
    dr_norm = pd.DatetimeIndex(
        pd.to_datetime(date_range).tz_localize(None) if date_range.tz is not None else pd.to_datetime(date_range)
    ).normalize()
    cash_delta = pd.Series(0.0, index=dr_norm)
    for _, row in tx.iterrows():
        d = row["date"]; amt = float(row["total"]) if pd.notna(row["total"]) else 0.0; act = row["action"]
        mask = dr_norm >= d
        if not mask.any(): continue
        target = dr_norm[mask][0]
        if act in ("deposit","dividend"):  cash_delta[target] += amt
        elif act=="sell":                   cash_delta[target] += amt
        elif act=="buy":                    cash_delta[target] -= amt
        elif act=="withdrawal":             cash_delta[target] -= amt
    return cash_delta.cumsum()

def _portfolio_returns(transactions) -> Dict:
    empty = dict(xirr=np.nan,total_return_pct=0.0,total_return_dollars=0.0,net_invested=0.0,current_value=0.0)
    if transactions.empty: return empty
    holdings    = _current_holdings(transactions)
    stock_value = sum(h["current_value"] for h in holdings.values())
    tx_sorted   = transactions.sort_values("date").copy()
    cash = 0.0
    for _, row in tx_sorted.iterrows():
        act = row["action"].lower(); amt = float(row["total"])
        if act in ("deposit","dividend"): cash += amt
        elif act=="sell":                  cash += amt
        elif act=="buy":                   cash -= amt
        elif act=="withdrawal":            cash -= amt
    current_value = stock_value + max(cash,0.0)
    dates, cfs = [], []
    for _, row in tx_sorted.iterrows():
        act = row["action"].lower(); amt = float(row["total"])
        if act=="deposit":      dates.append(row["date"]); cfs.append(-amt)
        elif act=="withdrawal": dates.append(row["date"]); cfs.append(+amt)
    dates.append(datetime.now()); cfs.append(current_value)
    xirr_val  = xirr(dates, cfs) if len(dates) >= 2 else np.nan
    invested  = sum(float(r["total"]) for _,r in tx_sorted.iterrows() if r["action"].lower()=="deposit")
    divested  = sum(float(r["total"]) for _,r in tx_sorted.iterrows() if r["action"].lower()=="withdrawal")
    net = invested - divested; gain = current_value - net
    return dict(xirr=xirr_val,total_return_pct=(gain/net*100) if net>0 else 0.0,
                total_return_dollars=gain,net_invested=net,current_value=current_value)


# ── Historical value (log-return TWR) ────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _historical_value(_tx_hash, tx_csv, start, end) -> pd.DataFrame:
    tx = pd.read_csv(StringIO(tx_csv))
    tx["date"]   = pd.to_datetime(tx["date"]).dt.tz_localize(None).dt.normalize()
    tx["shares"] = pd.to_numeric(tx["shares"], errors="coerce").fillna(0)
    tx["price"]  = pd.to_numeric(tx["price"],  errors="coerce").fillna(0)
    tx["action"] = tx["action"].astype(str).str.strip().str.lower()
    tx["ticker"] = tx["ticker"].astype(str).str.strip().str.upper()
    cash_actions = {"deposit","withdrawal","dividend"}
    is_cash_row  = tx["action"].isin(cash_actions) & (tx["shares"]==0)
    tx["total"]  = np.where(is_cash_row, tx["price"], tx["shares"]*tx["price"])
    stock_tickers = [t for t in tx["ticker"].unique() if _is_valid_ticker(t)]

    today       = pd.Timestamp(datetime.now()).tz_localize(None).normalize()
    fetch_start = start - timedelta(days=10); fetch_end = today + timedelta(days=2)
    price_df    = pd.DataFrame(index=pd.date_range(fetch_start, fetch_end, freq="D"))
    price_df.index = price_df.index.tz_localize(None).normalize()
    for ticker in stock_tickers:
        try:
            h = yf.Ticker(ticker).history(start=fetch_start, end=fetch_end, auto_adjust=True)
            if h.empty: continue
            s = h["Close"].copy()
            if s.index.tz is not None: s.index = s.index.tz_localize(None)
            s.index = pd.to_datetime(s.index).normalize()
            price_df[ticker] = s.reindex(price_df.index).ffill()
        except: pass

    def _price_on(ticker, dt):
        if ticker not in price_df.columns: return 0.0
        col = price_df[ticker]; dt_n = pd.Timestamp(dt).tz_localize(None).normalize()
        subset = col[col.index <= dt_n].dropna()
        return float(subset.iloc[-1]) if not subset.empty else 0.0

    date_range = pd.date_range(start, today, freq="B")
    date_range = pd.DatetimeIndex(
        pd.to_datetime(date_range).tz_localize(None) if date_range.tz is not None else pd.to_datetime(date_range)
    ).normalize()
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    tx = tx[tx["date"]>=start_ts].copy().reset_index(drop=True)

    shares_df = pd.DataFrame(0.0, index=date_range, columns=stock_tickers)
    for ticker in stock_tickers:
        t_tx = tx[tx["ticker"]==ticker].sort_values("date")
        if t_tx.empty: continue
        for _, row in t_tx.iterrows():
            d = pd.Timestamp(row["date"]).tz_localize(None).normalize(); act = row["action"]
            mask = date_range>=d
            if not mask.any(): continue
            snap = date_range[mask][0]
            if act=="buy":        shares_df.loc[snap:,ticker] += float(row["shares"])
            elif act=="sell":     shares_df.loc[snap:,ticker] -= float(row["shares"])
        shares_df[ticker] = shares_df[ticker].clip(lower=0)

    cash_ledger = _build_cash_ledger(tx, date_range)
    ext_cf = pd.Series(0.0, index=date_range)
    for _, row in tx.iterrows():
        d = pd.Timestamp(row["date"]).tz_localize(None).normalize()
        amt = float(row["total"]) if pd.notna(row["total"]) else 0.0; act = row["action"]
        mask = date_range>=d
        if not mask.any(): continue
        snap = date_range[mask][0]
        if act=="deposit":      ext_cf[snap] += amt
        elif act=="withdrawal": ext_cf[snap] -= amt

    stock_val = pd.Series(0.0, index=date_range)
    for ticker in stock_tickers:
        if ticker not in shares_df.columns: continue
        prices = pd.Series([_price_on(ticker,dt) for dt in date_range], index=date_range)
        stock_val += shares_df[ticker] * prices

    cash_series = cash_ledger.reindex(date_range).ffill().fillna(0.0)
    total_val   = stock_val + cash_series
    df = pd.DataFrame({"date":date_range,"stock_value":stock_val.values,"cash_balance":cash_series.values,
                        "total_value":total_val.values,"ext_cash_flow":ext_cf.reindex(date_range).fillna(0.0).values}
    ).sort_values("date").reset_index(drop=True)

    positive_mask = df["total_value"].gt(0)
    if not positive_mask.any(): df["twr_cum"] = np.nan; return df
    df = df.loc[positive_mask.idxmax():].copy().reset_index(drop=True)

    log_r  = np.zeros(len(df))
    v_arr  = df["total_value"].values.astype(np.float64)
    cf_arr = df["ext_cash_flow"].values.astype(np.float64)
    for i in range(1, len(df)):
        base = v_arr[i-1] + cf_arr[i]
        if base>1e-6 and v_arr[i]>1e-6:
            log_r[i] = np.log(v_arr[i]/base); log_r[i] = max(log_r[i], -np.log(2))
    df["twr_cum"] = np.exp(np.cumsum(log_r)); df["twr_cum"] = df["twr_cum"].ffill()
    return df


# ── Benchmark replication ─────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _benchmark_replicated(_tx_hash, tx_csv, start, end, symbol="SPY") -> pd.DataFrame:
    tx = pd.read_csv(StringIO(tx_csv))
    tx["date"]   = pd.to_datetime(tx["date"]).dt.tz_localize(None).dt.normalize()
    tx["action"] = tx["action"].astype(str).str.strip().str.lower()
    cash_actions = {"deposit","withdrawal","dividend"}
    is_cash_row  = tx["action"].isin(cash_actions) & (tx["shares"]==0)
    tx["total"]  = np.where(is_cash_row,
        pd.to_numeric(tx["price"], errors="coerce").fillna(0),
        pd.to_numeric(tx["shares"],errors="coerce").fillna(0)*pd.to_numeric(tx["price"],errors="coerce").fillna(0))
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    tx = tx[tx["date"]>=start_ts].copy()
    bm = _benchmark_history(start, end, symbol)
    if bm.empty: return pd.DataFrame(columns=["date","bm_value","bm_twr_cum"])
    bm = bm.set_index("date")["close"]
    today      = pd.Timestamp(datetime.now()).tz_localize(None).normalize()
    date_range = pd.DatetimeIndex(pd.to_datetime(pd.date_range(start,today,freq="B")).tz_localize(None)).normalize()
    spy_prices = bm.reindex(date_range).ffill()
    units = 0.0; daily_values = []
    for dt in date_range:
        spy_px = float(spy_prices.get(dt, np.nan))
        if np.isnan(spy_px) or spy_px<=0: daily_values.append(np.nan); continue
        day_tx = tx[tx["date"]==dt]
        for _, row in day_tx.iterrows():
            act = row["action"]; amt = float(row["total"]) if pd.notna(row["total"]) else 0.0
            if act=="deposit":              units += amt/spy_px
            elif act=="withdrawal" and units>0:
                frac = min(amt/(units*spy_px),1.0); units -= frac*units
        daily_values.append(units*spy_px)
    result = pd.DataFrame({"date":date_range,"bm_value":daily_values}).dropna(subset=["bm_value"]).reset_index(drop=True)
    if len(result)<2: result["bm_twr_cum"] = np.nan; return result
    bm_cf = pd.Series(0.0, index=date_range)
    for _, row in tx.iterrows():
        d = pd.Timestamp(row["date"]).tz_localize(None).normalize()
        amt = float(row["total"]) if pd.notna(row["total"]) else 0.0; act = row["action"]
        mask = date_range>=d
        if not mask.any(): continue
        snap = date_range[mask][0]
        if act=="deposit":      bm_cf[snap] += amt
        elif act=="withdrawal": bm_cf[snap] -= amt
    v_arr  = result["bm_value"].values.astype(np.float64)
    cf_arr = bm_cf.reindex(result["date"]).fillna(0.0).values.astype(np.float64)
    log_r  = np.zeros(len(result))
    for i in range(1,len(result)):
        base = v_arr[i-1]+cf_arr[i]
        if base>1e-6 and v_arr[i]>1e-6: log_r[i] = np.log(v_arr[i]/base)
    result["bm_twr_cum"] = np.exp(np.cumsum(log_r))
    return result


# ── Risk metrics ──────────────────────────────────────────────────────────────

def compute_risk_metrics(hist, bench_hist=None, rf=0.0, periods_per_year=252) -> Dict:
    result = {k:np.nan for k in ["cagr","ann_vol","sharpe","sortino","max_drawdown","calmar",
                                   "beta","alpha","tracking_error","info_ratio","excess_return_ann"]}
    if hist is None or hist.empty or "twr_cum" not in hist.columns: return result
    twr   = hist["twr_cum"].dropna()
    dates = hist.loc[twr.index,"date"] if "date" in hist.columns else None
    if len(twr)<10: return result
    n_years = (dates.iloc[-1]-dates.iloc[0]).days/365.25 if dates is not None and len(dates)>=2 else len(twr)/periods_per_year
    if n_years>0: result["cagr"] = (float(twr.iloc[-1])/float(twr.iloc[0]))**(1/n_years)-1.0
    dr    = twr.pct_change().fillna(0.0)
    log_r = np.log1p(dr.replace(-1.0,np.nan)).dropna()
    if len(log_r)>=2: result["ann_vol"] = float(log_r.std(ddof=1))*np.sqrt(periods_per_year)
    if not np.isnan(result["cagr"]) and not np.isnan(result["ann_vol"]) and result["ann_vol"]>0:
        result["sharpe"] = (result["cagr"]-rf)/result["ann_vol"]
    negative_r = dr[dr<0]
    if len(negative_r)>=2:
        dsd = float(negative_r.std(ddof=1))*np.sqrt(periods_per_year)
        if dsd>0 and not np.isnan(result["cagr"]): result["sortino"] = (result["cagr"]-rf)/dsd
    rolling_max = twr.cummax(); drawdown = (twr-rolling_max)/rolling_max
    result["max_drawdown"] = float(drawdown.min())
    if not np.isnan(result["cagr"]) and result["max_drawdown"]<0:
        result["calmar"] = result["cagr"]/abs(result["max_drawdown"])
    if bench_hist is not None and not bench_hist.empty and "close" in bench_hist.columns and "date" in bench_hist.columns:
        bm     = bench_hist.set_index("date")["close"]
        port_s = hist[["date","twr_cum"]].dropna().set_index("date")["twr_cum"]
        aligned = port_s.align(bm, join="inner"); port_a, bm_a = aligned[0], aligned[1]
        if len(port_a)>=30:
            port_dr = port_a.pct_change().dropna(); bm_dr = bm_a.pct_change().dropna()
            common  = port_dr.align(bm_dr, join="inner"); p_r, b_r = common[0].values, common[1].values
            if len(p_r)>=30:
                slope,_,_,_,_ = stats.linregress(b_r,p_r); result["beta"] = float(slope)
                rf_daily = (1+rf)**(1/periods_per_year)-1
                alpha_daily = float(np.mean(p_r))-(rf_daily+slope*(float(np.mean(b_r))-rf_daily))
                result["alpha"] = float((1+alpha_daily)**periods_per_year-1)
                active_r = p_r-b_r
                result["tracking_error"] = float(np.std(active_r,ddof=1))*np.sqrt(periods_per_year)
                if result["tracking_error"]>0:
                    result["info_ratio"] = float(np.mean(active_r))*periods_per_year/result["tracking_error"]
                bm_cagr = (float(bm_a.iloc[-1])/float(bm_a.iloc[0]))**(1/n_years)-1 if n_years>0 else np.nan
                result["excess_return_ann"] = (result["cagr"]-bm_cagr
                    if not np.isnan(bm_cagr) and not np.isnan(result["cagr"]) else np.nan)
    return result

def compute_rolling_metrics(hist, bench_hist=None, window=252, periods_per_year=252) -> pd.DataFrame:
    if hist is None or hist.empty or "twr_cum" not in hist.columns:
        return pd.DataFrame(columns=["date","roll_return","roll_vol","roll_beta"])
    twr = hist[["date","twr_cum"]].dropna().sort_values("date").reset_index(drop=True)
    dr  = twr["twr_cum"].pct_change().fillna(0.0)
    out = pd.DataFrame({"date":twr["date"],
                         "roll_return":(1+dr).rolling(window).apply(np.prod,raw=True)-1,
                         "roll_vol":dr.rolling(window).std(ddof=1)*np.sqrt(periods_per_year)})
    if bench_hist is not None and not bench_hist.empty and "close" in bench_hist.columns:
        bm    = bench_hist.set_index("date")["close"]; twr_s = twr.set_index("date")["twr_cum"]
        aligned = twr_s.align(bm, join="inner"); p_dr = aligned[0].pct_change().dropna(); b_dr = aligned[1].pct_change().dropna()
        common = p_dr.align(b_dr, join="inner"); p_r, b_r = common[0], common[1]
        if len(p_r)>=window:
            betas = []
            for i in range(len(p_r)):
                if i<window-1: betas.append(np.nan)
                else:
                    y=p_r.iloc[i-window+1:i+1].values; x=b_r.iloc[i-window+1:i+1].values
                    betas.append(stats.linregress(x,y)[0] if np.std(x)>1e-10 else np.nan)
            beta_s = pd.Series(betas, index=p_r.index)
            out = out.set_index("date"); out["roll_beta"] = beta_s.reindex(out.index); out = out.reset_index()
        else: out["roll_beta"] = np.nan
    else: out["roll_beta"] = np.nan
    return out

def compute_drawdown_series(hist) -> pd.DataFrame:
    if hist is None or hist.empty or "twr_cum" not in hist.columns:
        return pd.DataFrame(columns=["date","twr_cum","drawdown"])
    df   = hist[["date","twr_cum"]].dropna().sort_values("date")
    peak = df["twr_cum"].cummax()
    df["drawdown"] = (df["twr_cum"]-peak)/peak*100.0
    return df.reset_index(drop=True)

def compute_reconciliation(transactions, hist) -> Dict:
    empty = dict(beg_nav=0.0,contributions=0.0,withdrawals=0.0,net_gain=0.0,end_nav=0.0,check=0.0)
    if transactions.empty or hist is None or hist.empty: return empty
    tx = transactions.copy(); tx["date"] = pd.to_datetime(tx["date"]).dt.tz_localize(None).dt.normalize()
    beg_nav = float(hist["total_value"].iloc[0]); end_nav = float(hist["total_value"].iloc[-1])
    contributions = float(tx.loc[tx["action"]=="deposit","total"].sum())
    withdrawals   = float(tx.loc[tx["action"]=="withdrawal","total"].sum())
    net_gain      = end_nav-beg_nav-contributions+withdrawals
    check         = beg_nav+contributions-withdrawals+net_gain-end_nav
    return dict(beg_nav=beg_nav,contributions=contributions,withdrawals=withdrawals,net_gain=net_gain,end_nav=end_nav,check=check)


# ── Period helpers ────────────────────────────────────────────────────────────

_PERIOD_NAMES = {"1M":"1M","3M":"3M","6M":"6M","YTD":"YTD","1Y":"1Y","3Y":"3Y","5Y":"5Y","MAX":"MAX"}

def _has_full_period_history(df: pd.DataFrame, period: str) -> bool:
    if df is None or df.empty or "date" not in df.columns:
        return False
    if period == "MAX" or period == "YTD":
        return True
    offsets = {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5),
    }
    offset = offsets.get(period)
    if offset is None:
        return True
    dates = pd.to_datetime(df["date"]).dropna()
    if dates.empty:
        return False
    end = dates.max()
    required_start = end - offset
    return dates.min() <= required_start

def _slice_period(df, period) -> pd.DataFrame:
    if df is None or df.empty: return df if df is not None else pd.DataFrame()
    end = df["date"].max()
    if period=="MAX": return df.copy()
    if period=="YTD": return df[df["date"]>=pd.Timestamp(end.year,1,1)].copy()
    offsets = {"1M":pd.DateOffset(months=1),"3M":pd.DateOffset(months=3),"6M":pd.DateOffset(months=6),
               "1Y":pd.DateOffset(years=1),"3Y":pd.DateOffset(years=3),"5Y":pd.DateOffset(years=5)}
    offset = offsets.get(period)
    return df[df["date"]>=end-offset].copy() if offset else df.copy()

def _available_periods(start, end) -> List[str]:
    span = (datetime.now()-start).days
    order = ["1M","3M","6M","1Y","3Y","5Y"]; thresholds = {"1M":30,"3M":91,"6M":182,"1Y":365,"3Y":365*3,"5Y":365*5}
    opts = [k for k in order if k == "1Y" or span>=thresholds[k]]
    ytd_insert = next((i for i,k in enumerate(opts) if k=="1Y"), len(opts))
    opts.insert(ytd_insert,"YTD"); opts.append("MAX")
    return opts

def _period_return(df, period) -> float:
    sliced = _slice_period(df, period)
    if sliced is None or len(sliced)<2 or "twr_cum" not in sliced.columns: return 0.0
    v0 = float(sliced["twr_cum"].iloc[0]); v1 = float(sliced["twr_cum"].iloc[-1])
    return (v1/v0-1.0)*100.0 if v0>0 else 0.0


# ── Charts ────────────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    template="plotly_white", hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'SF Pro Display', 'Segoe UI', sans-serif", size=12, color="#ffffff"),
)

def _chart_performance(hist, bench, period) -> Tuple[Optional[go.Figure], Optional[str]]:
    if hist is None or hist.empty or "twr_cum" not in hist.columns:
        return None, "No portfolio data yet."
    port = hist[["date","twr_cum"]].copy()
    port["date"] = pd.to_datetime(port["date"]).dt.tz_localize(None).dt.normalize()
    port = port.dropna(subset=["date","twr_cum"]).sort_values("date").reset_index(drop=True)
    if len(port)<2: return None, "Not enough data to draw a chart yet."
    if period == "1Y" and not _has_full_period_history(port, period):
        return None, "Not enough data for a full '1Y' view."

    port_sliced = _slice_period(port, period)
    if port_sliced is None or len(port_sliced)<2:
        return None, f"Not enough data for the '{_PERIOD_NAMES.get(period,period)}' view."

    bm_sliced = None
    if bench is not None and not bench.empty and "close" in bench.columns:
        bm = bench[["date","close"]].copy()
        bm["date"] = pd.to_datetime(bm["date"]).dt.tz_localize(None).dt.normalize()
        bm = bm.dropna().sort_values("date").reset_index(drop=True)
        merged = pd.merge(port_sliced[["date","twr_cum"]], bm, on="date", how="inner").sort_values("date").reset_index(drop=True)
        if len(merged)>=2:
            bm_sliced   = merged[["date","close"]].copy()
            port_sliced = merged[["date","twr_cum"]].copy()

    if len(port_sliced)<2: return None, "Not enough overlapping data to show both lines."
    anchor = float(port_sliced["twr_cum"].iloc[0])
    if anchor==0: return None, "Cannot draw chart (anchor is zero)."

    port_ret  = (port_sliced["twr_cum"]/anchor-1.0)*100.0
    final_ret = float(port_ret.iloc[-1])
    port_color = UP if final_ret>=0 else DOWN
    r_hex = port_color.lstrip("#")
    rc, gc, bc = int(r_hex[0:2],16), int(r_hex[2:4],16), int(r_hex[4:6],16)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=port_sliced["date"].values, y=port_ret.values,
        name="Your Portfolio", mode="lines",
        line=dict(color=port_color, width=3),
        fill="tozeroy", fillcolor=f"rgba({rc},{gc},{bc},0.08)",
        hovertemplate="<b>%{x|%B %d, %Y}</b><br>Your portfolio: <b>%{y:+.2f}%</b><extra></extra>",
        showlegend=False,
    ))

    port_sign = "▲" if final_ret>=0 else "▼"
    fig.add_annotation(
        xref="paper", x=1.01,
        yref="y",     y=final_ret,
        text=f"{port_sign} {final_ret:+.1f}%",
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(size=12, color=port_color, family="'SF Pro Display','Segoe UI',sans-serif"),
        bgcolor="rgba(0,0,0,0.45)", borderpad=4,
    )

    if bm_sliced is not None:
        anchor_bm = float(bm_sliced["close"].iloc[0])
        if anchor_bm>0:
            bm_ret       = (bm_sliced["close"]/anchor_bm-1.0)*100.0
            bm_final_ret = float(bm_ret.iloc[-1])
            bm_sign      = "▲" if bm_final_ret>=0 else "▼"

            fig.add_trace(go.Scatter(
                x=bm_sliced["date"].values, y=bm_ret.values,
                name="S&P 500", mode="lines",
                line=dict(color=ORANGE, width=3),
                hovertemplate="<b>%{x|%B %d, %Y}</b><br>S&P 500: <b>%{y:+.2f}%</b><extra></extra>",
                showlegend=False,
            ))

            fig.add_annotation(
                xref="paper", x=1.01,
                yref="y",     y=bm_final_ret,
                text=f"{bm_sign} {bm_final_ret:+.1f}%",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=12, color=ORANGE, family="'SF Pro Display','Segoe UI',sans-serif"),
                bgcolor="rgba(0,0,0,0.45)", borderpad=4,
            )

    fig.update_layout(
        **_CHART_LAYOUT, height=380,
        margin=dict(l=55, r=120, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
                   tickfont=dict(color="#ffffff")),
        yaxis=dict(title=dict(text="", font=dict(color="#ffffff")),
                   ticksuffix="%", tickformat="+.1f",
                   showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
                   tickfont=dict(color="#ffffff")),
    )
    return fig, None


def _chart_drawdown(dd_df, bench_df: Optional[pd.DataFrame] = None, period: str = "MAX") -> go.Figure:
    if dd_df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_df["date"], y=dd_df["drawdown"], fill="tozeroy",
        fillcolor="rgba(255,59,48,0.15)", line=dict(color=DOWN, width=3),
        hovertemplate="<b>%{x|%B %d, %Y}</b><br>Down <b>%{y:.1f}%</b> from peak<extra></extra>",
    ))
    port_final = float(dd_df["drawdown"].iloc[-1])
    port_sign = "▲" if port_final >= 0 else "▼"
    fig.add_annotation(
        xref="paper", x=1.01,
        yref="y",     y=port_final,
        text=f"{port_sign} {port_final:+.1f}%",
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(size=12, color=DOWN, family="'SF Pro Display','Segoe UI',sans-serif"),
        bgcolor="rgba(0,0,0,0.45)", borderpad=4,
    )
    if bench_df is not None and not bench_df.empty and "close" in bench_df.columns:
        bm = bench_df[["date","close"]].copy()
        bm["date"] = pd.to_datetime(bm["date"]).dt.tz_localize(None).dt.normalize()
        bm = bm.dropna().sort_values("date").reset_index(drop=True)
        bm = _slice_period(bm, period)
        if len(bm) >= 2:
            peak = bm["close"].cummax()
            bm["drawdown"] = (bm["close"] - peak) / peak * 100.0
            bm_final = float(bm["drawdown"].iloc[-1])
            bm_sign = "▲" if bm_final >= 0 else "▼"
            fig.add_trace(go.Scatter(
                x=bm["date"].values, y=bm["drawdown"].values,
                name="S&P 500", mode="lines",
                line=dict(color=ORANGE, width=3),
                hovertemplate="<b>%{x|%B %d, %Y}</b><br>S&P 500 down <b>%{y:.1f}%</b> from peak<extra></extra>",
                showlegend=False,
            ))
            fig.add_annotation(
                xref="paper", x=1.01,
                yref="y",     y=bm_final,
                text=f"{bm_sign} {bm_final:+.1f}%",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=12, color=ORANGE, family="'SF Pro Display','Segoe UI',sans-serif"),
                bgcolor="rgba(0,0,0,0.45)", borderpad=4,
            )

    fig.update_layout(
        **_CHART_LAYOUT, height=380,
        margin=dict(l=55, r=120, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
                   tickfont=dict(color="#ffffff")),
        yaxis=dict(title=dict(text="", font=dict(color="#ffffff")),
                   ticksuffix="%", tickformat=".1f",
                   showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False,
                   tickfont=dict(color="#ffffff")),
    )
    return fig

def _chart_rolling(roll_df) -> go.Figure:
    if roll_df.empty: return go.Figure()
    has_beta = "roll_beta" in roll_df.columns and roll_df["roll_beta"].notna().any()
    rows = 3 if has_beta else 2
    titles = ["12-Month Return (%)","12-Month Volatility"] + (["12-Month Beta"] if has_beta else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=roll_df["date"],y=(roll_df["roll_return"]*100).values,mode="lines",
        line=dict(color=BLUE,width=1.8),name="12M Return",
        hovertemplate="12M return: <b>%{y:+.2f}%</b><extra></extra>"),row=1,col=1)
    fig.add_hline(y=0,line=dict(color="rgba(255,255,255,0.2)",width=1,dash="dash"),row=1,col=1)
    fig.add_trace(go.Scatter(x=roll_df["date"],y=(roll_df["roll_vol"]*100).values,mode="lines",
        line=dict(color=ORANGE,width=1.8),name="12M Vol",
        hovertemplate="Volatility: <b>%{y:.2f}%</b><extra></extra>"),row=2,col=1)
    if has_beta:
        fig.add_trace(go.Scatter(x=roll_df["date"],y=roll_df["roll_beta"].values,mode="lines",
            line=dict(color=GREY,width=1.8),name="12M Beta",
            hovertemplate="Beta: <b>%{y:.2f}×</b><extra></extra>"),row=3,col=1)
        fig.add_hline(y=1,line=dict(color="rgba(255,255,255,0.2)",width=1,dash="dash"),row=3,col=1)
    fig.update_layout(**_CHART_LAYOUT,height=220*rows,margin=dict(l=70,r=20,t=40,b=50),showlegend=False)
    for i in range(1,rows+1):
        fig.update_xaxes(showgrid=True,gridcolor="rgba(255,255,255,0.1)",row=i,col=1,tickfont=dict(color="#ffffff"))
        fig.update_yaxes(showgrid=True,gridcolor="rgba(255,255,255,0.1)",row=i,col=1,
                         tickfont=dict(color="#ffffff"),title=dict(font=dict(color="#ffffff")))
    for annotation in fig.layout.annotations:
        annotation.font.color = "#ffffff"
    return fig

def _chart_sector(sector_df) -> go.Figure:
    if sector_df.empty: return go.Figure()
    n = len(sector_df)
    blues = [f"rgba(10,{int(124+80*(1-i/max(n-1,1)))},{int(255*(0.7+0.3*(1-i/max(n-1,1))))},0.85)" for i in range(n)]
    fig = go.Figure(go.Bar(y=sector_df["Sector"],x=sector_df["Percentage"],orientation="h",
        marker=dict(color=blues,line=dict(color="rgba(0,0,0,0.1)",width=0.5)),
        text=sector_df["Percentage"].map(lambda v: f"{v:.1f}%"),
        textposition="inside",insidetextanchor="middle",textfont=dict(color="white",size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%  ·  $%{customdata:,.0f}<extra></extra>",
        customdata=sector_df["Value"]))
    fig.update_layout(**_CHART_LAYOUT,height=max(200,n*42),margin=dict(l=140,r=20,t=10,b=20),
                      xaxis=dict(visible=False),yaxis=dict(showgrid=False,tickfont=dict(color="#ffffff")),showlegend=False)
    return fig

def _chart_sector_fixed_height(sector_df, height: int) -> go.Figure:
    """Same as _chart_sector but with a fixed height to match an adjacent element."""
    if sector_df.empty: return go.Figure()
    n = len(sector_df)
    blues = [f"rgba(10,{int(124+80*(1-i/max(n-1,1)))},{int(255*(0.7+0.3*(1-i/max(n-1,1))))},0.85)" for i in range(n)]
    fig = go.Figure(go.Bar(y=sector_df["Sector"],x=sector_df["Percentage"],orientation="h",
        marker=dict(color=blues,line=dict(color="rgba(0,0,0,0.1)",width=0.5)),
        text=sector_df["Percentage"].map(lambda v: f"{v:.1f}%"),
        textposition="inside",insidetextanchor="middle",textfont=dict(color="white",size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%  ·  $%{customdata:,.0f}<extra></extra>",
        customdata=sector_df["Value"]))
    fig.update_layout(**_CHART_LAYOUT, height=height, margin=dict(l=140,r=20,t=10,b=20),
                      xaxis=dict(visible=False),yaxis=dict(showgrid=False,tickfont=dict(color="#ffffff")),showlegend=False)
    return fig


# ── Verdict helpers ───────────────────────────────────────────────────────────

def _sharpe_verdict(s):
    if np.isnan(s): return "Not enough data", GREY
    if s>=2.0:      return "Excellent", UP
    if s>=1.0:      return "Good", UP
    if s>=0.5:      return "Acceptable", ORANGE
    if s>=0.0:      return "Below average", ORANGE
    return "Poor", DOWN

def _mdd_verdict(mdd):
    if np.isnan(mdd): return "Not enough data", GREY
    m = abs(mdd*100)
    if m<5:  return "Barely any drops", UP
    if m<15: return "Manageable drops", UP
    if m<30: return "Some big drops", ORANGE
    return "Large drops", DOWN

def _alpha_verdict(a):
    if np.isnan(a): return "Not enough data", GREY
    ap = a*100
    if ap>3:  return "Beating the market", UP
    if ap>0:  return "Slightly ahead", UP
    if ap>-3: return "Roughly matched", ORANGE
    return "Trailing the market", DOWN

def _beta_verdict(b):
    if np.isnan(b): return "Not enough data", GREY
    if b<0.7:  return "Lower risk than market", UP
    if b<1.1:  return "Moves with the market", ORANGE
    if b<1.5:  return "Slightly more volatile", ORANGE
    return "Much more volatile", DOWN

def _return_verdict(pct):
    if pct>10:  return "Strong gain", UP
    if pct>0:   return "In profit", UP
    if pct>-10: return "Small loss", ORANGE
    return "Significant loss", DOWN


# ── Empty state ───────────────────────────────────────────────────────────────

SAMPLE_CSV = """date,ticker,action,shares,price
2023-01-10,AAPL,buy,10,130.73
2023-01-10,MSFT,buy,5,239.23
2023-01-10,CASH,deposit,0,3700
2023-06-15,NVDA,buy,3,423.02
2023-06-15,CASH,deposit,0,1270
2023-12-01,AAPL,sell,5,191.24
2024-03-01,GOOGL,buy,8,138.50
2024-03-01,CASH,deposit,0,1108
"""

def _render_empty_state() -> None:
    st.markdown("""
<div style="max-width:680px;margin:40px auto;text-align:center;">
  <div style="font-size:60px;margin-bottom:16px;">📊</div>
  <h2 style="font-size:28px;font-weight:700;color:#ffffff;margin-bottom:8px;">Let\'s see how your investments are doing</h2>
  <p style="font-size:16px;color:#ffffff;opacity:0.8;line-height:1.6;margin-bottom:32px;">
    Upload a simple spreadsheet of your investment history and we\'ll show you exactly how your money
    has grown — compared to just putting it in an index fund.
  </p>
</div>""", unsafe_allow_html=True)

    with st.expander("📖 What file do I need to upload?", expanded=True):
        st.markdown("""
**You need a CSV file** (a simple spreadsheet saved as text) with one row per transaction.

| Column | What to put | Example |
|--------|-------------|---------|
| `date` | When the trade happened | `2024-01-15` |
| `ticker` | The stock\'s 3-5 letter symbol | `AAPL` for Apple |
| `action` | What you did | `buy`, `sell`, `deposit` |
| `shares` | How many shares (0 for deposits) | `10` |
| `price` | Price per share (or total cash for deposits) | `185.50` |

For cash deposits use: ticker = `CASH`, action = `deposit`, shares = `0`, price = total amount.

Your brokerage (Fidelity, Schwab, Robinhood, etc.) can export your transaction history as CSV.
        """)

    st.download_button(label="⬇ Download sample CSV to try", data=SAMPLE_CSV,
                       file_name="sample_portfolio.csv", mime="text/csv")
    st.caption("Sample includes Apple, Microsoft, Nvidia and Google trades from 2023–2024.")


# ── Main render ───────────────────────────────────────────────────────────────

def render_performance() -> None:

    st.markdown(f"""
    <style>
    @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap\');
    html, body, [class*="css"] {{ font-family: \'Inter\', \'SF Pro Display\', -apple-system, sans-serif; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {BORDER}; background: transparent !important; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 14px; padding: 10px 18px; color: #ffffff; border-radius: 8px 8px 0 0; background: transparent !important; }}
    .stTabs [aria-selected="true"] {{ color: {BLUE} !important; border-bottom: 2px solid {BLUE} !important; background: transparent !important; }}
    div[data-testid="metric-container"] {{ background: transparent !important; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px; }}
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label {{ color: #ffffff !important; }}
    div[data-testid="stMetricValue"] > div {{ color: #ffffff !important; }}
    div[data-testid="stMetricDelta"] > div {{ color: #ffffff !important; opacity: 0.7; }}
    div[data-testid="stExpander"] {{ background: transparent !important; border: 1px solid {BORDER} !important; border-radius: 10px; }}
    div[data-testid="stExpander"] > details > summary {{ color: #ffffff !important; font-weight: 600; }}
    div[data-testid="stExpander"] > details > summary svg {{ fill: #ffffff !important; }}
    div[data-testid="stAlert"] {{ background: transparent !important; }}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] div {{ color: #ffffff !important; }}
    div[data-testid="stDataFrame"] {{ background: transparent !important; }}
    .stRadio > label, .stMultiSelect > label {{ color: #ffffff !important; }}
    .stRadio div[role="radiogroup"] label, .stRadio div[role="radiogroup"] span {{ color: #ffffff !important; }}
    .stCaption p, small {{ color: #ffffff !important; opacity: 0.7; }}
    .stDownloadButton button {{ background: transparent !important; border: 1px solid {BORDER} !important; color: #ffffff !important; }}
    .block-container, section[data-testid="stSidebar"] > div {{ background: transparent !important; }}
    p, span, label, div {{ color: #ffffff; }}
    .stSelectbox label {{ color: #ffffff !important; }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {{ color: #ffffff !important; }}
    </style>""", unsafe_allow_html=True)

    st.markdown('<h1 style="font-size:32px;font-weight:800;color:#ffffff;margin-bottom:4px;">Performance</h1>',
                unsafe_allow_html=True)

    _init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    # Controls (in-page, collapsed) to keep sidebar minimal.
    debug_mode = False

    # ── Empty state ───────────────────────────────────────────────────────────
    transactions = st.session_state.transactions
    if transactions.empty:
        _render_empty_state(); return

    # ── Compute ───────────────────────────────────────────────────────────────
    with st.spinner("Loading your portfolio data…"):
        try:
            returns  = _portfolio_returns(transactions)
            holdings = _current_holdings(transactions)
        except Exception as e:
            st.error(f"Something went wrong reading your portfolio: {e}")
            if debug_mode: st.code(traceback.format_exc())
            return

    buy_dates  = transactions.loc[transactions["action"]=="buy","date"]
    start_date = (pd.to_datetime(buy_dates).min().to_pydatetime() if not buy_dates.empty
                  else pd.to_datetime(transactions["date"]).min().to_pydatetime())
    end_date   = datetime.now()
    tx_csv     = transactions.to_csv(index=False)
    tx_hash    = hashlib.md5(tx_csv.encode()).hexdigest()

    with st.spinner("Calculating performance history…"):
        try:
            hist = _historical_value(tx_hash, tx_csv, start_date, end_date)
        except Exception as e:
            st.error(f"Error computing portfolio history: {e}")
            if debug_mode: st.code(traceback.format_exc())
            hist = pd.DataFrame()

    bench = _benchmark_history(start_date, end_date)
    with st.spinner("Running risk analysis…"):
        rf   = _risk_free_rate()
        risk = compute_risk_metrics(hist, bench, rf=rf)

    PERIODS = _available_periods(start_date, end_date)
    if "chart_period" not in st.session_state or st.session_state.chart_period not in PERIODS:
        st.session_state.chart_period = "YTD"
    if "risk_chart_period" not in st.session_state or st.session_state.risk_chart_period not in PERIODS:
        st.session_state.risk_chart_period = st.session_state.chart_period if st.session_state.chart_period in PERIODS else "YTD"

    # ── Narrative ─────────────────────────────────────────────────────────────
    narrative = _generate_narrative(returns, risk, start_date, transactions)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(10,124,255,0.15),rgba(10,124,255,0.08));'
        f'border-radius:14px;padding:20px 24px;margin-bottom:20px;border:1px solid rgba(10,124,255,0.4);">'
        f'<div style="font-size:11px;font-weight:700;color:#ffffff;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:8px;">My Portfolio at a Glance</div>'
        f'<div style="font-size:15px;color:#ffffff;line-height:1.7;">{narrative}</div></div>',
        unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Summary","Risk & Drops","What Do I Own?","Where Did My Money Go?","Portfolio"])

    # ════════════ TAB 0 — SUMMARY ════════════════════════════════════════════
    with tabs[0]:
        gain  = returns["total_return_dollars"]; g_col = UP if gain>=0 else DOWN
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.markdown(_metric_card("net_invested", f"${returns['net_invested']:,.0f}",
                                          UP if returns['net_invested'] >= 0 else DOWN), unsafe_allow_html=True)
        with c2: st.markdown(_metric_card("current_value",  f"${returns['current_value']:,.0f}",
                                          UP if returns['current_value'] >= returns['net_invested'] else DOWN), unsafe_allow_html=True)
        with c3:
            sign = "+" if gain>=0 else ""
            st.markdown(_metric_card("total_return_dollars", f"{sign}${abs(gain):,.0f}", g_col), unsafe_allow_html=True)
        with c4:
            cagr = risk.get("cagr",np.nan)
            st.markdown(_metric_card("cagr", f"{cagr*100:+.1f}%/yr" if not np.isnan(cagr) else "N/A",
                                     UP if not np.isnan(cagr) and cagr>=0 else DOWN), unsafe_allow_html=True)
        with c5:
            xv = returns["xirr"]
            st.markdown(_metric_card("xirr", f"{xv*100:+.1f}%/yr" if not np.isnan(xv) else "N/A",
                                     UP if not np.isnan(xv) and xv>=0 else DOWN), unsafe_allow_html=True)
        st.radio("Period:", options=PERIODS, format_func=lambda k: _PERIOD_NAMES.get(k,k),
                 index=PERIODS.index(st.session_state.chart_period), horizontal=True,
                 key="chart_period", label_visibility="collapsed")

        with st.spinner("Drawing your chart…"):
            try:
                fig, err = _chart_performance(hist, bench, st.session_state.chart_period)
                if err:   st.info(f"Chart not available yet: {err}")
                elif fig: st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Chart error: {e}")
                if debug_mode: st.code(traceback.format_exc())

        alpha  = risk.get("alpha",np.nan);  beta  = risk.get("beta",np.nan)
        excess = risk.get("excess_return_ann",np.nan); ir = risk.get("info_ratio",np.nan); te = risk.get("tracking_error",np.nan)
        a_label, a_color = _alpha_verdict(alpha); b_label, b_color = _beta_verdict(beta)
        ca, cb = st.columns(2)
        with ca:
            if not np.isnan(alpha):
                st.markdown(_verdict_card("Am I beating the market? (Alpha)", a_label, a_color,
                    f"I'm earning {alpha*100:+.2f}% per year above what the market alone would have given me. Positive alpha means my stock picks are adding value."), unsafe_allow_html=True)
            if not np.isnan(excess):
                st.markdown(_verdict_card("Extra return vs S&P 500",
                    f"{'+' if excess>0 else '-'} {excess*100:+.2f}% per year", UP if excess>0 else DOWN,
                    f"On average I'm earning {excess*100:+.2f}% per year more (or less) than the S&P 500."), unsafe_allow_html=True)
        with cb:
            if not np.isnan(beta):
                st.markdown(_verdict_card("How closely do I track the market? (Beta)", b_label, b_color,
                    f"Beta = {beta:.2f}x. When the market moves 10%, my portfolio tends to move about {beta*10:.1f}%. "
                    f"{'Less risk but also less upside.' if beta<0.9 else 'Market-like behaviour.' if beta<1.2 else 'More upside but also more downside.'}"), unsafe_allow_html=True)
            if not np.isnan(ir):
                st.markdown(_verdict_card("How consistently am I beating it? (IR)",
                    f"{'Consistent' if ir>0.5 else 'Inconsistent' if ir>0 else 'Lagging'} ({ir:.2f})",
                    UP if ir>0.5 else (ORANGE if ir>0 else DOWN),
                    f"IR = {ir:.2f}. Measures how reliably I outperform — not just whether I do on average. Above 0.5 is considered good."), unsafe_allow_html=True)
        if not np.isnan(te):
            st.info(f"My portfolio differs from the S&P 500 by about **{te*100:.1f}%** per year. "
                    f"A higher number means I'm making more independent bets. A very low number means I'm essentially tracking the index.")

    # ════════════ TAB 1 — RISK & DROPS ═══════════════════════════════════════
    with tabs[1]:
        vol=risk.get("ann_vol",np.nan); sharpe=risk.get("sharpe",np.nan); sortino=risk.get("sortino",np.nan)
        calmar=risk.get("calmar",np.nan); mdd=risk.get("max_drawdown",np.nan)
        s_label,s_color = _sharpe_verdict(sharpe); m_label,m_color = _mdd_verdict(mdd)

        st.radio("Period:", options=PERIODS, format_func=lambda k: _PERIOD_NAMES.get(k,k),
                 index=PERIODS.index(st.session_state.risk_chart_period), horizontal=True,
                 key="risk_chart_period", label_visibility="collapsed")

        if st.session_state.risk_chart_period == "1Y" and not _has_full_period_history(hist, "1Y"):
            st.info("Chart not available yet: Not enough data for a full '1Y' view.")
            dd_df = pd.DataFrame()
        else:
            dd_hist = _slice_period(hist, st.session_state.risk_chart_period)
            dd_df = compute_drawdown_series(dd_hist)

        if not dd_df.empty:
            max_dd=float(dd_df["drawdown"].min()); max_dd_d=dd_df.loc[dd_df["drawdown"].idxmin(),"date"]
            trough_idx=dd_df["drawdown"].idxmin(); post_trough=dd_df.loc[trough_idx:]
            recovery=post_trough[post_trough["drawdown"]>=-0.001]
            recovery_date=recovery["date"].iloc[0] if not recovery.empty else None
            ca,cb,cc = st.columns(3)
            with ca: st.markdown(_metric_card("max_drawdown", f"{max_dd:.1f}%", DOWN,
                override_label="Biggest Ever Drop",
                override_tooltip="Largest fall from a peak to a trough in my portfolio."), unsafe_allow_html=True)
            with cb: st.markdown(_metric_card("max_drawdown", max_dd_d.strftime("%b %Y"), "#ffffff",
                override_label="When It Happened",
                override_tooltip="Date my portfolio hit its lowest point."), unsafe_allow_html=True)
            with cc: st.markdown(_metric_card("max_drawdown",
                recovery_date.strftime("%b %Y") if recovery_date else "Still recovering", "#ffffff",
                override_label="When It Recovered",
                override_tooltip="Date my portfolio returned to its previous all-time high."), unsafe_allow_html=True)
            st.plotly_chart(_chart_drawdown(dd_df, bench, st.session_state.risk_chart_period), width="stretch")

            # Build episodes list
            in_dd=dd_df["drawdown"]<-1.0; episodes=[]; start_ep=None
            for i,is_in in enumerate(in_dd):
                if is_in and start_ep is None:        start_ep=i
                elif not is_in and start_ep is not None:
                    sub=dd_df.iloc[start_ep:i]; wi=sub["drawdown"].idxmin(); dur=len(sub)
                    episodes.append({"Worst point":dd_df.loc[wi,"date"].strftime("%B %Y"),
                                     "Biggest drop":f"{dd_df.loc[wi,'drawdown']:.1f}%",
                                     "How long it lasted":f"{dur} trading days (~{dur//21} months)"}); start_ep=None

            # 50/50 split: Notable drop periods | Risk Quality
            episodes_df = pd.DataFrame(episodes).head(8) if episodes else pd.DataFrame()
            table_rows = len(episodes_df)
            table_height = 46 + 36 * table_rows if table_rows > 0 else 220
            risk_card_height = max(98, int((table_height - 24) / 2))
            col_drops, col_risk = st.columns(2)

            with col_drops:
                if episodes:
                    st.dataframe(episodes_df, hide_index=True, width="stretch", height=table_height)

            with col_risk:
                rq1, rq2 = st.columns(2)
                with rq1:
                    if not np.isnan(sharpe):
                        st.markdown(_verdict_card("Quality of Return (Sharpe)",f"{s_label} ({sharpe:.2f})",s_color,
                            f"For every unit of risk I take, I earn a Sharpe of {sharpe:.2f}. Above 1.0 is good",
                            fixed_height=risk_card_height), unsafe_allow_html=True)
                    if not np.isnan(vol):
                        vl = ("Very calm",UP) if vol<0.08 else ("Moderate",UP) if vol<0.15 else ("Quite volatile",ORANGE) if vol<0.25 else ("Very volatile",DOWN)
                        st.markdown(_verdict_card("Daily Swings (Volatility)",f"{vl[0]} ({vol*100:.1f}%/yr)",vl[1],
                            f"My portfolio swings {vol*100:.1f}% per year. "
                            f"{'Lower volatility = easier to hold emotionally.' if vol<0.20 else 'Higher = bigger swings both ways.'}",
                            fixed_height=risk_card_height), unsafe_allow_html=True)
                with rq2:
                    if not np.isnan(mdd):
                        st.markdown(_verdict_card("Worst-Ever Drop (Max DD)",f"{m_label} ({mdd*100:.1f}%)",m_color,
                            f"At worst my portfolio was {abs(mdd*100):.1f}% below its peak. "
                            f"{'Within a normal range.' if abs(mdd*100)<30 else 'This was a significant drop.'}",
                            fixed_height=risk_card_height), unsafe_allow_html=True)
                    if not np.isnan(sortino):
                        sl = "Good" if sortino>=1 else ("OK" if sortino>=0.5 else "Low")
                        st.markdown(_verdict_card("Downside Risk (Sortino)",f"{sl} ({sortino:.2f})",
                            UP if sortino>=1 else (ORANGE if sortino>=0.5 else DOWN),
                            f"Like Sharpe but only penalizes negative days. Sortino = {sortino:.2f}. Above 1 is healthy.",
                            fixed_height=risk_card_height), unsafe_allow_html=True)

        else: st.info("Not enough data to show drawdown history yet.")

    # ════════════ TAB 2 — WHAT DO I OWN? ════════════════════════════════════
    with tabs[2]:
        if holdings:
            rows = []
            for ticker,d in holdings.items():
                gl=d["current_value"]-d["cost_basis"]; glp=(gl/d["cost_basis"]*100) if d["cost_basis"]>0 else 0.0
                rows.append({"Stock":ticker,"Company":d["name"][:24],"Shares owned":f"{d['shares']:.2f}",
                             "I paid each":f"${d['avg_cost']:.2f}","Worth now":f"${d['current_price']:.2f}",
                             "Total value":f"${d['current_value']:,.2f}","Profit / Loss":f"{'+'if gl>=0 else ''}${gl:,.2f}","Return %":f"{glp:+.1f}%"})
            hdf = pd.DataFrame(rows)
            def _color_str(s):
                raw = s.replace("$","").replace("%","").replace(",","").replace("+","").strip()
                try:
                    v = float(raw); return f"color: {UP}" if v>0 else (f"color: {DOWN}" if v<0 else "")
                except: return ""
            styled = hdf.style.map(_color_str, subset=["Profit / Loss","Return %"])
            table_height = 46 + 36 * len(hdf)
            ca, cb = st.columns([1.3, 0.7])
            with ca:
                st.dataframe(styled, width="stretch", hide_index=True, height=table_height)
            with cb:
                total_val=sum(h["current_value"] for h in holdings.values()); sector_agg: Dict[str,float]={}
                for d in holdings.values():
                    sector_agg.setdefault(d["sector"],0.0); sector_agg[d["sector"]] += d["current_value"]
                sdf = pd.DataFrame([{"Sector":k,"Value":v,"Percentage":v/total_val*100} for k,v in sector_agg.items()]
                ).sort_values("Value",ascending=False).reset_index(drop=True)
                chart_height = max(table_height, 200)
                fig_sector = _chart_sector_fixed_height(sdf, chart_height)
                st.plotly_chart(fig_sector, width="stretch")
        else:
            st.info("I don't have any open positions right now.")

    # ════════════ TAB 3 — WHERE DID MY MONEY GO? ═════════════════════════════
    with tabs[3]:
        recon = compute_reconciliation(transactions, hist)
        col_stmt, col_tx = st.columns(2)

        with col_stmt:
            _section_header("Portfolio Statement","A complete accounting of every dollar — what went in, what came out, what's left")
            steps = [
                ("I started with",       f"${recon['beg_nav']:,.2f}",      "#ffffff"),
                ("+ Money I added",      f"+ ${recon['contributions']:,.2f}", UP),
                ("− Money I withdrew",   f"− ${recon['withdrawals']:,.2f}",  DOWN if recon["withdrawals"]>0 else "#ffffff"),
                ("+ Investment gains",   f"{'+' if recon['net_gain']>=0 else ''}${recon['net_gain']:,.2f}", UP if recon["net_gain"]>=0 else DOWN),
                ("= My portfolio today", f"${recon['end_nav']:,.2f}",        UP),
            ]
            for label,value,color in steps:
                is_total = label.startswith("="); border = f"2px solid {UP}" if is_total else f"1px solid {BORDER}"
                st.markdown(f'<div style="background:{CARD_BG};border-radius:10px;padding:14px 20px;margin-bottom:8px;border:{border};display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:{"16px" if is_total else "14px"};font-weight:{"700" if is_total else "500"};color:#ffffff;">{label}</span>'
                    f'<span style="font-size:{"20px" if is_total else "16px"};font-weight:700;color:{color};">{value}</span></div>', unsafe_allow_html=True)
            check = recon["check"]
            if abs(check)<1.0: st.success(f"✅ Everything checks out (rounding: ${check:.2f})")
            else: st.warning(f"⚠️ Small discrepancy of ${check:.2f} — check that all deposits and withdrawals are in the CSV.")

        with col_tx:
            _section_header("All Transactions")
            filtered = transactions.copy()
            filtered = filtered.sort_values("date", ascending=False)
            filtered["date"]   = filtered["date"].dt.strftime("%b %d, %Y")
            filtered["price"]  = filtered["price"].map(lambda x: f"${float(x):.2f}")
            filtered["total"]  = filtered["total"].map(lambda x: f"${float(x):,.2f}")
            filtered["action"] = filtered["action"].map({"buy":"Buy","sell":"Sell","deposit":"Deposit","withdrawal":"Withdrawal","dividend":"Dividend"}).fillna(filtered["action"])
            filtered = filtered.rename(columns={"date":"Date","ticker":"Stock","action":"Type","shares":"Shares","price":"Price Each","total":"Total Value"})
            st.dataframe(filtered[["Date","Stock","Type","Shares","Price Each","Total Value"]],width="stretch",hide_index=True)
            st.download_button(label="⬇ Download as CSV",data=transactions.to_csv(index=False),
                               file_name=f"my_portfolio_{datetime.now().strftime('%Y-%m-%d')}.csv",mime="text/csv")

    # ════════════ TAB 4 — PORTFOLIO ══════════════════════════════════════════
    with tabs[4]:
        from ui.portfolio_monte_carlo import render_portfolio_monte_carlo
        render_portfolio_monte_carlo(holdings=holdings)

    # ── Debug ─────────────────────────────────────────────────────────────────
    if debug_mode:
        with st.expander("🔧 Diagnostics"):
            if not hist.empty:
                twr_vals = hist["twr_cum"].dropna()
                st.info(f"History rows: {len(hist)} | Date range: {hist['date'].min().date()} → {hist['date'].max().date()} | TWR range: [{twr_vals.min():.4f}, {twr_vals.max():.4f}] | Risk-free rate: {rf*100:.2f}% | Data source: {st.session_state.get('_source', 'unknown')}")
            else: st.warning("Portfolio history DataFrame is empty.")


if __name__ == "__main__":
    render_performance()
