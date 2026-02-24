# capital_iq_style_peer_finder_UPGRADED.py
"""
Capital IQ-Style Automated Peer Finder - UPGRADED VERSION
==========================================================

TRUE Capital IQ Public Comps Methodology:
1. Business Description Similarity (NLP-based with TF-IDF)
2. Financial Multiple Matching (EV/EBITDA, EV/Revenue, P/E, P/B, FCF Yield)
3. Proper Revenue CAGR from SEC (not price proxy)
4. Deterministic SIC Index (no sampling)
5. Float-Adjusted Market Cap Quality
6. Weighted Composite Scoring Model

Key Improvements:
- NLP-based business description similarity using TF-IDF + cosine similarity
- True multiples-based peer matching
- Proper revenue CAGR calculation from SEC companyfacts
- Complete SIC index without sampling bias
- Float quality adjustments
- Configurable weighted scoring model
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import requests
from datetime import datetime, timedelta
import json
import re
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from sec_engine.sec_fetch import fetch_company_submissions, fetch_company_facts
from sec_engine.cik_loader import load_full_cik_map
from sec_engine.normalize import GAAP_MAP
from sec_engine.ltm import extract_annual_series

# ============================================================================
# CACHES
# ============================================================================
PEER_CACHE = {}
COMPANY_DATA_CACHE = {}
SCREENER_CACHE = {}
SIC_INDEX_CACHE = {}
BUSINESS_DESC_CACHE = {}
REVENUE_CAGR_CACHE = {}
SIC_INDEX_CACHE_FILE = Path(__file__).resolve().parent / ".cache" / "sic_index.json"

# ---------------------------------------------------------------------------
# Session-aware peer cache helpers
# ---------------------------------------------------------------------------
# PEER_CACHE is module-level and persists for the lifetime of the Streamlit
# worker process.  That means a result cached in one browser session (with a
# small uploaded_universe) can be returned to a different session that expects
# a richer set of comps.
#
# We solve this by storing peers in st.session_state when available, falling
# back to the module-level dict otherwise.  The cache key includes a hash of
# the current universe so a change in the upload always triggers a re-run.

def _peer_cache_get(key: str):
    """Return cached peers or None.  Prefers session_state over module dict."""
    try:
        cache = st.session_state.setdefault("_peer_cache", {})
        return cache.get(key)
    except Exception:
        return PEER_CACHE.get(key)

def _peer_cache_set(key: str, value) -> None:
    """Store peers in both session_state and module dict."""
    try:
        st.session_state.setdefault("_peer_cache", {})[key] = value
    except Exception:
        pass
    PEER_CACHE[key] = value

# ============================================================================
# SCORING WEIGHTS (Configurable)
# ============================================================================
SCORING_WEIGHTS = {
    "business_description_similarity": 0.35,
    "multiples_similarity": 0.20,
    "revenue_scale_similarity": 0.15,
    "profitability_similarity": 0.10,
    "return_correlation": 0.10,
    "liquidity_float_quality": 0.10,
}

# S&P 500 / large-cap seed list guaranteed into every candidate pool.
# These are the most-compared names on CapIQ/Bloomberg and must never be
# excluded by ETF-holdings gaps.  Organised by GICS sector.
_SECTOR_SEED_UNIVERSE: Dict[str, List[str]] = {
    "Technology": [
        "AAPL","MSFT","NVDA","GOOGL","GOOG","META","AVGO","ORCL","AMD","INTC",
        "QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL","ADI","NXPI","ON",
        "CRM","NOW","ADBE","INTU","SNPS","CDNS","ANSS","PTC","CTSH","EPAM",
        "IBM","HPQ","HPE","DELL","NTAP","STX","WDC","PSTG","ANET","CSCO",
        "JNPR","FFIV","AKAM","NET","ZS","CRWD","PANW","OKTA","S","SNOW",
        "DDOG","MDB","CFLT","HUBS","TEAM","GTLB","ZM","TWLO","PLTR","AI",
        "PATH","APPN","PEGA","SMCI","FSLR","UBER","LYFT","ABNB","DASH",
    ],
    "Communication Services": [
        "GOOGL","GOOG","META","NFLX","DIS","CMCSA","CHTR","VZ","T","TMUS",
        "WBD","PARA","FOXA","FOX","NYT","IAC","MTCH","SNAP","PINS","RDDT",
        "SPOT","TTWO","EA","RBLX","U",
    ],
    "Consumer Cyclical": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","MAR",
        "HLT","GM","F","RIVN","APTV","LEA","BWA","LVS","MGM","WYNN",
        "RCL","CCL","NCLH","DKNG","PENN","PHM","LEN","DHI","TOL","NVR",
        "ORLY","AZO","GPC","TSCO","DG","DLTR","WMT","TGT","COST",
        "BBY","ROST","BURL","URBN","ANF","AEO","RL",
    ],
    "Consumer Defensive": [
        "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","GIS","K",
        "CPB","HRL","CAG","MKC","CHD","CLX","EL","KVUE","KHC","STZ",
        "TAP","SAM","CELH","MNST","KDP",
    ],
    "Healthcare": [
        "JNJ","LLY","ABT","TMO","MRK","AZN","BMY","AMGN","GILD","VRTX",
        "REGN","BIIB","MRNA","BNTX","PFE","NVO","NVS","ABBV","MDT","SYK",
        "BSX","EW","ISRG","ZBH","BAX","BDX","DHR","A","WAT","IQV","CVS",
        "WBA","MCK","ABC","CAH","HUM","UNH","CI","ELV","CNC","MOH","HCA",
        "THC","UHS","INCY","ALNY","IONS","SRPT","EXEL","BMRN",
    ],
    "Financials": [
        "BRK-B","JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","COF",
        "DFS","SYF","ALLY","MET","PRU","AFL","AIG","TRV","PGR","CB","ALL",
        "HIG","ICE","CME","CBOE","NDAQ","V","MA","PYPL","FIS","FI","GPN",
        "SPGI","MCO","MSCI","FDS","VRSK","SOFI","NU","AFRM",
    ],
    "Industrials": [
        "GE","HON","MMM","RTX","LMT","BA","GD","NOC","TDG","TXT","HEI",
        "KTOS","SAIC","LDOS","CACI","CAT","DE","PCAR","CMI","AGCO","ITW",
        "EMR","ETN","ROK","PH","IR","XYL","GNRC","IDEX","GWW","MSC","FAST",
        "SNA","SWK","TT","CARR","OTIS","JCI","UPS","FDX","EXPD","CHRW",
        "XPO","SAIA","ODFL","JBHT","KNX","UAL","DAL","AAL","LUV","CSX",
        "UNP","NSC","WAB",
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","OXY","HAL","BKR","DVN","HES",
        "FANG","MRO","APA","VLO","MPC","PSX","EQT","RRC","AR","WMB",
        "ET","KMI","EPD","OKE","TRP","ENB",
    ],
    "Basic Materials": [
        "LIN","APD","ECL","DD","DOW","LYB","EMN","CE","NEM","GOLD","AEM",
        "NUE","STLD","CLF","AA","FCX","SCCO","ALB","SQM","MLM","VMC",
        "CRH","EXP","PKG","IP","WRK",
    ],
    "Real Estate": [
        "AMT","PLD","EQIX","CCI","SPG","EQR","AVB","MAA","UDR","CPT",
        "ESS","INVH","NNN","O","WPC","ADC","VICI","GLPI","BXP","SLG",
        "CBRE","JLL","VTR","PEAK","WELL","PSA","EXR","CUBE","DLR",
    ],
    "Utilities": [
        "NEE","DUK","SO","D","EXC","AEP","SRE","PCG","ED","XEL","WEC",
        "DTE","CMS","LNT","EVRG","ES","ETR","PPL","EIX","PNW","AWK","WTRG",
    ],
}

CIK_MAP_GLOBAL = load_full_cik_map()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _normalize_ticker_list(tickers: Optional[List[str]]) -> List[str]:
    if not tickers:
        return []
    cleaned = []
    seen = set()
    for ticker in tickers:
        t = str(ticker).strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned


def _industry_keyword_overlap(a: str, b: str) -> float:
    """Return 0..1 overlap score based on normalized industry keywords."""
    if not a or not b:
        return 0.0
    stop_words = {"services", "service", "systems", "group", "holdings", "international", "inc", "corp", "corporation"}
    a_tokens = {t for t in a.lower().replace("&", " ").replace("/", " ").split() if len(t) > 2 and t not in stop_words}
    b_tokens = {t for t in b.lower().replace("&", " ").replace("/", " ").split() if len(t) > 2 and t not in stop_words}
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens.intersection(b_tokens))
    union = len(a_tokens.union(b_tokens))
    return inter / union if union > 0 else 0.0


def _safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _load_sic_index_from_disk(max_age_hours: int = 24) -> Dict[str, List[str]]:
    try:
        if not SIC_INDEX_CACHE_FILE.exists():
            return {}
        age = datetime.now() - datetime.fromtimestamp(SIC_INDEX_CACHE_FILE.stat().st_mtime)
        if age > timedelta(hours=max_age_hours):
            return {}
        payload = json.loads(SIC_INDEX_CACHE_FILE.read_text(encoding="utf-8"))
        index = payload.get("sic_index", {})
        if not isinstance(index, dict):
            return {}
        return {str(k): list(dict.fromkeys([str(t).upper() for t in v if t])) for k, v in index.items()}
    except Exception:
        return {}


def _save_sic_index_to_disk(index: Dict[str, List[str]]) -> None:
    try:
        SIC_INDEX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "sic_index": index,
        }
        SIC_INDEX_CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_return_series(ticker: str, period: str = "2y") -> pd.Series:
    """Cached daily return series for correlation scoring."""
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return pd.Series(dtype=float)
        returns = hist["Close"].pct_change().dropna()
        if returns.empty:
            return pd.Series(dtype=float)
        returns.index = pd.to_datetime(returns.index).tz_localize(None)
        return returns.astype(float)
    except Exception:
        return pd.Series(dtype=float)


def calculate_return_correlation(
    target_ticker: str,
    peer_ticker: str,
    min_overlap_days: int = 120
) -> Optional[float]:
    """Pearson correlation on daily returns, bounded to [-1, 1]."""
    if not target_ticker or not peer_ticker:
        return None
    if target_ticker.upper() == peer_ticker.upper():
        return 1.0

    target_returns = get_daily_return_series(target_ticker.upper(), period="2y")
    peer_returns = get_daily_return_series(peer_ticker.upper(), period="2y")
    if target_returns.empty or peer_returns.empty:
        return None

    aligned = pd.concat([target_returns, peer_returns], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < min_overlap_days:
        return None

    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="pearson")
    if pd.isna(corr):
        return None
    return float(max(-1.0, min(1.0, corr)))


# ============================================================================
# NEW: BUSINESS DESCRIPTION SIMILARITY (NLP-based)
# ============================================================================

def clean_business_description(text: str) -> str:
    """Clean and normalize business description text."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common stop words and corporate terms
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'inc', 'corporation', 'corp',
        'company', 'ltd', 'llc', 'limited', 'plc', 'sa', 'nv', 'ag'
    }
    
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)


@st.cache_data(ttl=3600, show_spinner=False)
def get_business_description(ticker: str) -> str:
    """Fetch and cache business description from yfinance."""
    if ticker in BUSINESS_DESC_CACHE:
        return BUSINESS_DESC_CACHE[ticker]
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        desc = info.get('longBusinessSummary', '')
        
        # Fallback to sector + industry if no description
        if not desc:
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            desc = f"{sector} {industry}"
        
        cleaned = clean_business_description(desc)
        BUSINESS_DESC_CACHE[ticker] = cleaned
        return cleaned
    except Exception:
        return ""


def calculate_business_description_similarity(
    target_ticker: str,
    peer_ticker: str,
    vectorizer: Optional[TfidfVectorizer] = None
) -> float:
    """
    Calculate business description similarity using TF-IDF and cosine similarity.
    Returns a score from 0 to 100.
    """
    target_desc = get_business_description(target_ticker)
    peer_desc = get_business_description(peer_ticker)
    
    if not target_desc or not peer_desc:
        return 0.0
    
    try:
        if TfidfVectorizer is None or cosine_similarity is None:
            target_tokens = set(target_desc.split())
            peer_tokens = set(peer_desc.split())
            if not target_tokens or not peer_tokens:
                return 0.0
            overlap = len(target_tokens.intersection(peer_tokens)) / len(target_tokens.union(peer_tokens))
            return float(overlap * 100)

        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=700,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )

        tfidf_matrix = vectorizer.fit_transform([target_desc, peer_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity * 100)
    except Exception:
        return 0.0


def calculate_business_description_similarity_batch(
    target_ticker: str,
    peer_tickers: List[str],
) -> Dict[str, float]:
    """
    Compute TF-IDF business description similarity in one vectorization pass.
    Returns {peer_ticker: score_0_to_100}.
    """
    peers = [p for p in _normalize_ticker_list(peer_tickers) if p != target_ticker.upper()]
    if not peers:
        return {}

    target_desc = get_business_description(target_ticker.upper())
    if not target_desc:
        return {p: 0.0 for p in peers}

    peer_descs = [get_business_description(p) for p in peers]
    if TfidfVectorizer is None or cosine_similarity is None:
        target_tokens = set(target_desc.split())
        out = {}
        for p, desc in zip(peers, peer_descs):
            tokens = set(desc.split())
            if not tokens or not target_tokens:
                out[p] = 0.0
                continue
            out[p] = float(100.0 * len(tokens.intersection(target_tokens)) / len(tokens.union(target_tokens)))
        return out

    corpus = [target_desc] + peer_descs
    try:
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(corpus)
        target_vec = matrix[0:1]
        peer_vecs = matrix[1:]
        sims = cosine_similarity(target_vec, peer_vecs).flatten()
        return {p: float(max(0.0, min(1.0, s)) * 100.0) for p, s in zip(peers, sims)}
    except Exception:
        return {p: 0.0 for p in peers}


# ============================================================================
# NEW: PROPER REVENUE CAGR FROM SEC
# ============================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_revenue_cagr_from_sec(ticker: str, cik: str, years: int = 3) -> float:
    """
    Calculate proper revenue CAGR from SEC companyfacts.
    Returns annual CAGR as a decimal (e.g., 0.15 for 15%).
    """
    cache_key = f"{ticker}_{years}y"
    if cache_key in REVENUE_CAGR_CACHE:
        return REVENUE_CAGR_CACHE[cache_key]
    
    try:
        # Fetch SEC facts
        facts = fetch_company_facts(cik)
        
        # Extract annual revenue series
        revenue_series = extract_annual_series(facts, GAAP_MAP.get("revenue", []))
        
        if revenue_series is None or revenue_series.empty or len(revenue_series) < years + 1:
            return np.nan
        
        # Sort by date and get last years+1 observations
        revenue_series = revenue_series.sort_index()
        
        # Get start and end values
        start_value = float(revenue_series.iloc[-(years + 1)])
        end_value = float(revenue_series.iloc[-1])
        
        if start_value <= 0 or end_value <= 0:
            return np.nan
        
        # Calculate CAGR
        cagr = (end_value / start_value) ** (1 / years) - 1
        
        REVENUE_CAGR_CACHE[cache_key] = cagr
        return cagr
        
    except Exception:
        return np.nan


# ============================================================================
# NEW: FINANCIAL MULTIPLES SIMILARITY
# ============================================================================

def calculate_multiples_similarity(
    target_profile: dict,
    peer_profile: dict
) -> float:
    """
    Calculate similarity based on financial multiples.
    
    Compares:
    - EV / EBITDA
    - EV / Revenue
    - P / E (LTM)
    - Price / Book
    - FCF Yield
    - EBITDA Margin
    
    Returns score from 0 to 100 based on normalized dispersion.
    """
    multiples_scores = []

    def ratio_similarity(target_val, peer_val, max_ratio_gap=1.5):
        t = _safe_float(target_val)
        p = _safe_float(peer_val)
        if pd.isna(t) or pd.isna(p) or t <= 0 or p <= 0:
            return None
        ratio = max(t, p) / min(t, p)
        gap = ratio - 1.0
        score = max(0.0, 100.0 * (1.0 - min(gap / max_ratio_gap, 1.0)))
        return score

    def delta_similarity(target_val, peer_val, max_abs_gap):
        t = _safe_float(target_val)
        p = _safe_float(peer_val)
        if pd.isna(t) or pd.isna(p):
            return None
        gap = abs(t - p)
        score = max(0.0, 100.0 * (1.0 - min(gap / max_abs_gap, 1.0)))
        return score
    
    # 1. EV / EBITDA
    target_ev = target_profile.get("enterprise_value", 0)
    peer_ev = peer_profile.get("enterprise_value", 0)
    target_ebitda = target_profile.get("ebitda", 0)
    peer_ebitda = peer_profile.get("ebitda", 0)
    
    if target_ev and target_ebitda and peer_ev and peer_ebitda:
        target_ev_ebitda = abs(target_ev / target_ebitda)
        peer_ev_ebitda = abs(peer_ev / peer_ebitda)
        score = ratio_similarity(target_ev_ebitda, peer_ev_ebitda, max_ratio_gap=2.0)
        if score is not None:
            multiples_scores.append(score)
    
    # 2. EV / Revenue
    target_rev = target_profile.get("revenue", 0)
    peer_rev = peer_profile.get("revenue", 0)
    
    if target_ev and target_rev and peer_ev and peer_rev:
        target_ev_rev = abs(target_ev / target_rev)
        peer_ev_rev = abs(peer_ev / peer_rev)
        score = ratio_similarity(target_ev_rev, peer_ev_rev, max_ratio_gap=1.5)
        if score is not None:
            multiples_scores.append(score)
    
    # 3. P/E (LTM)
    target_pe = target_profile.get("pe_ltm", 0)
    peer_pe = peer_profile.get("pe_ltm", 0)
    
    if target_pe and peer_pe and target_pe > 0 and peer_pe > 0:
        score = ratio_similarity(target_pe, peer_pe, max_ratio_gap=2.0)
        if score is not None:
            multiples_scores.append(score)
    
    # 4. Price / Book
    target_pb = target_profile.get("price_to_book", 0)
    peer_pb = peer_profile.get("price_to_book", 0)
    
    if target_pb and peer_pb and target_pb > 0 and peer_pb > 0:
        score = ratio_similarity(target_pb, peer_pb, max_ratio_gap=1.5)
        if score is not None:
            multiples_scores.append(score)
    
    # 5. FCF Yield
    target_fcf_yield = target_profile.get("fcf_yield", 0)
    peer_fcf_yield = peer_profile.get("fcf_yield", 0)
    
    if target_fcf_yield or peer_fcf_yield:
        score = delta_similarity(target_fcf_yield, peer_fcf_yield, max_abs_gap=12.0)
        if score is not None:
            multiples_scores.append(score)
    
    # 6. EBITDA Margin
    target_margin = target_profile.get("ebitda_margin", 0)
    peer_margin = peer_profile.get("ebitda_margin", 0)
    
    if target_margin or peer_margin:
        score = delta_similarity(target_margin, peer_margin, max_abs_gap=0.30)
        if score is not None:
            multiples_scores.append(score)
    
    # Return average score
    if multiples_scores:
        return float(np.mean(multiples_scores))
    return 0.0


# ============================================================================
# NEW: DETERMINISTIC SIC INDEX
# ============================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def build_complete_sic_index() -> Dict[str, List[str]]:
    """
    Build a complete, deterministic SIC code → tickers index.
    No sampling, no bias. Complete coverage.
    
    Returns:
        Dict mapping SIC codes to lists of tickers
    """
    if SIC_INDEX_CACHE:
        return SIC_INDEX_CACHE

    disk_index = _load_sic_index_from_disk(max_age_hours=24)
    if disk_index:
        SIC_INDEX_CACHE.update(disk_index)
        return disk_index

    # Fast deterministic path from the currently loaded screener universe.
    session_sic_map = st.session_state.get("sic_map", {}) if hasattr(st, "session_state") else {}
    if session_sic_map:
        sic_index = {}
        for ticker, sic_code in sorted(session_sic_map.items()):
            sic = str(sic_code).strip()
            if not sic or sic == "0":
                continue
            sic_index.setdefault(sic, []).append(str(ticker).upper())
        for sic in list(sic_index.keys()):
            sic_index[sic] = sorted(list(dict.fromkeys(sic_index[sic])))
        SIC_INDEX_CACHE.update(sic_index)
        _save_sic_index_to_disk(sic_index)
        return sic_index

    # Full refresh fallback (expensive): deterministic over the SEC ticker list.
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': 'Automated Peer Finder research@example.com'}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return {}

        companies_data = response.json()
        cik_to_ticker = {
            str(c['cik_str']).zfill(10): c['ticker'].upper()
            for c in companies_data.values()
            if c.get("ticker")
        }
        ordered_items = sorted(cik_to_ticker.items(), key=lambda x: x[1])
        sic_index = {}

        def _fetch_sic(item):
            cik, ticker = item
            try:
                submissions = fetch_company_submissions(cik)
                sic_code = str(submissions.get('sic', '')).strip()
                if sic_code and sic_code != '0':
                    return sic_code, ticker
            except Exception:
                return None
            return None

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(_fetch_sic, it) for it in ordered_items]
            for future in as_completed(futures):
                result = future.result()
                if not result:
                    continue
                sic_code, ticker = result
                sic_index.setdefault(sic_code, []).append(ticker)

        for sic in list(sic_index.keys()):
            sic_index[sic] = sorted(list(dict.fromkeys(sic_index[sic])))
        SIC_INDEX_CACHE.update(sic_index)
        _save_sic_index_to_disk(sic_index)
        return sic_index
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def get_peers_by_sic_deterministic(sic_code: str, max_results: int = 200) -> List[str]:
    """
    Get all companies with matching SIC code (deterministic, no sampling).
    
    Returns exact matches first, then 3-digit matches, then 2-digit matches.
    """
    if not sic_code:
        return []
    
    sic_index = build_complete_sic_index()
    
    if not sic_index:
        return []
    
    matching_tickers = []
    
    # Exact match
    if sic_code in sic_index:
        matching_tickers.extend(sic_index[sic_code])
    
    # 3-digit match
    if len(matching_tickers) < max_results and len(sic_code) >= 3:
        sic_3digit = sic_code[:3]
        for sic, tickers in sorted(sic_index.items(), key=lambda x: x[0]):
            if sic.startswith(sic_3digit) and sic != sic_code:
                matching_tickers.extend(tickers)
                if len(matching_tickers) >= max_results:
                    break
    
    # 2-digit match
    if len(matching_tickers) < max_results and len(sic_code) >= 2:
        sic_2digit = sic_code[:2]
        for sic, tickers in sorted(sic_index.items(), key=lambda x: x[0]):
            if sic.startswith(sic_2digit) and not sic.startswith(sic_code[:3]):
                matching_tickers.extend(tickers)
                if len(matching_tickers) >= max_results:
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for ticker in matching_tickers:
        if ticker not in seen:
            seen.add(ticker)
            result.append(ticker)
    
    return sorted(result[:max_results])


# ============================================================================
# NEW: FLOAT-ADJUSTED MARKET CAP QUALITY
# ============================================================================

def calculate_float_quality_score(profile: dict) -> float:
    """
    Calculate liquidity and float quality score.
    
    Factors:
    - Float ratio (floatShares / sharesOutstanding)
    - Average volume
    - Market cap
    
    Returns score from 0 to 100.
    """
    score = 50.0  # Base score
    
    # 1. Float ratio
    float_shares = profile.get("float_shares", 0)
    shares_outstanding = profile.get("shares_outstanding", 0)
    
    if float_shares and shares_outstanding and shares_outstanding > 0:
        float_ratio = float_shares / shares_outstanding
        
        # Penalize low float
        if float_ratio < 0.3:
            score -= 30 * (0.3 - float_ratio) / 0.3
        elif float_ratio > 0.7:
            score += 20
        else:
            score += 10
    
    # 2. Average volume (liquidity)
    avg_volume = profile.get("avg_volume", 0)
    market_cap = profile.get("market_cap", 0)
    
    if avg_volume and market_cap and market_cap > 0:
        # Dollar volume as % of market cap
        # Assume average price ~= sqrt(market_cap) for rough estimation
        avg_price = max(1, (market_cap / 1e6) ** 0.5)
        dollar_volume = avg_volume * avg_price
        liquidity_ratio = dollar_volume / (market_cap / 252)  # Daily $ volume vs 1-day market cap
        
        # Higher liquidity = better
        if liquidity_ratio > 0.01:  # >1% daily turnover
            score += 20
        elif liquidity_ratio > 0.005:  # >0.5% daily turnover
            score += 10
        elif liquidity_ratio < 0.001:  # <0.1% daily turnover
            score -= 20
    
    # 3. Absolute volume check
    if avg_volume:
        if avg_volume > 1_000_000:
            score += 10
        elif avg_volume < 100_000:
            score -= 15
    
    return max(0, min(100, score))


# ============================================================================
# COMPANY PROFILE EXTRACTION (Enhanced)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_comprehensive_company_profile(ticker: str) -> dict:
    """
    Extract comprehensive company profile for peer analysis.
    
    Enhanced with:
    - Business description
    - Proper revenue CAGR
    - Additional multiples
    - Float metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        # Get CIK for SEC data
        cik = CIK_MAP_GLOBAL.get(ticker.upper())
        
        # Basic Info
        profile = {
            "ticker": ticker,
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "industry_key": info.get("industryKey", ""),
            "sector_key": info.get("sectorKey", ""),
            
            # Size Metrics
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "revenue": info.get("totalRevenue", 0),
            "employees": info.get("fullTimeEmployees", 0),
            
            # Geographic
            "country": info.get("country", ""),
            "city": info.get("city", ""),
            "state": info.get("state", ""),
            
            # Liquidity Metrics (Enhanced)
            "avg_volume": info.get("averageVolume", 0),
            "avg_volume_10d": info.get("averageVolume10days", 0),
            "float_shares": info.get("floatShares", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            
            # Exchange/Listing
            "exchange": info.get("exchange", ""),
            "quote_type": info.get("quoteType", ""),
            
            # Financial Metrics (Enhanced)
            "profit_margins": info.get("profitMargins", 0),
            "operating_margins": info.get("operatingMargins", 0),
            "ebitda_margin": info.get("ebitdaMargins", 0),
            "roe": info.get("returnOnEquity", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "current_ratio": info.get("currentRatio", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
            
            # Multiples (Enhanced)
            "pe_ltm": info.get("trailingPE", 0),
            "price_to_book": info.get("priceToBook", 0),
            "ebitda": info.get("ebitda", 0),
            
            # FCF
            "free_cash_flow": info.get("freeCashflow", 0),
            
            # Flags
            "is_active": info.get("regularMarketPrice", 0) > 0,
            "market_cap_category": categorize_market_cap(info.get("marketCap", 0)),
            "last_updated": datetime.now().isoformat(),
        }
        
        # Calculate FCF Yield
        if profile["free_cash_flow"] and profile["market_cap"] and profile["market_cap"] > 0:
            profile["fcf_yield"] = (profile["free_cash_flow"] / profile["market_cap"]) * 100
        else:
            profile["fcf_yield"] = 0
        
        # Get proper revenue CAGR from SEC
        if cik:
            revenue_cagr = get_revenue_cagr_from_sec(ticker, cik, years=3)
            profile["revenue_cagr_3y"] = revenue_cagr if not pd.isna(revenue_cagr) else 0
        else:
            profile["revenue_cagr_3y"] = 0
        
        return profile
        
    except Exception as e:
        # Return minimal profile if data fetch fails
        return {
            "ticker": ticker,
            "name": ticker,
            "sector": "",
            "industry": "",
            "market_cap": 0,
            "country": "",
            "is_active": False,
        }


def categorize_market_cap(market_cap: float) -> str:
    """Categorize market cap into bands (Capital IQ style)"""
    if market_cap >= 200e9:
        return "mega_cap"  # $200B+
    elif market_cap >= 10e9:
        return "large_cap"  # $10B - $200B
    elif market_cap >= 2e9:
        return "mid_cap"    # $2B - $10B
    elif market_cap >= 300e6:
        return "small_cap"  # $300M - $2B
    elif market_cap >= 50e6:
        return "micro_cap"  # $50M - $300M
    else:
        return "nano_cap"   # < $50M


# ============================================================================
# INDUSTRY PEER DISCOVERY (Keep existing yfinance method)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def discover_industry_peers_from_yfinance(
    sector: str,
    industry: str,
    max_results: int = 100
) -> List[str]:
    """
    Discover industry peers using yfinance screener.
    Capital IQ uses proprietary screener - we replicate with yfinance + SEC.
    """
    discovered_tickers = []

    try:
        # Method 1: Get holdings from sector ETFs
        industry_etf_map = {
            "Technology": ["XLK", "VGT", "QQQ", "SOXX", "SMH", "IGV", "SKYY", "BUG"],
            "Healthcare": ["XLV", "VHT", "IHI", "IBB", "XBI", "ARKG", "PSCH"],
            "Financials": ["XLF", "VFH", "KBE", "KIE", "KBWP", "IAI", "KCE"],
            "Consumer Cyclical": ["XLY", "VCR", "CARZ", "IBUY", "ONLN"],
            "Consumer Defensive": ["XLP", "VDC", "PBJ"],
            "Energy": ["XLE", "VDE", "OIH", "AMLP", "MLPA"],
            "Industrials": ["XLI", "VIS", "ITA", "XAR", "DFEN", "JETS"],
            "Basic Materials": ["XLB", "VAW", "GDX", "GDXJ", "SIL"],
            "Real Estate": ["XLRE", "VNQ", "IYR", "REM", "MORT"],
            "Communication Services": ["XLC", "VOX", "FIVG", "WCLD"],
            "Utilities": ["XLU", "VPU", "FUTY", "UTES"]
        }

        sector_etfs = industry_etf_map.get(sector, [])
        for etf_ticker in sector_etfs:
            try:
                etf = yf.Ticker(etf_ticker)
                if hasattr(etf, 'funds_data'):
                    funds_data = etf.funds_data
                    holdings = []
                    if hasattr(funds_data, "top_holdings"):
                        try:
                            top_holdings = funds_data.top_holdings
                            if isinstance(top_holdings, pd.DataFrame) and "Symbol" in top_holdings.columns:
                                holdings = top_holdings["Symbol"].dropna().astype(str).tolist()
                        except Exception:
                            holdings = []

                    if not holdings and isinstance(funds_data, dict):
                        maybe_holdings = funds_data.get("holdings", [])
                        for holding in maybe_holdings[:100]:
                            symbol = str(holding.get("symbol", "")).strip().upper()
                            if symbol:
                                holdings.append(symbol)

                    for symbol in holdings[:100]:
                        t = str(symbol).strip().upper()
                        if t and t not in discovered_tickers:
                            discovered_tickers.append(t)
            except Exception:
                continue

        # Method 2: yfinance screener — query by sector/industry directly.
        # This surfaces companies not in any ETF top-holdings list.
        try:
            screener = yf.Screener()
            body = {
                "offset": 0,
                "size": 100,
                "sortField": "marketcap",
                "sortType": "DESC",
                "quoteType": "EQUITY",
                "query": {
                    "operator": "AND",
                    "operands": [
                        {"operator": "eq", "operands": ["sector", sector]},
                    ],
                },
                "userId": "",
                "userIdType": "guid",
            }
            response = screener.set_body(body).fetch()
            quotes = response.get("quotes", []) if isinstance(response, dict) else []
            for q in quotes:
                sym = str(q.get("symbol", "")).strip().upper()
                if sym and sym not in discovered_tickers:
                    discovered_tickers.append(sym)
        except Exception:
            pass

    except Exception:
        pass

    return discovered_tickers[:max_results]


# ============================================================================
# SCREENING & FILTERING
# ============================================================================

def screen_peers_capital_iq_style(
    target_profile: dict,
    candidate_tickers: List[str],
    criteria: dict = None
) -> List[Tuple[str, float, dict]]:
    """
    Apply Capital IQ-style screening criteria to filter peers.
    
    Returns:
        List of (ticker, score, profile) tuples sorted by relevance
    """
    if criteria is None:
        criteria = {
            "market_cap_min_ratio": 0.25,
            "market_cap_max_ratio": 4.0,
            "same_country_only": False,
            "min_avg_volume": 100000,
            "require_financials": True,
            "exclude_otc": True,
            "same_sector_required": False,
            "same_industry_preferred": True,
        }

    target_market_cap = target_profile.get("market_cap", 0)
    target_sector = target_profile.get("sector", "")
    target_industry = target_profile.get("industry", "")
    target_country = target_profile.get("country", "")

    # Calculate market cap screening bands
    min_market_cap = target_market_cap * criteria["market_cap_min_ratio"]
    max_market_cap = target_market_cap * criteria["market_cap_max_ratio"]

    screened_peers = []

    # Process candidates in parallel for speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(get_comprehensive_company_profile, ticker): ticker
            for ticker in candidate_tickers
            if ticker != target_profile["ticker"]
        }

        for future in as_completed(future_to_ticker):
            try:
                peer_profile = future.result()

                # Apply screening criteria
                score = 0
                pass_screen = True

                # 1. Sector matching (required if specified)
                if criteria["same_sector_required"]:
                    if peer_profile.get("sector") != target_sector:
                        continue
                    score += 18
                elif peer_profile.get("sector") == target_sector and target_sector:
                    score += 8

                # 2. Industry matching (preferred)
                if criteria["same_industry_preferred"]:
                    if peer_profile.get("industry") == target_industry:
                        score += 12
                    else:
                        overlap = _industry_keyword_overlap(target_industry, peer_profile.get("industry", ""))
                        score += overlap * 8

                # 3. Market cap screening
                peer_cap = peer_profile.get("market_cap", 0)
                if target_market_cap > 0 and peer_cap > 0:
                    if peer_cap < min_market_cap or peer_cap > max_market_cap:
                        continue

                # Score based on market cap similarity
                if target_market_cap > 0 and peer_cap > 0:
                    cap_ratio = min(peer_cap, target_market_cap) / max(peer_cap, target_market_cap)
                    score += cap_ratio * 24
                    if peer_profile.get("market_cap_category") == target_profile.get("market_cap_category"):
                        score += 4

                # Revenue scale similarity
                target_rev = target_profile.get("revenue", 0)
                peer_rev = peer_profile.get("revenue", 0)
                if target_rev and peer_rev and target_rev > 0 and peer_rev > 0:
                    rev_ratio = min(peer_rev, target_rev) / max(peer_rev, target_rev)
                    score += rev_ratio * 8

                # 4. Geographic matching
                peer_country = peer_profile.get("country", "")
                if criteria["same_country_only"] and peer_country != target_country:
                    continue

                if peer_country == target_country:
                    score += 6

                # 5. Liquidity screening
                peer_volume = peer_profile.get("avg_volume", 0)
                if peer_volume < criteria["min_avg_volume"]:
                    continue

                score += min(peer_volume / 1000000, 10)

                # 6. Exchange screening
                if criteria["exclude_otc"]:
                    exchange = peer_profile.get("exchange", "")
                    if "OTC" in exchange.upper() or "PINK" in exchange.upper():
                        continue
                    if exchange.upper() in {"NMS", "NYQ", "NGM", "ASE"}:
                        score += 4

                # 7. Active trading
                if not peer_profile.get("is_active"):
                    continue

                if criteria.get("require_financials", False):
                    has_core = (
                        _safe_float(peer_profile.get("enterprise_value"), default=0) > 0
                        and _safe_float(peer_profile.get("revenue"), default=0) > 0
                    )
                    if not has_core:
                        continue

                # Passed all screens - add to results
                screened_peers.append((peer_profile["ticker"], score, peer_profile))

            except Exception:
                continue

    # Sort by score (highest first)
    screened_peers.sort(key=lambda x: x[1], reverse=True)

    return screened_peers


# ============================================================================
# UPGRADED: WEIGHTED COMPOSITE SCORING MODEL
# ============================================================================

def calculate_weighted_composite_score(
    target_ticker: str,
    target_profile: dict,
    peer_ticker: str,
    peer_profile: dict,
    base_screening_score: float,
    weights: Dict[str, float] = None,
    precomputed_business_similarity: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate weighted composite score using Capital IQ methodology.
    
    Components (configurable weights):
    1. Business description similarity (30%)
    2. Multiples similarity (25%)
    3. Revenue scale similarity (15%)
    4. Profitability similarity (10%)
    5. Return correlation (10%)
    6. Liquidity & float quality (10%)
    
    Returns dict with all component scores and final weighted score.
    """
    if weights is None:
        weights = SCORING_WEIGHTS
    
    scores = {}
    
    # 1. Business Description Similarity (30%)
    business_sim = (
        float(precomputed_business_similarity)
        if precomputed_business_similarity is not None
        else calculate_business_description_similarity(target_ticker, peer_ticker)
    )
    scores["business_description"] = business_sim
    
    # 2. Multiples Similarity (25%)
    multiples_sim = calculate_multiples_similarity(target_profile, peer_profile)
    scores["multiples"] = multiples_sim
    
    # 3. Revenue Scale Similarity (15%)
    target_rev = _safe_float(target_profile.get("revenue"), default=0)
    peer_rev = _safe_float(peer_profile.get("revenue"), default=0)
    rev_scale_score = 0.0
    if target_rev > 0 and peer_rev > 0:
        rev_ratio = min(peer_rev, target_rev) / max(peer_rev, target_rev)
        rev_scale_score = rev_ratio * 100.0

    target_cagr = _safe_float(target_profile.get("revenue_cagr_3y"), default=np.nan)
    peer_cagr = _safe_float(peer_profile.get("revenue_cagr_3y"), default=np.nan)
    rev_growth_score = 50.0
    if not pd.isna(target_cagr) and not pd.isna(peer_cagr):
        rev_growth_gap = abs(target_cagr - peer_cagr)
        rev_growth_score = max(0.0, 100.0 * (1.0 - min(rev_growth_gap / 0.35, 1.0)))
    scores["revenue_scale"] = (0.7 * rev_scale_score) + (0.3 * rev_growth_score)
    
    # 4. Profitability Similarity (10%)
    prof_scores = []
    margin_pairs = [
        ("operating_margins", 0.40),
        ("ebitda_margin", 0.40),
        ("roe", 0.60),
    ]
    for key, max_gap in margin_pairs:
        t = _safe_float(target_profile.get(key), default=np.nan)
        p = _safe_float(peer_profile.get(key), default=np.nan)
        if pd.isna(t) or pd.isna(p):
            continue
        gap = abs(t - p)
        prof_scores.append(max(0.0, 100.0 * (1.0 - min(gap / max_gap, 1.0))))
    scores["profitability"] = float(np.mean(prof_scores)) if prof_scores else 50.0
    
    # 5. Return Correlation (10%)
    corr = calculate_return_correlation(target_ticker, peer_ticker)
    if corr is not None:
        # Transform [-1, 1] to [0, 100]
        scores["correlation"] = (corr + 1) * 50
    else:
        scores["correlation"] = 50  # Neutral if no data
    
    # 6. Liquidity & Float Quality (10%)
    float_quality = calculate_float_quality_score(peer_profile)
    scores["liquidity_float"] = float_quality
    
    # Calculate weighted composite
    composite = (
        scores["business_description"] * weights["business_description_similarity"] +
        scores["multiples"] * weights["multiples_similarity"] +
        scores["revenue_scale"] * weights["revenue_scale_similarity"] +
        scores["profitability"] * weights["profitability_similarity"] +
        scores["correlation"] * weights["return_correlation"] +
        scores["liquidity_float"] * weights["liquidity_float_quality"]
    )
    
    scores["composite"] = composite
    scores["base_screening"] = base_screening_score
    
    return scores


# ============================================================================
# STRUCTURAL ANALOGUES — cross-sector peers analysts always compare
# ============================================================================

def _get_structural_analogues(ticker: str, industry: str, sector: str) -> List[str]:
    """
    Return a curated list of tickers that analysts structurally compare to
    companies in this industry/sector, even when they span different GICS codes.

    Examples:
    - Semiconductor designers → also compare to semiconductor equipment (AMAT, LRCX, ASML)
    - Cloud hyperscalers → also compare to CDN/networking (ANET, CSCO)
    - EV makers → also compare to traditional OEM and battery suppliers
    - Streaming → also compare to traditional media and gaming

    This replicates the "related companies" and "cross-industry comp" tab that
    CapIQ/Bloomberg surface alongside the primary peer set.
    """
    analogues: List[str] = []
    ind_lower = (industry or "").lower()
    sec_lower = (sector or "").lower()

    # ── Semiconductors ────────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("semiconductor", "chip", "integrated circuit")):
        analogues += ["NVDA","AMD","INTC","QCOM","AVGO","TXN","MU","MRVL","ADI",
                      "NXPI","ON","AMAT","LRCX","KLAC","ASML","SNPS","CDNS","ARM"]

    # ── Semiconductor Equipment ───────────────────────────────────────────────
    if any(k in ind_lower for k in ("semiconductor equipment", "wafer", "etch", "deposition")):
        analogues += ["AMAT","LRCX","KLAC","ASML","TER","ONTO","FORM","ACMR",
                      "NVDA","AMD","INTC","TSM","SMCI"]

    # ── Cloud / SaaS / Enterprise Software ───────────────────────────────────
    if any(k in ind_lower for k in ("software", "saas", "cloud", "application")):
        analogues += ["MSFT","GOOGL","AMZN","CRM","NOW","ADBE","INTU","ORCL",
                      "SAP","SNOW","DDOG","MDB","HUBS","TEAM","WDAY","ZM","PANW"]

    # ── Networking / Cybersecurity ────────────────────────────────────────────
    if any(k in ind_lower for k in ("network", "cybersecurity", "security", "firewall")):
        analogues += ["CSCO","ANET","JNPR","PANW","CRWD","ZS","FTNT","NET",
                      "OKTA","S","TENB","QLYS","RPD"]

    # ── AI / Machine Learning Infrastructure ─────────────────────────────────
    if any(k in ind_lower for k in ("artificial intelligence", "machine learning", "data center")):
        analogues += ["NVDA","AMD","INTC","SMCI","DELL","HPE","ANET","CSCO",
                      "MSFT","GOOGL","AMZN","META","PLTR","AI","C3AI"]

    # ── Internet / E-Commerce ─────────────────────────────────────────────────
    if any(k in ind_lower for k in ("internet", "e-commerce", "online retail", "marketplace")):
        analogues += ["AMZN","GOOGL","META","NFLX","BKNG","ABNB","DASH","UBER",
                      "LYFT","ETSY","EBAY","SHOP","MELI","SE","CPNG","JD","BABA"]

    # ── Streaming / Digital Media ─────────────────────────────────────────────
    if any(k in ind_lower for k in ("streaming", "media", "entertainment", "content")):
        analogues += ["NFLX","DIS","CMCSA","WBD","PARA","SPOT","TTWO","EA",
                      "RBLX","U","SNAP","PINS","META","GOOGL","RDDT"]

    # ── Electric Vehicles ─────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("electric vehicle", "ev ", "autonomous", "battery")):
        analogues += ["TSLA","RIVN","LCID","NIO","LI","XPEV","GM","F","STLA",
                      "TM","HMC","APTV","LEA","ALB","SQM","LTHM","LAC"]

    # ── Biotech / Pharmaceuticals ─────────────────────────────────────────────
    if any(k in ind_lower for k in ("biotech", "biopharmaceutical", "drug", "therapeutics")):
        analogues += ["AMGN","GILD","VRTX","REGN","BIIB","MRNA","BNTX","ABBV",
                      "LLY","PFE","MRK","BMY","AZN","NVO","INCY","ALNY","IONS",
                      "SRPT","EXEL","BMRN","RARE","UTHR"]

    # ── Medical Devices ───────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("medical device", "medical equipment", "diagnostic", "surgical")):
        analogues += ["MDT","SYK","BSX","EW","ISRG","ZBH","BAX","BDX","DHR",
                      "A","WAT","DXCM","PODD","TNDM","NVCR","ILMN","EXAS"]

    # ── Managed Care / Health Insurance ──────────────────────────────────────
    if any(k in ind_lower for k in ("managed care", "health insurance", "health plan")):
        analogues += ["UNH","CI","ELV","HUM","CNC","MOH","CVS","WBA"]

    # ── Banks / Financial Services ────────────────────────────────────────────
    if any(k in ind_lower for k in ("bank", "financial services", "lending", "mortgage")):
        analogues += ["JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","FITB",
                      "HBAN","CFG","KEY","RF","ZION","WBS","COLB","EWBC"]

    # ── Asset Management ──────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("asset management", "investment management", "fund")):
        analogues += ["BLK","SCHW","MS","GS","AMP","IVZ","BEN","TROW","WDR",
                      "LM","AMG","VCTR","APAM","STEP","HLNE","ARCC","FSK"]

    # ── Payments / Fintech ────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("payment", "transaction", "fintech", "credit card")):
        analogues += ["V","MA","PYPL","AXP","COF","DFS","SYF","FIS","FI","GPN",
                      "FOUR","WEX","PAYO","AFRM","SOFI","NU","ADYEY","SQ","BILL"]

    # ── Insurance ─────────────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("insurance", "reinsurance", "underwriting")):
        analogues += ["BRK-B","PGR","CB","ALL","TRV","HIG","AFL","MET","PRU",
                      "AIG","MKL","RE","CINF","WRB","RNR","EG","ACGL"]

    # ── Oil & Gas Exploration ─────────────────────────────────────────────────
    if any(k in ind_lower for k in ("oil", "gas", "exploration", "production", "e&p")):
        analogues += ["XOM","CVX","COP","EOG","DVN","HES","FANG","MRO","APA",
                      "OXY","OVV","SM","MTDR","CTRA","PDCE","ESTE","VTLE"]

    # ── Oil Services ──────────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("oil field", "drilling", "oilfield services")):
        analogues += ["SLB","HAL","BKR","NOV","WTTR","NINE","OII","LBRT","PUMP"]

    # ── Aerospace & Defense ───────────────────────────────────────────────────
    if any(k in ind_lower for k in ("aerospace", "defense", "military", "missile", "aircraft")):
        analogues += ["LMT","RTX","NOC","GD","BA","L3H","HWM","TDG","TXT",
                      "HEI","KTOS","BWXT","CW","DRS","SAIC","LDOS","CACI"]

    # ── Logistics / Supply Chain ──────────────────────────────────────────────
    if any(k in ind_lower for k in ("logistics", "freight", "trucking", "shipping", "supply chain")):
        analogues += ["UPS","FDX","EXPD","CHRW","XPO","SAIA","ODFL","JBHT",
                      "KNX","WERN","ARCB","GXO","RXO","ECHO","FWRD"]

    # ── Renewable Energy ──────────────────────────────────────────────────────
    if any(k in ind_lower for k in ("solar", "wind", "renewable", "clean energy")):
        analogues += ["NEE","ENPH","SEDG","FSLR","ARRY","RUN","SPWR","CSIQ",
                      "JKS","DQ","BE","PLUG","BLDP","HYLN","EVGO","BLNK"]

    # Remove duplicates and the target itself
    seen = set()
    result = []
    for t in analogues:
        if t != ticker and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ============================================================================
# MAIN PEER DISCOVERY FUNCTION (UPGRADED)
# ============================================================================

def discover_peers_capital_iq_style(
    ticker: str,
    uploaded_universe: List[str] = None,
    min_peers: int = 5,
    max_peers: int = 15,
    screening_criteria: dict = None,
    scoring_weights: Dict[str, float] = None
) -> List[Dict]:
    """
    UPGRADED: Main function to discover peers using TRUE Capital IQ methodology.

    Key Improvements:
    1. NLP-based business description similarity
    2. Proper revenue CAGR from SEC (not price proxy)
    3. Financial multiples matching
    4. Deterministic SIC index (no sampling)
    5. Float-adjusted quality scoring
    6. Weighted composite scoring model

    Args:
        ticker: Target company ticker
        uploaded_universe: Optional list of tickers to include as ADDITIONAL
            SEEDS in the candidate pool.  This list is NEVER used as a
            constraint — companies outside it are always considered.
            Passing None (or an empty list) has no effect on the breadth
            of peer discovery; the ETF holdings, sector seed universe, SIC
            index, and structural-analogues layers always run regardless.
        min_peers: Minimum number of peers to return
        max_peers: Maximum number of peers to return
        screening_criteria: Custom screening rules (or use defaults)
        scoring_weights: Custom scoring weights (or use defaults)

    Returns:
        List of peer dictionaries with scores and profiles
    """
    # ── Cache key includes universe hash so a change in the upload
    #    always triggers a fresh run (fixes stale cross-session results)
    uploaded_universe_norm = _normalize_ticker_list(uploaded_universe)
    universe_signature = "|".join(sorted(uploaded_universe_norm)[:200])
    cache_key = f"{ticker.upper()}_{max_peers}_{min_peers}_{hash(universe_signature)}_v3"

    cached = _peer_cache_get(cache_key)
    if cached is not None:
        return cached

    # Use default weights if not provided
    if scoring_weights is None:
        scoring_weights = SCORING_WEIGHTS

    # Step 1: Get target company profile
    target_profile = get_comprehensive_company_profile(ticker)

    if not target_profile.get("is_active"):
        return []

    target_sector = target_profile.get("sector", "")
    target_industry = target_profile.get("industry", "")

    # Step 2: Build candidate universe
    candidate_tickers = set()

    # 2a. Add uploaded universe first (if provided)
    if uploaded_universe_norm:
        candidate_tickers.update([t for t in uploaded_universe_norm if t != ticker])

    # 2b. Discover via yfinance industry search
    industry_peers = discover_industry_peers_from_yfinance(
        target_sector,
        target_industry,
        max_results=180
    )
    candidate_tickers.update(industry_peers)

    # 2b2. Inject curated sector seed universe — guarantees that the biggest
    #       names in each sector (S&P 500 / Russell 1000) are always considered,
    #       regardless of ETF holdings gaps.
    seed_peers = _SECTOR_SEED_UNIVERSE.get(target_sector, [])
    candidate_tickers.update([t for t in seed_peers if t != ticker.upper()])

    # 2c. UPGRADED: Discover via deterministic SIC index
    cik_map = load_full_cik_map()
    cik = cik_map.get(ticker)
    if cik and len(candidate_tickers) < 250:
        try:
            submissions = fetch_company_submissions(cik)
            sic = submissions.get('sic', '')
            if sic:
                sic_peers = get_peers_by_sic_deterministic(sic, max_results=250)
                candidate_tickers.update(sic_peers)
        except:
            pass

    # 2d. Cross-sector structural analogues — companies that analysts always
    #     compare despite different GICS sectors (e.g. semiconductor equipment
    #     vs chip designers, cloud infra vs SaaS, etc.).  Derived from the
    #     target's industry keyword so it adapts dynamically.
    candidate_tickers.update(
        _get_structural_analogues(ticker.upper(), target_industry, target_sector)
    )

    candidate_tickers.discard(ticker.upper())

    # Step 3: Screen candidates (progressive relaxation)
    # ── Adaptive market-cap bands ──────────────────────────────────────────
    # Fixed 0.25×–4× ratio bands break for mega-caps (NVDA at $3T would
    # require peers ≥ $750B, instantly excluding GOOGL at $2T) and for
    # micro-caps (a $500M company would exclude a $150M peer that analysts
    # routinely compare it to).
    #
    # Capital IQ uses log-scale proximity: companies within ±1.5 log-deciles
    # are comparable regardless of absolute size.  We replicate this with
    # an adaptive ratio that widens for large companies and narrows for small
    # ones, matching the actual analyst workflow.
    def _adaptive_cap_tiers(market_cap: float):
        """Return (min_ratio, max_ratio) tuned to the company's size tier."""
        if market_cap >= 500e9:       # Mega-cap: very wide — NVDA vs MSFT vs GOOGL
            return 0.08, 12.0
        elif market_cap >= 50e9:      # Large-cap: wide
            return 0.12, 8.0
        elif market_cap >= 10e9:      # Mid-large: standard
            return 0.15, 6.5
        elif market_cap >= 2e9:       # Mid-cap
            return 0.20, 5.0
        elif market_cap >= 500e6:     # Small-cap
            return 0.25, 4.0
        else:                         # Micro-cap: tight but still ±2 deciles
            return 0.30, 3.5

    base_min_r, base_max_r = _adaptive_cap_tiers(target_profile.get("market_cap", 0))

    if screening_criteria is not None:
        criteria_tiers = [screening_criteria]
    else:
        criteria_tiers = [
            {
                "market_cap_min_ratio": base_min_r,
                "market_cap_max_ratio": base_max_r,
                "same_country_only": False,
                "min_avg_volume": 100000,
                "require_financials": True,
                "exclude_otc": True,
                "same_sector_required": False,
                "same_industry_preferred": True,
            },
            # Tier 2: relax financials requirement, keep liquidity bar
            {
                "market_cap_min_ratio": base_min_r * 0.6,
                "market_cap_max_ratio": base_max_r * 1.5,
                "same_country_only": False,
                "min_avg_volume": 50000,
                "require_financials": False,
                "exclude_otc": True,
                "same_sector_required": False,
                "same_industry_preferred": True,
            },
            # Tier 3: widest net — used only if < min_peers found above
            {
                "market_cap_min_ratio": base_min_r * 0.3,
                "market_cap_max_ratio": base_max_r * 2.5,
                "same_country_only": False,
                "min_avg_volume": 25000,
                "require_financials": False,
                "exclude_otc": True,
                "same_sector_required": False,
                "same_industry_preferred": False,
            },
        ]

    screened_map: Dict[str, Tuple[float, dict]] = {}
    for tier_idx, criteria in enumerate(criteria_tiers):
        screened_peers_tier = screen_peers_capital_iq_style(
            target_profile,
            list(candidate_tickers),
            criteria=criteria
        )
        for peer_ticker, score, profile in screened_peers_tier:
            tier_adjusted_score = score - (tier_idx * 10)
            current = screened_map.get(peer_ticker)
            if current is None or tier_adjusted_score > current[0]:
                screened_map[peer_ticker] = (tier_adjusted_score, profile)
        if len(screened_map) >= max(min_peers, max_peers):
            break

    screened_peers = [(t, s, p) for t, (s, p) in screened_map.items()]
    screened_peers.sort(key=lambda x: x[1], reverse=True)

    # Step 4: UPGRADED - Apply weighted composite scoring
    business_similarity_map = calculate_business_description_similarity_batch(
        ticker,
        [p[0] for p in screened_peers]
    )

    enhanced_peers = []
    for peer_ticker, base_score, peer_profile in screened_peers:
        # Calculate all component scores
        component_scores = calculate_weighted_composite_score(
            ticker,
            target_profile,
            peer_ticker,
            peer_profile,
            base_score,
            weights=scoring_weights,
            precomputed_business_similarity=business_similarity_map.get(peer_ticker),
        )
        
        enhanced_peers.append({
            "ticker": peer_ticker,
            "name": peer_profile.get("name", peer_ticker),
            "sector": peer_profile.get("sector", ""),
            "industry": peer_profile.get("industry", ""),
            "market_cap": peer_profile.get("market_cap", 0),
            "country": peer_profile.get("country", ""),
            "profile": peer_profile,
            
            # Component scores
            "business_description_score": component_scores["business_description"],
            "multiples_similarity_score": component_scores["multiples"],
            "revenue_scale_score": component_scores["revenue_scale"],
            "profitability_score": component_scores["profitability"],
            "correlation_score": component_scores["correlation"],
            "liquidity_float_score": component_scores["liquidity_float"],
            
            # Final scores
            "composite_score": component_scores["composite"],
            "base_screening_score": component_scores["base_screening"],
        })

    # Sort by composite score
    enhanced_peers.sort(key=lambda x: x["composite_score"], reverse=True)

    # Sort purely by composite score — same-industry peers naturally surface at
    # the top because business_description_similarity (35%) and sector/industry
    # bonuses in the base screening score already reward them.  A hard bucket
    # sort here would bury high-quality cross-industry comparables that analysts
    # routinely use (e.g., NVDA vs TSMC in semiconductors vs comms).
    enhanced_peers.sort(key=lambda x: -float(x.get("composite_score", 0.0)))

    result = enhanced_peers[:max_peers]

    # Store in session-aware cache
    _peer_cache_set(cache_key, result)

    return result


# ============================================================================
# CONVENIENCE FUNCTION (backward compatible)
# ============================================================================

def find_best_peers_automated(
    ticker: str,
    uploaded_universe: List[str] = None,
    max_peers: int = 10
) -> List[str]:
    """
    Simplified interface — returns a ranked ticker list of public comparables.

    This is the function called by multiples.py, tearsheet.py, and ratios.py.

    Discovery is ALWAYS open-universe: the function searches ETF holdings,
    a curated sector seed list (~80–120 S&P 500 names per sector), the SEC
    EDGAR SIC index, and a structural-analogues layer (cross-sector comps
    that analysts routinely use).  Uploaded_universe is treated as an
    ADDITIONAL SEED, never as a whitelist or constraint — companies not in
    the uploaded list are always considered and will appear in results if
    they score higher than listed alternatives.

    Args:
        ticker:            Target company ticker.
        uploaded_universe: Optional extra seeds (e.g. the screener universe
                           the user uploaded).  Pass None to rely entirely
                           on automated discovery — the result will be the
                           same or richer either way.
        max_peers:         Maximum peers to return to the caller.

    Returns:
        List of ticker strings, best comps first.
    """
    # Request more candidates internally than we return so the scoring layer
    # has enough headroom to surface the truly best matches even when many
    # candidates from automated discovery turn out to be borderline.
    internal_max = max(max_peers * 3, 30)

    peer_data = discover_peers_capital_iq_style(
        ticker=ticker,
        uploaded_universe=uploaded_universe,   # treated as seed, not constraint
        min_peers=5,
        max_peers=internal_max,
    )

    return [
        p["ticker"] for p in peer_data
        if p.get("ticker") and p.get("ticker") != ticker.upper()
    ][:max_peers]


# Export main functions
__all__ = [
    'discover_peers_capital_iq_style',
    'find_best_peers_automated',
    'get_comprehensive_company_profile',
    'screen_peers_capital_iq_style',
    'calculate_weighted_composite_score',
    'calculate_business_description_similarity',
    'calculate_business_description_similarity_batch',
    'calculate_multiples_similarity',
    'get_revenue_cagr_from_sec',
    'build_complete_sic_index',
    'calculate_float_quality_score',
    '_get_structural_analogues',
    '_SECTOR_SEED_UNIVERSE',
    'SCORING_WEIGHTS',
]
