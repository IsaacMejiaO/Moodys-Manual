import pandas as pd
import numpy as np
from typing import List, Dict
from sec_engine.sec_fetch import fetch_company_submissions


def _parse_market_cap(value) -> float:
    """
    Robustly parse a market-cap value that may be a float, int, or a
    formatted string (e.g. "1,234,567" or "1234567").

    Returns np.nan for any value that cannot be converted to a finite float,
    so that callers can safely filter on pd.notna() rather than catching
    exceptions.
    """
    if value is None:
        return np.nan
    try:
        result = pd.to_numeric(str(value).replace(",", "").strip(), errors="coerce")
        return float(result) if pd.notna(result) else np.nan
    except (TypeError, ValueError):
        return np.nan

def get_company_sic(cik: str) -> str:
    """
    Extract SIC code for a company
    """
    try:
        submissions = fetch_company_submissions(cik)
        return submissions.get('sic', '')
    except:
        return ''

def build_sic_map(universe: List[str], cik_map: Dict[str, str]) -> Dict[str, str]:
    """
    Build a mapping of ticker -> SIC code for entire universe
    """
    sic_map = {}
    for ticker in universe:
        cik = cik_map.get(ticker)
        if cik:
            sic = get_company_sic(cik)
            if sic:
                sic_map[ticker] = sic
    return sic_map

def find_peers_by_sic(
    ticker: str,
    sic_map: Dict[str, str],
    df: pd.DataFrame,
    min_peers: int = 3,
    max_peers: int = 8
) -> List[str]:
    """
    Find peer companies based on SIC code and similar market cap
    
    Args:
        ticker: Target company ticker
        sic_map: Dictionary mapping tickers to SIC codes
        df: DataFrame with company data (must include 'Ticker' and 'Market Cap (M)')
        min_peers: Minimum number of peers to return
        max_peers: Maximum number of peers to return
    """
    target_sic = sic_map.get(ticker)
    
    if not target_sic:
        # Fallback: return all other companies
        return [t for t in df['Ticker'].tolist() if t != ticker][:max_peers]
    
    # Get target company's market cap
    target_row = df[df['Ticker'] == ticker]
    if target_row.empty:
        return []
    
    target_cap_str = target_row['Market Cap (M)'].iloc[0]
    # Robust parsing: handles floats, ints, comma-formatted strings, None, NaN.
    target_cap = _parse_market_cap(target_cap_str)
    if np.isnan(target_cap):
        # Cannot compute cap-distance without a valid target market cap;
        # fall back to returning all same-industry peers unsorted.
        return [t for t in df['Ticker'].tolist() if t != ticker][:max_peers]
    
    # Find companies with same SIC (first 3 digits for broader matching)
    sic_prefix = target_sic[:3] if len(target_sic) >= 3 else target_sic
    
    same_industry = []
    for t, sic in sic_map.items():
        if t == ticker:
            continue
        if sic.startswith(sic_prefix):
            same_industry.append(t)
    
    if len(same_industry) < min_peers:
        # Broaden search to 2-digit SIC
        sic_prefix = target_sic[:2] if len(target_sic) >= 2 else target_sic
        same_industry = []
        for t, sic in sic_map.items():
            if t == ticker:
                continue
            if sic.startswith(sic_prefix):
                same_industry.append(t)
    
    if not same_industry:
        # Ultimate fallback
        return [t for t in df['Ticker'].tolist() if t != ticker][:max_peers]

    # Optional deterministic expansion via Capital IQ-style SIC index.
    try:
        from sec_engine.capital_iq_style_peer_finder import get_peers_by_sic_deterministic
        deterministic_pool = get_peers_by_sic_deterministic(str(target_sic), max_results=200)
        deterministic_pool = [t for t in deterministic_pool if t != ticker]
        if deterministic_pool:
            same_industry = list(dict.fromkeys(same_industry + deterministic_pool))
    except Exception:
        pass
    
    # Filter df to same industry and calculate market cap similarity
    industry_df = df[df['Ticker'].isin(same_industry)].copy()
    
    # Parse market cap for sorting — use robust helper so bad values become NaN
    # and are then excluded from the distance calculation rather than crashing.
    industry_df['market_cap_numeric'] = industry_df['Market Cap (M)'].apply(_parse_market_cap)
    industry_df = industry_df[pd.notna(industry_df['market_cap_numeric'])]
    
    # Calculate distance from target market cap
    industry_df['cap_distance'] = abs(
        industry_df['market_cap_numeric'] - target_cap
    )
    
    # Sort by market cap similarity and return top peers
    peers = industry_df.nsmallest(max_peers, 'cap_distance')['Ticker'].tolist()
    
    return peers
