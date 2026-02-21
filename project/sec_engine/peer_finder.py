import pandas as pd
from typing import List, Dict
from sec_engine.sec_fetch import fetch_company_submissions

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
    # Handle formatted string (e.g., "1,234,567")
    if isinstance(target_cap_str, str):
        target_cap = float(target_cap_str.replace(',', ''))
    else:
        target_cap = float(target_cap_str)
    
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
    
    # Parse market cap for sorting
    industry_df['market_cap_numeric'] = industry_df['Market Cap (M)'].apply(
        lambda x: float(str(x).replace(',', '')) if pd.notna(x) else 0
    )
    
    # Calculate distance from target market cap
    industry_df['cap_distance'] = abs(
        industry_df['market_cap_numeric'] - target_cap
    )
    
    # Sort by market cap similarity and return top peers
    peers = industry_df.nsmallest(max_peers, 'cap_distance')['Ticker'].tolist()
    
    return peers
