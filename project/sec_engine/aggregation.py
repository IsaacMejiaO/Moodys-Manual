import pandas as pd
import numpy as np
from typing import List, Dict
from sec_engine.sec_fetch import fetch_company_submissions


# ── Peer Override Registry ────────────────────────────────────────────────────
#
# Maps a ticker to an explicit ordered list of peer tickers.
# When present, find_peers_by_sic() returns this list DIRECTLY, bypassing all
# SIC-matching and market-cap heuristics.
#
# Use cases:
#   - Conglomerates (e.g. BRK-B) where SIC codes are meaningless
#   - Holding companies that span multiple industries
#   - Tickers whose EDGAR SIC code is incorrect or extremely broad
#   - Any company where analyst judgment should override the algorithm
#
# Populate either by editing the dict below or by calling set_peer_override()
# at runtime (e.g. in a notebook or app startup).
#
# Format: { "TICKER": ["PEER1", "PEER2", ...] }  — all values stored uppercase.
#
PEER_OVERRIDES: Dict[str, List[str]] = {
    # ── Examples (commented out — uncomment and customise) ──────────────────
    # "BRK-B": ["JPM", "BAC", "AIG", "MET", "PRU", "GS", "MS"],
    # "GE":    ["HON", "MMM", "RTX", "EMR", "ITW", "ETN", "PH"],
}


def set_peer_override(ticker: str, peers: List[str]) -> None:
    """
    Register an explicit peer list for a ticker at runtime.

    The provided list replaces ALL algorithmic peer-finding for this ticker.
    Peers are deduplicated and upper-cased automatically. Passing an empty
    list clears the override (same as clear_peer_override).

    Args:
        ticker: Target ticker (case-insensitive).
        peers:  Ordered list of peer tickers.

    Example:
        set_peer_override("BRK-B", ["JPM", "BAC", "GS", "AIG"])
    """
    ticker = ticker.upper().strip()
    if not peers:
        PEER_OVERRIDES.pop(ticker, None)
        return
    seen = set()
    cleaned = []
    for p in peers:
        p = str(p).upper().strip()
        if p and p != ticker and p not in seen:
            seen.add(p)
            cleaned.append(p)
    PEER_OVERRIDES[ticker] = cleaned


def clear_peer_override(ticker: str) -> None:
    """
    Remove the peer override for a ticker, reverting to heuristic discovery.

    Args:
        ticker: Target ticker (case-insensitive).
    """
    PEER_OVERRIDES.pop(ticker.upper().strip(), None)


def list_peer_overrides() -> Dict[str, List[str]]:
    """Return a copy of the current peer override registry."""
    return {k: list(v) for k, v in PEER_OVERRIDES.items()}


def get_peer_override(ticker: str):
    """
    Return the explicit peer list for a ticker, or None if not overridden.

    Args:
        ticker: Target ticker (case-insensitive).

    Returns:
        List of peer tickers if an override is registered, else None.
    """
    return PEER_OVERRIDES.get(ticker.upper().strip())


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
    Find peer companies based on SIC code and similar market cap.

    Override check (first):
        If an explicit peer list has been registered for this ticker via
        set_peer_override() or PEER_OVERRIDES, it is returned immediately
        without any heuristic logic. The returned list is capped at
        max_peers but otherwise returned verbatim.

    Heuristic fallback (when no override exists):
        1. Match on the first 3 digits of the SIC code.
        2. If fewer than min_peers found, broaden to 2 digits.
        3. Sort by absolute market-cap distance from the target.

    Args:
        ticker: Target company ticker
        sic_map: Dictionary mapping tickers to SIC codes
        df: DataFrame with company data (must include 'Ticker' and 'Market Cap (M)')
        min_peers: Minimum number of peers to return
        max_peers: Maximum number of peers to return
    """
    ticker_upper = ticker.upper().strip()

    # ── Analyst override check ────────────────────────────────────────────────
    # If an explicit peer list is registered, return it immediately.
    # This bypasses all SIC/market-cap heuristics for tickers where analyst
    # judgment is more reliable (conglomerates, holding companies, etc.).
    override = get_peer_override(ticker_upper)
    if override is not None:
        # Optionally filter to tickers known in df (safety check, not mandatory)
        all_known = set(df['Ticker'].str.upper()) if not df.empty else set()
        filtered = [p for p in override if not all_known or p in all_known]
        result = filtered if filtered else override
        return result[:max_peers]

    target_sic = sic_map.get(ticker)
    
    if not target_sic:
        # Fallback: return all other companies
        return [t for t in df['Ticker'].tolist() if t not in (ticker, ticker_upper)][:max_peers]
    
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
