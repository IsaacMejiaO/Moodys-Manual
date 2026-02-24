# metadata_fetch.py
# ------------------------------------------------------------------
# DEPRECATED — this module is no longer used by the application.
#
# Metadata extraction has been fully inlined into app.py inside the
# fetch_company_data_unified() function, which also applies the same
# yfinance → SEC EDGAR fallback pattern used for financial data.
#
# This file emits a DeprecationWarning on import so that accidental
# references are surfaced at runtime rather than silently using stale
# logic. It will be removed in a future cleanup.
# ------------------------------------------------------------------

import warnings

warnings.warn(
    "metadata_fetch.py is deprecated and will be removed. "
    "Metadata is now fetched inline in app.py via fetch_company_data_unified().",
    DeprecationWarning,
    stacklevel=2,
)


def fetch_metadata(ticker: str) -> dict:
    """
    Deprecated stub — do not use.
    Kept temporarily to avoid AttributeError if anything references
    this function by name before the file is deleted.
    """
    warnings.warn(
        "fetch_metadata() is deprecated. Use fetch_company_data_unified() in app.py.",
        DeprecationWarning,
        stacklevel=2,
    )
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    eps_growth = info.get("earningsQuarterlyGrowth")
    dividend_yield = info.get("dividendYield")
    return {
        "name": info.get("longName") or ticker,
        "industry": info.get("industry"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "pe_ltm": info.get("trailingPE"),
        "eps_growth_pct": eps_growth * 100 if eps_growth is not None else None,
        "dividend_yield_pct": dividend_yield * 100 if dividend_yield is not None else 0.0,
    }
