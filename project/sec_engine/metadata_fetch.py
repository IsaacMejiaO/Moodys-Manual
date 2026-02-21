import yfinance as yf

def fetch_metadata(ticker: str) -> dict:
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
