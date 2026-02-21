# sec_engine/cik_loader.py
import requests
import pandas as pd

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "Isaac Mejia Ortiz <email@example.com>"}

def load_full_cik_map() -> dict:
    """
    Loads the full SEC ticker → CIK mapping.
    Returns a dict: { "AAPL": "0000320193", ... }
    """
    response = requests.get(SEC_TICKER_URL, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame.from_dict(data, orient="index")

    # Normalize ticker
    df["ticker"] = df["ticker"].str.upper()

    # Convert cik_str → zero padded string
    df["cik_str"] = df["cik_str"].astype(int).astype(str).str.zfill(10)

    return dict(zip(df["ticker"], df["cik_str"]))
