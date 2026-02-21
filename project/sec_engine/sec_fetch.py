import requests
import pandas as pd

SEC_HEADERS = {
    "User-Agent": "Isaac Mejia Ortiz isaac@email.com"
}

def fetch_company_facts(cik: str) -> dict:
    """
    Pulls all XBRL facts from the SEC
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=SEC_HEADERS)
    response.raise_for_status()
    return response.json()

def fetch_company_submissions(cik: str) -> dict:
    """
    Get company submission history including SIC code
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    response = requests.get(url, headers=SEC_HEADERS)
    response.raise_for_status()
    return response.json()
