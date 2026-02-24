# sec_fetch.py
# ------------------------------------------------------------------
# HTTP client for SEC EDGAR data APIs.
#
# Fixes vs. prior version:
#   - All requests now have an explicit timeout (10 s connect, 30 s read).
#     Previously a slow EDGAR response would hang the thread indefinitely.
#   - Exponential-backoff retry on transient failures (429, 503, network
#     errors). SEC EDGAR rate-limits to 10 req/s; a 429 means we need
#     to back off, not crash.
#   - A shared threading.Semaphore caps concurrent SEC requests at 8,
#     staying safely under the 10 req/s EDGAR limit even with 20 worker
#     threads in the parallel loader.
#   - Structured exceptions (SECFetchError) replace bare raise_for_status()
#     so callers can distinguish a 404 (ticker not in EDGAR) from a 503
#     (transient outage) and handle them differently.
#   - User-Agent is now read from the SEC_USER_AGENT environment variable
#     so that personal contact details are not hard-coded in source code.
#     Falls back to a generic placeholder if the variable is not set.
#   - Hardcoded "Host: data.sec.gov" header removed — it was only valid
#     for fetch_company_facts() but not for fetch_company_submissions()
#     (which also hits data.sec.gov), and a static Host header causes
#     400 errors on certain proxy configurations. requests sets Host
#     automatically from the URL.
# ------------------------------------------------------------------

import os
import time
import threading
import requests

# SEC requires a descriptive User-Agent. Format: "Name email"
# Set the SEC_USER_AGENT environment variable to avoid storing contact
# details in source code, e.g.:
#   export SEC_USER_AGENT="Jane Smith jane@example.com"
_DEFAULT_USER_AGENT = "EquityAnalyzer contact@example.com"
_USER_AGENT = os.environ.get("SEC_USER_AGENT", _DEFAULT_USER_AGENT)

SEC_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    # Note: "Host" header intentionally omitted — requests derives it
    # automatically from the URL, which is correct for all endpoints.
}

# Semaphore: at most 8 concurrent requests to EDGAR.
# With max_workers=20 in the parallel loader, this prevents bursting
# above the 10 req/s limit that EDGAR enforces.
_SEC_SEMAPHORE = threading.Semaphore(8)

# Request timeouts: (connect_timeout_s, read_timeout_s)
_TIMEOUT = (10, 30)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.5   # seconds; doubles each retry


class SECFetchError(Exception):
    """Raised when an SEC EDGAR request fails after all retries."""
    def __init__(self, url: str, status_code: int | None, message: str):
        self.url = url
        self.status_code = status_code
        super().__init__(f"SEC fetch failed [{status_code}] {url}: {message}")


def _get_with_retry(url: str) -> dict:
    """
    GET a JSON endpoint with timeout, retry, and rate-limit awareness.

    Retries on:
      - 429 Too Many Requests (backs off longer)
      - 503 Service Unavailable
      - requests.ConnectionError / requests.Timeout

    Does NOT retry on:
      - 404 Not Found (company not in EDGAR — caller should handle)
      - 400 Bad Request
    """
    last_exc = None

    with _SEC_SEMAPHORE:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, headers=SEC_HEADERS, timeout=_TIMEOUT)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 404:
                    raise SECFetchError(url, 404, "Not found in EDGAR")

                if resp.status_code in (429, 503):
                    # Rate limited or server busy — back off before retrying
                    wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    if resp.status_code == 429:
                        # Respect Retry-After header if present
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = max(wait, float(retry_after))
                            except ValueError:
                                pass
                    time.sleep(wait)
                    last_exc = SECFetchError(url, resp.status_code, resp.reason)
                    continue

                # Other 4xx/5xx — not retriable
                raise SECFetchError(url, resp.status_code, resp.reason)

            except SECFetchError:
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                last_exc = exc
                continue

    raise SECFetchError(url, None, f"Failed after {_MAX_RETRIES} retries: {last_exc}")


def fetch_company_facts(cik: str) -> dict:
    """
    Pull all XBRL facts for a company from SEC EDGAR.

    Args:
        cik: CIK as a string (leading zeros optional — we zero-pad internally).

    Returns:
        Parsed JSON dict from the companyfacts endpoint.

    Raises:
        SECFetchError: On 404 (company not in EDGAR) or persistent failure.
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    return _get_with_retry(url)


def fetch_company_submissions(cik: str) -> dict:
    """
    Fetch company submission history (includes SIC code, entity name, filings).

    Args:
        cik: CIK as a string.

    Returns:
        Parsed JSON dict from the submissions endpoint.

    Raises:
        SECFetchError: On failure.
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    return _get_with_retry(url)
