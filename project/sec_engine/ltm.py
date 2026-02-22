# ltm.py
# ------------------------------------------------------------------
# LTM (Last Twelve Months) construction from SEC XBRL facts.
#
# Key design decisions:
#   1. Deduplication by (period-end, period-length) — not just date —
#      so that a 10-K annual entry and a 10-Q quarterly entry that
#      share the same end-date are not collapsed into one.
#   2. When multiple filings report the same (end, duration) pair,
#      we keep the one with the highest accession number (latest filed),
#      which picks up amendments / restatements automatically.
#   3. LTM is built only from entries whose duration is ~90 days
#      (quarterly entries). Annual 10-K entries are explicitly excluded
#      so we never mix a 365-day value with three 90-day values.
#   4. A gap check ensures the 4 chosen quarters span no more than
#      ~370 days total (allowing for slightly unequal fiscal quarters).
#      If the span is too wide, we return NaN rather than silently
#      produce a number that covers 18+ months.
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Dict, List

# Tolerance: a "quarterly" entry must cover between 75 and 105 days.
_Q_MIN_DAYS = 75
_Q_MAX_DAYS = 105

# Maximum allowable span for 4 quarters (must cover ~12 months).
_LTM_MAX_SPAN_DAYS = 380


def _is_quarterly(days: int) -> bool:
    return _Q_MIN_DAYS <= days <= _Q_MAX_DAYS


def extract_quarterly_series(
    facts: dict,
    tag_list: List[str],
    unit: str = "USD",
) -> pd.Series:
    """
    Extract a time series of *quarterly* values from SEC XBRL facts.

    For each (period-end, period-length-in-days) pair, the entry with
    the highest accession number is kept (latest amendment wins).

    Only entries whose period spans ~90 days are included, filtering
    out any annual 10-K cumulative entries that share the same tag.

    Returns a Series indexed by period-end date, sorted ascending.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tag_list:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get(unit, [])

        rows = []
        for e in entries:
            if e.get("form") not in ("10-Q", "10-K"):
                continue
            end_str = e.get("end")
            start_str = e.get("start")
            if not end_str or not start_str:
                continue

            try:
                end_dt = pd.to_datetime(end_str)
                start_dt = pd.to_datetime(start_str)
            except Exception:
                continue

            days = (end_dt - start_dt).days

            # Only keep entries that represent a single quarter
            if not _is_quarterly(days):
                continue

            rows.append({
                "end": end_dt,
                "days": days,
                "accn": e.get("accn", ""),
                "val": e["val"],
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Keep latest-filed entry for each (end-date, period-length) pair
        df = (
            df.sort_values("accn", ascending=True)
            .drop_duplicates(subset=["end", "days"], keep="last")
        )

        series = df.set_index("end")["val"].sort_index()
        return series.astype(float)

    return pd.Series(dtype="float64")


def extract_annual_series(
    facts: dict,
    tag_list: List[str],
    unit: str = "USD",
) -> pd.Series:
    """
    Extract a time series of *annual* values from 10-K filings only.

    Keeps the latest-filed entry per fiscal-year-end date.
    Returns a Series indexed by fiscal-year-end date, sorted ascending.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tag_list:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get(unit, [])

        rows = []
        for e in entries:
            if e.get("form") != "10-K":
                continue
            end_str = e.get("end")
            if not end_str:
                continue
            try:
                end_dt = pd.to_datetime(end_str)
            except Exception:
                continue

            rows.append({
                "end": end_dt,
                "accn": e.get("accn", ""),
                "val": e["val"],
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Keep latest-filed entry per fiscal-year-end
        df = (
            df.sort_values("accn", ascending=True)
            .drop_duplicates(subset=["end"], keep="last")
        )

        series = df.set_index("end")["val"].sort_index()
        return series.astype(float)

    return pd.Series(dtype="float64")


def build_ltm(series: pd.Series) -> float:
    """
    Sum the last 4 quarterly entries to produce a Last Twelve Months value.

    Returns NaN if:
      - Fewer than 4 quarterly observations are available.
      - The 4 quarters span more than ~380 days (non-contiguous data).
      - Any of the 4 values is NaN.

    This prevents silently producing LTM figures that actually cover
    15–18 months due to missing intermediate quarters.
    """
    if series is None or series.empty:
        return float("nan")

    series = series.dropna().sort_index()

    if len(series) < 4:
        return float("nan")

    last4 = series.iloc[-4:]

    # Gap check: span from start of earliest quarter to end of latest
    span_days = (last4.index[-1] - last4.index[0]).days
    if span_days > _LTM_MAX_SPAN_DAYS:
        return float("nan")

    return float(last4.sum())


def latest_balance(series: pd.Series) -> float:
    """Return the most recent non-NaN value from a series."""
    if series is None or series.empty:
        return float("nan")
    clean = series.dropna().sort_index()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[-1])


def build_ltm_financials(
    facts: dict,
    gaap_map: Dict[str, List[str]],
    ltm_metrics=None,
) -> Dict[str, float]:
    """
    Build a dict of LTM (flow) and latest (stock) values for all metrics
    in gaap_map, using the corrected quarterly extraction logic.
    """
    if ltm_metrics is None:
        ltm_metrics = {
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "ocf",
            "capex",
            "depreciation",
            "ebitda",
            "sga",
            "rd",
            "cogs",
            "interest_expense",
            "amortization",
        }

    ltm_data = {}
    for metric, tags in gaap_map.items():
        series = extract_quarterly_series(facts, tags)
        if metric in ltm_metrics:
            ltm_data[metric] = build_ltm(series)
        else:
            ltm_data[metric] = latest_balance(series)

    return ltm_data
