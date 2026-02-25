# tests/test_ltm.py
# -----------------------------------------------------------------------
# Unit tests for sec_engine/ltm.py
#
# These tests exercise the deduplication, quarterly-filter, gap-check,
# and LTM summation logic WITHOUT hitting EDGAR.  All facts dicts are
# constructed inline to match the exact schema that EDGAR returns.
# -----------------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sec_engine.ltm import (
    extract_quarterly_series,
    extract_annual_series,
    build_ltm,
    latest_balance,
    build_ltm_financials,
    _Q_MIN_DAYS,
    _Q_MAX_DAYS,
    _LTM_MAX_SPAN_DAYS,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_entry(start: str, end: str, val: float, form: str = "10-Q", accn: str = "0001") -> dict:
    """Build a minimal EDGAR units entry dict."""
    return {"start": start, "end": end, "val": val, "form": form, "accn": accn}


def _make_facts(tag: str, entries: list, unit: str = "USD") -> dict:
    """Build a minimal EDGAR companyfacts dict with a single tag."""
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        unit: entries
                    }
                }
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# extract_quarterly_series
# ═══════════════════════════════════════════════════════════════════════

class TestExtractQuarterlySeries:
    def test_four_clean_quarters(self):
        entries = [
            _make_entry("2023-01-01", "2023-03-31", 100),
            _make_entry("2023-04-01", "2023-06-30", 110),
            _make_entry("2023-07-01", "2023-09-30", 120),
            _make_entry("2023-10-01", "2023-12-31", 130),
        ]
        facts = _make_facts("Revenues", entries)
        s = extract_quarterly_series(facts, ["Revenues"])
        assert len(s) == 4
        assert list(s.values) == [100.0, 110.0, 120.0, 130.0]

    def test_annual_entries_excluded(self):
        """10-K annual entries must not appear in quarterly series."""
        entries = [
            _make_entry("2022-01-01", "2022-12-31", 400, form="10-K"),  # 364 days — annual
            _make_entry("2023-01-01", "2023-03-31", 100),                # quarterly — include
        ]
        facts = _make_facts("Revenues", entries)
        s = extract_quarterly_series(facts, ["Revenues"])
        assert len(s) == 1
        assert s.iloc[0] == pytest.approx(100)

    def test_deduplication_keeps_latest_accession(self):
        """When two entries share (end, days), the higher accn wins."""
        entries = [
            _make_entry("2023-01-01", "2023-03-31", 100, accn="0001"),  # original
            _make_entry("2023-01-01", "2023-03-31", 105, accn="0002"),  # amendment
        ]
        facts = _make_facts("Revenues", entries)
        s = extract_quarterly_series(facts, ["Revenues"])
        assert len(s) == 1
        assert s.iloc[0] == pytest.approx(105)  # amendment wins

    def test_tag_fallback(self):
        """Second tag in tag_list used when first is absent."""
        entries = [_make_entry("2023-01-01", "2023-03-31", 200)]
        facts = _make_facts("SalesRevenueNet", entries)
        s = extract_quarterly_series(facts, ["Revenues", "SalesRevenueNet"])
        assert s.iloc[0] == pytest.approx(200)

    def test_missing_tag_returns_empty_series(self):
        facts = {"facts": {"us-gaap": {}}}
        s = extract_quarterly_series(facts, ["Revenues"])
        assert s.empty

    def test_too_short_period_excluded(self):
        """A 30-day period is not a quarter."""
        entries = [_make_entry("2023-01-01", "2023-01-31", 999)]
        facts = _make_facts("Revenues", entries)
        s = extract_quarterly_series(facts, ["Revenues"])
        assert s.empty

    def test_too_long_period_excluded(self):
        """A 180-day period is not a quarter."""
        entries = [_make_entry("2023-01-01", "2023-06-30", 999)]
        facts = _make_facts("Revenues", entries)
        s = extract_quarterly_series(facts, ["Revenues"])
        assert s.empty

    def test_boundary_days_included(self):
        """Entries exactly at _Q_MIN_DAYS and _Q_MAX_DAYS must be included."""
        from datetime import date, timedelta
        start = date(2023, 1, 1)
        # min boundary
        end_min = start + timedelta(days=_Q_MIN_DAYS)
        entries = [_make_entry(str(start), str(end_min), 50)]
        facts = _make_facts("Revenues", entries)
        s = extract_quarterly_series(facts, ["Revenues"])
        assert not s.empty

        # max boundary
        end_max = start + timedelta(days=_Q_MAX_DAYS)
        entries2 = [_make_entry(str(start), str(end_max), 50)]
        facts2 = _make_facts("Revenues", entries2)
        s2 = extract_quarterly_series(facts2, ["Revenues"])
        assert not s2.empty


# ═══════════════════════════════════════════════════════════════════════
# extract_annual_series
# ═══════════════════════════════════════════════════════════════════════

class TestExtractAnnualSeries:
    def test_extracts_10k_entries(self):
        entries = [
            _make_entry("2020-01-01", "2020-12-31", 400, form="10-K"),
            _make_entry("2021-01-01", "2021-12-31", 450, form="10-K"),
        ]
        facts = _make_facts("Revenues", entries)
        s = extract_annual_series(facts, ["Revenues"])
        assert len(s) == 2
        assert s.iloc[-1] == pytest.approx(450)

    def test_excludes_10q_entries(self):
        entries = [
            _make_entry("2023-01-01", "2023-03-31", 100, form="10-Q"),  # quarterly — exclude
            _make_entry("2022-01-01", "2022-12-31", 400, form="10-K"),  # annual — include
        ]
        facts = _make_facts("Revenues", entries)
        s = extract_annual_series(facts, ["Revenues"])
        assert len(s) == 1
        assert s.iloc[0] == pytest.approx(400)

    def test_deduplication_keeps_latest_accession(self):
        entries = [
            _make_entry("2022-01-01", "2022-12-31", 400, form="10-K", accn="0001"),
            _make_entry("2022-01-01", "2022-12-31", 410, form="10-K", accn="0002"),  # amendment
        ]
        facts = _make_facts("Revenues", entries)
        s = extract_annual_series(facts, ["Revenues"])
        assert len(s) == 1
        assert s.iloc[0] == pytest.approx(410)


# ═══════════════════════════════════════════════════════════════════════
# build_ltm
# ═══════════════════════════════════════════════════════════════════════

class TestBuildLTM:
    def _make_quarterly_series(self, values: list, start_date: str = "2023-03-31") -> pd.Series:
        """Create a quarterly pd.Series with quarterly DatetimeIndex."""
        idx = pd.date_range(start_date, periods=len(values), freq="QE")
        return pd.Series(values, index=idx, dtype=float)

    def test_sum_of_four_quarters(self):
        s = self._make_quarterly_series([100, 110, 120, 130])
        assert build_ltm(s) == pytest.approx(460.0)

    def test_more_than_four_quarters_uses_last_four(self):
        # 8 quarters; LTM = last 4 = 500+510+520+530 = 2060
        s = self._make_quarterly_series([100, 110, 120, 130, 500, 510, 520, 530])
        assert build_ltm(s) == pytest.approx(2060.0)

    def test_fewer_than_four_returns_nan(self):
        s = self._make_quarterly_series([100, 110, 120])
        assert math.isnan(build_ltm(s))

    def test_empty_series_returns_nan(self):
        assert math.isnan(build_ltm(pd.Series(dtype=float)))

    def test_none_returns_nan(self):
        assert math.isnan(build_ltm(None))

    def test_gap_too_wide_returns_nan(self):
        """If the 4 quarters span >_LTM_MAX_SPAN_DAYS, result must be NaN."""
        # Create quarters with a 12-month gap between Q2 and Q3
        idx = pd.DatetimeIndex([
            "2021-03-31",
            "2021-06-30",
            # Gap: Q3 is 18 months later than Q2 → span of 4 quarters ≈ 27 months
            "2022-12-31",
            "2023-03-31",
        ])
        s = pd.Series([100, 110, 120, 130], index=idx, dtype=float)
        assert math.isnan(build_ltm(s))

    def test_nan_values_in_series_propagate(self):
        s = self._make_quarterly_series([100, float("nan"), 120, 130])
        assert math.isnan(build_ltm(s))


# ═══════════════════════════════════════════════════════════════════════
# latest_balance
# ═══════════════════════════════════════════════════════════════════════

class TestLatestBalance:
    def test_returns_most_recent(self):
        s = pd.Series([100, 200, 300], index=pd.date_range("2020", periods=3, freq="YE"))
        assert latest_balance(s) == pytest.approx(300.0)

    def test_ignores_nans(self):
        s = pd.Series([100, float("nan"), 200],
                      index=pd.date_range("2020", periods=3, freq="YE"))
        assert latest_balance(s) == pytest.approx(200.0)

    def test_empty_returns_nan(self):
        assert math.isnan(latest_balance(pd.Series(dtype=float)))

    def test_none_returns_nan(self):
        assert math.isnan(latest_balance(None))

    def test_all_nan_returns_nan(self):
        s = pd.Series([float("nan"), float("nan")])
        assert math.isnan(latest_balance(s))


# ═══════════════════════════════════════════════════════════════════════
# build_ltm_financials
# ═══════════════════════════════════════════════════════════════════════

class TestBuildLTMFinancials:
    def test_revenue_ltm_computed(self):
        """End-to-end: four clean quarterly revenue entries → correct LTM."""
        entries = [
            _make_entry("2023-01-01", "2023-03-31", 100),
            _make_entry("2023-04-01", "2023-06-30", 110),
            _make_entry("2023-07-01", "2023-09-30", 120),
            _make_entry("2023-10-01", "2023-12-31", 130),
        ]
        facts = _make_facts("Revenues", entries)
        gaap_map = {
            "revenue": ["Revenues"],
        }
        result = build_ltm_financials(facts, gaap_map, ltm_metrics={"revenue"})
        assert result["revenue"] == pytest.approx(460.0)

    def test_balance_sheet_metric_uses_latest_balance(self):
        """Balance sheet metrics should return latest, not LTM sum."""
        entries = [
            _make_entry("2023-01-01", "2023-03-31", 500),
            _make_entry("2023-04-01", "2023-06-30", 600),
            _make_entry("2023-07-01", "2023-09-30", 700),
            _make_entry("2023-10-01", "2023-12-31", 800),
        ]
        facts = _make_facts("Assets", entries)
        gaap_map = {"total_assets": ["Assets"]}
        # total_assets is NOT in ltm_metrics → should use latest_balance
        result = build_ltm_financials(facts, gaap_map, ltm_metrics={"revenue"})
        assert result["total_assets"] == pytest.approx(800.0)

    def test_missing_tag_returns_nan(self):
        facts = {"facts": {"us-gaap": {}}}
        gaap_map = {"revenue": ["Revenues"]}
        result = build_ltm_financials(facts, gaap_map)
        assert math.isnan(result["revenue"])
