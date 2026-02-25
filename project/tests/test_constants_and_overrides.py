# tests/test_constants_and_overrides.py
# -----------------------------------------------------------------------
# Tests for the per-ticker tax rate registry (constants.py) and the
# peer override registry (peer_finder.py).
#
# These tests are fully isolated: they use set/clear helpers so that
# any mutations to the registry are cleaned up after each test.
# -----------------------------------------------------------------------

import math
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sec_engine.constants import (
    NOPAT_TAX_RATE,
    get_effective_tax_rate,
    set_ticker_tax_rate,
    clear_ticker_tax_rate,
    list_tax_rate_overrides,
    TICKER_TAX_RATE_OVERRIDES,
)
from sec_engine.peer_finder import (
    get_peer_override,
    set_peer_override,
    clear_peer_override,
    list_peer_overrides,
    find_peers_by_sic,
    PEER_OVERRIDES,
)


# ═══════════════════════════════════════════════════════════════════════
# Tax rate registry
# ═══════════════════════════════════════════════════════════════════════

class TestTaxRateRegistry:
    def teardown_method(self):
        """Clean up any overrides set during tests."""
        for ticker in ["TESTCO", "LOWCO", "HIGHCO", "ZEROCO"]:
            clear_ticker_tax_rate(ticker)

    def test_default_returns_statutory_rate(self):
        assert get_effective_tax_rate("TESTCO") == pytest.approx(NOPAT_TAX_RATE)

    def test_set_and_get_override(self):
        set_ticker_tax_rate("TESTCO", 0.12)
        assert get_effective_tax_rate("TESTCO") == pytest.approx(0.12)

    def test_case_insensitive(self):
        set_ticker_tax_rate("lowco", 0.08)
        assert get_effective_tax_rate("LOWCO") == pytest.approx(0.08)
        assert get_effective_tax_rate("lowco") == pytest.approx(0.08)
        assert get_effective_tax_rate("LowCo") == pytest.approx(0.08)

    def test_clear_reverts_to_default(self):
        set_ticker_tax_rate("TESTCO", 0.15)
        clear_ticker_tax_rate("TESTCO")
        assert get_effective_tax_rate("TESTCO") == pytest.approx(NOPAT_TAX_RATE)

    def test_zero_rate_valid(self):
        set_ticker_tax_rate("ZEROCO", 0.0)
        assert get_effective_tax_rate("ZEROCO") == pytest.approx(0.0)

    def test_one_rate_valid(self):
        set_ticker_tax_rate("HIGHCO", 1.0)
        assert get_effective_tax_rate("HIGHCO") == pytest.approx(1.0)

    def test_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            set_ticker_tax_rate("TESTCO", 1.01)

    def test_rate_below_zero_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            set_ticker_tax_rate("TESTCO", -0.01)

    def test_percentage_raises(self):
        """Passing 12 instead of 0.12 must raise, not silently store wrong value."""
        with pytest.raises(ValueError):
            set_ticker_tax_rate("TESTCO", 12.0)

    def test_empty_ticker_returns_default(self):
        assert get_effective_tax_rate("") == pytest.approx(NOPAT_TAX_RATE)

    def test_none_ticker_returns_default(self):
        assert get_effective_tax_rate(None) == pytest.approx(NOPAT_TAX_RATE)

    def test_list_overrides_returns_copy(self):
        """list_tax_rate_overrides() must return a copy, not the live dict."""
        set_ticker_tax_rate("TESTCO", 0.14)
        snapshot = list_tax_rate_overrides()
        # Mutating the snapshot must not affect the registry
        snapshot["TESTCO"] = 0.99
        assert get_effective_tax_rate("TESTCO") == pytest.approx(0.14)

    def test_clear_nonexistent_does_not_raise(self):
        """Clearing a ticker with no override must be a no-op."""
        clear_ticker_tax_rate("DOESNOTEXIST")  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# Peer override registry
# ═══════════════════════════════════════════════════════════════════════

class TestPeerOverrideRegistry:
    def teardown_method(self):
        for ticker in ["BRKB", "BRK-B", "GE", "TESTCO"]:
            clear_peer_override(ticker)

    def test_no_override_returns_none(self):
        assert get_peer_override("TESTCO") is None

    def test_set_and_get_override(self):
        set_peer_override("GE", ["HON", "MMM", "RTX"])
        result = get_peer_override("GE")
        assert result == ["HON", "MMM", "RTX"]

    def test_case_insensitive_set(self):
        set_peer_override("brk-b", ["JPM", "BAC", "GS"])
        assert get_peer_override("BRK-B") == ["JPM", "BAC", "GS"]

    def test_clear_reverts_to_none(self):
        set_peer_override("GE", ["HON", "MMM"])
        clear_peer_override("GE")
        assert get_peer_override("GE") is None

    def test_deduplication(self):
        set_peer_override("GE", ["HON", "MMM", "HON", "RTX"])
        result = get_peer_override("GE")
        assert result == ["HON", "MMM", "RTX"]  # HON appears once

    def test_self_removed_from_peers(self):
        set_peer_override("GE", ["GE", "HON", "MMM"])
        result = get_peer_override("GE")
        assert "GE" not in result

    def test_empty_list_clears_override(self):
        set_peer_override("GE", ["HON", "MMM"])
        set_peer_override("GE", [])
        assert get_peer_override("GE") is None

    def test_list_overrides_returns_copy(self):
        set_peer_override("GE", ["HON", "MMM"])
        snapshot = list_peer_overrides()
        snapshot["GE"] = ["DUMMY"]
        assert get_peer_override("GE") == ["HON", "MMM"]

    def test_clear_nonexistent_does_not_raise(self):
        clear_peer_override("DOESNOTEXIST")


# ═══════════════════════════════════════════════════════════════════════
# find_peers_by_sic respects override
# ═══════════════════════════════════════════════════════════════════════

class TestFindPeersBySicOverride:
    """
    Test that find_peers_by_sic() returns the override list immediately
    without running SIC matching logic.
    """

    def _make_df(self, tickers, caps=None):
        if caps is None:
            caps = [1000.0] * len(tickers)
        return pd.DataFrame({"Ticker": tickers, "Market Cap (M)": caps})

    def teardown_method(self):
        clear_peer_override("CONGL")

    def test_override_bypasses_sic_matching(self):
        """
        CONGL has no SIC entry — normally would fall back to returning
        all other tickers. With an override, must return the exact list.
        """
        set_peer_override("CONGL", ["JPM", "BAC", "GS"])
        df = self._make_df(["CONGL", "JPM", "BAC", "GS", "UNRELATED"])
        sic_map = {}  # deliberately empty — override must win without SIC

        result = find_peers_by_sic("CONGL", sic_map, df, max_peers=10)
        assert result == ["JPM", "BAC", "GS"]

    def test_override_respects_max_peers(self):
        set_peer_override("CONGL", ["A", "B", "C", "D", "E"])
        df = self._make_df(["CONGL", "A", "B", "C", "D", "E"])
        sic_map = {}

        result = find_peers_by_sic("CONGL", sic_map, df, max_peers=3)
        assert len(result) <= 3
        assert result == ["A", "B", "C"]

    def test_no_override_still_runs_sic_logic(self):
        """Without an override, SIC matching logic runs normally."""
        # Two tickers with matching SIC code
        sic_map = {"TARGET": "7372", "PEER1": "7372", "OTHER": "1234"}
        df = self._make_df(["TARGET", "PEER1", "OTHER"], [1000, 900, 500])

        result = find_peers_by_sic("TARGET", sic_map, df, max_peers=5)
        assert "PEER1" in result
        assert "OTHER" not in result  # different SIC

    def test_override_excludes_self(self):
        """The target ticker must never appear in its own peer list."""
        set_peer_override("CONGL", ["CONGL", "JPM", "BAC"])
        df = self._make_df(["CONGL", "JPM", "BAC"])
        sic_map = {}

        result = find_peers_by_sic("CONGL", sic_map, df, max_peers=10)
        assert "CONGL" not in result


# ═══════════════════════════════════════════════════════════════════════
# Integration: tax rate flows through ROIC correctly
# ═══════════════════════════════════════════════════════════════════════

class TestTaxRateIntegration:
    """
    Verify that a per-ticker tax rate override actually changes ROIC
    when build_company_summary is called, without any direct import of
    the override mechanism inside aggregation.py.
    """

    def teardown_method(self):
        clear_ticker_tax_rate("AMZN_TEST")

    def test_roic_changes_with_tax_override(self):
        from sec_engine.aggregation import build_company_summary
        import numpy as np

        def _make_summary(tax_rate=None):
            ltm = {k: np.nan for k in [
                "gross_profit", "net_income", "sga", "rd", "cogs",
                "interest_expense", "ocf", "capex", "ebitda",
                "depreciation", "amortization",
            ]}
            ltm.update({"revenue": 1000, "operating_income": 300})
            balance = {k: np.nan for k in [
                "equity", "cash", "total_assets", "total_liabilities",
                "current_assets", "current_liabilities", "accounts_receivable",
                "inventory", "accounts_payable", "long_term_debt",
                "retained_earnings", "ppe",
            ]}
            balance.update({"debt": 200, "equity": 800, "cash": 100})
            metadata = {"name": "Test", "industry": "Tech",
                        "market_cap": np.nan, "pe_ltm": np.nan,
                        "eps_growth_pct": np.nan, "dividend_yield_pct": np.nan}
            rev_hist = pd.Series(
                [900.0, 920.0, 960.0, 1000.0],
                index=pd.date_range("2020-03-31", periods=4, freq="QE")
            )
            kwargs = {}
            if tax_rate is not None:
                kwargs["tax_rate"] = tax_rate
            return build_company_summary(
                ticker="AMZN_TEST",
                ltm_data=ltm,
                balance_data=balance,
                metadata=metadata,
                revenue_history=rev_hist,
                lfcf_history=rev_hist,
                capex_from_yfinance=False,
                **kwargs,
            )

        # Default 21%
        result_default = _make_summary()
        # Low tax 12%
        result_low = _make_summary(tax_rate=0.12)
        # High tax 35%
        result_high = _make_summary(tax_rate=0.35)

        assert result_low["ROIC %"] > result_default["ROIC %"] > result_high["ROIC %"]
        assert result_default["Effective Tax Rate"] == pytest.approx(0.21)
        assert result_low["Effective Tax Rate"] == pytest.approx(0.12)
        assert result_high["Effective Tax Rate"] == pytest.approx(0.35)

    def test_registry_override_flows_through(self):
        """Registry-level override (no explicit arg) still affects ROIC."""
        from sec_engine.aggregation import build_company_summary
        import numpy as np

        def _make_summary_no_arg():
            ltm = {k: np.nan for k in [
                "gross_profit", "net_income", "sga", "rd", "cogs",
                "interest_expense", "ocf", "capex", "ebitda",
                "depreciation", "amortization",
            ]}
            ltm.update({"revenue": 1000, "operating_income": 300})
            balance = {k: np.nan for k in [
                "equity", "cash", "total_assets", "total_liabilities",
                "current_assets", "current_liabilities", "accounts_receivable",
                "inventory", "accounts_payable", "long_term_debt",
                "retained_earnings", "ppe",
            ]}
            balance.update({"debt": 200, "equity": 800, "cash": 100})
            metadata = {"name": "Test", "industry": "Tech",
                        "market_cap": np.nan, "pe_ltm": np.nan,
                        "eps_growth_pct": np.nan, "dividend_yield_pct": np.nan}
            rev_hist = pd.Series(
                [900.0, 920.0, 960.0, 1000.0],
                index=pd.date_range("2020-03-31", periods=4, freq="QE")
            )
            return build_company_summary(
                ticker="AMZN_TEST",
                ltm_data=ltm,
                balance_data=balance,
                metadata=metadata,
                revenue_history=rev_hist,
                lfcf_history=rev_hist,
                capex_from_yfinance=False,
            )

        result_before = _make_summary_no_arg()
        set_ticker_tax_rate("AMZN_TEST", 0.08)
        result_after = _make_summary_no_arg()

        # Lower tax → higher ROIC
        assert result_after["ROIC %"] > result_before["ROIC %"]
        assert result_after["Effective Tax Rate"] == pytest.approx(0.08)
