# tests/test_aggregation.py
# -----------------------------------------------------------------------
# Unit tests for sec_engine/aggregation.py
#
# Focuses on:
#   - NOPAT / ROIC computation with both default and custom tax rates
#   - UFCF vs LFCF distinction (the bug that was fixed)
#   - Net debt guard (debt=NaN → net_debt=NaN, not -cash)
#   - Working capital / capital_employed NaN guards
#   - EBITDA fallback path
#   - FCF and margin calculations
#
# build_company_summary() is tested via its returned dict keys —
# no UI/Streamlit dependency.
# -----------------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sec_engine.aggregation import build_company_summary, nan_to_zero, nz
from sec_engine.constants import NOPAT_TAX_RATE


# ── Fixtures ────────────────────────────────────────────────────────────

def _make_quarterly_history(values: list, start: str = "2020-03-31") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="QE")
    return pd.Series(values, index=idx, dtype=float)


def _minimal_summary(
    revenue=1000,
    gross_profit=700,
    operating_income=300,
    net_income=250,
    sga=100,
    rd=50,
    cogs=300,
    interest_expense=20,
    ocf=280,
    capex=80,
    ebitda=np.nan,
    depreciation=40,
    amortization=10,
    debt=200,
    equity=800,
    cash=100,
    total_assets=1200,
    total_liabilities=400,
    current_assets=400,
    current_liabilities=150,
    accounts_receivable=120,
    inventory=60,
    accounts_payable=80,
    lt_debt=180,
    retained_earnings=500,
    ppe=300,
    market_cap=5000,
    pe_ltm=20,
    eps_growth_pct=15,
    dividend_yield_pct=1.5,
    capex_from_yfinance=False,   # SEC path: capex already positive
    tax_rate=None,               # None → uses NOPAT_TAX_RATE default
):
    """
    Build a minimal but complete set of arguments for build_company_summary.
    Callers can override any field via keyword args.
    """
    rev_hist = _make_quarterly_history([900, 920, 960, 1000])
    lfcf_hist = _make_quarterly_history([150, 160, 190, 200])

    ltm_data = {
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "sga": sga,
        "rd": rd,
        "cogs": cogs,
        "interest_expense": interest_expense,
        "ocf": ocf,
        "capex": capex,
        "ebitda": ebitda,
        "depreciation": depreciation,
        "amortization": amortization,
    }

    balance_data = {
        "debt": debt,
        "equity": equity,
        "cash": cash,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "accounts_receivable": accounts_receivable,
        "inventory": inventory,
        "accounts_payable": accounts_payable,
        "long_term_debt": lt_debt,
        "retained_earnings": retained_earnings,
        "ppe": ppe,
    }

    metadata = {
        "name": "Test Corp",
        "industry": "Software",
        "market_cap": market_cap,
        "pe_ltm": pe_ltm,
        "eps_growth_pct": eps_growth_pct,
        "dividend_yield_pct": dividend_yield_pct,
    }

    kwargs = {}
    if tax_rate is not None:
        kwargs["tax_rate"] = tax_rate

    return build_company_summary(
        ticker="TEST",
        ltm_data=ltm_data,
        balance_data=balance_data,
        metadata=metadata,
        revenue_history=rev_hist,
        lfcf_history=lfcf_hist,
        capex_from_yfinance=capex_from_yfinance,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# nan_to_zero and nz helpers
# ═══════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_nan_to_zero_converts_nan(self):
        assert nan_to_zero(float("nan")) == 0.0

    def test_nan_to_zero_converts_none(self):
        assert nan_to_zero(None) == 0.0

    def test_nan_to_zero_preserves_zero(self):
        # float 0.0 is a real value — must NOT be truthy-collapsed
        assert nan_to_zero(0.0) == 0.0

    def test_nan_to_zero_preserves_value(self):
        assert nan_to_zero(42.5) == pytest.approx(42.5)

    def test_nz_converts_none_to_nan(self):
        assert math.isnan(nz(None))

    def test_nz_passes_through_float(self):
        assert nz(3.14) == pytest.approx(3.14)

    def test_nz_passes_through_zero(self):
        assert nz(0.0) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# ROIC / NOPAT with default tax rate
# ═══════════════════════════════════════════════════════════════════════

class TestROICDefaultTax:
    def test_roic_uses_21pct_by_default(self):
        """ROIC must use the 21% statutory rate when no override is given."""
        ebit = 300
        debt = 200
        equity = 800
        cash = 100
        invested_capital = debt + equity - cash  # = 900

        result = _minimal_summary(
            operating_income=ebit,
            debt=debt,
            equity=equity,
            cash=cash,
        )

        expected_nopat = ebit * (1 - NOPAT_TAX_RATE)          # 300 * 0.79 = 237
        expected_roic = (expected_nopat / invested_capital) * 100  # 237/900 * 100 = 26.33%

        assert result["ROIC %"] == pytest.approx(expected_roic, rel=1e-4)

    def test_nopat_tax_rate_constant_is_021(self):
        """Guard against accidental constant changes."""
        assert NOPAT_TAX_RATE == pytest.approx(0.21)


# ═══════════════════════════════════════════════════════════════════════
# ROIC with custom per-ticker tax rate
# ═══════════════════════════════════════════════════════════════════════

class TestROICCustomTax:
    def test_custom_12pct_tax_rate(self):
        """
        A company with NOLs might have a 12% effective rate.
        ROIC with 12% must differ from ROIC with 21%.
        """
        ebit = 300
        debt = 200
        equity = 800
        cash = 100
        invested_capital = debt + equity - cash

        result_default = _minimal_summary(operating_income=ebit, debt=debt, equity=equity, cash=cash)
        result_low_tax = _minimal_summary(operating_income=ebit, debt=debt, equity=equity, cash=cash,
                                          tax_rate=0.12)

        expected_nopat_low = ebit * (1 - 0.12)
        expected_roic_low = (expected_nopat_low / invested_capital) * 100

        assert result_low_tax["ROIC %"] == pytest.approx(expected_roic_low, rel=1e-4)
        # Must be different from the 21% result
        assert result_low_tax["ROIC %"] != pytest.approx(result_default["ROIC %"], rel=1e-3)

    def test_custom_35pct_tax_rate(self):
        """A high-tax jurisdiction: ROIC must be lower than at 21%."""
        result_default = _minimal_summary()
        result_high_tax = _minimal_summary(tax_rate=0.35)
        assert result_high_tax["ROIC %"] < result_default["ROIC %"]

    def test_zero_tax_rate(self):
        """Tax-free: NOPAT = EBIT."""
        ebit = 300
        invested_capital = 200 + 800 - 100
        result = _minimal_summary(operating_income=ebit, tax_rate=0.0)
        expected_roic = (ebit / invested_capital) * 100
        assert result["ROIC %"] == pytest.approx(expected_roic, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# UFCF vs LFCF distinction
# ═══════════════════════════════════════════════════════════════════════

class TestUFCFvsLFCF:
    def test_ufcf_greater_than_lfcf_when_interest_nonzero(self):
        """UFCF adds back after-tax interest → always ≥ LFCF for positive interest."""
        result = _minimal_summary(ocf=280, capex=80, interest_expense=20)
        assert result["UFCF Margin %"] >= result["LFCF Margin %"]

    def test_ufcf_equals_lfcf_plus_after_tax_interest(self):
        """
        Verify the exact arithmetic:
        LFCF = OCF - CapEx = 280 - 80 = 200
        UFCF = LFCF + |interest| * (1 - t) = 200 + 20 * 0.79 = 215.8
        """
        result = _minimal_summary(
            revenue=1000, ocf=280, capex=80, interest_expense=20,
            capex_from_yfinance=False,
        )
        lfcf = 280 - 80          # = 200
        ufcf = lfcf + 20 * (1 - NOPAT_TAX_RATE)   # = 215.8

        assert result["LFCF Margin %"] == pytest.approx(lfcf / 1000 * 100, rel=1e-4)
        assert result["UFCF Margin %"] == pytest.approx(ufcf / 1000 * 100, rel=1e-4)

    def test_ufcf_nan_when_interest_missing(self):
        """If interest expense is NaN we cannot compute UFCF."""
        result = _minimal_summary(interest_expense=np.nan)
        assert math.isnan(result["UFCF Margin %"])

    def test_ufcf_nan_when_interest_zero(self):
        """Interest=0 is ambiguous — could be unreported. Must not silently alias LFCF."""
        result = _minimal_summary(interest_expense=0)
        assert math.isnan(result["UFCF Margin %"])


# ═══════════════════════════════════════════════════════════════════════
# Net debt guard
# ═══════════════════════════════════════════════════════════════════════

class TestNetDebtGuard:
    def test_net_debt_nan_when_debt_nan(self):
        """
        If debt is unknown, net_debt = 0 - cash would be misleading.
        Must return NaN, not -cash.
        """
        result = _minimal_summary(debt=np.nan, cash=500)
        # Net debt is used in Net Debt/Interest — that should be NaN
        assert math.isnan(result["Net Debt/Interest"])

    def test_net_debt_computed_correctly_when_debt_known(self):
        # debt=200, cash=100 → net_debt=100; interest=20 → ratio=5
        result = _minimal_summary(debt=200, cash=100, interest_expense=20)
        assert result["Net Debt/Interest"] == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════
# Working capital and capital employed guards
# ═══════════════════════════════════════════════════════════════════════

class TestWorkingCapitalGuards:
    def test_working_capital_nan_when_current_assets_nan(self):
        """Missing current_assets must propagate NaN, not substitute 0."""
        result = _minimal_summary(current_assets=np.nan, current_liabilities=150)
        # Quick ratio uses current_assets — should be NaN
        assert math.isnan(result["Current Ratio"])

    def test_working_capital_nan_when_current_liabilities_nan(self):
        result = _minimal_summary(current_assets=400, current_liabilities=np.nan)
        assert math.isnan(result["Current Ratio"])

    def test_working_capital_correct_when_both_present(self):
        result = _minimal_summary(current_assets=400, current_liabilities=150)
        assert result["Current Ratio"] == pytest.approx(400 / 150, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# EBITDA fallback
# ═══════════════════════════════════════════════════════════════════════

class TestEBITDAFallback:
    def test_ebitda_computed_from_ebit_plus_da(self):
        """When ebitda tag is NaN, EBITDA = EBIT + D&A."""
        result = _minimal_summary(
            operating_income=300,
            ebitda=np.nan,    # force fallback path
            depreciation=40,
            amortization=10,
        )
        expected_ebitda_margin = (300 + 40 + 10) / 1000 * 100  # 35%
        assert result["EBITDA Margin %"] == pytest.approx(expected_ebitda_margin, rel=1e-4)

    def test_ebitda_direct_takes_precedence(self):
        """When ebitda is available directly, D&A fallback must not override it."""
        result = _minimal_summary(
            operating_income=300,
            ebitda=380,   # direct value
            depreciation=40,
            amortization=10,
        )
        assert result["EBITDA Margin %"] == pytest.approx(38.0, rel=1e-4)

    def test_ebitda_nan_when_ebit_nan(self):
        result = _minimal_summary(operating_income=np.nan, ebitda=np.nan)
        assert math.isnan(result["EBITDA Margin %"])


# ═══════════════════════════════════════════════════════════════════════
# capex sign normalization
# ═══════════════════════════════════════════════════════════════════════

class TestCapexSignNormalization:
    def test_yfinance_negative_capex_flipped(self):
        """yfinance returns capex as negative; we must abs() it."""
        result_pos = _minimal_summary(capex=80, capex_from_yfinance=False)
        result_neg = _minimal_summary(capex=-80, capex_from_yfinance=True)
        assert result_pos["LFCF Margin %"] == pytest.approx(result_neg["LFCF Margin %"], rel=1e-4)

    def test_sec_positive_capex_not_flipped(self):
        """SEC capex is already positive; must not double-abs."""
        result = _minimal_summary(capex=80, capex_from_yfinance=False)
        lfcf = (280 - 80) / 1000 * 100
        assert result["LFCF Margin %"] == pytest.approx(lfcf, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Summary dict completeness
# ═══════════════════════════════════════════════════════════════════════

class TestSummaryCompleteness:
    REQUIRED_KEYS = [
        "Company", "Ticker", "Industry", "Market Cap (M)", "Data As Of",
        "ROA %", "ROIC %", "ROE %", "RCE %",
        "Gross Margin %", "SG&A Margin %", "R&D Margin %", "EBITDA Margin %",
        "EBIT Margin %", "Net Margin %", "LFCF Margin %", "UFCF Margin %",
        "CapEx % Revenue",
        "Total Asset Turnover", "AR Turnover", "Inventory Turnover",
        "Current Ratio", "Quick Ratio",
        "Avg Days Sales Outstanding", "Avg Days Inventory Outstanding",
        "Avg Days Payable Outstanding", "Cash Conversion Cycle",
        "Total D/E", "Total D/Capital", "LT D/E", "LT D/Capital",
        "Total Liab/Assets", "EBIT/Interest", "EBITDA/Interest",
        "Total Debt/Interest", "Net Debt/Interest", "Altman Z-Score",
        "Revenue YoY %",
        "Revenue 2yr CAGR %", "Revenue 3yr CAGR %", "Revenue 5yr CAGR %",
        "PEG (PE LTM)", "PEG (Lynch)", "FCF Yield %",
    ]

    def test_all_required_keys_present(self):
        result = _minimal_summary()
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_ticker_and_company_set(self):
        result = _minimal_summary()
        assert result["Ticker"] == "TEST"
        assert result["Company"] == "Test Corp"

    def test_market_cap_in_millions(self):
        # market_cap passed as 5_000_000_000 (5B) → should be 5000M
        result = _minimal_summary(market_cap=5_000_000_000)
        assert result["Market Cap (M)"] == pytest.approx(5000.0)
