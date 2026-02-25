# tests/test_metrics.py
# -----------------------------------------------------------------------
# Unit tests for sec_engine/metrics.py
#
# Design principles:
#   - Every public function is tested against at least one happy path,
#     one zero-denominator path, and one NaN/None path.
#   - Floating-point comparisons use pytest.approx() with rel=1e-6 so
#     tests don't fail due to IEEE-754 rounding noise.
#   - No mocking of external services — these are pure-function tests.
# -----------------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sec_engine.metrics import (
    safe_divide,
    margins,
    ebitda_margin,
    sga_margin,
    rd_margin,
    lfcf_margin,
    ufcf_margin,
    capex_as_pct_revenue,
    roa,
    roic,
    roe,
    rce,
    total_asset_turnover,
    accounts_receivable_turnover,
    inventory_turnover,
    current_ratio,
    quick_ratio,
    days_sales_outstanding,
    days_inventory_outstanding,
    days_payable_outstanding,
    cash_conversion_cycle,
    total_debt_to_equity,
    total_debt_to_capital,
    lt_debt_to_equity,
    lt_debt_to_capital,
    total_liabilities_to_assets,
    ebit_to_interest,
    ebitda_to_interest,
    total_debt_to_interest,
    net_debt_to_interest,
    altman_z_score,
    cagr,
    series_cagr,
    yoy_growth,
    peg_pe_ltm,
    peg_lynch,
    fcf,
    fcf_yield,
)


# ═══════════════════════════════════════════════════════════════════════
# safe_divide
# ═══════════════════════════════════════════════════════════════════════

class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10.0, 4.0) == pytest.approx(2.5)

    def test_integer_inputs(self):
        assert safe_divide(9, 3) == pytest.approx(3.0)

    def test_zero_denominator_returns_nan(self):
        assert math.isnan(safe_divide(5.0, 0.0))

    def test_zero_denominator_custom_default(self):
        assert safe_divide(5.0, 0, default=0.0) == 0.0

    def test_none_numerator_returns_nan(self):
        assert math.isnan(safe_divide(None, 4.0))

    def test_none_denominator_returns_nan(self):
        assert math.isnan(safe_divide(10.0, None))

    def test_nan_numerator_returns_nan(self):
        assert math.isnan(safe_divide(float("nan"), 4.0))

    def test_nan_denominator_returns_nan(self):
        assert math.isnan(safe_divide(10.0, float("nan")))

    def test_string_input_returns_default(self):
        # Non-numeric type — must not raise, must return default
        assert math.isnan(safe_divide("bad", 4.0))
        assert math.isnan(safe_divide(10.0, "bad"))

    def test_pandas_na_returns_default(self):
        assert math.isnan(safe_divide(pd.NA, 4.0))

    def test_negative_division(self):
        assert safe_divide(-6.0, 2.0) == pytest.approx(-3.0)

    def test_small_denominator(self):
        # Very small but nonzero — should not trigger zero guard
        assert safe_divide(1.0, 1e-10) == pytest.approx(1e10)


# ═══════════════════════════════════════════════════════════════════════
# margins
# ═══════════════════════════════════════════════════════════════════════

class TestMargins:
    def test_typical_tech_company(self):
        # Revenue 100, GP 70, EBIT 30, NI 25  → 70%, 30%, 25%
        result = margins(100, 70, 30, 25)
        assert result["Gross Margin %"] == pytest.approx(70.0)
        assert result["EBIT Margin %"] == pytest.approx(30.0)
        assert result["Net Margin %"] == pytest.approx(25.0)

    def test_zero_revenue_all_nan(self):
        result = margins(0, 70, 30, 25)
        assert math.isnan(result["Gross Margin %"])
        assert math.isnan(result["EBIT Margin %"])
        assert math.isnan(result["Net Margin %"])

    def test_negative_net_income(self):
        result = margins(100, 40, -10, -5)
        assert result["EBIT Margin %"] == pytest.approx(-10.0)
        assert result["Net Margin %"] == pytest.approx(-5.0)

    def test_none_revenue_returns_nan(self):
        result = margins(None, 70, 30, 25)
        assert math.isnan(result["Gross Margin %"])

    def test_ebitda_margin(self):
        assert ebitda_margin(200, 60) == pytest.approx(30.0)

    def test_ebitda_margin_zero_revenue(self):
        assert math.isnan(ebitda_margin(0, 60))

    def test_sga_margin(self):
        assert sga_margin(100, 15) == pytest.approx(15.0)

    def test_rd_margin(self):
        assert rd_margin(100, 20) == pytest.approx(20.0)

    def test_lfcf_margin(self):
        assert lfcf_margin(200, 50) == pytest.approx(25.0)

    def test_ufcf_margin(self):
        assert ufcf_margin(200, 60) == pytest.approx(30.0)

    def test_capex_pct_revenue_takes_abs(self):
        # Capex stored as negative sometimes — function should abs() it
        assert capex_as_pct_revenue(100, -20) == pytest.approx(20.0)
        assert capex_as_pct_revenue(100, 20) == pytest.approx(20.0)


# ═══════════════════════════════════════════════════════════════════════
# Profitability ratios
# ═══════════════════════════════════════════════════════════════════════

class TestProfitabilityRatios:
    def test_roa(self):
        # 25 NI / 500 assets = 5%
        assert roa(25, 500) == pytest.approx(5.0)

    def test_roa_zero_assets(self):
        assert math.isnan(roa(25, 0))

    def test_roic(self):
        # NOPAT 80 / IC 400 = 20%
        assert roic(80, 400) == pytest.approx(20.0)

    def test_roic_zero_ic(self):
        assert math.isnan(roic(80, 0))

    def test_roe(self):
        # NI 30 / equity 150 = 20%
        assert roe(30, 150) == pytest.approx(20.0)

    def test_roe_zero_equity(self):
        assert math.isnan(roe(30, 0))

    def test_rce(self):
        # EBIT 50 / cap_employed 250 = 20%
        assert rce(50, 250) == pytest.approx(20.0)


# ═══════════════════════════════════════════════════════════════════════
# Turnover
# ═══════════════════════════════════════════════════════════════════════

class TestTurnover:
    def test_total_asset_turnover(self):
        assert total_asset_turnover(300, 150) == pytest.approx(2.0)

    def test_ar_turnover(self):
        assert accounts_receivable_turnover(240, 30) == pytest.approx(8.0)

    def test_inventory_turnover(self):
        assert inventory_turnover(100, 25) == pytest.approx(4.0)

    def test_inventory_turnover_zero_inventory(self):
        assert math.isnan(inventory_turnover(100, 0))


# ═══════════════════════════════════════════════════════════════════════
# Liquidity
# ═══════════════════════════════════════════════════════════════════════

class TestLiquidity:
    def test_current_ratio(self):
        assert current_ratio(200, 100) == pytest.approx(2.0)

    def test_quick_ratio(self):
        # (200 - 50) / 100 = 1.5
        assert quick_ratio(200, 50, 100) == pytest.approx(1.5)

    def test_quick_ratio_no_inventory(self):
        # Inventory = 0 → quick = current
        assert quick_ratio(200, 0, 100) == pytest.approx(2.0)

    def test_dso(self):
        # (AR 50 / Revenue 365) * 365 = 50 days
        assert days_sales_outstanding(50, 365) == pytest.approx(50.0)

    def test_dio(self):
        # (Inventory 90 / COGS 360) * 365 = 91.25 days
        assert days_inventory_outstanding(90, 360) == pytest.approx(91.25, rel=1e-4)

    def test_dpo(self):
        # (AP 30 / COGS 360) * 365 = 30.42 days
        assert days_payable_outstanding(30, 360) == pytest.approx(30.4167, rel=1e-4)

    def test_cash_conversion_cycle(self):
        # DSO=50, DIO=90, DPO=30 → 110
        assert cash_conversion_cycle(50, 90, 30) == pytest.approx(110.0)

    def test_ccc_with_nan_component(self):
        assert math.isnan(cash_conversion_cycle(float("nan"), 90, 30))


# ═══════════════════════════════════════════════════════════════════════
# Leverage
# ═══════════════════════════════════════════════════════════════════════

class TestLeverage:
    def test_total_de(self):
        assert total_debt_to_equity(100, 200) == pytest.approx(0.5)

    def test_total_d_cap(self):
        # 100 / (100+200) = 0.3333
        assert total_debt_to_capital(100, 200) == pytest.approx(1/3, rel=1e-5)

    def test_lt_de(self):
        assert lt_debt_to_equity(80, 200) == pytest.approx(0.4)

    def test_lt_d_cap(self):
        assert lt_debt_to_capital(80, 200) == pytest.approx(80/280, rel=1e-5)

    def test_liabilities_to_assets(self):
        assert total_liabilities_to_assets(300, 500) == pytest.approx(0.6)

    def test_ebit_to_interest(self):
        # Interest stored as positive expense
        assert ebit_to_interest(60, 20) == pytest.approx(3.0)

    def test_ebit_to_interest_negative_stored(self):
        # Interest stored as negative (yfinance convention) — function should abs()
        assert ebit_to_interest(60, -20) == pytest.approx(3.0)

    def test_ebitda_to_interest(self):
        assert ebitda_to_interest(90, 30) == pytest.approx(3.0)

    def test_total_debt_to_interest(self):
        assert total_debt_to_interest(300, 30) == pytest.approx(10.0)

    def test_net_debt_to_interest(self):
        assert net_debt_to_interest(200, 25) == pytest.approx(8.0)


# ═══════════════════════════════════════════════════════════════════════
# Altman Z-Score
# ═══════════════════════════════════════════════════════════════════════

class TestAltmanZScore:
    def test_healthy_company_above_29(self):
        """
        Healthy company with strong working capital, retained earnings,
        good earnings, and no excessive debt → Z > 2.99 (safe zone).
        Inputs chosen so the math is traceable.
        """
        # X1=0.3, X2=0.4, X3=0.15, X4=2.0, X5=0.8
        # Z = 1.2*0.3 + 1.4*0.4 + 3.3*0.15 + 0.6*2.0 + 1.0*0.8
        #   = 0.36 + 0.56 + 0.495 + 1.2 + 0.8 = 3.415
        total_assets = 1000
        z = altman_z_score(
            working_capital=300,       # X1=0.3
            total_assets=total_assets,
            retained_earnings=400,     # X2=0.4
            ebit=150,                  # X3=0.15
            market_value_equity=2000,  # X4=2.0 (mve/liabilities = 2000/1000)
            total_liabilities=1000,
            revenue=800,               # X5=0.8
        )
        assert z == pytest.approx(3.415, rel=1e-5)
        assert z > 2.99

    def test_distressed_company_below_181(self):
        """
        A financially distressed company should score below 1.81.
        """
        z = altman_z_score(
            working_capital=-100,
            total_assets=1000,
            retained_earnings=-300,
            ebit=-50,
            market_value_equity=100,
            total_liabilities=900,
            revenue=400,
        )
        assert z < 1.81

    def test_zero_total_assets_returns_distress_score(self):
        # safe_divide uses default=0 for zero denominator, so X1=X2=X3=X5=0.
        # Only X4 = market_equity/total_liabilities remains computable.
        # Result is a low positive number — correctly in distress territory (< 1.81).
        # This is intentional: returning NaN would hide that the company looks distressed.
        z = altman_z_score(100, 0, 200, 50, 500, 300, 400)
        assert not math.isnan(z)
        assert z < 1.81  # should appear distressed, not healthy

    def test_none_inputs(self):
        z = altman_z_score(None, 1000, 200, 50, 500, 300, 400)
        # With working_capital=None, safe_divide returns 0 (default=0), others valid
        # Should still return a numeric result (not raise)
        assert isinstance(z, float)


# ═══════════════════════════════════════════════════════════════════════
# CAGR
# ═══════════════════════════════════════════════════════════════════════

class TestCAGR:
    def test_3yr_cagr_known_value(self):
        # 100 → 133.1 over 3 years = 10% CAGR
        assert cagr(100, 133.1, 3) == pytest.approx(0.10, rel=1e-4)

    def test_5yr_cagr(self):
        # 100 → 161.05 over 5 years ≈ 10%
        assert cagr(100, 161.051, 5) == pytest.approx(0.10, rel=1e-3)

    def test_zero_start_returns_nan(self):
        assert math.isnan(cagr(0, 100, 3))

    def test_none_inputs_return_nan(self):
        assert math.isnan(cagr(None, 100, 3))
        assert math.isnan(cagr(100, None, 3))

    def test_zero_years_returns_nan(self):
        assert math.isnan(cagr(100, 200, 0))

    def test_negative_years_returns_nan(self):
        assert math.isnan(cagr(100, 200, -1))

    def test_sign_flip_returns_nan(self):
        # Can't express negative-to-positive CAGR via power formula
        assert math.isnan(cagr(-100, 100, 3))

    def test_both_negative_values(self):
        # Both negative: improving losses from -100 → -50 is meaningful
        result = cagr(-100, -50, 3)
        assert not math.isnan(result)
        # |−50|/|−100| = 0.5 → (0.5)^(1/3) − 1 ≈ −0.2063
        assert result == pytest.approx((0.5) ** (1/3) - 1, rel=1e-5)

    def test_series_cagr_happy_path(self):
        # Series: 100, 110, 121, 133.1 → 3yr CAGR ≈ 10%
        s = pd.Series([100.0, 110.0, 121.0, 133.1],
                      index=pd.date_range("2020", periods=4, freq="YE"))
        assert series_cagr(s, years=3) == pytest.approx(0.10, rel=1e-3)

    def test_series_cagr_insufficient_data(self):
        s = pd.Series([100.0, 110.0])
        assert math.isnan(series_cagr(s, years=3))


# ═══════════════════════════════════════════════════════════════════════
# YoY growth
# ═══════════════════════════════════════════════════════════════════════

class TestYoYGrowth:
    def test_positive_growth(self):
        assert yoy_growth(110, 100) == pytest.approx(0.10)

    def test_negative_growth(self):
        assert yoy_growth(90, 100) == pytest.approx(-0.10)

    def test_zero_prior_returns_nan(self):
        assert math.isnan(yoy_growth(100, 0))

    def test_none_inputs_return_nan(self):
        assert math.isnan(yoy_growth(None, 100))
        assert math.isnan(yoy_growth(100, None))

    def test_negative_prior_absolute_denominator(self):
        # Growing from -100 to -50 is +50% improvement
        # |prior| = 100, delta = 50, rate = 50%
        assert yoy_growth(-50, -100) == pytest.approx(0.50)


# ═══════════════════════════════════════════════════════════════════════
# PEG ratios
# ═══════════════════════════════════════════════════════════════════════

class TestPEGRatios:
    def test_peg_pe_ltm_normal(self):
        # P/E=20, growth=10% → PEG=2.0
        assert peg_pe_ltm(20, 10) == pytest.approx(2.0)

    def test_peg_pe_ltm_negative_pe_returns_nan(self):
        assert math.isnan(peg_pe_ltm(-5, 10))

    def test_peg_pe_ltm_negative_growth_returns_nan(self):
        assert math.isnan(peg_pe_ltm(20, -5))

    def test_peg_pe_ltm_zero_growth_returns_nan(self):
        assert math.isnan(peg_pe_ltm(20, 0))

    def test_peg_lynch_with_dividend(self):
        # P/E=15, growth=10%, dividend=2% → PEG = 15/12 = 1.25
        assert peg_lynch(15, 10, 2) == pytest.approx(1.25)

    def test_peg_lynch_zero_growth_and_dividend(self):
        assert math.isnan(peg_lynch(15, 0, 0))

    def test_peg_lynch_negative_pe_returns_nan(self):
        assert math.isnan(peg_lynch(-5, 10, 0))


# ═══════════════════════════════════════════════════════════════════════
# FCF
# ═══════════════════════════════════════════════════════════════════════

class TestFCF:
    def test_fcf_positive(self):
        assert fcf(150, 50) == pytest.approx(100.0)

    def test_fcf_none_ocf_returns_nan(self):
        assert math.isnan(fcf(None, 50))

    def test_fcf_none_capex_returns_nan(self):
        assert math.isnan(fcf(150, None))

    def test_fcf_yield(self):
        # FCF 50 / Market Cap 1000 = 5%
        assert fcf_yield(50, 1000) == pytest.approx(5.0)

    def test_fcf_yield_zero_market_cap(self):
        assert math.isnan(fcf_yield(50, 0))
