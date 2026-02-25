# constants.py
# ------------------------------------------------------------------
# Shared financial constants used across sec_engine modules.
#
# Centralizing these values ensures that all modules (aggregation,
# ratios, multiples, any future DCF model) use a consistent set of
# assumptions. Change a value here and it propagates everywhere.
# ------------------------------------------------------------------

# Effective tax rate used to compute:
#   NOPAT  = EBIT × (1 - NOPAT_TAX_RATE)
#   UFCF   = LFCF + Interest Expense × (1 - NOPAT_TAX_RATE)
#
# The US federal statutory corporate rate is 21% (Tax Cuts and Jobs Act,
# 2017). Companies with significant deferred taxes, NOLs, or material
# international operations may have effective rates that differ.
# This is the blended rate used as a practical approximation for
# publicly-traded US equities; adjust for specific DCF or WACC work.
NOPAT_TAX_RATE: float = 0.21


# ------------------------------------------------------------------
# Per-Ticker Effective Tax Rate Override Registry
# ------------------------------------------------------------------
# This registry allows analysts to specify a company-specific effective
# tax rate that replaces the statutory default of 21%.
#
# When to use this:
#   - Companies with large NOL (net operating loss) carryforwards that
#     substantially reduce their cash tax rate (e.g. 8-12% effective rate).
#   - Companies with significant R&D tax credits or accelerated
#     depreciation benefits.
#   - Capital-intensive industries (utilities, mining) where deferred
#     tax assets are structurally large.
#   - Companies operating primarily in high-tax foreign jurisdictions.
#
# How to populate:
#   You can override at runtime via set_ticker_tax_rate(), or you can
#   hardcode a curated list below for tickers you analyze repeatedly.
#
# Format: { "TICKER": effective_rate_as_float }
#   e.g. { "AMZN": 0.12 } means 12% effective rate for Amazon
#
# Sources for effective rates:
#   - Company 10-K, footnote "Income Taxes" (3-year average cash tax rate)
#   - SEC EDGAR: IncomeTaxesPaid / IncomeBeforeIncomeTaxesDomestic
#
# Note: This dict is intentionally empty by default. The statutory 21%
# is the correct starting point when no better information is available.
# Populating it with wrong numbers is worse than using the default.
# ------------------------------------------------------------------
TICKER_TAX_RATE_OVERRIDES: dict = {
    # ── Examples (commented out — uncomment and adjust as needed) ──────────
    # "AMZN": 0.12,   # Heavy NOL carryforwards + accelerated depreciation
    # "TSLA": 0.10,   # Significant deferred tax assets from prior losses
    # "GOOGL": 0.17,  # International income taxed at lower rates
    # "ORCL": 0.19,   # Moderate international benefit
    # "XOM":  0.26,   # Resource extraction, typically above statutory
}


def get_effective_tax_rate(ticker: str) -> float:
    """
    Return the effective tax rate for a given ticker.

    Priority order:
      1. Value in TICKER_TAX_RATE_OVERRIDES (analyst-specified)
      2. NOPAT_TAX_RATE (21% statutory default)

    Args:
        ticker: Stock ticker symbol (case-insensitive).

    Returns:
        Effective tax rate as a float between 0.0 and 1.0.
    """
    if not ticker:
        return NOPAT_TAX_RATE
    return TICKER_TAX_RATE_OVERRIDES.get(ticker.upper().strip(), NOPAT_TAX_RATE)


def set_ticker_tax_rate(ticker: str, rate: float) -> None:
    """
    Register a per-ticker effective tax rate at runtime.

    Use this in notebooks, tests, or app startup to override rates
    without editing the source file.

    Args:
        ticker: Stock ticker symbol (case-insensitive).
        rate:   Effective tax rate as a decimal fraction [0.0, 1.0].
                Values outside [0, 1] raise ValueError.

    Raises:
        ValueError: If rate is not in [0.0, 1.0].

    Example:
        set_ticker_tax_rate("AMZN", 0.12)
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(
            f"Tax rate {rate!r} for '{ticker}' is out of range [0.0, 1.0]. "
            "Provide a decimal fraction, not a percentage (e.g. 0.12, not 12)."
        )
    TICKER_TAX_RATE_OVERRIDES[ticker.upper().strip()] = rate


def clear_ticker_tax_rate(ticker: str) -> None:
    """
    Remove a per-ticker override, reverting to the 21% statutory default.

    Args:
        ticker: Stock ticker symbol (case-insensitive).
    """
    TICKER_TAX_RATE_OVERRIDES.pop(ticker.upper().strip(), None)


def list_tax_rate_overrides() -> dict:
    """Return a copy of the current per-ticker tax rate registry."""
    return dict(TICKER_TAX_RATE_OVERRIDES)
