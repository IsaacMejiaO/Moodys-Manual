# calculations.py
# ------------------------------------------------------------------
# DEPRECATED — this file is not imported anywhere in the application
# and will be removed in a future cleanup.
#
# All metric calculations have been moved to:
#   sec_engine/metrics.py      ← pure financial metric functions
#   sec_engine/aggregation.py  ← per-company summary builder
#
# Notable bugs in the original that are fixed in metrics.py:
#   - revenue_cagr() used stock *price* history as a proxy for revenue
#     growth, which is conceptually incorrect. Use metrics.series_cagr()
#     with a proper SEC or yfinance revenue series instead.
#   - safe_divide() did not handle float NaN inputs (only None and 0).
#     metrics.safe_divide() handles all three cases.
#   - peg_ratios() mixed up earningsQuarterlyGrowth (a decimal, e.g. 0.15)
#     with a percent argument and double-counted the ×100 conversion.
#
# IMPORTANT: This module previously raised ImportError at load time,
# which broke pytest discovery for the entire sec_engine package.
# It now emits a DeprecationWarning instead so that:
#   - Accidental imports produce a visible warning rather than a crash.
#   - Test runners can still traverse the package without errors.
# ------------------------------------------------------------------

import warnings

warnings.warn(
    "calculations.py is deprecated and will be removed. "
    "Import from sec_engine.metrics or sec_engine.aggregation instead.",
    DeprecationWarning,
    stacklevel=2,
)
