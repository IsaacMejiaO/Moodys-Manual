# sec_engine/universe.py
# ------------------------------------------------------------------
# Default ticker universe for the SEC-Based Equity Analyzer.
#
# This list is used as a fallback when no CustomScreener.csv is found
# on disk and no universe has been uploaded via the UI.
#
# To override at runtime, either:
#   1. Drop a CustomScreener.csv next to app.py (preferred)
#   2. Upload a CSV via the "Upload Universe" widget in the sidebar
#   3. Edit this list directly
# ------------------------------------------------------------------

UNIVERSE = [
    "NVDA", "GOOGL", "MSFT", "AMZN", "META", "NFLX",
]
