# universe.py
# ------------------------------------------------------------------
# DEPRECATED — this 5-ticker stub is no longer the operative universe.
#
# The application's real default universe is the _DEFAULT_SCREENER_TICKERS
# list defined in app.py (200+ tickers). Users can override it by uploading
# a CSV or placing a CustomScreener.csv file alongside app.py.
#
# This file is retained only for backward compatibility with any external
# script that imports UNIVERSE from sec_engine.universe. It emits a
# DeprecationWarning on import and will be removed in a future cleanup.
# ------------------------------------------------------------------

import warnings

warnings.warn(
    "sec_engine.universe is deprecated. The operative screener universe is "
    "_DEFAULT_SCREENER_TICKERS in app.py (200+ tickers). "
    "This stub will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Kept for backward compatibility only — do not extend this list here.
UNIVERSE = ["NVDA", "GOOGL", "MSFT", "AMZN", "META"]
