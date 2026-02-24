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
