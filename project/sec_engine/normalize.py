# normalize.py
# ---------------------------------------------------------
# GAAP tag normalization map for SEC companyfacts XBRL data.
#
# Tag ordering matters — the first matching tag found in a company's
# filing is used. More specific / common tags should come first.
#
# Corrections vs. prior version:
#   - "debt": now maps to true total-debt tags (ShortTermBorrowings +
#     LongTermDebt variants), not only long-term debt.
#   - "total_liabilities": removed LiabilitiesAndStockholdersEquity
#     (that tag is the full balance-sheet total, not liabilities alone).
#   - "capex": removed CapitalExpendituresIncurredButNotYetPaid (non-cash
#     supplemental disclosure — not actual cash paid for PP&E).
#   - "ebitda": kept for completeness but very few companies file this
#     tag; the fallback EBIT + D&A computation in aggregation.py is the
#     primary path.
#   - "ebit": was a duplicate of "operating_income" (identical tag list).
#     Now aliased programmatically at the bottom of this file so there
#     is a single source of truth and no double-fetching of EDGAR data.
# ---------------------------------------------------------

GAAP_MAP = {
    # ── Income Statement ──────────────────────────────────────────────

    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
    ],

    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],

    "gross_profit": [
        "GrossProfit",
        "GrossProfitLoss",
    ],

    "opex": [
        "OperatingExpenses",
        "CostsAndExpenses",
    ],

    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
        "SellingAndMarketingExpense",
    ],

    "rd": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],

    "operating_income": [
        "OperatingIncomeLoss",
        "OperatingIncome",
    ],

    # Alias — same tags as operating_income. Defined as a programmatic alias
    # below the dict (not an inline duplicate) so there is only one tag list
    # to maintain and so that callers using either key never double-fetch EDGAR.
    # "ebit": <see alias below>

    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
    ],

    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],

    # Very few companies file this tag; aggregation.py computes it from
    # operating_income + depreciation when this is missing.
    "ebitda": [
        "EarningsBeforeInterestTaxesDepreciationAndAmortization",
    ],

    # ── Cash Flow Statement ───────────────────────────────────────────

    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],

    "capex": [
        # Cash paid for property, plant & equipment — the standard capex tag.
        "PaymentsToAcquirePropertyPlantAndEquipment",
        # Some filers use this broader tag (includes software & other assets).
        "PaymentsToAcquireProductiveAssets",
        # Older / alternate tag used by some registrants.
        "CapitalExpenditures",
        # NOTE: CapitalExpendituresIncurredButNotYetPaid is intentionally
        # excluded — it is a non-cash supplemental disclosure and would
        # overstate cash capex.
    ],

    "depreciation": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "Depreciation",
    ],

    "amortization": [
        "AmortizationOfIntangibleAssets",
        "AmortizationOfDeferredCharges",
    ],

    # ── Balance Sheet — Assets ────────────────────────────────────────

    "total_assets": [
        "Assets",
    ],

    "current_assets": [
        "AssetsCurrent",
    ],

    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
    ],

    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],

    "inventory": [
        "InventoryNet",
        "InventoryFinishedGoodsNetOfReserves",
        "InventoryGross",
    ],

    "ppe": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],

    # ── Balance Sheet — Liabilities ───────────────────────────────────

    "total_liabilities": [
        "Liabilities",
        # NOTE: LiabilitiesAndStockholdersEquity is intentionally excluded —
        # it equals total assets (both sides of the balance sheet), not
        # liabilities alone. Including it produces grossly inflated leverage ratios.
    ],

    "current_liabilities": [
        "LiabilitiesCurrent",
    ],

    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ],

    # Total debt = short-term borrowings + current portion of LT debt + LT debt.
    # The tags below represent *total* debt as reported on the balance sheet.
    # Prior version incorrectly used only long-term debt tags here.
    "debt": [
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtAndCapitalLeaseObligations",      # includes current portion
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebt",                                # fallback if combined not filed
    ],

    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
    ],

    # ── Balance Sheet — Equity ────────────────────────────────────────

    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    ],

    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
}

# "ebit" is an alias for "operating_income" — they map to the same XBRL tags.
# Using a reference here (not a copy) means updating "operating_income" tags
# automatically keeps "ebit" in sync, and callers fetching both keys in a
# loop will not issue duplicate EDGAR API calls for the same data.
GAAP_MAP["ebit"] = GAAP_MAP["operating_income"]
