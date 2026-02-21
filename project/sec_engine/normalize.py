# normalize.py
# ---------------------------------------------------------
# GAAP tag normalization map for SEC companyfacts.
# ---------------------------------------------------------

GAAP_MAP = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
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

    "operating_income": [
        "OperatingIncomeLoss",
        "OperatingIncome",
    ],

    "ebit": [
        "OperatingIncomeLoss",
        "OperatingIncome",
    ],

    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],

    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],

    "depreciation": [
        "DepreciationAndAmortization",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
    ],

    "ebitda": [
        "EarningsBeforeInterestTaxesDepreciationAndAmortization",
        "EarningsBeforeInterestTaxesDepreciationAndAmortizationAndRent",
    ],

    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],

    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
        "PaymentsForProceedsFromProductiveAssets",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],

    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    ],

    "debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "DebtCurrent",
        "DebtNoncurrent",
        "ShortTermBorrowings",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
    ],

    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],

    # ---------- Keys referenced by SEC backup in app.py ----------

    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
        "SellingAndMarketingExpense",
    ],

    "rd": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],

    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
    ],

    "total_assets": [
        "Assets",
    ],

    "current_assets": [
        "AssetsCurrent",
    ],

    "current_liabilities": [
        "LiabilitiesCurrent",
    ],

    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
    ],

    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],

    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ],

    "inventory": [
        "InventoryNet",
        "InventoryFinishedGoodsNetOfReserves",
    ],

    "ppe": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],

    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
}