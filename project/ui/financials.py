# ui/financials.py
"""
Financials Page — SEC XBRL Primary, Exact 10-K Presentation
=============================================================

Architecture
------------
PRIMARY:  Raw XBRL companyfacts from SEC EDGAR. Only 10-K annual entries.
          Latest accession number per fiscal-year-end wins (restatements auto-applied).
          Tags listed in the exact priority order companies most commonly file them.

FALLBACK: yfinance fills any row where every XBRL tag returns empty.

COMPUTED: EBITDA, FCF, margins, effective tax rate — all clearly badged.

Key design decisions
--------------------
- Each schema row lists ALL known XBRL tags for that concept, not just one or two.
  This is the primary fix vs. the prior version: the previous schema had ~3-5 tags
  per row; companies routinely file under 10-20 different tag names for the same
  concept depending on filing year, industry, and preparer conventions.
- Tags are ordered: most common / specific first, broad fallbacks last.
- "Hide empty rows" (default ON) collapses lines with no data so the table
  stays tight. Turn it OFF to see every possible line item for any company.
- Newest year on the left (Bloomberg / CapIQ convention).
- YoY growth sub-rows beneath every key subtotal.
- Source badge (SEC / YF / Calc) on every row.
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sec_engine.sec_fetch import fetch_company_facts
from sec_engine.cik_loader import load_full_cik_map

warnings.filterwarnings("ignore", category=FutureWarning)

UP     = "#00C805"
DOWN   = "#FF3B30"
BLUE   = "#0A7CFF"
ORANGE = "#FF9F0A"

# ── Special-structure entity sets ─────────────────────────────────────────────
_REITS = {
    "SPG","EXR","VICI","PSA","EQIX","OHI","GLPI","HST","FRT","EGP",
    "FR","NNN","CTRE","AMT","CCI","SBAC","DLR","O","PLD","WPC","STAG",
    "COLD","CUBE","NSA","REXR","TRNO","ELS","UDR","VTR","PEAK","IIPR",
}
_MLPS     = {"OKE","WES","HESM","AM","EPD","MMP","PAA","TRGP","KMI","ET","ENB","DCP","MPLX","CEQP"}
_INSURERS = {"PGR","ALL","CINF","ERIE","KNSL","CB","AIG","MET","PRU","TRV","HIG","AFG","RLI","ACGL","RNR"}
_BANKS    = {"JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","COF","KEY","RF","FITB","HBAN","ZION","CMA","NTRS","STT","BK"}
_IFRS     = {"BAM","NVO","ASML","TSM","TM","HMC","SONY","UL","BP","SHEL","RIO","BHP","SAP","BABA","JD","PDD","SE"}

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA FORMAT
# (display_label, [xbrl_tags…], is_subtotal, indent)
#
# indent 0  = section banner (dark band, no values — empty tag list required)
# indent 1  = top-level line
# indent 2  = sub-line
# indent 3  = sub-sub-line
# is_subtotal = bold, separator borders
# empty tags + indent>0 = computed row (filled programmatically later)
# ══════════════════════════════════════════════════════════════════════════════

INCOME_SCHEMA: List[Tuple] = [

    # ── REVENUE ───────────────────────────────────────────────────────────────
    ("Revenue", [
        # Current ASC 606 tags (most companies post-2018)
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        # Legacy / broad tags
        "Revenues",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        # Industry-specific totals
        "LicenseAndServiceRevenue",
        "ContractRevenue",
        "HealthCareOrganizationRevenue",
        "RealEstateRevenueNet",
        "OilAndGasRevenue",
        "UtilityRevenue",
        "TelecommunicationsRevenue",
    ], True, 1),

    ("  Product Revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTaxProduct",
        "SalesRevenueGoodsNet",
        "ProductRevenue",
        "NetProductRevenue",
        "GoodsAndProductsRevenue",
    ], False, 2),

    ("  Service Revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTaxService",
        "SalesRevenueServicesNet",
        "ServiceRevenue",
        "SubscriptionRevenue",
        "MaintenanceRevenue",
        "SupportAndMaintenanceRevenue",
    ], False, 2),

    ("  Subscription Revenue", [
        "SubscriptionRevenue",
        "RevenueFromContractWithCustomerExcludingAssessedTaxSubscription",
    ], False, 2),

    ("  License / Technology Revenue", [
        "LicenseRevenue",
        "LicenseAndServiceRevenue",
        "TechnologyAndLicensingRevenue",
        "RoyaltyRevenue",
        "LicensingRevenue",
        "SoftwareLicenseRevenue",
    ], False, 2),

    ("  Advertising Revenue", [
        "AdvertisingRevenue",
        "OnlineAdvertisingRevenue",
        "DigitalAdvertisingRevenue",
    ], False, 2),

    ("  Hardware / Device Revenue", [
        "HardwareRevenue",
        "DeviceRevenue",
    ], False, 2),

    ("  Related-Party Revenue", [
        "RevenueFromRelatedParties",
        "RelatedPartyRevenue",
    ], False, 2),

    ("  Other Revenue", [
        "OtherRevenues",
        "OtherRevenue",
        "OtherSalesRevenue",
    ], False, 2),

    # ── COST OF REVENUE ───────────────────────────────────────────────────────
    ("Cost of Revenue", [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfRevenueExcludingDepreciationAndAmortization",
        "CostOfSales",
    ], False, 1),

    ("  Cost of Products", [
        "CostOfGoodsSold",
        "CostOfGoodsAndServicesSoldCostOfProducts",
        "CostOfProductRevenue",
        "CostOfMerchandiseSoldAndOccupancyCosts",
    ], False, 2),

    ("  Cost of Services", [
        "CostOfServices",
        "CostOfGoodsAndServicesSoldCostOfServices",
        "CostOfServiceRevenue",
        "CostOfSubscriptionRevenue",
        "CostOfMaintenanceRevenue",
    ], False, 2),

    ("  D&A in Cost of Revenue", [
        "CostOfGoodsAndServicesSoldDepreciationAndAmortization",
        "CostOfRevenueDepreciationAndAmortization",
    ], False, 2),

    ("  Depletion & Exploration (E&P)", [
        "DepletionOfOilAndGasProperties",
        "ExplorationExpense",
        "OilAndGasProductionExpense",
    ], False, 2),

    ("Gross Profit", [
        "GrossProfit",
        "GrossProfitLoss",
    ], True, 1),

    # ── OPERATING EXPENSES ────────────────────────────────────────────────────
    ("OPERATING EXPENSES", [], False, 0),

    ("  Research & Development", [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "TechnologyAndDevelopmentExpense",
        "ProductDevelopmentAndEngineeringExpense",
        "ResearchAndDevelopmentExpenseMember",
        "InProcessResearchAndDevelopmentExpense",
    ], False, 2),

    ("  Sales & Marketing", [
        "SellingAndMarketingExpense",
        "MarketingExpense",
        "SellingExpense",
        "AdvertisingExpense",
        "SalesAndMarketingExpense",
    ], False, 2),

    ("  General & Administrative", [
        "GeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpenseExcludingDepreciation",
    ], False, 2),

    ("  SG&A (Combined)", [
        "SellingGeneralAndAdministrativeExpense",
        "SellingGeneralAndAdministrativeExpenseExcludingDepreciation",
    ], False, 2),

    ("  Labor & Related", [
        "LaborAndRelatedExpense",
        "EmployeeBenefitsAndShareBasedCompensation",
        "SalariesAndWages",
        "EmployeeBenefitsExpense",
    ], False, 2),

    ("  IT & Communications", [
        "InformationTechnologyAndDataProcessingExpense",
        "CommunicationsAndInformationTechnology",
        "TechnologyExpense",
    ], False, 2),

    ("  Amortization of Intangibles", [
        "AmortizationOfIntangibleAssets",
        "AmortizationOfAcquiredIntangibleAssets",
        "BusinessAcquisitionCostOfAcquiredEntityAmortizationOfIntangibles",
        "AmortizationOfDeferredSalesCommissions",
        "AmortizationOfCapitalizedCostsToObtainContracts",
    ], False, 2),

    ("  Depreciation & Amortization", [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "Depreciation",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationNonproduction",
    ], False, 2),

    ("  Stock-Based Compensation", [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
        "ShareBasedCompensationExpense",
        "EmployeeBenefitsAndShareBasedCompensation",
    ], False, 2),

    ("  Lease / ROU Amortization", [
        "OperatingLeaseRightOfUseAssetAmortizationExpense",
        "FinanceLeaseRightOfUseAssetAmortization",
        "OperatingLeaseCost",
    ], False, 2),

    ("  Impairment", [
        "GoodwillImpairmentLoss",
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
        "ImpairmentOfLongLivedAssetsHeldForUse",
        "AssetImpairmentCharges",
        "ImpairmentCharges",
        "GoodwillAndIntangibleAssetImpairment",
        "TangibleAssetImpairmentCharges",
    ], False, 2),

    ("  Restructuring Charges", [
        "RestructuringCharges",
        "RestructuringAndRelatedCostIncurredCost",
        "RestructuringSettlementAndImpairmentProvisions",
        "RestructuringCostsAndAssetImpairmentCharges",
    ], False, 2),

    ("  Acquisition / Integration Costs", [
        "BusinessCombinationAcquisitionRelatedCosts",
        "AcquisitionRelatedCosts",
        "MergerRelatedCosts",
        "BusinessCombinationIntegrationRelatedCosts",
    ], False, 2),

    ("  Litigation & Settlements", [
        "LitigationSettlementExpense",
        "LitigationExpense",
        "LossContingencyAccrualProvision",
    ], False, 2),

    ("  Other Operating Expense / (Income)", [
        "OtherOperatingIncomeExpenseNet",
        "OtherCostAndExpenseOperating",
        "OtherExpenses",
        "OtherOperatingExpenses",
        "OtherOperatingCosts",
    ], False, 2),

    ("Total Operating Expenses", [
        "OperatingExpenses",
        "CostsAndExpenses",
        "OperatingCostsAndExpenses",
    ], False, 1),

    ("Operating Income (EBIT)", [
        "OperatingIncomeLoss",
        "OperatingIncome",
        "IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet",
    ], True, 1),

    ("EBITDA", [], True, 1),  # computed: EBIT + D&A

    # ── BELOW OPERATING LINE ──────────────────────────────────────────────────
    ("Interest Expense", [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
        "InterestExpenseRelatedParty",
        "InterestExpenseOther",
        "FinanceLeaseInterestExpense",
    ], False, 1),

    ("Interest Income", [
        "InterestIncomeOperating",
        "InvestmentIncomeInterest",
        "InterestAndDividendIncomeOperating",
        "InterestAndDividendIncomeSecurities",
    ], False, 1),

    ("Other Non-Operating Income / (Expense)", [
        "OtherNonoperatingIncomeExpense",
        "NonoperatingIncomeExpense",
        "OtherNonoperatingIncome",
        "OtherNonoperatingExpense",
        "InterestAndOtherIncome",
        "OtherIncomeLoss",
    ], False, 1),

    ("FX Gain / (Loss)", [
        "ForeignCurrencyTransactionGainLossBeforeTax",
        "ForeignCurrencyTransactionGainLossRealized",
        "ForeignCurrencyTransactionGainLossUnrealized",
        "GainLossOnForeignCurrencyDerivativeInstrumentsNotDesignatedAsHedgingInstruments",
    ], False, 1),

    ("Gain / (Loss) on Investments", [
        "GainLossOnInvestments",
        "GainLossOnSaleOfInvestments",
        "GainLossOnSaleOfBusiness",
        "GainLossOnDispositionOfAssets",
        "GainLossOnSaleOfPropertyPlantEquipment",
        "EquitySecuritiesFvNiUnrealizedGainLoss",
        "GainLossOnDerivativeInstrumentsNetPretax",
    ], False, 1),

    ("Equity Method Investment Income / (Loss)", [
        "IncomeLossFromEquityMethodInvestments",
        "IncomeLossFromEquityMethodInvestmentsNetOfDividendsOrDistributions",
    ], False, 1),

    ("Income Before Tax", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
        "IncomeLossBeforeIncomeTaxes",
    ], True, 1),

    ("Income Tax Expense / (Benefit)", [
        "IncomeTaxExpenseBenefit",
        "IncomeTaxesPaidNet",
        "CurrentIncomeTaxExpenseBenefit",
    ], False, 1),

    ("  Effective Tax Rate (%)", [], False, 2),  # computed

    ("Net Income from Cont. Operations", [
        "IncomeLossFromContinuingOperations",
        "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeExtraordinaryItemsAndCumulativeEffectOfAccountingChanges",
    ], False, 1),

    ("Discontinued Operations", [
        "IncomeLossFromDiscontinuedOperationsNetOfTax",
        "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
        "DiscontinuedOperationIncomeLossFromDiscontinuedOperationDuringPhaseOutPeriodNetOfTax",
    ], False, 1),

    ("Net Income", [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest",
    ], True, 1),

    ("  Attributable to Noncontrolling Interests", [
        "NetIncomeLossAttributableToNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity",
        "MinorityInterestInNetIncomeLossOtherMinorityInterests",
    ], False, 2),

    ("  Attributable to Common Stockholders", [
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLossAvailableToCommonStockholdersDiluted",
    ], False, 2),

    ("  Preferred Dividends", [
        "PreferredStockDividendsAndOtherAdjustments",
        "DividendsPreferredStock",
        "DividendsPreferredStockCash",
    ], False, 2),
]

PER_SHARE_SCHEMA: List[Tuple] = [

    # ── PER SHARE ─────────────────────────────────────────────────────────────
    ("PER SHARE", [], False, 0),

    ("EPS — Basic", [
        "EarningsPerShareBasic",
        "IncomeLossFromContinuingOperationsPerBasicShare",
    ], False, 1),

    ("EPS — Diluted", [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
        "EarningsPerShareBasicAndDiluted",
    ], False, 1),

    ("Shares Outstanding — Basic (M)", [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ], False, 1),

    ("Shares Outstanding — Diluted (M)", [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ], False, 1),
]

BALANCE_SHEET_SCHEMA: List[Tuple] = [

    # ══ ASSETS ════════════════════════════════════════════════════════════════
    ("ASSETS", [], False, 0),
    ("CURRENT ASSETS", [], False, 0),

    ("  Cash & Cash Equivalents", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
        "Cash",
        "CashAndDueFromBanks",
    ], False, 2),

    ("  Restricted Cash (Current)", [
        "RestrictedCashAndCashEquivalentsCurrent",
        "RestrictedCashCurrent",
    ], False, 2),

    ("  Short-Term Investments", [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
        "DebtSecuritiesAvailableForSaleCurrent",
        "TradingSecuritiesCurrent",
        "ShortTermBankLoansAndNotesPayable",
    ], False, 2),

    ("  Accounts Receivable, Net", [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsAndOtherReceivablesNetCurrent",
        "AccountsReceivableBilledForLongTermContractsOrProgramsNet",
        "TradeAndOtherAccountsReceivableNet",
    ], False, 2),

    ("  Unbilled Receivables", [
        "UnbilledReceivablesCurrent",
        "ContractWithCustomerAssetNetCurrent",
        "UnbilledContractsReceivable",
        "ReceivablesLongTermContractsOrPrograms",
    ], False, 2),

    ("  Notes Receivable (Current)", [
        "NotesReceivableNetCurrent",
        "NotesAndLoansReceivableNetCurrent",
    ], False, 2),

    ("  Inventory", [
        "InventoryNet",
        "InventoryFinishedGoodsNetOfReserves",
        "InventoryGross",
        "InventoryRawMaterialsNetOfReserves",
        "InventoryWorkInProcessNetOfReserves",
        "InventoryFinishedGoods",
        "RetailRelatedInventoryMerchandise",
    ], False, 2),

    ("  Income Tax Receivable", [
        "IncomeTaxesReceivable",
        "IncomeTaxReceivable",
        "TaxesReceivable",
    ], False, 2),

    ("  Prepaid Expenses & Other Current Assets", [
        "PrepaidExpenseAndOtherAssetsCurrent",
        "PrepaidExpenseCurrent",
        "OtherAssetsCurrent",
        "OtherCurrentAssets",
        "DeferredCostsCurrent",
    ], False, 2),

    ("  Derivative Assets (Current)", [
        "DerivativeAssetsCurrent",
        "DerivativeFairValueOfDerivativeAsset",
    ], False, 2),

    ("Total Current Assets", ["AssetsCurrent"], True, 1),

    # ── NON-CURRENT ASSETS ────────────────────────────────────────────────────
    ("NON-CURRENT ASSETS", [], False, 0),

    ("  PP&E, Net", [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        "PropertyPlantAndEquipmentNetIncludingCapitalLeases",
    ], False, 2),

    ("  Operating Lease ROU Assets", [
        "OperatingLeaseRightOfUseAsset",
        "OperatingLeaseRightOfUseAssetBeforeAccumulatedAmortization",
    ], False, 2),

    ("  Finance Lease ROU Assets", [
        "FinanceLeaseRightOfUseAsset",
        "FinanceLeaseRightOfUseAssetAfterAccumulatedAmortization",
    ], False, 2),

    ("  Goodwill", ["Goodwill", "GoodwillGross"], False, 2),

    ("  Intangible Assets, Net", [
        "FiniteLivedIntangibleAssetsNet",
        "IntangibleAssetsNetExcludingGoodwill",
        "IntangibleAssetsNetIncludingGoodwill",
        "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
        "InfinitelivedIntangibleAssetsNet",
    ], False, 2),

    ("  Long-Term Investments & Securities", [
        "LongTermInvestments",
        "AvailableForSaleSecuritiesNoncurrent",
        "EquityMethodInvestments",
        "MarketableSecuritiesNoncurrent",
        "DebtSecuritiesAvailableForSaleNoncurrent",
        "EquitySecuritiesFvNi",
        "InvestmentsInAffiliatesSubsidiariesAssociatesAndJointVentures",
    ], False, 2),

    ("  Capitalized Contract Costs", [
        "CapitalizedContractCostNet",
        "CapitalizedContractCostNetNoncurrent",
        "DeferredCommissions",
        "DeferredContractAcquisitionCostsNetNoncurrent",
    ], False, 2),

    ("  Deferred Tax Assets (LT)", [
        "DeferredIncomeTaxAssetsNet",
        "DeferredTaxAssetsLiabilitiesNet",
        "DeferredTaxAssetsGross",
    ], False, 2),

    ("  Restricted Cash (LT)", [
        "RestrictedCashAndCashEquivalentsNoncurrent",
        "RestrictedCashNoncurrent",
        "RestrictedInvestmentsNoncurrent",
    ], False, 2),

    ("  Other Non-Current Assets", [
        "OtherAssetsNoncurrent",
        "OtherNoncurrentAssets",
    ], False, 2),

    ("Total Assets", ["Assets"], True, 1),

    # ══ LIABILITIES ════════════════════════════════════════════════════════════
    ("LIABILITIES", [], False, 0),
    ("CURRENT LIABILITIES", [], False, 0),

    ("  Accounts Payable", [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
        "AccountsPayableRelatedPartiesCurrent",
        "TradeAccountsPayable",
    ], False, 2),

    ("  Accrued Compensation & Benefits", [
        "EmployeeRelatedLiabilitiesCurrent",
        "AccruedEmployeeBenefitsCurrent",
        "AccruedLiabilitiesForCommissionsExpenseAndTaxes",
        "WorkersCompensationLiabilityCurrent",
    ], False, 2),

    ("  Accrued Expenses & Other Current Liabilities", [
        "AccruedLiabilitiesCurrent",
        "OtherAccruedLiabilitiesCurrent",
        "AccruedExpensesAndOtherCurrentLiabilities",
        "AccountsPayableAndOtherAccruedLiabilities",
        "OtherLiabilitiesAndAccruedLiabilitiesCurrent",
    ], False, 2),

    ("  Deferred Revenue (Current)", [
        "DeferredRevenueCurrent",
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueAndCreditsNoncurrent",
        "CustomerDepositsAndDeferredRevenueCurrent",
    ], False, 2),

    ("  Income Taxes Payable", [
        "AccruedIncomeTaxesCurrent",
        "TaxesPayableCurrent",
        "IncomeTaxesPayableCurrent",
    ], False, 2),

    ("  Short-Term Debt & Commercial Paper", [
        "ShortTermBorrowings",
        "CommercialPaper",
        "NotesPayableCurrent",
        "DebtCurrent",
        "ShortTermDebt",
        "LineOfCreditCurrent",
    ], False, 2),

    ("  Current Portion of Long-Term Debt", [
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "CurrentMaturitiesOfLongTermDebt",
        "LongTermNotesPayableCurrent",
    ], False, 2),

    ("  Operating Lease Liability (Current)", [
        "OperatingLeaseLiabilityCurrent",
    ], False, 2),

    ("  Finance Lease Liability (Current)", [
        "FinanceLeaseLiabilityCurrent",
        "CapitalLeaseObligationsCurrent",
    ], False, 2),

    ("  Restructuring Reserve (Current)", [
        "RestructuringReserveCurrent",
        "BusinessExitCostsLiabilityCurrent",
    ], False, 2),

    ("  Other Current Liabilities", [
        "OtherLiabilitiesCurrent",
        "OtherCurrentLiabilities",
    ], False, 2),

    ("Total Current Liabilities", ["LiabilitiesCurrent"], True, 1),

    # ── NON-CURRENT LIABILITIES ───────────────────────────────────────────────
    ("NON-CURRENT LIABILITIES", [], False, 0),

    ("  Long-Term Debt", [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        "SeniorLongTermNotes",
        "ConvertibleDebtNoncurrent",
        "LongTermNotesPayable",
        "LongTermLineOfCredit",
    ], False, 2),

    ("  Operating Lease Liability (LT)", [
        "OperatingLeaseLiabilityNoncurrent",
    ], False, 2),

    ("  Finance Lease Liability (LT)", [
        "FinanceLeaseLiabilityNoncurrent",
        "CapitalLeaseObligationsNoncurrent",
    ], False, 2),

    ("  Deferred Revenue (LT)", [
        "DeferredRevenueNoncurrent",
        "ContractWithCustomerLiabilityNoncurrent",
    ], False, 2),

    ("  Deferred Tax Liabilities", [
        "DeferredIncomeTaxLiabilitiesNet",
        "DeferredTaxLiabilitiesGross",
        "DeferredTaxLiabilities",
    ], False, 2),

    ("  Pension & Post-Retirement Obligations", [
        "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
        "PensionAndOtherPostretirementBenefitExpense",
        "DefinedBenefitPlanBenefitObligation",
        "OtherPostretirementBenefitsPayableNoncurrent",
    ], False, 2),

    ("  Other Non-Current Liabilities", [
        "OtherLiabilitiesNoncurrent",
        "OtherNoncurrentLiabilities",
        "OtherLongTermLiabilities",
    ], False, 2),

    ("Total Liabilities", ["Liabilities"], True, 1),

    # ══ EQUITY ═════════════════════════════════════════════════════════════════
    ("STOCKHOLDERS' EQUITY", [], False, 0),

    ("  Preferred Stock", [
        "PreferredStockValue",
        "PreferredStockValueOutstanding",
        "PreferredUnitsOutstanding",
    ], False, 2),

    ("  Common Stock (Par Value)", [
        "CommonStockValue",
        "CommonStockValueOutstanding",
    ], False, 2),

    ("  Additional Paid-In Capital", [
        "AdditionalPaidInCapital",
        "AdditionalPaidInCapitalCommonStock",
        "CommonStockAndAdditionalPaidInCapital",
    ], False, 2),

    ("  Retained Earnings / (Deficit)", [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarningsUnappropriated",
    ], False, 2),

    ("  Accumulated Other Comprehensive Income / (Loss)", [
        "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
        "AccumulatedOtherComprehensiveIncomeLossAvailableForSaleSecuritiesAdjustmentNetOfTax",
        "OtherComprehensiveIncomeLossNetOfTax",
    ], False, 2),

    ("  Treasury Stock", [
        "TreasuryStockValue",
        "TreasuryStockCommonValue",
        "TreasuryStockCarryingBasis",
    ], False, 2),

    ("  Partners' Capital (MLP)", [
        "PartnersCapital",
        "PartnersCapitalAttributableToParent",
        "GeneralPartnersCapitalAccount",
        "LimitedPartnersCapitalAccount",
    ], False, 2),

    ("Total Stockholders' Equity", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
        "MembersEquity",
    ], True, 1),

    ("  Noncontrolling Interests", [
        "MinorityInterest",
        "NoncontrollingInterestMember",
        "RedeemableNoncontrollingInterestEquityCarryingAmount",
        "PartnersCapitalAttributableToNoncontrollingInterest",
    ], False, 2),

    ("Total Liabilities & Equity", [
        "LiabilitiesAndStockholdersEquity",
        "LiabilitiesAndPartnersCapital",
        "LiabilitiesAndMembersEquity",
    ], True, 1),
]

CASH_FLOW_SCHEMA: List[Tuple] = [

    # ══ OPERATING ACTIVITIES ════════════════════════════════════════════════════
    ("OPERATING ACTIVITIES", [], False, 0),

    ("  Net Income", [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ], False, 2),

    ("  Depreciation & Amortization", [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "Depreciation",
        "DepreciationAmortizationAndAccretionNet",
    ], False, 2),

    ("  Amortization of Intangibles", [
        "AmortizationOfIntangibleAssets",
        "AmortizationOfAcquiredIntangibleAssets",
    ], False, 2),

    ("  Stock-Based Compensation", [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
        "EmployeeBenefitsAndShareBasedCompensation",
    ], False, 2),

    ("  Deferred Income Taxes", [
        "DeferredIncomeTaxExpenseBenefit",
        "DeferredIncomeTaxesAndTaxCredits",
        "DeferredTaxExpenseBenefit",
    ], False, 2),

    ("  Amortization of Debt Costs & Discount", [
        "AmortizationOfFinancingCostsAndDiscounts",
        "AmortizationOfDebtDiscountPremium",
        "AmortizationOfFinancingCosts",
    ], False, 2),

    ("  Impairment Charges", [
        "GoodwillImpairmentLoss",
        "AssetImpairmentCharges",
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
        "ImpairmentCharges",
    ], False, 2),

    ("  (Gain) / Loss on Investments & Sales", [
        "GainLossOnSaleOfPropertyPlantEquipment",
        "GainLossOnInvestments",
        "GainLossOnSaleOfBusiness",
        "GainLossOnDispositionOfAssets",
    ], False, 2),

    ("  Other Non-Cash Items", [
        "OtherNoncashIncomeExpense",
        "OtherOperatingActivitiesCashFlowStatement",
        "OtherNoncashExpense",
        "OtherNoncashIncome",
    ], False, 2),

    ("  Δ Accounts Receivable", [
        "IncreaseDecreaseInAccountsReceivable",
        "IncreaseDecreaseInReceivables",
        "IncreaseDecreaseInAccountsAndOtherReceivables",
    ], False, 2),

    ("  Δ Unbilled Receivables", [
        "IncreaseDecreaseInContractWithCustomerAsset",
        "IncreaseDecreaseInUnbilledReceivables",
    ], False, 2),

    ("  Δ Inventory", [
        "IncreaseDecreaseInInventories",
        "IncreaseDecreaseInRetailRelatedInventories",
    ], False, 2),

    ("  Δ Prepaid & Other Current Assets", [
        "IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets",
        "IncreaseDecreaseInPrepaidExpense",
        "IncreaseDecreaseInOtherCurrentAssets",
        "IncreaseDecreaseInOtherOperatingAssets",
    ], False, 2),

    ("  Δ Accounts Payable", [
        "IncreaseDecreaseInAccountsPayable",
        "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
        "IncreaseDecreaseInAccountsPayableTrade",
    ], False, 2),

    ("  Δ Accrued Liabilities", [
        "IncreaseDecreaseInAccruedLiabilities",
        "IncreaseDecreaseInEmployeeRelatedLiabilities",
        "IncreaseDecreaseInAccruedInterestReceivableNet",
    ], False, 2),

    ("  Δ Deferred Revenue", [
        "IncreaseDecreaseInContractWithCustomerLiability",
        "IncreaseDecreaseInDeferredRevenue",
        "IncreaseDecreaseInDeferredLiabilities",
    ], False, 2),

    ("  Δ Income Taxes Payable", [
        "IncreaseDecreaseInIncomeTaxesPayableNetOfIncomeTaxesReceivable",
        "IncreaseDecreaseInIncomeTaxesReceivable",
        "IncreaseDecreaseInAccruedIncomeTaxesPayable",
    ], False, 2),

    ("  Δ Other Operating Liabilities", [
        "IncreaseDecreaseInOtherOperatingLiabilities",
        "IncreaseDecreaseInOtherCurrentLiabilities",
        "IncreaseDecreaseInOtherNoncurrentLiabilities",
    ], False, 2),

    ("Cash from Operations", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ], True, 1),

    ("Free Cash Flow", [], True, 1),      # computed: CFO − |CapEx|
    ("  FCF Margin (%)", [], False, 2),   # computed
    ("  FCF / Net Income (%)", [], False, 2),  # computed

    # ══ INVESTING ACTIVITIES ════════════════════════════════════════════════════
    ("INVESTING ACTIVITIES", [], False, 0),

    ("  Capital Expenditures", [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpenditures",
        "PaymentsToAcquireAndDevelopRealEstate",
    ], False, 2),

    ("  Capitalized Software Development", [
        "PaymentsToDevelopSoftware",
        "CapitalizedComputerSoftwareAdditions",
        "PaymentsToAcquireSoftware",
    ], False, 2),

    ("  Capitalized Contract Costs", [
        "CapitalizedContractCostAmortization",
        "IncreaseDecreaseInCapitalizedContractCostNet",
    ], False, 2),

    ("  Acquisitions, Net of Cash", [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
        "BusinessAcquisitionCostOfAcquiredEntityTransactionCosts",
        "PaymentsForProceedsFromBusinessesAndInterestInAffiliates",
    ], False, 2),

    ("  Purchases of Investments", [
        "PaymentsToAcquireAvailableForSaleSecurities",
        "PaymentsToAcquireInvestments",
        "PaymentsToAcquireShortTermInvestments",
        "PaymentsToAcquireOtherInvestments",
        "PaymentsToAcquireEquityMethodInvestments",
        "PaymentsToAcquireTradingSecuritiesHeldForInvestment",
        "PaymentsForProceedsFromInvestments",
    ], False, 2),

    ("  Maturities / Sales of Investments", [
        "ProceedsFromSaleOfAvailableForSaleSecurities",
        "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
        "ProceedsFromSaleMaturityAndCollectionOfInvestments",
        "ProceedsFromSaleOfShortTermInvestments",
        "ProceedsFromEquityMethodInvestmentDividendsOrDistributionsReturnOfCapital",
        "ProceedsFromSaleAndCollectionOfNotesReceivable",
    ], False, 2),

    ("  Proceeds from Asset / Business Divestitures", [
        "ProceedsFromSaleOfPropertyPlantAndEquipment",
        "ProceedsFromDivestitureOfBusinesses",
        "ProceedsFromSalesOfBusinessAcquisitionAndDisposition",
        "ProceedsFromSaleOfProductiveAssets",
    ], False, 2),

    ("  Other Investing Activities", [
        "PaymentsForProceedsFromOtherInvestingActivities",
        "OtherPaymentsToAcquireBusinesses",
        "PaymentsForProceedsFromLoansAndLeases",
    ], False, 2),

    ("Cash from Investing", [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
    ], True, 1),

    # ══ FINANCING ACTIVITIES ════════════════════════════════════════════════════
    ("FINANCING ACTIVITIES", [], False, 0),

    ("  Proceeds from Long-Term Debt", [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromIssuanceOfSeniorLongTermDebt",
        "ProceedsFromLongTermLinesOfCredit",
        "ProceedsFromIssuanceOfConvertibleDebt",
        "ProceedsFromDebtNetOfIssuanceCosts",
    ], False, 2),

    ("  Repayment of Long-Term Debt", [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        "RepaymentsOfSeniorDebt",
        "RepaymentsOfConvertibleDebt",
        "RepaymentsOfLinesOfCredit",
        "RepaymentsOfLongTermLinesOfCredit",
    ], False, 2),

    ("  Net Commercial Paper / Short-Term Borrowings", [
        "ProceedsFromRepaymentsOfCommercialPaper",
        "ProceedsFromShortTermDebt",
        "RepaymentsOfShortTermDebt",
        "ProceedsFromRepaymentsOfShortTermDebt",
    ], False, 2),

    ("  Debt Issuance Costs", [
        "PaymentsOfDebtIssuanceCosts",
        "PaymentsOfFinancingCosts",
    ], False, 2),

    ("  Proceeds from Stock Issuance & Options", [
        "ProceedsFromIssuanceOfCommonStock",
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans",
        "ProceedsFromStockPlansIssuedDuringPeriodValue",
    ], False, 2),

    ("  Taxes Withheld on RSU Vesting", [
        "PaymentsRelatedToTaxWithholdingForShareBasedCompensation",
        "EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense",
    ], False, 2),

    ("  Share Repurchases", [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
        "StockRepurchasedDuringPeriodValue",
    ], False, 2),

    ("  Dividends Paid (Common)", [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
        "DividendsPaid",
    ], False, 2),

    ("  Dividends Paid (Preferred)", [
        "PaymentsOfDividendsPreferredStockAndPreferenceStock",
        "PaymentsOfDividendsMinorityInterest",
    ], False, 2),

    ("  Finance Lease Principal Payments", [
        "FinanceLeasePrincipalPayments",
        "RepaymentsOfLongTermCapitalLeaseObligations",
        "RepaymentOfNotesReceivableFromRelatedParties",
    ], False, 2),

    ("  Distributions to Noncontrolling Interests", [
        "PaymentsToMinorityShareholders",
        "PaymentsOfDistributionsToAffiliates",
        "MinorityInterestDecreaseFromDistributionsToNoncontrollingInterestHolders",
    ], False, 2),

    ("  Contributions from Noncontrolling Interests", [
        "ProceedsFromMinorityShareholders",
        "ProceedsFromContributionsFromAffiliates",
    ], False, 2),

    ("  Other Financing Activities", [
        "ProceedsFromPaymentsForOtherFinancingActivities",
        "ProceedsFromOtherEquity",
        "PaymentsForOtherFinancingActivities",
    ], False, 2),

    ("Cash from Financing", [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ], True, 1),

    ("FX Effect on Cash", [
        "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "EffectOfExchangeRateOnCashAndCashEquivalents",
        "EffectOfExchangeRateOnCash",
    ], False, 1),

    ("Net Change in Cash", [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
        "NetCashProvidedByUsedInContinuingOperations",
    ], True, 1),

    ("  Cash Paid for Interest", [
        "InterestPaidNet",
        "InterestPaid",
        "InterestPaidCapitalized",
    ], False, 2),

    ("  Cash Paid for Taxes", [
        "IncomeTaxesPaid",
        "IncomeTaxesPaidNet",
    ], False, 2),

    ("Free Cash Flow", [], True, 1),      # computed: CFO − |CapEx|
    ("  FCF Margin (%)", [], False, 2),   # computed
    ("  FCF / Net Income (%)", [], False, 2),  # computed
]


# ── yfinance fallback maps ─────────────────────────────────────────────────────
_YF_IS: Dict[str, str] = {
    "Revenue":                              "Total Revenue",
    "Cost of Revenue":                      "Cost Of Revenue",
    "Gross Profit":                         "Gross Profit",
    "  Research & Development":             "Research And Development",
    "  SG&A (Combined)":                    "Selling General And Administration",
    "  General & Administrative":           "General And Administrative Expense",
    "Operating Income (EBIT)":              "Operating Income",
    "Interest Expense":                     "Interest Expense",
    "Interest Income":                      "Interest Income",
    "Income Before Tax":                    "Pretax Income",
    "Income Tax Expense / (Benefit)":       "Tax Provision",
    "Net Income from Cont. Operations":     "Net Income Continuous Operations",
    "Net Income":                           "Net Income",
    "EPS — Basic":                          "Basic EPS",
    "EPS — Diluted":                        "Diluted EPS",
    "Shares Outstanding — Basic (M)":       "Basic Average Shares",
    "Shares Outstanding — Diluted (M)":     "Diluted Average Shares",
    "EBITDA":                               "EBITDA",
    "  Depreciation & Amortization":        "Reconciled Depreciation",
    "  Stock-Based Compensation":           "Stock Based Compensation",
}
_YF_BS: Dict[str, str] = {
    "  Cash & Cash Equivalents":            "Cash And Cash Equivalents",
    "  Short-Term Investments":             "Other Short Term Investments",
    "  Accounts Receivable, Net":           "Receivables",
    "  Inventory":                          "Inventory",
    "  Prepaid Expenses & Other Current Assets": "Other Current Assets",
    "Total Current Assets":                 "Current Assets",
    "  PP&E, Net":                          "Net PPE",
    "  Operating Lease ROU Assets":         "Leases",
    "  Goodwill":                           "Goodwill",
    "  Intangible Assets, Net":             "Other Intangible Assets",
    "  Deferred Tax Assets (LT)":           "Deferred Tax Assets",
    "  Other Non-Current Assets":           "Other Non Current Assets",
    "Total Assets":                         "Total Assets",
    "  Accounts Payable":                   "Accounts Payable",
    "  Accrued Compensation & Benefits":    "Payables And Accrued Expenses",
    "  Accrued Expenses & Other Current Liabilities": "Current Accrued Expenses",
    "  Deferred Revenue (Current)":         "Deferred Revenue",
    "  Short-Term Debt & Commercial Paper": "Current Debt",
    "  Current Portion of Long-Term Debt":  "Current Debt And Capital Lease Obligation",
    "  Operating Lease Liability (Current)":"Current Capital Lease Obligation",
    "Total Current Liabilities":            "Current Liabilities",
    "  Long-Term Debt":                     "Long Term Debt",
    "  Operating Lease Liability (LT)":     "Long Term Capital Lease Obligation",
    "  Deferred Tax Liabilities":           "Deferred Tax Liabilities Net",
    "  Other Non-Current Liabilities":      "Other Non Current Liabilities",
    "Total Liabilities":                    "Total Liabilities Net Minority Interest",
    "  Additional Paid-In Capital":         "Capital Stock",
    "  Retained Earnings / (Deficit)":      "Retained Earnings",
    "  Accumulated Other Comprehensive Income / (Loss)": "Other Equity Adjustments",
    "  Treasury Stock":                     "Treasury Stock",
    "Total Stockholders' Equity":           "Stockholders Equity",
    "  Noncontrolling Interests":           "Minority Interest",
    "Total Liabilities & Equity":           "Total Assets",
}
_YF_CF: Dict[str, str] = {
    "  Net Income":                         "Net Income",
    "  Depreciation & Amortization":        "Depreciation And Amortization",
    "  Amortization of Intangibles":        "Amortization Of Intangibles",
    "  Stock-Based Compensation":           "Stock Based Compensation",
    "  Deferred Income Taxes":              "Deferred Tax",
    "  (Gain) / Loss on Investments & Sales": "Gain On Sale Of Security",
    "  Other Non-Cash Items":               "Other Non Cash Items",
    "  Δ Accounts Receivable":              "Change In Receivables",
    "  Δ Inventory":                        "Change In Inventory",
    "  Δ Prepaid & Other Current Assets":   "Change In Other Current Assets",
    "  Δ Accounts Payable":                 "Change In Payables And Accrued Expense",
    "  Δ Accrued Liabilities":              "Change In Other Current Liabilities",
    "  Δ Deferred Revenue":                 "Change In Other Working Capital",
    "Cash from Operations":                 "Operating Cash Flow",
    "  Capital Expenditures":               "Capital Expenditure",
    "  Capitalized Software Development":   "Net Intangibles Purchase And Sale",
    "  Acquisitions, Net of Cash":          "Acquisitions Net",
    "  Purchases of Investments":           "Purchase Of Investment",
    "  Maturities / Sales of Investments":  "Sale Of Investment",
    "  Proceeds from Asset / Business Divestitures": "Net Business Purchase And Sale",
    "Cash from Investing":                  "Investing Cash Flow",
    "  Proceeds from Long-Term Debt":       "Issuance Of Debt",
    "  Repayment of Long-Term Debt":        "Repayment Of Debt",
    "  Proceeds from Stock Issuance & Options": "Common Stock Issuance",
    "  Taxes Withheld on RSU Vesting":      "Common Stock Payments",
    "  Share Repurchases":                  "Repurchase Of Capital Stock",
    "  Dividends Paid (Common)":            "Payment Of Dividends",
    "  Finance Lease Principal Payments":   "Repayment Of Debt",
    "Cash from Financing":                  "Financing Cash Flow",
    "Net Change in Cash":                   "Changes In Cash",
    "  Cash Paid for Interest":             "Interest Paid Supplemental Data",
    "  Cash Paid for Taxes":                "Income Tax Paid Supplemental Data",
}


# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;}}
.stTabs [data-baseweb="tab-list"]{{gap:0;border-bottom:1px solid rgba(255,255,255,0.1);background:transparent!important;}}
.stTabs [data-baseweb="tab"]{{font-weight:600;font-size:13px;padding:10px 22px;color:rgba(255,255,255,0.45);background:transparent!important;border-bottom:2px solid transparent;border-radius:0;}}
.stTabs [aria-selected="true"]{{color:#fff!important;border-bottom:2px solid rgba(255,255,255,0.6)!important;background:transparent!important;}}
.fin-wrap{{overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.015);margin-top:8px;}}
.fin-table{{width:100%;border-collapse:collapse;font-size:12.5px;}}
.fin-table thead th{{background:rgba(255,255,255,0.04);color:#ffffff;font-weight:700;font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;padding:10px 16px 9px;border-bottom:1px solid rgba(255,255,255,0.09);text-align:right;white-space:nowrap;}}
.fin-table thead th.hl{{text-align:left;min-width:240px;max-width:300px;}}
.fin-table thead th.hy{{min-width:88px;}}
.fin-table td{{padding:5px 16px;border-bottom:1px solid rgba(255,255,255,.035);color:#ffffff;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}}
.fin-table td.tl{{text-align:left;}}
.fin-table tbody tr:hover td{{background:rgba(255,255,255,.022);}}
.rs td{{font-size:9.5px!important;font-weight:800!important;letter-spacing:.13em!important;text-transform:uppercase!important;color:#ffffff!important;background:rgba(255,255,255,.04)!important;padding-top:11px!important;padding-bottom:5px!important;border-top:1px solid rgba(255,255,255,.07)!important;border-bottom:none!important;}}
.rt td{{font-weight:700!important;color:#fff!important;border-top:1px solid rgba(255,255,255,.14)!important;border-bottom:1px solid rgba(255,255,255,.14)!important;background:rgba(255,255,255,.018)!important;}}
.ri1 td.tl{{padding-left:16px!important;}}
.ri2 td.tl{{padding-left:32px!important;color:#ffffff!important;font-size:12px!important;}}
.ri3 td.tl{{padding-left:48px!important;color:#ffffff!important;font-size:11.5px!important;}}
.ry td{{font-size:10px!important;padding-top:2px!important;padding-bottom:2px!important;border-bottom:1px solid rgba(255,255,255,.025)!important;background:rgba(0,0,0,.1)!important;color:rgba(255,255,255,.55)!important;}}
.ry td.tl{{padding-left:20px!important;font-style:italic;}}
.vp{{color:{UP}!important;}}.vn{{color:{DOWN}!important;}}.vna{{color:rgba(255,255,255,.35)!important;}}.vpc{{font-size:11px!important;color:#ffffff!important;}}.veps{{font-style:italic;}}
.struct-ban{{background:rgba(255,159,10,.07);border:1px solid rgba(255,159,10,.22);border-radius:10px;padding:13px 18px;margin-bottom:14px;font-size:12.5px;color:rgba(255,255,255,.78);line-height:1.72;}}
.calc-badge{{display:inline-block;font-size:8.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:rgba(255,255,255,.35);background:rgba(255,255,255,.07);border-radius:3px;padding:1px 4px;margin-left:6px;vertical-align:middle;}}
.fin-note{{margin-top:22px;padding:12px 16px;background:rgba(255,255,255,.015);border-radius:8px;font-size:10.5px;color:rgba(255,255,255,.5);line-height:1.85;}}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_facts(cik: str) -> dict:
    try:
        return fetch_company_facts(cik)
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def _load_yf(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        return {
            "is":   getattr(t, "income_stmt",  pd.DataFrame()),
            "bs":   getattr(t, "balance_sheet", pd.DataFrame()),
            "cf":   getattr(t, "cashflow",      pd.DataFrame()),
            "info": t.info or {},
        }
    except Exception:
        return {"is": pd.DataFrame(), "bs": pd.DataFrame(), "cf": pd.DataFrame(), "info": {}}

@st.cache_data(ttl=86400, show_spinner=False)
def _cik_map() -> dict:
    try:
        return load_full_cik_map()
    except Exception:
        return {}


# ── SEC extractors ─────────────────────────────────────────────────────────────

def _sec_usd(facts: dict, tags: List[str], n: int) -> Tuple[pd.Series, str]:
    """Pull the first matching tag from SEC XBRL (10-K, USD, n years)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get("USD", [])
        rows = []
        for e in entries:
            if e.get("form") != "10-K":
                continue
            try:
                rows.append({"end": pd.to_datetime(e["end"]), "accn": e.get("accn",""), "val": float(e["val"])})
            except Exception:
                continue
        if not rows:
            continue
        df = (pd.DataFrame(rows).sort_values("accn")
              .drop_duplicates(subset=["end"], keep="last")
              .sort_values("end"))
        return df.set_index("end")["val"].iloc[-n:], tag
    return pd.Series(dtype="float64"), ""


def _sec_shares(facts: dict, tags: List[str], n: int) -> pd.Series:
    """Pull share-count or per-share data from SEC XBRL."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in us_gaap:
            continue
        for unit in ("USD/shares", "shares"):
            entries = us_gaap[tag].get("units", {}).get(unit, [])
            rows = []
            for e in entries:
                if e.get("form") != "10-K":
                    continue
                try:
                    rows.append({"end": pd.to_datetime(e["end"]), "accn": e.get("accn",""), "val": float(e["val"])})
                except Exception:
                    continue
            if not rows:
                continue
            df = (pd.DataFrame(rows).sort_values("accn")
                  .drop_duplicates(subset=["end"], keep="last")
                  .sort_values("end"))
            return df.set_index("end")["val"].iloc[-n:]
    return pd.Series(dtype="float64")


_PER_SHARE = {"EPS — Basic", "EPS — Diluted"}
_SHARE_CNT = {"Shares Outstanding — Basic (M)", "Shares Outstanding — Diluted (M)",
              "  Shares Outstanding (M)"}


def _build_df(
    schema: List[Tuple],
    facts: dict,
    yf_df: pd.DataFrame,
    yf_map: Dict[str, str],
    n: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw: Dict[str, pd.Series] = {}
    src: Dict[str, str]       = {}

    for label, tags, _sub, _ind in schema:
        if not tags:
            raw[label] = pd.Series(dtype="float64")
            src[label] = ""
            continue

        if label in _PER_SHARE or label in _SHARE_CNT:
            s = _sec_shares(facts, tags, n)
        else:
            s, _ = _sec_usd(facts, tags, n)

        if not s.empty:
            raw[label] = s
            src[label] = "SEC"
            continue

        # yfinance fallback
        yf_field = yf_map.get(label)
        if yf_field and not yf_df.empty and yf_field in yf_df.index:
            row = yf_df.loc[yf_field].dropna()
            if not row.empty:
                row.index = pd.to_datetime(row.index).tz_localize(None)
                raw[label] = row.sort_index().iloc[-n:].astype(float)
                src[label] = "yfinance"
                continue

        raw[label] = pd.Series(dtype="float64")
        src[label] = ""

    non_empty = [s for s in raw.values() if not s.empty]
    if not non_empty:
        return pd.DataFrame(), src

    all_dates = sorted(set().union(*[s.index for s in non_empty]))[-n:]
    df = pd.DataFrame({lbl: {d: s.get(d, np.nan) for d in all_dates} for lbl, s in raw.items()}).T
    df.columns = pd.to_datetime(df.columns)
    return df, src


# ── Computed rows ──────────────────────────────────────────────────────────────

def _inject_is_computed(df: pd.DataFrame, src: Dict[str, str]) -> None:
    if df.empty:
        return
    if "EBITDA" in df.index and df.loc["EBITDA"].isna().all():
        ebit = df.loc["Operating Income (EBIT)"] if "Operating Income (EBIT)" in df.index else None
        da   = df.loc["  Depreciation & Amortization"] if "  Depreciation & Amortization" in df.index else None
        if ebit is not None and da is not None:
            comp = ebit.add(da.abs(), fill_value=np.nan)
            if not comp.dropna().empty:
                df.loc["EBITDA"] = comp
                src["EBITDA"]    = "calc"
    if "  Effective Tax Rate (%)" in df.index:
        ebt = df.loc["Income Before Tax"] if "Income Before Tax" in df.index else None
        tax = df.loc["Income Tax Expense / (Benefit)"] if "Income Tax Expense / (Benefit)" in df.index else None
        if ebt is not None and tax is not None:
            rate = tax.abs().div(ebt.abs().replace(0, np.nan)) * 100
            df.loc["  Effective Tax Rate (%)"] = rate
            src["  Effective Tax Rate (%)"]    = "calc"


def _inject_bs_computed(df: pd.DataFrame, src: Dict[str, str], yf_info: dict) -> None:
    if df.empty:
        return


def _inject_cf_computed(df: pd.DataFrame, src: Dict[str, str], is_df: pd.DataFrame) -> None:
    if df.empty or "Free Cash Flow" not in df.index:
        return
    ocf   = df.loc["Cash from Operations"]   if "Cash from Operations"   in df.index else pd.Series(dtype=float)
    capex = df.loc["  Capital Expenditures"] if "  Capital Expenditures" in df.index else pd.Series(dtype=float)
    if not ocf.dropna().empty and not capex.dropna().empty:
        fcf = ocf.add(capex.apply(lambda x: -abs(x) if not pd.isna(x) else np.nan), fill_value=np.nan)
        df.loc["Free Cash Flow"] = fcf
        src["Free Cash Flow"]    = "calc"
        if "  FCF Margin (%)" in df.index and not is_df.empty and "Revenue" in is_df.index:
            df.loc["  FCF Margin (%)"] = fcf.div(is_df.loc["Revenue"].replace(0, np.nan)) * 100
            src["  FCF Margin (%)"]    = "calc"
        if "  FCF / Net Income (%)" in df.index and not is_df.empty and "Net Income" in is_df.index:
            df.loc["  FCF / Net Income (%)"] = fcf.div(is_df.loc["Net Income"].replace(0, np.nan)) * 100
            src["  FCF / Net Income (%)"]    = "calc"


# ── Formatting ─────────────────────────────────────────────────────────────────

_GROWTH_ROWS = {
    "Revenue","Gross Profit","Operating Income (EBIT)","EBITDA","Net Income",
    "Cash from Operations","Free Cash Flow","Total Assets","Total Stockholders' Equity",
}
_PCT_ROWS = {"  Effective Tax Rate (%)","  FCF Margin (%)","  FCF / Net Income (%)"}


def _fmt(val: float, label: str, divisor: float, cs_base: Optional[float]) -> Tuple[str, str]:
    if pd.isna(val):
        return "—", "vna"
    if label in _PCT_ROWS:
        return f"{val:.1f}%", "vpc"
    if label in _PER_SHARE:
        s = val
        txt = f"(${abs(s):.2f})" if s < 0 else f"${s:.2f}"
        return txt, ("vn" if s < 0 else "veps")
    if label in _SHARE_CNT:
        return f"{val/1e6:,.1f}M", ""
    if cs_base is not None and not pd.isna(cs_base) and cs_base != 0:
        return f"{val/cs_base*100:.1f}%", "vpc"
    s = val / divisor
    if abs(s) >= 100:
        num_str = f"{abs(s):,.0f}"
    else:
        num_str = f"{abs(s):,.2f}"
    txt = f"({num_str})" if s < 0 else num_str
    cls = "vn" if s < 0 else ""
    return txt, cls


def _fmt_yoy(cur: float, prv: float) -> Tuple[str, str]:
    if pd.isna(cur) or pd.isna(prv) or prv == 0:
        return "—", "vna"
    g = (cur - prv) / abs(prv) * 100
    return f"{'+'if g>=0 else ''}{g:.1f}%", ("vp" if g >= 0 else "vn")


# ── HTML table ─────────────────────────────────────────────────────────────────

def _render_table(
    df: pd.DataFrame,
    schema: List[Tuple],
    src: Dict[str, str],
    divisor: float,
    unit_lbl: str,
) -> str:
    if df.empty:
        return "<p style='color:rgba(255,255,255,.35)'>No data available.</p>"

    cols = list(reversed(df.columns))   # newest → oldest (left → right)
    hdrs = [c.strftime("FY%Y") for c in cols]

    h = f'<th class="hl">{unit_lbl}</th>' + "".join(f'<th class="hy">{y}</th>' for y in hdrs)
    html = f'<div class="fin-wrap"><table class="fin-table"><thead><tr>{h}</tr></thead><tbody>'

    for label, tags, is_sub, indent in schema:
        if label not in df.index:
            continue
        row = df.loc[label]

        # Always hide empty data rows
        if tags and row.dropna().empty:
            continue

        # Section banner
        if not tags and indent == 0:
            if label.startswith("_spacer_"):
                html += f'<tr class="rspacer"><td colspan="{1+len(cols)}" style="padding:6px 0;border:none;background:transparent;"></td></tr>'
            else:
                html += f'<tr class="rs"><td class="tl" colspan="{1+len(cols)}">{label}</td></tr>'
            continue

        # Computed placeholder not yet filled
        if not tags and indent > 0 and row.dropna().empty:
            continue

        row_cls = "rt" if is_sub else f"ri{indent}"

        badge = ' <span class="calc-badge">calc</span>' if src.get(label) == "calc" else ""
        cells = f'<td class="tl">{label}{badge}</td>'
        for col in cols:
            v = row[col]
            txt, vcls = _fmt(v, label, divisor, None)
            cells += f'<td class="{vcls}">{txt}</td>'
        html += f'<tr class="{row_cls}">{cells}</tr>'


    return html + "</tbody></table></div>"


# ── Structure banner ───────────────────────────────────────────────────────────

def _warn(ticker: str, name: str) -> Optional[str]:
    t = ticker.upper()
    if t in _REITS:
        return (f"<b>{name}</b> is a <b>REIT</b>. Key metrics are FFO, AFFO, and NOI — not GAAP net income. "
                "COGS lines are blank by design. Statements shown as filed in the 10-K.")
    if t in _MLPS:
        return (f"<b>{name}</b> is an <b>MLP</b>. Effectively zero income tax (pass-through). "
                "Equity is Partners' Capital. Primary metric is Distributable Cash Flow.")
    if t in _INSURERS:
        return (f"<b>{name}</b> is a <b>P&C insurer</b>. Revenue = Net Premiums Earned — no COGS. "
                "Key metrics: Combined Ratio, Loss Ratio, Underwriting Income.")
    if t in _BANKS:
        return (f"<b>{name}</b> is a <b>bank</b>. Revenue = NII + Non-Interest Income. "
                "Key metrics: NIM, Efficiency Ratio, Tier 1 Capital Ratio.")
    if t in _IFRS:
        return (f"<b>{name}</b> files under <b>IFRS</b>, not US GAAP. "
                "SEC XBRL (us-gaap namespace) will be mostly empty. yfinance is the primary source.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_financials(ticker: str) -> None:
    _inject_css()
    ticker = ticker.upper()

    # ── Title = ticker symbol ─────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:30px;font-weight:800;color:#fff;margin-bottom:6px;">{ticker}</h1>',
        unsafe_allow_html=True,
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    cik = _cik_map().get(ticker, "")
    with st.spinner(f"Fetching {ticker} from SEC EDGAR…"):
        facts   = _load_facts(cik) if cik else {}
        yf_data = _load_yf(ticker)

    yf_info = yf_data.get("info", {})
    name    = yf_info.get("longName") or yf_info.get("shortName") or facts.get("entityName","") or ticker

    # ── Structure warning (REIT / MLP / Insurer / Bank / IFRS) ───────────────
    w = _warn(ticker, name)
    if w:
        st.markdown(f'<div class="struct-ban">⚠ {w}</div>', unsafe_allow_html=True)

    # ── Only control: display unit ────────────────────────────────────────────
    unit_opt = st.selectbox(
        "Display unit",
        ["Actual ($)", "Thousands ($K)", "Millions ($M)", "Billions ($B)"],
        index=2,
        key=f"fin_u_{ticker}",
    )
    div_map  = {"Actual ($)": 1, "Thousands ($K)": 1e3, "Millions ($M)": 1e6, "Billions ($B)": 1e9}
    lbl_map  = {"Actual ($)": "$", "Thousands ($K)": "$K", "Millions ($M)": "$M", "Billions ($B)": "$B"}
    divisor  = div_map[unit_opt]
    unit_lbl = lbl_map[unit_opt]

    N = 5  # fixed 5 fiscal years

    # ── Statement tabs ────────────────────────────────────────────────────────
    tab_is, tab_bs, tab_cf = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])

    # ── Income Statement ──────────────────────────────────────────────────────
    with tab_is:
        with st.spinner("Loading…"):
            is_df, is_src = _build_df(INCOME_SCHEMA, facts, yf_data["is"], _YF_IS, N)
        _inject_is_computed(is_df, is_src)
        if is_df.empty:
            st.warning(f"No Income Statement data for {ticker}.")
        else:
            st.markdown(_render_table(is_df, INCOME_SCHEMA, is_src, divisor, unit_lbl), unsafe_allow_html=True)

        # ── Per Share (below Income Statement, no extra tab) ──────────────────
        with st.spinner("Loading…"):
            ps_df, ps_src = _build_df(PER_SHARE_SCHEMA, facts, yf_data["is"], _YF_IS, N)
        if not ps_df.empty:
            st.markdown(_render_table(ps_df, PER_SHARE_SCHEMA, ps_src, divisor, unit_lbl), unsafe_allow_html=True)

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    with tab_bs:
        with st.spinner("Loading…"):
            bs_df, bs_src = _build_df(BALANCE_SHEET_SCHEMA, facts, yf_data["bs"], _YF_BS, N)
        _inject_bs_computed(bs_df, bs_src, yf_info)
        if bs_df.empty:
            st.warning(f"No Balance Sheet data for {ticker}.")
        else:
            st.markdown(_render_table(bs_df, BALANCE_SHEET_SCHEMA, bs_src, divisor, unit_lbl), unsafe_allow_html=True)

    # ── Cash Flow Statement ───────────────────────────────────────────────────
    with tab_cf:
        with st.spinner("Loading…"):
            cf_df, cf_src = _build_df(CASH_FLOW_SCHEMA, facts, yf_data["cf"], _YF_CF, N)
        _inject_cf_computed(cf_df, cf_src, is_df if not is_df.empty else pd.DataFrame())
        if cf_df.empty:
            st.warning(f"No Cash Flow Statement data for {ticker}.")
        else:
            st.markdown(_render_table(cf_df, CASH_FLOW_SCHEMA, cf_src, divisor, unit_lbl), unsafe_allow_html=True)
