# ui/financials.py
"""
Financials Page — SEC XBRL Primary, Exact 10-K Presentation
=============================================================

Architecture
------------
1. PRIMARY: Fetch raw XBRL companyfacts from SEC EDGAR. Extract every
   annual (10-K) filing for each concept. Deduplicate by accession number
   (latest amendment wins). Present statements in the same section / line
   order that companies use in their actual 10-K filings.

2. FALLBACK: yfinance annual DataFrames fill any gap where an XBRL tag
   returns no data (sparse filers, recent IPOs, IFRS filers).

3. COMPUTED: EBITDA = EBIT + D&A when not filed. FCF = CFO − |CapEx|.
   FCF Margin = FCF / Revenue. All computed values are badged visibly.

Design principles
-----------------
- The XBRL tag priority lists mirror the order SEC filers actually use,
  sourced from the EDGAR GAAP taxonomy. More specific / dominant tags first.
- Statements display newest-year-left so analysts scan left-to-right
  for the most recent period, exactly like Bloomberg / CapIQ.
- YoY % growth sub-rows appear below every key subtotal automatically.
- Source provenance (SEC / yfinance / Computed) is shown per row.
- Special-structure banners (REIT / MLP / Insurer / IFRS) inform the user
  when a standard IS/BS/CF layout does not fully apply.
- Unit scaler (Actual / K / M / B) and common-size toggle ship out of the box.
- "Hide empty rows" collapses any all-NaN line to keep the table dense.
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

# ── Design tokens ─────────────────────────────────────────────────────────────
UP      = "#00C805"
DOWN    = "#FF3B30"
BLUE    = "#0A7CFF"
ORANGE  = "#FF9F0A"

# ── Special-structure sets ────────────────────────────────────────────────────
_REITS = {
    "SPG","EXR","VICI","PSA","EQIX","OHI","GLPI","HST","FRT","EGP",
    "FR","NNN","CTRE","AMT","CCI","SBAC","DLR","O","PLD","WPC","STAG",
    "COLD","CUBE","NSA","REXR","TRNO","ELS","UDR","VTR","PEAK","IIPR",
}
_MLPS = {
    "OKE","WES","HESM","AM","EPD","MMP","PAA","TRGP","KMI","ET","ENB",
    "DCP","MPLX","CEQP","BPL",
}
_INSURERS = {
    "PGR","ALL","CINF","ERIE","KNSL","CB","AIG","MET","PRU","TRV",
    "HIG","AFG","RLI","ACGL","RNR",
}
_BANKS = {
    "JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","COF",
    "KEY","RF","FITB","HBAN","ZION","CMA","NTRS","STT","BK",
}
_IFRS_FILERS = {
    "BAM","NVO","ASML","TSM","TM","HMC","SONY","UL","BP","SHEL",
    "RIO","BHP","SAP","BABA","JD","PDD","SE","GRAB",
}

# ══════════════════════════════════════════════════════════════════════════════
# STATEMENT SCHEMAS
# Each row: (display_label, [xbrl_tags_priority_order], is_subtotal, indent_level)
#
# indent_level:
#   0 = section banner (dark band, no values)
#   1 = top-level line item
#   2 = sub-line (greyed, indented)
#   3 = sub-sub-line
#
# Empty tag list  →  either section header (indent=0) or computed row (indent>0).
# is_subtotal     →  bold + separator lines above & below.
# ══════════════════════════════════════════════════════════════════════════════

INCOME_SCHEMA: List[Tuple] = [
    # ── Revenue ───────────────────────────────────────────────────────────────
    ("Revenue",
     ["Revenues",
      "RevenueFromContractWithCustomerExcludingAssessedTax",
      "RevenueFromContractWithCustomerIncludingAssessedTax",
      "SalesRevenueNet",
      "RevenuesNetOfInterestExpense",
      "SalesRevenueGoodsNet"],
     True, 1),
    ("  Product Revenue",
     ["SalesRevenueGoodsNet",
      "RevenueFromContractWithCustomerExcludingAssessedTaxProduct"],
     False, 2),
    ("  Service Revenue",
     ["SalesRevenueServicesNet",
      "RevenueFromContractWithCustomerExcludingAssessedTaxService"],
     False, 2),
    ("  Other Revenue",
     ["OtherRevenues","OtherOperatingIncome"],
     False, 2),

    # ── Cost of Revenue ───────────────────────────────────────────────────────
    ("Cost of Revenue",
     ["CostOfRevenue",
      "CostOfGoodsAndServicesSold",
      "CostOfGoodsSold",
      "CostOfServices"],
     False, 1),
    ("  Cost of Products",
     ["CostOfGoodsSold",
      "CostOfGoodsAndServicesSoldCostOfProducts"],
     False, 2),
    ("  Cost of Services",
     ["CostOfServices",
      "CostOfGoodsAndServicesSoldCostOfServices"],
     False, 2),

    ("Gross Profit",
     ["GrossProfit","GrossProfitLoss"],
     True, 1),

    # ── Operating Expenses ────────────────────────────────────────────────────
    ("  Research & Development",
     ["ResearchAndDevelopmentExpense",
      "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
     False, 2),
    ("  Sales & Marketing",
     ["SellingAndMarketingExpense","MarketingExpense"],
     False, 2),
    ("  General & Administrative",
     ["GeneralAndAdministrativeExpense"],
     False, 2),
    ("  SG&A",
     ["SellingGeneralAndAdministrativeExpense"],
     False, 2),
    ("  Amortization of Intangibles",
     ["AmortizationOfIntangibleAssets",
      "AmortizationOfAcquiredIntangibleAssets"],
     False, 2),
    ("  Restructuring & Impairment",
     ["RestructuringCharges",
      "RestructuringAndRelatedCostIncurredCost",
      "GoodwillImpairmentLoss"],
     False, 2),
    ("  Other Operating Expense / (Income)",
     ["OtherOperatingIncomeExpenseNet",
      "OtherCostAndExpenseOperating"],
     False, 2),
    ("Total Operating Expenses",
     ["OperatingExpenses","CostsAndExpenses"],
     False, 1),

    ("Operating Income (EBIT)",
     ["OperatingIncomeLoss","OperatingIncome"],
     True, 1),

    # ── Below Operating Line ──────────────────────────────────────────────────
    ("  Interest Expense",
     ["InterestExpense","InterestAndDebtExpense","InterestExpenseDebt"],
     False, 2),
    ("  Interest & Other Income",
     ["InterestAndDividendIncomeOperating",
      "InvestmentIncomeInterest",
      "InterestIncomeExpenseNet",
      "NonoperatingIncomeExpense",
      "OtherNonoperatingIncomeExpense"],
     False, 2),
    ("  Equity Method Investments",
     ["IncomeLossFromEquityMethodInvestments"],
     False, 2),

    ("Income Before Tax",
     ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
     True, 1),
    ("  Income Tax Expense / (Benefit)",
     ["IncomeTaxExpenseBenefit"],
     False, 2),
    ("  Effective Tax Rate (%)",       [], False, 2),   # computed

    ("Net Income (Cont. Operations)",
     ["IncomeLossFromContinuingOperations",
      "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"],
     False, 1),
    ("  Disc. Operations",
     ["IncomeLossFromDiscontinuedOperationsNetOfTax"],
     False, 2),

    ("Net Income",
     ["NetIncomeLoss",
      "ProfitLoss",
      "NetIncomeLossAvailableToCommonStockholdersBasic"],
     True, 1),
    ("  Attributable to NCI",
     ["NetIncomeLossAttributableToNoncontrollingInterest"],
     False, 2),
    ("  Attributable to Common",
     ["NetIncomeLossAvailableToCommonStockholdersBasic"],
     False, 2),

    # ── Per Share ─────────────────────────────────────────────────────────────
    ("PER SHARE",                      [], False, 0),
    ("EPS — Basic",                    ["EarningsPerShareBasic"],       False, 1),
    ("EPS — Diluted",                  ["EarningsPerShareDiluted"],     False, 1),
    ("Shares (Basic, M)",              ["WeightedAverageNumberOfSharesOutstandingBasic"],       False, 1),
    ("Shares (Diluted, M)",            ["WeightedAverageNumberOfDilutedSharesOutstanding"],     False, 1),

    # ── Supplemental ─────────────────────────────────────────────────────────
    ("SUPPLEMENTAL",                   [], False, 0),
    ("EBITDA",                         [],   True, 1),   # computed
    ("  D&A",
     ["DepreciationAndAmortization",
      "DepreciationDepletionAndAmortization",
      "Depreciation"],
     False, 2),
    ("  Stock-Based Compensation",
     ["ShareBasedCompensation",
      "AllocatedShareBasedCompensationExpense"],
     False, 2),
]

BALANCE_SHEET_SCHEMA: List[Tuple] = [
    # ══ ASSETS ════════════════════════════════════════════════════════════════
    ("ASSETS",                          [], False, 0),
    ("CURRENT ASSETS",                  [], False, 0),
    ("  Cash & Cash Equivalents",
     ["CashAndCashEquivalentsAtCarryingValue",
      "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
      "Cash"],
     False, 2),
    ("  Short-Term Investments",
     ["ShortTermInvestments",
      "AvailableForSaleSecuritiesCurrent",
      "MarketableSecuritiesCurrent"],
     False, 2),
    ("  Accounts Receivable, Net",
     ["AccountsReceivableNetCurrent",
      "ReceivablesNetCurrent"],
     False, 2),
    ("  Inventory",
     ["InventoryNet",
      "InventoryFinishedGoodsNetOfReserves",
      "InventoryGross"],
     False, 2),
    ("  Prepaid Expenses & Other",
     ["PrepaidExpenseAndOtherAssetsCurrent",
      "PrepaidExpenseCurrent",
      "OtherAssetsCurrent"],
     False, 2),
    ("Total Current Assets",           ["AssetsCurrent"],               True, 1),

    ("NON-CURRENT ASSETS",             [], False, 0),
    ("  PP&E, Net",
     ["PropertyPlantAndEquipmentNet",
      "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"],
     False, 2),
    ("  Operating Lease ROU Assets",   ["OperatingLeaseRightOfUseAsset"], False, 2),
    ("  Goodwill",                     ["Goodwill"],                    False, 2),
    ("  Intangible Assets, Net",
     ["FiniteLivedIntangibleAssetsNet",
      "IntangibleAssetsNetExcludingGoodwill"],
     False, 2),
    ("  Long-Term Investments",
     ["LongTermInvestments",
      "AvailableForSaleSecuritiesNoncurrent",
      "EquityMethodInvestments",
      "MarketableSecuritiesNoncurrent"],
     False, 2),
    ("  Deferred Tax Assets (LT)",
     ["DeferredIncomeTaxAssetsNet"],
     False, 2),
    ("  Other Non-Current Assets",     ["OtherAssetsNoncurrent"],       False, 2),

    ("Total Assets",                   ["Assets"],                      True, 1),

    # ══ LIABILITIES ════════════════════════════════════════════════════════════
    ("LIABILITIES",                    [], False, 0),
    ("CURRENT LIABILITIES",            [], False, 0),
    ("  Accounts Payable",
     ["AccountsPayableCurrent",
      "AccountsPayableAndAccruedLiabilitiesCurrent"],
     False, 2),
    ("  Accrued Liabilities",
     ["AccruedLiabilitiesCurrent",
      "EmployeeRelatedLiabilitiesCurrent",
      "OtherAccruedLiabilitiesCurrent"],
     False, 2),
    ("  Deferred Revenue (Current)",
     ["DeferredRevenueCurrent",
      "ContractWithCustomerLiabilityCurrent"],
     False, 2),
    ("  Short-Term Debt",
     ["ShortTermBorrowings",
      "CommercialPaper",
      "NotesPayableCurrent",
      "DebtCurrent"],
     False, 2),
    ("  Current Portion of LT Debt",
     ["LongTermDebtCurrent",
      "LongTermDebtAndCapitalLeaseObligationsCurrent"],
     False, 2),
    ("  Operating Lease Liability (Current)",
     ["OperatingLeaseLiabilityCurrent"],
     False, 2),
    ("  Other Current Liabilities",    ["OtherLiabilitiesCurrent"],    False, 2),

    ("Total Current Liabilities",      ["LiabilitiesCurrent"],         True, 1),

    ("NON-CURRENT LIABILITIES",        [], False, 0),
    ("  Long-Term Debt",
     ["LongTermDebtNoncurrent",
      "LongTermDebt",
      "LongTermDebtAndCapitalLeaseObligations",
      "LongTermDebtAndFinanceLeaseObligations"],
     False, 2),
    ("  Operating Lease Liability (LT)",
     ["OperatingLeaseLiabilityNoncurrent"],
     False, 2),
    ("  Deferred Revenue (LT)",
     ["DeferredRevenueNoncurrent",
      "ContractWithCustomerLiabilityNoncurrent"],
     False, 2),
    ("  Deferred Tax Liabilities",
     ["DeferredIncomeTaxLiabilitiesNet"],
     False, 2),
    ("  Other Non-Current Liabilities",["OtherLiabilitiesNoncurrent"], False, 2),

    ("Total Liabilities",              ["Liabilities"],                 True, 1),

    # ══ EQUITY ═════════════════════════════════════════════════════════════════
    ("STOCKHOLDERS' EQUITY",           [], False, 0),
    ("  Common Stock & APIC",
     ["AdditionalPaidInCapital",
      "CommonStockAndAdditionalPaidInCapital",
      "AdditionalPaidInCapitalCommonStock"],
     False, 2),
    ("  Retained Earnings (Deficit)",  ["RetainedEarningsAccumulatedDeficit"], False, 2),
    ("  Accumulated OCI",              ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"], False, 2),
    ("  Treasury Stock",
     ["TreasuryStockValue","TreasuryStockCommonValue"],
     False, 2),
    ("Total Stockholders' Equity",
     ["StockholdersEquity",
      "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
     True, 1),
    ("  Non-Controlling Interests",
     ["MinorityInterest","RedeemableNoncontrollingInterestEquityCarryingAmount"],
     False, 2),

    ("Total Liabilities & Equity",     ["LiabilitiesAndStockholdersEquity"], True, 1),

    # ── Supplemental ─────────────────────────────────────────────────────────
    ("SUPPLEMENTAL",                   [], False, 0),
    ("Total Debt",
     ["DebtLongtermAndShorttermCombinedAmount",
      "LongTermDebtAndCapitalLeaseObligations",
      "LongTermDebt"],
     False, 1),
    ("Net Cash & Investments",         [], False, 1),   # computed
    ("Book Value per Share",           [], False, 1),   # computed
]

CASH_FLOW_SCHEMA: List[Tuple] = [
    # ══ OPERATING ══════════════════════════════════════════════════════════════
    ("OPERATING ACTIVITIES",           [], False, 0),
    ("  Net Income",
     ["NetIncomeLoss","ProfitLoss"],
     False, 2),
    ("  Depreciation & Amortization",
     ["DepreciationAndAmortization",
      "DepreciationDepletionAndAmortization",
      "Depreciation"],
     False, 2),
    ("  Stock-Based Compensation",
     ["ShareBasedCompensation",
      "AllocatedShareBasedCompensationExpense"],
     False, 2),
    ("  Deferred Income Taxes",
     ["DeferredIncomeTaxExpenseBenefit",
      "DeferredIncomeTaxesAndTaxCredits"],
     False, 2),
    ("  Amortization of Debt Costs",
     ["AmortizationOfFinancingCostsAndDiscounts",
      "AmortizationOfDebtDiscountPremium"],
     False, 2),
    ("  Other Non-Cash Items",
     ["OtherNoncashIncomeExpense"],
     False, 2),
    ("  Δ Accounts Receivable",
     ["IncreaseDecreaseInAccountsReceivable",
      "IncreaseDecreaseInReceivables"],
     False, 2),
    ("  Δ Inventory",
     ["IncreaseDecreaseInInventories"],
     False, 2),
    ("  Δ Accounts Payable",
     ["IncreaseDecreaseInAccountsPayable",
      "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities"],
     False, 2),
    ("  Δ Deferred Revenue",
     ["IncreaseDecreaseInContractWithCustomerLiability",
      "IncreaseDecreaseInDeferredRevenue"],
     False, 2),
    ("  Δ Other Working Capital",
     ["IncreaseDecreaseInOtherOperatingLiabilities",
      "IncreaseDecreaseInOtherCurrentLiabilities"],
     False, 2),

    ("Cash from Operations",
     ["NetCashProvidedByUsedInOperatingActivities",
      "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
     True, 1),

    # ══ INVESTING ══════════════════════════════════════════════════════════════
    ("INVESTING ACTIVITIES",           [], False, 0),
    ("  Capital Expenditures",
     ["PaymentsToAcquirePropertyPlantAndEquipment",
      "PaymentsToAcquireProductiveAssets",
      "CapitalExpenditures"],
     False, 2),
    ("  Capitalized Software",
     ["PaymentsToDevelopSoftware",
      "CapitalizedComputerSoftwareAdditions"],
     False, 2),
    ("  Acquisitions, Net of Cash",
     ["PaymentsToAcquireBusinessesNetOfCashAcquired",
      "PaymentsToAcquireBusinessesGross"],
     False, 2),
    ("  Purchases of Investments",
     ["PaymentsToAcquireAvailableForSaleSecurities",
      "PaymentsToAcquireInvestments",
      "PaymentsToAcquireShortTermInvestments"],
     False, 2),
    ("  Maturities / Sales of Investments",
     ["ProceedsFromSaleOfAvailableForSaleSecurities",
      "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
      "ProceedsFromSaleMaturityAndCollectionOfInvestments"],
     False, 2),
    ("  Proceeds from Asset Sales",
     ["ProceedsFromSaleOfPropertyPlantAndEquipment",
      "ProceedsFromSalesOfBusinessAcquisitionAndDisposition"],
     False, 2),
    ("  Other Investing Activities",
     ["PaymentsForProceedsFromOtherInvestingActivities"],
     False, 2),

    ("Cash from Investing",
     ["NetCashProvidedByUsedInInvestingActivities",
      "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
     True, 1),

    # ══ FINANCING ══════════════════════════════════════════════════════════════
    ("FINANCING ACTIVITIES",           [], False, 0),
    ("  Proceeds from Debt Issuance",
     ["ProceedsFromIssuanceOfLongTermDebt",
      "ProceedsFromIssuanceOfDebt",
      "ProceedsFromLongTermLinesOfCredit"],
     False, 2),
    ("  Repayment of Debt",
     ["RepaymentsOfLongTermDebt",
      "RepaymentsOfDebt",
      "RepaymentsOfLinesOfCredit"],
     False, 2),
    ("  Proceeds from Stock Issuance",
     ["ProceedsFromIssuanceOfCommonStock",
      "ProceedsFromStockOptionsExercised"],
     False, 2),
    ("  Share Repurchases",
     ["PaymentsForRepurchaseOfCommonStock",
      "PaymentsForRepurchaseOfEquity"],
     False, 2),
    ("  Dividends Paid",
     ["PaymentsOfDividends",
      "PaymentsOfDividendsCommonStock"],
     False, 2),
    ("  Finance Lease Payments",
     ["FinanceLeasePrincipalPayments",
      "RepaymentsOfLongTermCapitalLeaseObligations"],
     False, 2),
    ("  Other Financing Activities",
     ["ProceedsFromPaymentsForOtherFinancingActivities"],
     False, 2),

    ("Cash from Financing",
     ["NetCashProvidedByUsedInFinancingActivities",
      "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
     True, 1),

    # ══ SUMMARY ════════════════════════════════════════════════════════════════
    ("SUMMARY",                        [], False, 0),
    ("  FX Effect on Cash",
     ["EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
      "EffectOfExchangeRateOnCashAndCashEquivalents"],
     False, 2),
    ("Net Change in Cash",
     ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
      "CashAndCashEquivalentsPeriodIncreaseDecrease"],
     True, 1),

    ("SUPPLEMENTAL",                   [], False, 0),
    ("Free Cash Flow",                 [],   True, 1),  # computed: CFO − |CapEx|
    ("  FCF Margin (%)",               [], False, 2),   # computed
    ("  FCF Conversion (%)",           [], False, 2),   # computed: FCF / Net Income
]

# ── yfinance fallback maps ─────────────────────────────────────────────────────
_YF_IS: Dict[str, str] = {
    "Revenue":                          "Total Revenue",
    "Cost of Revenue":                  "Cost Of Revenue",
    "Gross Profit":                     "Gross Profit",
    "  Research & Development":         "Research And Development",
    "  SG&A":                           "Selling General And Administration",
    "Operating Income (EBIT)":          "Operating Income",
    "  Interest Expense":               "Interest Expense",
    "Income Before Tax":                "Pretax Income",
    "  Income Tax Expense / (Benefit)": "Tax Provision",
    "Net Income":                       "Net Income",
    "EPS — Basic":                      "Basic EPS",
    "EPS — Diluted":                    "Diluted EPS",
    "Shares (Basic, M)":                "Basic Average Shares",
    "Shares (Diluted, M)":              "Diluted Average Shares",
    "EBITDA":                           "EBITDA",
    "  D&A":                            "Reconciled Depreciation",
    "  Stock-Based Compensation":       "Stock Based Compensation",
}
_YF_BS: Dict[str, str] = {
    "  Cash & Cash Equivalents":        "Cash And Cash Equivalents",
    "  Short-Term Investments":         "Other Short Term Investments",
    "  Accounts Receivable, Net":       "Receivables",
    "  Inventory":                      "Inventory",
    "  Prepaid Expenses & Other":       "Other Current Assets",
    "Total Current Assets":             "Current Assets",
    "  PP&E, Net":                      "Net PPE",
    "  Goodwill":                       "Goodwill",
    "  Intangible Assets, Net":         "Other Intangible Assets",
    "Total Assets":                     "Total Assets",
    "  Accounts Payable":               "Accounts Payable",
    "  Accrued Liabilities":            "Current Accrued Expenses",
    "  Short-Term Debt":                "Current Debt",
    "Total Current Liabilities":        "Current Liabilities",
    "  Long-Term Debt":                 "Long Term Debt",
    "Total Liabilities":                "Total Liabilities Net Minority Interest",
    "  Retained Earnings (Deficit)":    "Retained Earnings",
    "Total Stockholders' Equity":       "Stockholders Equity",
    "Total Liabilities & Equity":       "Total Assets",
    "Total Debt":                       "Total Debt",
}
_YF_CF: Dict[str, str] = {
    "  Net Income":                     "Net Income",
    "  Depreciation & Amortization":    "Depreciation And Amortization",
    "  Stock-Based Compensation":       "Stock Based Compensation",
    "  Deferred Income Taxes":          "Deferred Tax",
    "  Δ Accounts Receivable":          "Change In Receivables",
    "  Δ Inventory":                    "Change In Inventory",
    "  Δ Accounts Payable":             "Change In Payables And Accrued Expense",
    "  Δ Deferred Revenue":             "Change In Other Current Liabilities",
    "Cash from Operations":             "Operating Cash Flow",
    "  Capital Expenditures":           "Capital Expenditure",
    "  Acquisitions, Net of Cash":      "Acquisitions Net",
    "  Purchases of Investments":       "Purchase Of Investment",
    "  Maturities / Sales of Investments": "Sale Of Investment",
    "Cash from Investing":              "Investing Cash Flow",
    "  Proceeds from Debt Issuance":    "Issuance Of Debt",
    "  Repayment of Debt":              "Repayment Of Debt",
    "  Share Repurchases":              "Repurchase Of Capital Stock",
    "  Dividends Paid":                 "Payment Of Dividends",
    "Cash from Financing":              "Financing Cash Flow",
    "Net Change in Cash":               "Changes In Cash",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600; font-size: 13px; padding: 10px 22px;
    color: rgba(255,255,255,0.45);
    background: transparent !important;
    border-bottom: 2px solid transparent;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: #ffffff !important;
    border-bottom: 2px solid {BLUE} !important;
    background: transparent !important;
}}
.fin-wrap {{
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.015);
    margin-top: 8px;
}}
.fin-table {{
    width: 100%; border-collapse: collapse;
    font-size: 12.5px;
}}
.fin-table thead th {{
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.4);
    font-weight: 700; font-size: 10.5px;
    letter-spacing: 0.07em; text-transform: uppercase;
    padding: 10px 16px 9px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.09);
    text-align: right; white-space: nowrap;
}}
.fin-table thead th.hdr-lbl {{
    text-align: left; min-width: 230px; max-width: 290px;
}}
.fin-table thead th.hdr-yr {{ min-width: 88px; }}
.fin-table td {{
    padding: 5px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    color: rgba(255,255,255,0.85);
    text-align: right;
    font-variant-numeric: tabular-nums; white-space: nowrap;
}}
.fin-table td.td-lbl {{ text-align: left; }}
.fin-table tbody tr:hover td {{
    background: rgba(255,255,255,0.022);
}}

/* Section banner rows */
.row-sec td {{
    font-size: 9.5px !important; font-weight: 800 !important;
    letter-spacing: 0.13em !important; text-transform: uppercase !important;
    color: rgba(255,255,255,0.3) !important;
    background: rgba(10,124,255,0.055) !important;
    padding-top: 11px !important; padding-bottom: 5px !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    border-bottom: none !important;
}}

/* Subtotal rows */
.row-sub td {{
    font-weight: 700 !important; color: #ffffff !important;
    border-top: 1px solid rgba(255,255,255,0.14) !important;
    border-bottom: 1px solid rgba(255,255,255,0.14) !important;
    background: rgba(255,255,255,0.018) !important;
}}

/* Indent levels */
.row-i1 td.td-lbl {{ padding-left: 16px !important; }}
.row-i2 td.td-lbl {{
    padding-left: 32px !important;
    color: rgba(255,255,255,0.58) !important;
    font-size: 12px !important;
}}
.row-i3 td.td-lbl {{
    padding-left: 48px !important;
    color: rgba(255,255,255,0.42) !important;
    font-size: 11.5px !important;
}}

/* YoY growth rows */
.row-yoy td {{
    font-size: 10px !important;
    padding-top: 2px !important; padding-bottom: 2px !important;
    border-bottom: 1px solid rgba(255,255,255,0.025) !important;
    background: rgba(0,0,0,0.1) !important;
    color: rgba(255,255,255,0.38) !important;
}}
.row-yoy td.td-lbl {{
    padding-left: 20px !important;
    font-style: italic;
}}

/* Value states */
.vp  {{ color: {UP} !important; }}
.vn  {{ color: {DOWN} !important; }}
.vna {{ color: rgba(255,255,255,0.18) !important; }}
.vpc {{ font-size: 11px !important; color: rgba(255,255,255,0.45) !important; }}
.veps {{ font-style: italic; }}

/* Source badges */
.sb {{
    display: inline-block; font-size: 8.5px; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
    padding: 1px 5px; border-radius: 3px; margin-left: 5px;
    vertical-align: middle; opacity: 0.65;
}}
.sb-sec  {{ background: rgba(10,124,255,0.2);  color: {BLUE}; }}
.sb-yf   {{ background: rgba(255,159,10,0.2);  color: {ORANGE}; }}
.sb-calc {{ background: rgba(0,200,5,0.12);    color: {UP}; }}

/* Special-structure banner */
.struct-ban {{
    background: rgba(255,159,10,0.07);
    border: 1px solid rgba(255,159,10,0.22);
    border-radius: 10px; padding: 13px 18px; margin-bottom: 14px;
    font-size: 12.5px; color: rgba(255,255,255,0.78); line-height: 1.72;
}}
.hist-pill {{
    display: inline-block; font-size: 11px; font-weight: 700;
    padding: 3px 13px; border-radius: 20px; margin-bottom: 10px;
    border: 1px solid currentColor;
}}
.fin-footnote {{
    margin-top: 22px; padding: 12px 16px;
    background: rgba(255,255,255,0.015);
    border-radius: 8px; font-size: 10.5px;
    color: rgba(255,255,255,0.28); line-height: 1.85;
}}
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
        return {"is": pd.DataFrame(), "bs": pd.DataFrame(),
                "cf": pd.DataFrame(), "info": {}}


@st.cache_data(ttl=86400, show_spinner=False)
def _cik_map() -> dict:
    try:
        return load_full_cik_map()
    except Exception:
        return {}


# ── SEC XBRL extractors ───────────────────────────────────────────────────────

def _sec_annual(
    facts: dict,
    tags: List[str],
    unit: str = "USD",
    n: int = 10,
) -> Tuple[pd.Series, str]:
    """
    Pull annual (10-K only) series from raw XBRL facts.
    Returns (series indexed by fy-end date, tag_used).
    Latest-filed accession wins per fiscal-year-end date.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get(unit, [])
        rows = []
        for e in entries:
            if e.get("form") != "10-K":
                continue
            end_str = e.get("end")
            if not end_str:
                continue
            try:
                rows.append({
                    "end":  pd.to_datetime(end_str),
                    "accn": e.get("accn", ""),
                    "val":  float(e["val"]),
                })
            except Exception:
                continue
        if not rows:
            continue
        df = (
            pd.DataFrame(rows)
            .sort_values("accn")
            .drop_duplicates(subset=["end"], keep="last")
            .sort_values("end")
        )
        return df.set_index("end")["val"].iloc[-n:], tag
    return pd.Series(dtype="float64"), ""


def _sec_pershare(
    facts: dict,
    tags: List[str],
    n: int = 10,
) -> pd.Series:
    """Extract per-share (USD/shares) or share-count (shares) annual series."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in us_gaap:
            continue
        for unit_key in ("USD/shares", "shares"):
            entries = us_gaap[tag].get("units", {}).get(unit_key, [])
            rows = []
            for e in entries:
                if e.get("form") != "10-K":
                    continue
                end_str = e.get("end")
                if not end_str:
                    continue
                try:
                    rows.append({
                        "end":  pd.to_datetime(end_str),
                        "accn": e.get("accn", ""),
                        "val":  float(e["val"]),
                    })
                except Exception:
                    continue
            if not rows:
                continue
            df = (
                pd.DataFrame(rows)
                .sort_values("accn")
                .drop_duplicates(subset=["end"], keep="last")
                .sort_values("end")
            )
            return df.set_index("end")["val"].iloc[-n:]
    return pd.Series(dtype="float64")


# ── Statement DataFrame builder ───────────────────────────────────────────────

_PER_SHARE = {"EPS — Basic", "EPS — Diluted"}
_SHARE_CNT = {"Shares (Basic, M)", "Shares (Diluted, M)"}


def _build_df(
    schema: List[Tuple],
    facts: dict,
    yf_df: pd.DataFrame,
    yf_map: Dict[str, str],
    n: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Build wide DataFrame (rows=labels, cols=fy-end dates) + sources dict.
    Column order: oldest-first (caller can reverse for display).
    """
    raw: Dict[str, pd.Series] = {}
    sources: Dict[str, str]   = {}

    for label, tags, _sub, _ind in schema:
        if not tags:
            raw[label]     = pd.Series(dtype="float64")
            sources[label] = ""
            continue

        # ── Per-share / share-count rows ──────────────────────────────────────
        if label in _PER_SHARE or label in _SHARE_CNT:
            s = _sec_pershare(facts, tags, n)
            if not s.empty:
                raw[label]     = s
                sources[label] = "SEC"
                continue
        else:
            # ── Standard USD rows ─────────────────────────────────────────────
            s, _used = _sec_annual(facts, tags, n=n)
            if not s.empty:
                raw[label]     = s
                sources[label] = "SEC"
                continue

        # ── yfinance fallback ─────────────────────────────────────────────────
        yf_field = yf_map.get(label)
        if yf_field and not yf_df.empty and yf_field in yf_df.index:
            row = yf_df.loc[yf_field].dropna()
            if not row.empty:
                row.index = pd.to_datetime(row.index).tz_localize(None)
                row = row.sort_index().iloc[-n:].astype(float)
                raw[label]     = row
                sources[label] = "yfinance"
                continue

        raw[label]     = pd.Series(dtype="float64")
        sources[label] = ""

    # ── Align all series to a common column set ───────────────────────────────
    non_empty = [s for s in raw.values() if not s.empty]
    if not non_empty:
        return pd.DataFrame(), sources

    all_dates = sorted(set().union(*[s.index for s in non_empty]))
    all_dates = all_dates[-n:]

    out = {
        label: {d: s.get(d, np.nan) for d in all_dates}
        for label, s in raw.items()
    }
    df = pd.DataFrame(out).T
    df.columns = pd.to_datetime(df.columns)
    return df, sources


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTED ROW INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def _inject_is_computed(df: pd.DataFrame, src: Dict[str, str]) -> None:
    if df.empty:
        return
    # EBITDA = EBIT + D&A
    if "EBITDA" in df.index and df.loc["EBITDA"].isna().all():
        ebit = df.loc["Operating Income (EBIT)"] if "Operating Income (EBIT)" in df.index else None
        da   = df.loc["  D&A"]                   if "  D&A" in df.index else None
        if ebit is not None and da is not None:
            comp = ebit.add(da.abs(), fill_value=np.nan)
            if not comp.dropna().empty:
                df.loc["EBITDA"] = comp
                src["EBITDA"]    = "calc"
    # Effective Tax Rate
    if "  Effective Tax Rate (%)" in df.index:
        ebt = df.loc["Income Before Tax"]                   if "Income Before Tax"                   in df.index else None
        tax = df.loc["  Income Tax Expense / (Benefit)"]   if "  Income Tax Expense / (Benefit)"   in df.index else None
        if ebt is not None and tax is not None:
            rate = tax.abs().div(ebt.abs().replace(0, np.nan)) * 100
            df.loc["  Effective Tax Rate (%)"] = rate
            src["  Effective Tax Rate (%)"]    = "calc"


def _inject_bs_computed(
    df: pd.DataFrame,
    src: Dict[str, str],
    yf_info: dict,
) -> None:
    if df.empty:
        return
    # Net Cash = Cash + ST Inv − Total Debt
    if "Net Cash & Investments" in df.index:
        cash  = df.loc["  Cash & Cash Equivalents"] if "  Cash & Cash Equivalents" in df.index else pd.Series(dtype=float)
        stinv = df.loc["  Short-Term Investments"]  if "  Short-Term Investments"  in df.index else pd.Series(dtype=float)
        debt  = df.loc["Total Debt"]                if "Total Debt"                in df.index else pd.Series(dtype=float)
        if not cash.dropna().empty and not debt.dropna().empty:
            nc = cash.add(stinv.fillna(0), fill_value=0).sub(debt.abs().fillna(0), fill_value=0)
            df.loc["Net Cash & Investments"] = nc
            src["Net Cash & Investments"]    = "calc"
    # Book Value per Share
    if "Book Value per Share" in df.index:
        eq  = df.loc["Total Stockholders' Equity"] if "Total Stockholders' Equity" in df.index else pd.Series(dtype=float)
        shs = yf_info.get("sharesOutstanding", np.nan)
        if not eq.dropna().empty and not pd.isna(shs) and shs > 0:
            df.loc["Book Value per Share"] = eq.div(shs)
            src["Book Value per Share"]    = "calc"


def _inject_cf_computed(
    df: pd.DataFrame,
    src: Dict[str, str],
    is_df: pd.DataFrame,
) -> None:
    if df.empty or "Free Cash Flow" not in df.index:
        return
    ocf   = df.loc["Cash from Operations"]   if "Cash from Operations"   in df.index else pd.Series(dtype=float)
    capex = df.loc["  Capital Expenditures"] if "  Capital Expenditures" in df.index else pd.Series(dtype=float)
    if not ocf.dropna().empty and not capex.dropna().empty:
        fcf = ocf.add(capex.apply(lambda x: -abs(x) if not pd.isna(x) else np.nan), fill_value=np.nan)
        df.loc["Free Cash Flow"] = fcf
        src["Free Cash Flow"]    = "calc"
        # FCF Margin
        if "  FCF Margin (%)" in df.index and not is_df.empty and "Revenue" in is_df.index:
            rev = is_df.loc["Revenue"]
            df.loc["  FCF Margin (%)"] = fcf.div(rev.replace(0, np.nan)) * 100
            src["  FCF Margin (%)"]    = "calc"
        # FCF Conversion
        if "  FCF Conversion (%)" in df.index and not is_df.empty and "Net Income" in is_df.index:
            ni = is_df.loc["Net Income"]
            df.loc["  FCF Conversion (%)"] = fcf.div(ni.replace(0, np.nan)) * 100
            src["  FCF Conversion (%)"]    = "calc"


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

_GROWTH_AFTER = {
    "Revenue", "Gross Profit", "Operating Income (EBIT)", "EBITDA",
    "Net Income", "Cash from Operations", "Free Cash Flow",
    "Total Assets", "Total Stockholders' Equity", "Total Liabilities & Equity",
}
_PCT_ROWS = {"  Effective Tax Rate (%)", "  FCF Margin (%)", "  FCF Conversion (%)"}
_NEUTRAL  = {
    "  Δ Accounts Receivable", "  Δ Inventory", "  Δ Accounts Payable",
    "  Δ Deferred Revenue", "  Δ Other Working Capital",
    "Net Cash & Investments", "Book Value per Share",
    "  Effective Tax Rate (%)", "  FCF Margin (%)", "  FCF Conversion (%)",
}


def _fmt_val(
    val: float,
    label: str,
    divisor: float,
    cs_base: Optional[float],
) -> Tuple[str, str]:
    """Return (display_text, css_class)."""
    if pd.isna(val):
        return "—", "vna"

    if label in _PCT_ROWS:
        return f"{val:.1f}%", "vpc"
    if label in _PER_SHARE:
        return f"${val:.2f}", "veps"
    if label in _SHARE_CNT:
        return f"{val/1e6:,.1f}M", ""

    if cs_base is not None and not pd.isna(cs_base) and cs_base != 0:
        return f"{val/cs_base*100:.1f}%", "vpc"

    s = val / divisor
    if abs(s) >= 10_000:
        txt = f"{s:,.0f}"
    elif abs(s) >= 1_000:
        txt = f"{s:,.0f}"
    elif abs(s) >= 100:
        txt = f"{s:,.1f}"
    else:
        txt = f"{s:,.2f}"

    cls = "" if label in _NEUTRAL else ("vn" if val < 0 else "")
    return txt, cls


def _fmt_growth(cur: float, prv: float) -> Tuple[str, str]:
    if pd.isna(cur) or pd.isna(prv) or prv == 0:
        return "—", "vna"
    g   = (cur - prv) / abs(prv) * 100
    sgn = "+" if g >= 0 else ""
    return f"{sgn}{g:.1f}%", ("vp" if g >= 0 else "vn")


# ══════════════════════════════════════════════════════════════════════════════
# HTML TABLE RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_table(
    df: pd.DataFrame,
    schema: List[Tuple],
    src: Dict[str, str],
    divisor: float,
    unit_lbl: str,
    common_size: bool,
    cs_base_row: str,
    hide_empty: bool,
) -> str:
    if df.empty:
        return "<p style='color:rgba(255,255,255,0.35);'>No data available.</p>"

    # Newest year on the left
    cols = list(reversed(list(df.columns)))
    yr_hdrs = [c.strftime("FY%Y") for c in cols]

    # Header
    hdr = f'<th class="hdr-lbl">{unit_lbl}</th>'
    for yr in yr_hdrs:
        hdr += f'<th class="hdr-yr">{yr}</th>'
    html = f'<div class="fin-wrap"><table class="fin-table"><thead><tr>{hdr}</tr></thead><tbody>'

    cs_base_series = (
        df.loc[cs_base_row]
        if common_size and cs_base_row and cs_base_row in df.index
        else None
    )

    for label, tags, is_sub, indent in schema:
        if label not in df.index:
            continue

        row_vals = df.loc[label]

        # Hide empty data rows
        if hide_empty and tags and row_vals.dropna().empty:
            continue

        # ── Section banner ────────────────────────────────────────────────────
        if not tags and indent == 0:
            html += (
                f'<tr class="row-sec">'
                f'<td class="td-lbl" colspan="{1+len(cols)}">'
                f'{label}</td></tr>'
            )
            continue

        # ── Computed placeholder with no data yet ─────────────────────────────
        if not tags and indent > 0 and row_vals.dropna().empty:
            continue

        # Source badge
        s = src.get(label, "")
        badge = (
            '<span class="sb sb-sec">SEC</span>'  if s == "SEC"      else
            '<span class="sb sb-yf">YF</span>'    if s == "yfinance" else
            '<span class="sb sb-calc">Calc</span>'if s == "calc"     else
            ""
        )

        # Row class
        if is_sub:
            row_cls = "row-sub"
        elif indent == 2:
            row_cls = "row-i2"
        elif indent == 3:
            row_cls = "row-i3"
        else:
            row_cls = "row-i1"

        # Label cell
        cells = f'<td class="td-lbl">{label}{badge}</td>'

        # Value cells
        for col in cols:
            v  = row_vals[col]
            cb = cs_base_series[col] if cs_base_series is not None else None
            txt, vcls = _fmt_val(v, label, divisor, cb)
            cells += f'<td class="{vcls}">{txt}</td>'

        html += f'<tr class="{row_cls}">{cells}</tr>'

        # YoY growth sub-row for key subtotals
        if label in _GROWTH_AFTER and len(cols) >= 2:
            gcells = '<td class="td-lbl" style="padding-left:20px;font-style:italic;font-size:10px;">YoY</td>'
            for i, col in enumerate(cols):
                if i == len(cols) - 1:
                    gcells += '<td class="vna">—</td>'
                else:
                    prv = cols[i + 1]
                    gtxt, gcls = _fmt_growth(row_vals[col], row_vals[prv])
                    gcells += f'<td class="{gcls}" style="font-size:10px;">{gtxt}</td>'
            html += f'<tr class="row-yoy">{gcells}</tr>'

    html += '</tbody></table></div>'
    return html


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURE WARNING
# ══════════════════════════════════════════════════════════════════════════════

def _structure_warning(ticker: str, name: str) -> Optional[str]:
    t = ticker.upper()
    if t in _REITS:
        return (
            f"<b>{name}</b> is a <b>Real Estate Investment Trust (REIT)</b>. "
            "The GAAP statements below follow the exact as-filed 10-K presentation. "
            "Key REIT metrics — FFO, AFFO, and NOI — are non-GAAP and not shown here. "
            "Cost of goods sold will appear blank as REITs do not report COGS."
        )
    if t in _MLPS:
        return (
            f"<b>{name}</b> is a <b>Master Limited Partnership (MLP)</b>. "
            "Income tax is near-zero (pass-through entity). Equity is reported as "
            "<i>Partners' Capital</i>. The primary economic metric is "
            "<b>Distributable Cash Flow (DCF)</b>, not GAAP net income."
        )
    if t in _INSURERS:
        return (
            f"<b>{name}</b> is a <b>P&C insurer</b>. Revenue = Net Premiums Earned — "
            "there is no traditional COGS line. Key metrics (Combined Ratio, Loss Ratio, "
            "Underwriting Income) are not captured in the standard layout."
        )
    if t in _BANKS:
        return (
            f"<b>{name}</b> is a <b>bank / financial institution</b>. "
            "Revenue = Net Interest Income + Non-Interest Income. Key metrics "
            "(NIM, Efficiency Ratio, Tier 1 Capital) are not shown in the standard layout."
        )
    if t in _IFRS_FILERS:
        return (
            f"<b>{name}</b> files under <b>IFRS</b>, not US GAAP. "
            "SEC XBRL tags (us-gaap namespace) will largely return no data. "
            "yfinance is the primary source for this company. Some lines may be blank."
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_financials(ticker: str) -> None:
    _inject_css()
    ticker = ticker.upper()

    # ── Page title ────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:30px;font-weight:800;color:#ffffff;margin-bottom:6px;">'
        'Financials</h1>',
        unsafe_allow_html=True,
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    cik = _cik_map().get(ticker, "")
    with st.spinner(f"Fetching {ticker} from SEC EDGAR…"):
        facts   = _load_facts(cik) if cik else {}
        yf_data = _load_yf(ticker)

    yf_info = yf_data.get("info", {})
    name = (
        yf_info.get("longName")
        or yf_info.get("shortName")
        or facts.get("entityName", "")
        or ticker
    )
    currency = yf_info.get("financialCurrency", "USD")
    has_sec  = bool(facts.get("facts", {}).get("us-gaap"))
    has_yf   = any(not v.empty for v in [yf_data["is"], yf_data["bs"], yf_data["cf"]])

    # ── Header row ────────────────────────────────────────────────────────────
    badges = ""
    if has_sec:
        badges += f'<span class="sb sb-sec" style="font-size:10px;padding:3px 10px;opacity:1;">SEC XBRL — Primary</span> '
    if has_yf:
        badges += f'<span class="sb sb-yf" style="font-size:10px;padding:3px 10px;opacity:1;">yfinance — Fallback</span>'
    if not has_sec and not has_yf:
        badges = '<span style="color:#FF3B30;font-size:12px;">⚠ No data sources available</span>'

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap;">'
        f'<span style="font-size:20px;font-weight:700;color:#fff;">{name}</span>'
        f'<span style="color:rgba(255,255,255,0.25);">·</span>'
        f'<span style="color:rgba(255,255,255,0.42);font-size:13px;">{ticker}</span>'
        f'<span style="color:rgba(255,255,255,0.25);">·</span>'
        f'<span style="color:rgba(255,255,255,0.42);font-size:13px;">{currency}</span>'
        f'<span style="color:rgba(255,255,255,0.25);">·</span>'
        f'{badges}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Structure banner ──────────────────────────────────────────────────────
    warn = _structure_warning(ticker, name)
    if warn:
        st.markdown(f'<div class="struct-ban">⚠️  {warn}</div>', unsafe_allow_html=True)

    # ── Controls bar ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1.6, 1.6, 1.0, 1.0])
    with c1:
        n_yrs = st.select_slider(
            "Years of history",
            options=[5, 6, 7, 8, 9, 10],
            value=10,
            key=f"fin_n_{ticker}",
        )
    with c2:
        unit_opt = st.selectbox(
            "Display unit",
            ["Actual ($)", "Thousands ($K)", "Millions ($M)", "Billions ($B)"],
            index=2,
            key=f"fin_u_{ticker}",
        )
    with c3:
        cs_on = st.toggle("Common size", value=False, key=f"fin_cs_{ticker}")
    with c4:
        hide_e = st.toggle("Hide empty rows", value=True, key=f"fin_he_{ticker}")

    div_map  = {"Actual ($)": 1, "Thousands ($K)": 1e3,
                "Millions ($M)": 1e6, "Billions ($B)": 1e9}
    lbl_map  = {"Actual ($)": "$", "Thousands ($K)": "$K",
                "Millions ($M)": "$M", "Billions ($B)": "$B"}
    divisor  = div_map[unit_opt]
    unit_lbl = lbl_map[unit_opt]

    # ── Statement tabs ────────────────────────────────────────────────────────
    tab_is, tab_bs, tab_cf = st.tabs([
        "📋  Income Statement",
        "🏦  Balance Sheet",
        "💵  Cash Flow Statement",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # INCOME STATEMENT
    # ════════════════════════════════════════════════════════════════════════════
    with tab_is:
        with st.spinner("Pulling from SEC XBRL…"):
            is_df, is_src = _build_df(INCOME_SCHEMA, facts, yf_data["is"], _YF_IS, n_yrs)
        _inject_is_computed(is_df, is_src)

        if is_df.empty:
            st.warning(f"No Income Statement data available for {ticker}.")
        else:
            n_av = int(is_df.loc["Revenue"].dropna().shape[0]) if "Revenue" in is_df.index else len(is_df.columns)
            pc   = UP if n_av >= 8 else (ORANGE if n_av >= 5 else DOWN)
            st.markdown(
                f'<span class="hist-pill" style="color:{pc};">● {n_av} fiscal years</span>',
                unsafe_allow_html=True,
            )
            cs_row = "Revenue" if cs_on else ""
            st.markdown(
                _render_table(is_df, INCOME_SCHEMA, is_src, divisor, unit_lbl, cs_on, cs_row, hide_e),
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════════
    # BALANCE SHEET
    # ════════════════════════════════════════════════════════════════════════════
    with tab_bs:
        with st.spinner("Pulling from SEC XBRL…"):
            bs_df, bs_src = _build_df(BALANCE_SHEET_SCHEMA, facts, yf_data["bs"], _YF_BS, n_yrs)
        _inject_bs_computed(bs_df, bs_src, yf_info)

        if bs_df.empty:
            st.warning(f"No Balance Sheet data available for {ticker}.")
        else:
            n_av = int(bs_df.loc["Total Assets"].dropna().shape[0]) if "Total Assets" in bs_df.index else len(bs_df.columns)
            pc   = UP if n_av >= 8 else (ORANGE if n_av >= 5 else DOWN)
            st.markdown(
                f'<span class="hist-pill" style="color:{pc};">● {n_av} fiscal years</span>',
                unsafe_allow_html=True,
            )
            cs_row = "Total Assets" if cs_on else ""
            st.markdown(
                _render_table(bs_df, BALANCE_SHEET_SCHEMA, bs_src, divisor, unit_lbl, cs_on, cs_row, hide_e),
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════════
    # CASH FLOW STATEMENT
    # ════════════════════════════════════════════════════════════════════════════
    with tab_cf:
        with st.spinner("Pulling from SEC XBRL…"):
            cf_df, cf_src = _build_df(CASH_FLOW_SCHEMA, facts, yf_data["cf"], _YF_CF, n_yrs)
        _inject_cf_computed(cf_df, cf_src, is_df if not is_df.empty else pd.DataFrame())

        if cf_df.empty:
            st.warning(f"No Cash Flow Statement data available for {ticker}.")
        else:
            n_av = int(cf_df.loc["Cash from Operations"].dropna().shape[0]) if "Cash from Operations" in cf_df.index else len(cf_df.columns)
            pc   = UP if n_av >= 8 else (ORANGE if n_av >= 5 else DOWN)
            st.markdown(
                f'<span class="hist-pill" style="color:{pc};">● {n_av} fiscal years</span>',
                unsafe_allow_html=True,
            )
            cs_row = "Cash from Operations" if cs_on else ""
            st.markdown(
                _render_table(cf_df, CASH_FLOW_SCHEMA, cf_src, divisor, unit_lbl, cs_on, cs_row, hide_e),
                unsafe_allow_html=True,
            )

    # ── Footnotes ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="fin-footnote">'
        '<b style="color:rgba(255,255,255,0.45);">Data sources & methodology</b><br>'
        '<span class="sb sb-sec">SEC</span> '
        'Sourced directly from SEC EDGAR XBRL companyfacts endpoint (10-K annual filings only). '
        'For each concept, the latest-filed accession number per fiscal-year-end wins '
        '(picks up amendments and restatements automatically). '
        'Tags are queried from the us-gaap XBRL namespace in priority order as filed.<br>'
        '<span class="sb sb-yf">YF</span> '
        'Sourced from yfinance annual statements — used only when XBRL returns no data for that row.<br>'
        '<span class="sb sb-calc">Calc</span> '
        'Computed: EBITDA = EBIT + D&amp;A · FCF = CFO − |CapEx| · '
        'FCF Margin = FCF ÷ Revenue · FCF Conversion = FCF ÷ Net Income · '
        'Net Cash = Cash + ST Investments − Total Debt · '
        'Effective Tax Rate = Tax Expense ÷ Pre-tax Income.<br>'
        'Columns display fiscal-year-end dates, newest on the left. '
        'YoY growth rows appear beneath key subtotals. '
        'All figures in reported currency. '
        'Negative values shown in red only for P&amp;L losses; '
        'financing outflows are presented as filed (may be negative).'
        '</div>',
        unsafe_allow_html=True,
    )
