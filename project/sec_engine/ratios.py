# sec_engine/ratios.py
"""
Financial Ratios Analysis Display
Comprehensive ratio analysis organized into logical sections
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def format_ratio(value: float, decimal_places: int = 2, suffix: str = "") -> str:
    """Format a ratio value for display"""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    return f"{value:.{decimal_places}f}{suffix}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format a percentage value"""
    return format_ratio(value, decimal_places, "%")

def format_multiple(value: float, decimal_places: int = 2) -> str:
    """Format a multiple/ratio (e.g., 2.5x)"""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    return f"{value:.{decimal_places}f}x"

def build_ratios_display(summary: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a comprehensive ratios display DataFrame from company summary.
    
    Args:
        summary: Dictionary from build_company_summary()
    
    Returns:
        DataFrame with all ratios organized by category
    """
    
    ratios_data = []
    
    # ========== SECTION: PROFITABILITY RATIOS ==========
    ratios_data.append({"Category": "PROFITABILITY RATIOS", "Metric": "", "Value": ""})
    ratios_data.append({
        "Category": "Profitability",
        "Metric": "Return on Assets (ROA)",
        "Value": format_percentage(summary.get("ROA %"))
    })
    ratios_data.append({
        "Category": "Profitability",
        "Metric": "Return on Invested Capital (ROIC)",
        "Value": format_percentage(summary.get("ROIC %"))
    })
    ratios_data.append({
        "Category": "Profitability",
        "Metric": "Return on Equity (ROE)",
        "Value": format_percentage(summary.get("ROE %"))
    })
    ratios_data.append({
        "Category": "Profitability",
        "Metric": "Return on Capital Employed (RCE)",
        "Value": format_percentage(summary.get("RCE %"))
    })
    
    # ========== SECTION: MARGIN ANALYSIS ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "MARGIN ANALYSIS", "Metric": "", "Value": ""})
    ratios_data.append({
        "Category": "Margins",
        "Metric": "Gross Margin",
        "Value": format_percentage(summary.get("Gross Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "SG&A Margin",
        "Value": format_percentage(summary.get("SG&A Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "R&D Margin",
        "Value": format_percentage(summary.get("R&D Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "EBITDA Margin",
        "Value": format_percentage(summary.get("EBITDA Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "EBIT Margin",
        "Value": format_percentage(summary.get("EBIT Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "Net Income Margin",
        "Value": format_percentage(summary.get("Net Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "Levered Free Cash Flow Margin",
        "Value": format_percentage(summary.get("LFCF Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "Unlevered Free Cash Flow Margin",
        "Value": format_percentage(summary.get("UFCF Margin %"))
    })
    ratios_data.append({
        "Category": "Margins",
        "Metric": "CapEx as % of Revenue",
        "Value": format_percentage(summary.get("CapEx % Revenue"))
    })
    
    # ========== SECTION: ASSET TURNOVER ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "ASSET TURNOVER", "Metric": "", "Value": ""})
    ratios_data.append({
        "Category": "Asset Turnover",
        "Metric": "Total Asset Turnover",
        "Value": format_multiple(summary.get("Total Asset Turnover"))
    })
    ratios_data.append({
        "Category": "Asset Turnover",
        "Metric": "Accounts Receivable Turnover",
        "Value": format_multiple(summary.get("AR Turnover"))
    })
    ratios_data.append({
        "Category": "Asset Turnover",
        "Metric": "Inventory Turnover",
        "Value": format_multiple(summary.get("Inventory Turnover"))
    })
    
    # ========== SECTION: SHORT-TERM LIQUIDITY ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "SHORT-TERM LIQUIDITY", "Metric": "", "Value": ""})
    ratios_data.append({
        "Category": "Liquidity",
        "Metric": "Current Ratio",
        "Value": format_multiple(summary.get("Current Ratio"))
    })
    ratios_data.append({
        "Category": "Liquidity",
        "Metric": "Quick Ratio",
        "Value": format_multiple(summary.get("Quick Ratio"))
    })
    ratios_data.append({
        "Category": "Liquidity",
        "Metric": "Avg. Days Sales Outstanding",
        "Value": format_ratio(summary.get("Avg Days Sales Outstanding"), 1, " days")
    })
    ratios_data.append({
        "Category": "Liquidity",
        "Metric": "Avg. Days Inventory Outstanding",
        "Value": format_ratio(summary.get("Avg Days Inventory Outstanding"), 1, " days")
    })
    ratios_data.append({
        "Category": "Liquidity",
        "Metric": "Avg. Days Payable Outstanding",
        "Value": format_ratio(summary.get("Avg Days Payable Outstanding"), 1, " days")
    })
    ratios_data.append({
        "Category": "Liquidity",
        "Metric": "Cash Conversion Cycle",
        "Value": format_ratio(summary.get("Cash Conversion Cycle"), 1, " days")
    })
    
    # ========== SECTION: LONG-TERM LIQUIDITY (LEVERAGE) ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "LONG-TERM LIQUIDITY & LEVERAGE", "Metric": "", "Value": ""})
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Total Debt/Equity",
        "Value": format_multiple(summary.get("Total D/E"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Total Debt/Capital",
        "Value": format_multiple(summary.get("Total D/Capital"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Long-term Debt/Equity",
        "Value": format_multiple(summary.get("LT D/E"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Long-term Debt/Capital",
        "Value": format_multiple(summary.get("LT D/Capital"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Total Liabilities/Total Assets",
        "Value": format_multiple(summary.get("Total Liab/Assets"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "EBIT/Interest Expense",
        "Value": format_multiple(summary.get("EBIT/Interest"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "EBITDA/Interest Expense",
        "Value": format_multiple(summary.get("EBITDA/Interest"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Total Debt/Interest Expense",
        "Value": format_multiple(summary.get("Total Debt/Interest"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Net Debt/Interest Expense",
        "Value": format_multiple(summary.get("Net Debt/Interest"))
    })
    ratios_data.append({
        "Category": "Leverage",
        "Metric": "Altman Z-Score",
        "Value": format_ratio(summary.get("Altman Z-Score"))
    })
    
    # ========== SECTION: GROWTH OVER PRIOR YEAR ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "GROWTH OVER PRIOR YEAR (YoY)", "Metric": "", "Value": ""})
    
    yoy_metrics = [
        ("Total Revenue", "Revenue YoY %"),
        ("Gross Profit", "Gross Profit YoY %"),
        ("EBIT", "EBIT YoY %"),
        ("EBITDA", "EBITDA YoY %"),
        ("Net Income", "Net Income YoY %"),
        ("EPS (Basic)", "EPS YoY %"),
        ("EPS (Diluted)", "Diluted EPS YoY %"),
        ("Accounts Receivable", "AR YoY %"),
        ("Inventory", "Inventory YoY %"),
        ("Net PP&E", "Net PP&E YoY %"),
        ("Total Assets", "Total Assets YoY %"),
        ("Total Liabilities", "Total Liabilities YoY %"),
        ("Total Equity", "Total Equity YoY %"),
    ]
    
    for metric_name, key in yoy_metrics:
        ratios_data.append({
            "Category": "YoY Growth",
            "Metric": metric_name,
            "Value": format_percentage(summary.get(key))
        })
    
    # ========== SECTION: 2-YEAR CAGR ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "2-YEAR CAGR", "Metric": "", "Value": ""})
    
    cagr_2yr_metrics = [
        ("Total Revenue", "Revenue 2yr CAGR %"),
        ("Gross Profit", "Gross Profit 2yr CAGR %"),
        ("EBIT", "EBIT 2yr CAGR %"),
        ("EBITDA", "EBITDA 2yr CAGR %"),
        ("Net Income", "Net Income 2yr CAGR %"),
        ("EPS (Basic)", "EPS 2yr CAGR %"),
        ("EPS (Diluted)", "Diluted EPS 2yr CAGR %"),
        ("Accounts Receivable", "AR 2yr CAGR %"),
        ("Inventory", "Inventory 2yr CAGR %"),
        ("Net PP&E", "Net PP&E 2yr CAGR %"),
        ("Total Assets", "Total Assets 2yr CAGR %"),
        ("Total Liabilities", "Total Liabilities 2yr CAGR %"),
        ("Total Equity", "Total Equity 2yr CAGR %"),
    ]
    
    for metric_name, key in cagr_2yr_metrics:
        ratios_data.append({
            "Category": "2-Year CAGR",
            "Metric": metric_name,
            "Value": format_percentage(summary.get(key))
        })
    
    # ========== SECTION: 3-YEAR CAGR ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "3-YEAR CAGR", "Metric": "", "Value": ""})
    
    cagr_3yr_metrics = [
        ("Total Revenue", "Revenue 3yr CAGR %"),
        ("Gross Profit", "Gross Profit 3yr CAGR %"),
        ("EBIT", "EBIT 3yr CAGR %"),
        ("EBITDA", "EBITDA 3yr CAGR %"),
        ("Net Income", "Net Income 3yr CAGR %"),
        ("EPS (Basic)", "EPS 3yr CAGR %"),
        ("EPS (Diluted)", "Diluted EPS 3yr CAGR %"),
        ("Accounts Receivable", "AR 3yr CAGR %"),
        ("Inventory", "Inventory 3yr CAGR %"),
        ("Net PP&E", "Net PP&E 3yr CAGR %"),
        ("Total Assets", "Total Assets 3yr CAGR %"),
        ("Total Liabilities", "Total Liabilities 3yr CAGR %"),
        ("Total Equity", "Total Equity 3yr CAGR %"),
        ("Levered Free Cash Flow", "LFCF 3yr CAGR %"),
    ]
    
    for metric_name, key in cagr_3yr_metrics:
        ratios_data.append({
            "Category": "3-Year CAGR",
            "Metric": metric_name,
            "Value": format_percentage(summary.get(key))
        })
    
    # ========== SECTION: 5-YEAR CAGR ==========
    ratios_data.append({"Category": "", "Metric": "", "Value": ""})
    ratios_data.append({"Category": "5-YEAR CAGR", "Metric": "", "Value": ""})
    
    cagr_5yr_metrics = [
        ("Total Revenue", "Revenue 5yr CAGR %"),
        ("Gross Profit", "Gross Profit 5yr CAGR %"),
        ("EBIT", "EBIT 5yr CAGR %"),
        ("EBITDA", "EBITDA 5yr CAGR %"),
        ("Net Income", "Net Income 5yr CAGR %"),
        ("EPS (Basic)", "EPS 5yr CAGR %"),
        ("EPS (Diluted)", "Diluted EPS 5yr CAGR %"),
        ("Accounts Receivable", "AR 5yr CAGR %"),
        ("Inventory", "Inventory 5yr CAGR %"),
        ("Net PP&E", "Net PP&E 5yr CAGR %"),
        ("Total Assets", "Total Assets 5yr CAGR %"),
        ("Total Liabilities", "Total Liabilities 5yr CAGR %"),
        ("Total Equity", "Total Equity 5yr CAGR %"),
    ]
    
    for metric_name, key in cagr_5yr_metrics:
        ratios_data.append({
            "Category": "5-Year CAGR",
            "Metric": metric_name,
            "Value": format_percentage(summary.get(key))
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(ratios_data)
    
    return df

def print_ratios(summary: Dict[str, Any]) -> None:
    """
    Print formatted ratios analysis to console.
    
    Args:
        summary: Dictionary from build_company_summary()
    """
    df = build_ratios_display(summary)
    
    # Print header
    company_name = summary.get("Company", "Unknown")
    ticker = summary.get("Ticker", "N/A")
    
    print("=" * 80)
    print(f"FINANCIAL RATIOS ANALYSIS: {company_name} ({ticker})")
    print("=" * 80)
    print()
    
    # Print each section
    current_section = None
    for _, row in df.iterrows():
        category = row["Category"]
        metric = row["Metric"]
        value = row["Value"]
        
        # Section headers
        if category and not metric and not value:
            print()
            print(f"\n{category}")
            print("-" * 80)
            current_section = category
        # Regular metrics
        elif metric and value:
            print(f"  {metric:<50} {value:>20}")
    
    print()
    print("=" * 80)

def export_ratios_to_excel(summary: Dict[str, Any], filepath: str) -> None:
    """
    Export ratios to an Excel file with formatting.
    
    Args:
        summary: Dictionary from build_company_summary()
        filepath: Output Excel file path
    """
    df = build_ratios_display(summary)
    
    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Ratios', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Ratios']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        section_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D9E1F2',
            'border': 1
        })
        
        # Set column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 45)
        worksheet.set_column('C:C', 20)
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply section formatting
        for row_num, row in enumerate(df.itertuples(index=False), start=1):
            if row.Category and not row.Metric and not row.Value:
                worksheet.write(row_num, 0, row.Category, section_format)
    
    print(f"Ratios exported to: {filepath}")

# Example usage and interpretation guide
RATIO_INTERPRETATION = {
    "ROA": {
        "good": "> 5%",
        "average": "2-5%",
        "poor": "< 2%",
        "note": "Varies significantly by industry"
    },
    "ROIC": {
        "good": "> 10%",
        "average": "7-10%",
        "poor": "< 7%",
        "note": "Should exceed weighted average cost of capital (WACC)"
    },
    "ROE": {
        "good": "> 15%",
        "average": "10-15%",
        "poor": "< 10%",
        "note": "Can be inflated by high leverage"
    },
    "Current Ratio": {
        "good": "1.5 - 3.0",
        "average": "1.0 - 1.5",
        "poor": "< 1.0",
        "note": "< 1.0 indicates potential liquidity issues"
    },
    "Quick Ratio": {
        "good": "> 1.0",
        "average": "0.7 - 1.0",
        "poor": "< 0.7",
        "note": "More conservative than current ratio"
    },
    "Debt/Equity": {
        "good": "< 1.0",
        "average": "1.0 - 2.0",
        "poor": "> 2.0",
        "note": "Varies significantly by industry"
    },
    "Interest Coverage": {
        "good": "> 3.0x",
        "average": "1.5 - 3.0x",
        "poor": "< 1.5x",
        "note": "Measures ability to service debt"
    },
    "Altman Z-Score": {
        "good": "> 2.99",
        "average": "1.81 - 2.99",
        "poor": "< 1.81",
        "note": "< 1.81 indicates high bankruptcy risk"
    }
}

if __name__ == "__main__":
    # Example usage
    sample_summary = {
        "Company": "Example Corp",
        "Ticker": "EXAM",
        "Industry": "Technology",
        "ROA %": 12.5,
        "ROIC %": 18.3,
        "ROE %": 22.1,
        "RCE %": 16.7,
        "Gross Margin %": 65.2,
        "Current Ratio": 2.1,
        "Quick Ratio": 1.8,
        "Total D/E": 0.45,
        "Altman Z-Score": 3.5,
        "Revenue YoY %": 15.3,
        "Revenue 3yr CAGR %": 18.7,
    }
    
    print_ratios(sample_summary)
