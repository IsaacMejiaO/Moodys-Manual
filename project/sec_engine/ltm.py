import pandas as pd
from typing import Dict, List

def build_ltm(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    series = series.dropna().sort_index()
    if len(series) < 4:
        return float("nan")
    return float(series.iloc[-4:].sum())

def latest_balance(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    return float(series.dropna().sort_index().iloc[-1])

def extract_quarterly_series(facts: dict, tag_list: List[str], unit: str = "USD") -> pd.Series:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tag_list:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get(unit, [])
        rows = []
        for e in entries:
            if e.get("form") not in ("10-Q", "10-K"):
                continue
            if "end" not in e:
                continue
            rows.append({"date": pd.to_datetime(e["end"]), "value": e["val"]})
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset="date")
            return df.set_index("date")["value"].sort_index()
    return pd.Series(dtype="float64")

def extract_annual_series(facts: dict, tag_list: List[str], unit: str = "USD") -> pd.Series:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tag_list:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get(unit, [])
        rows = []
        for e in entries:
            if e.get("form") != "10-K":
                continue
            if "end" not in e:
                continue
            rows.append({"date": pd.to_datetime(e["end"]), "value": e["val"]})
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset="date")
            return df.set_index("date")["value"].sort_index()
    return pd.Series(dtype="float64")

def build_ltm_financials(facts: dict, gaap_map: Dict[str, List[str]], ltm_metrics=None) -> Dict[str, float]:
    if ltm_metrics is None:
        ltm_metrics = [
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "ocf",
            "capex",
            "depreciation",
            "ebitda",
        ]

    ltm_data = {}
    for metric, tags in gaap_map.items():
        series = extract_quarterly_series(facts, tags)
        if metric in ltm_metrics:
            ltm_data[metric] = build_ltm(series)
        else:
            ltm_data[metric] = latest_balance(series)
    return ltm_data
