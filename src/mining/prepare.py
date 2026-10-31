# src/mining/prepare.py
"""
Data-Mining preparation script.

- Reads processed CSVs from `Data/processed/`.
- Enriches the core monthly price table with auxiliary features.
- Generates three output CSVs for the three mining tasks:
    * classification.csv  - features + categorical target (Increase/Stable/Decrease)
    * regression.csv      - features + numeric target (next-month modal price)
    * clustering.csv      - aggregated commodity rows for K-Means
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2] / "Data" / "processed"
OUT  = BASE / "mining"
OUT.mkdir(parents=True, exist_ok=True)

# Input CSVs
COM_MONTH = BASE / "commodity_monthly_price.csv"
COM_STATS = BASE / "commodity_statistics.csv"
SEAS_SUM  = BASE / "seasonal_price_summary.csv"
STATE_COM = BASE / "state_commodity_price.csv"
TREND     = BASE / "trend_analysis.csv"

# =====================================================================
# 1. Load core monthly price table
# =====================================================================
# Columns: commodity, year, month, avg_modal_price, avg_min_price, avg_max_price
commodity = pd.read_csv(COM_MONTH)
commodity["year"]  = commodity["year"].astype(int)
commodity["month"] = commodity["month"].astype(int)
print(f"Core table loaded: {len(commodity)} rows, columns: {commodity.columns.tolist()}")

# =====================================================================
# 2. Enrich with auxiliary tables (safe left joins only)
# =====================================================================

# --- commodity_statistics: join on [commodity, year] ---
if COM_STATS.is_file():
    stats = pd.read_csv(COM_STATS)
    # Keep only useful columns to avoid name clashes
    stats_cols = ["commodity", "year", "variance", "standard_deviation", "coefficient_of_variation"]
    stats_cols = [c for c in stats_cols if c in stats.columns]
    stats = stats[stats_cols].drop_duplicates()
    commodity = commodity.merge(stats, on=["commodity", "year"], how="left")
    print(f"  + commodity_statistics merged -> {len(commodity)} rows")

# --- seasonal_price_summary: join on [commodity, year, season] ---
# But commodity table has no 'season' column yet, so we CREATE it from month
def month_to_season(m):
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"

commodity["season"] = commodity["month"].apply(month_to_season)

if SEAS_SUM.is_file():
    seas = pd.read_csv(SEAS_SUM)
    # Seasonal summary has: commodity, year, season, avg_modal_price, ...
    # Rename price columns to avoid clash with the core table
    seas = seas.rename(columns={
        "avg_modal_price": "seasonal_avg_price",
        "avg_min_price": "seasonal_min_price",
        "avg_max_price": "seasonal_max_price",
    })
    join_keys = [k for k in ["commodity", "year", "season"] if k in seas.columns]
    if join_keys:
        commodity = commodity.merge(seas[join_keys + ["seasonal_avg_price"]], on=join_keys, how="left")
        print(f"  + seasonal_price_summary merged -> {len(commodity)} rows")

# --- state_commodity_price: AGGREGATE to commodity-year level ---
# Original has one row per state-commodity-year; we average across states
if STATE_COM.is_file():
    state_raw = pd.read_csv(STATE_COM)
    state_agg = (
        state_raw
        .groupby(["commodity", "year"], as_index=False)
        .agg(
            num_states=("state", "nunique"),
            state_avg_price=("avg_modal_price", "mean"),
        )
    )
    commodity = commodity.merge(state_agg, on=["commodity", "year"], how="left")
    print(f"  + state_commodity_price (aggregated) merged -> {len(commodity)} rows")

# --- trend_analysis: join on [commodity, year, month] ---
if TREND.is_file():
    trend = pd.read_csv(TREND)
    trend_cols = ["commodity", "year", "month",
                  "month_over_month_percentage_change", "trend"]
    trend_cols = [c for c in trend_cols if c in trend.columns]
    trend = trend[trend_cols].drop_duplicates(subset=["commodity", "year", "month"])
    commodity = commodity.merge(trend, on=["commodity", "year", "month"], how="left")
    print(f"  + trend_analysis merged -> {len(commodity)} rows")

# =====================================================================
# 3. Create lagged features (previous month price)
# =====================================================================
commodity = commodity.sort_values(["commodity", "year", "month"]).reset_index(drop=True)
commodity["prev_price"] = commodity.groupby("commodity")["avg_modal_price"].shift(1)
# Drop rows with no previous month
commodity = commodity.dropna(subset=["prev_price"]).reset_index(drop=True)

# Price change percentage
commodity["price_change_pct"] = (
    (commodity["avg_modal_price"] - commodity["prev_price"]) / commodity["prev_price"] * 100
)

print(f"\nEnriched table: {len(commodity)} rows, {len(commodity.columns)} columns")
print(f"Columns: {commodity.columns.tolist()}")

# =====================================================================
# 4. Classification dataset
# =====================================================================
STABLE_THRESH = 5.0  # +/- 5% -> Stable

def label_movement(pct):
    if pct > STABLE_THRESH:
        return "Increase"
    if pct < -STABLE_THRESH:
        return "Decrease"
    return "Stable"

class_df = commodity.copy()
class_df["target"] = class_df["price_change_pct"].apply(label_movement)

# Select numeric + useful categorical features
class_keep = [
    "prev_price", "avg_modal_price", "price_change_pct",
    "variance", "standard_deviation", "coefficient_of_variation",
    "seasonal_avg_price", "num_states", "state_avg_price",
    "month_over_month_percentage_change",
    "year", "month", "season",
    "target",
]
class_keep = [c for c in class_keep if c in class_df.columns]
class_df = class_df[class_keep].dropna(subset=["target"])
class_df.to_csv(OUT / "classification.csv", index=False)

# =====================================================================
# 5. Regression dataset (predict next-month price)
# =====================================================================
reg_df = commodity.copy()
reg_df["target_price"] = reg_df.groupby("commodity")["avg_modal_price"].shift(-1)
reg_df = reg_df.dropna(subset=["target_price"])

reg_keep = [
    "prev_price", "price_change_pct",
    "variance", "standard_deviation", "coefficient_of_variation",
    "seasonal_avg_price", "num_states", "state_avg_price",
    "month_over_month_percentage_change",
    "year", "month",
    "target_price",
]
reg_keep = [c for c in reg_keep if c in reg_df.columns]
reg_df = reg_df[reg_keep]
reg_df.to_csv(OUT / "regression.csv", index=False)

# =====================================================================
# 6. Clustering dataset (one row per commodity)
# =====================================================================
cluster_df = (
    commodity.groupby("commodity")
    .agg(
        avg_price=("avg_modal_price", "mean"),
        volatility=("price_change_pct", "std"),
        mean_change=("price_change_pct", "mean"),
        num_months=("month", "count"),
    )
    .reset_index()
)
cluster_df.to_csv(OUT / "clustering.csv", index=False)

# =====================================================================
# Summary
# =====================================================================
print(f"\nDONE: Mining CSVs generated in {OUT}")
print(f"   classification.csv : {len(class_df)} rows, {len(class_df.columns)} cols")
print(f"   regression.csv     : {len(reg_df)} rows, {len(reg_df.columns)} cols")
print(f"   clustering.csv     : {len(cluster_df)} rows, {len(cluster_df.columns)} cols")
print(f"\nTarget distribution (classification):")
print(class_df["target"].value_counts().to_string())
