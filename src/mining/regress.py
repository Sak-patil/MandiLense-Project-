# src/mining/regress.py
"""
Simple Linear Regression for next‑month modal price prediction.
The script:
1️⃣ Loads `Data/processed/mining/regression.csv`.
2️⃣ Performs a chronological 80 % / 20 % train‑test split (by year/month if present).
3️⃣ Scales numeric features with `StandardScaler`.
4️⃣ Trains a `LinearRegression` model (you can swap in other regressors later).
5️⃣ Prints MAE, RMSE and R² on the test set.
6️⃣ Persists the trained model to `saved_models/linear_regression.joblib`.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2] / "Data" / "processed" / "mining"
DATA_PATH = BASE / "regression.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "saved_models" / "linear_regression.joblib"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Target column
TARGET_COL = "target_price"
feature_cols = [c for c in df.columns if c != TARGET_COL]

# Drop rows with any NaN in features or target
df = df.dropna().reset_index(drop=True)
print(f"Regression dataset: {len(df)} rows after dropping NaN")

X = df[feature_cols]
y = df[TARGET_COL]

# ---------------------------------------------------------------------
# Chronological train‑test split (if year/month exist)
# ---------------------------------------------------------------------
if {"year", "month"}.issubset(df.columns):
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    X = df[feature_cols]
    y = df[TARGET_COL]
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# ---------------------------------------------------------------------
# Scale numeric features (all are numeric after preprocessing)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# Train Linear Regression
# ---------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
y_pred = lr.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print("--- Regression Evaluation ---")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

# ---------------------------------------------------------------------
# Persist model + scaler as a dict
# ---------------------------------------------------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": lr, "scaler": scaler}, MODEL_PATH)
print(f"Regression model saved to {MODEL_PATH}")
