# src/mining/classify.py
"""
Random-Forest classification for price-movement (Increase / Stable / Decrease).

IMPORTANT: We deliberately EXCLUDE `price_change_pct` and `avg_modal_price`
from the features, because the target label is derived from them.
Including them would be data leakage (the model would get 100% accuracy
by simply reading the answer).

Features used:
  - prev_price           (last month's modal price)
  - variance, std_dev, cv (commodity volatility stats)
  - seasonal_avg_price   (average price for that season)
  - num_states, state_avg_price (geographic spread)
  - month_over_month_percentage_change (trend momentum)
  - year, month, season  (time context)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2] / "Data" / "processed" / "mining"
DATA_PATH = BASE / "classification.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "saved_models" / "random_forest.joblib"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

TARGET_COL = "target"

# DROP leaky columns -- these are used to CREATE the target, so the model
# must NOT see them at prediction time.
LEAKY_COLS = {"price_change_pct", "avg_modal_price", "target"}
feature_cols = [c for c in df.columns if c not in LEAKY_COLS]

X = df[feature_cols]
y = df[TARGET_COL]

print(f"Dataset : {len(df)} rows")
print(f"Features: {feature_cols}")
print(f"Target  : {y.value_counts().to_dict()}")

# ---------------------------------------------------------------------
# Chronological train-test split (80% train / 20% test)
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# ---------------------------------------------------------------------
# Detect numeric vs categorical columns
# ---------------------------------------------------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print(f"Numeric features : {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

pipeline = Pipeline([("preprocess", preprocess), ("model", clf)])

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
y_pred = pipeline.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC (one-vs-rest)
if len(pipeline.classes_) > 2:
    probas = pipeline.predict_proba(X_test)
    auc = roc_auc_score(y_test, probas, multi_class="ovr")
    print(f"ROC-AUC (OVR): {auc:.4f}")
else:
    probas = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probas)
    print(f"ROC-AUC: {auc:.4f}")

# ---------------------------------------------------------------------
# Feature importance (top 10)
# ---------------------------------------------------------------------
rf_model = pipeline.named_steps["model"]
ohe_names = []
if categorical_cols:
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(categorical_cols))
all_feature_names = numeric_cols + ohe_names

importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
print("\n--- Top 10 Feature Importances ---")
print(feat_imp.head(10).to_string())

# ---------------------------------------------------------------------
# Persist model
# ---------------------------------------------------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
