# src/mining/cluster.py
"""
K‑Means clustering for mandi‑commodity groups.
The script:
1️⃣ Loads `Data/processed/mining/clustering.csv`.
2️⃣ Scales numeric features (`avg_price`, `volatility`, `mean_change`).
3️⃣ Runs an *elbow* scan for k=2‑10 and prints the total inertia.
4️⃣ Computes the *silhouette* score for each k (requires at least 2 clusters).
5️⃣ Chooses the best k (largest silhouette) and fits the final model.
6️⃣ Saves the scaler + fitted KMeans pipeline to `saved_models/kmeans.joblib`.
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2] / "Data" / "processed" / "mining"
DATA_PATH = BASE / "clustering.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "saved_models" / "kmeans.joblib"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Features for clustering - numeric only
features = ["avg_price", "volatility", "mean_change"]
df = df.dropna(subset=features).reset_index(drop=True)
X = df[features].copy()
print(f"Clustering dataset: {len(df)} rows after dropping NaN")

# ---------------------------------------------------------------------
# Scale features
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# Elbow & silhouette scan
# ---------------------------------------------------------------------
inertias = []
silhouettes = []
K_RANGE = range(2, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil)
    print(f"k={k:2d} | inertia={km.inertia_:,.0f} | silhouette={sil:.4f}")

# Plot - save to file instead of opening a window
PLOT_PATH = BASE / "elbow_silhouette.png"
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(K_RANGE, inertias, marker="o")
plt.title("Elbow - Inertia")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.subplot(1, 2, 2)
plt.plot(K_RANGE, silhouettes, marker="o", color="orange")
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Elbow/Silhouette plot saved to {PLOT_PATH}")

# ---------------------------------------------------------------------
# Choose best k (max silhouette) and fit final model
# ---------------------------------------------------------------------
best_k = K_RANGE[silhouettes.index(max(silhouettes))]
print(f"\nSelecting k={best_k} (max silhouette)")
final_km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
final_km.fit(X_scaled)

# Attach cluster label to original dataframe and save
df["cluster"] = final_km.labels_
df.to_csv(BASE / "clustering_labeled.csv", index=False)
print(f"Clustering labeled CSV saved to {BASE / 'clustering_labeled.csv'}")

# ---------------------------------------------------------------------
# Persist pipeline (scaler + kmeans)
# ---------------------------------------------------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"scaler": scaler, "model": final_km}, MODEL_PATH)
print(f"K-Means model saved to {MODEL_PATH}")
