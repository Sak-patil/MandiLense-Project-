<h1 align="center">🌾 MandiLense</h1>
<h3 align="center">Smart Mandi Intelligence — AI-Powered Price Forecasting for Indian Agricultural Markets</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Plotly-Visualisation-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Records-6.5M+-green" alt="Records">
</p>

<p align="center">
  <strong>Summer Internship Project 2025–26</strong><br>
  Applied Data Mining &amp; Machine Learning on Indian Agricultural Mandi Transaction Data
</p>

---

## 📖 What is MandiLense?

Every day, lakhs of Indian farmers sell their crops at **Mandis** (government-regulated wholesale markets). But prices are unpredictable — tomatoes can go from ₹15/kg to ₹100/kg in a single month. Farmers have no way to know what's coming.

**MandiLense** uses **6.5 million+ real mandi transaction records** (2025–2026) from the [Agmarknet](https://agmarknet.gov.in/) portal to train AI models that answer three critical questions:

| # | Question | AI Model Used | Result |
|---|----------|---------------|--------|
| 1 | **Will the price go UP, DOWN, or stay SAME next month?** | Random Forest Classifier (200 trees) | **98% Accuracy** |
| 2 | **What will the exact price be in ₹/quintal?** | Linear Regression (baseline) | MAE: ₹7,377 |
| 3 | **Which crops behave similarly in the market?** | K-Means Clustering | **3 groups, Silhouette: 0.97** |

An interactive **Streamlit web dashboard** lets users enter market data and get instant AI predictions.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **pip** (Python package manager)
- **MySQL** (optional — only needed if you want to use the database upload feature)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "MandiLense Summer Internship Project"
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. That's it! 🎉

---

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
Pillow
python-dotenv
SQLAlchemy
PyMySQL
mysql-connector-python
matplotlib
seaborn
```

---

## 📁 Project Structure

```
MandiLense Summer Internship Project/
│
├── app.py                          # 🌐 Streamlit web application (main entry point)
├── MODEL_EXPLANATION.md            # 📝 Beginner-friendly guide to all 3 models
├── POC_DOCUMENT.md                 # 📋 Proof of Concept technical dossier
├── README.md                       # 📖 This file
├── .env                            # 🔑 Database credentials (gitignored)
├── .gitignore                      # Git ignore rules
│
├── Data/
│   ├── raw/                        # 📥 Raw Agmarknet CSV dumps
│   │   ├── 2025.csv                #     ~573 MB, ~5.9 million records
│   │   └── 2026.csv                #     ~62 MB, ~600K records
│   └── processed/                  # 🧹 Cleaned & aggregated data
│       ├── mandi_2025_cleaned.csv
│       ├── mandi_2026_cleaned.csv
│       ├── commodity_monthly_price.csv
│       ├── commodity_statistics.csv
│       ├── seasonal_price_summary.csv
│       ├── state_commodity_price.csv
│       ├── trend_analysis.csv
│       ├── market_price_summary.csv
│       ├── business_insights_summary.csv
│       └── mining/                 # 🧠 ML-ready datasets
│           ├── classification.csv      # 4,095 records with direction labels
│           ├── regression.csv          # 3,759 records with target prices
│           ├── clustering.csv          # 329 commodity feature vectors
│           ├── clustering_labeled.csv  # With cluster assignments
│           └── elbow_silhouette.png    # Cluster optimization chart
│
├── saved_models/                   # 💾 Trained model binaries
│   ├── random_forest.joblib        #     Classification model (98% accuracy)
│   ├── linear_regression.joblib    #     Regression model + scaler
│   └── kmeans.joblib               #     Clustering model (k=3)
│
├── src/                            # 🐍 Python source modules
│   ├── data_processing.py          #     Data cleaning & validation functions
│   ├── feature_engineering.py      #     Feature creation utilities
│   ├── database.py                 #     MySQL upload via SQLAlchemy
│   └── mining/                     #     ML training scripts
│       ├── prepare.py              #       Feature engineering & dataset creation
│       ├── classify.py             #       Random Forest training & evaluation
│       ├── regress.py              #       Linear Regression training & evaluation
│       └── cluster.py              #       K-Means clustering & optimization
│
├── Notebooks/                      # 📓 Jupyter analysis notebooks (step-by-step)
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_clean_test.ipynb
│   ├── 03_cleaning_process_full_2025.ipynb
│   ├── 04_cleaning_process_full_2026.ipynb
│   ├── 05_commodity_monthly_analysis.ipynb
│   ├── 06_state_market_seasonal_combined_analysis.ipynb
│   ├── 07_commodity_statistics.ipynb
│   ├── 08_trend_analysis.ipynb
│   ├── 09_business_insights.ipynb
│   ├── 10_data_mining_models.ipynb
│   └── csv_to_database.ipynb
│
└── outputs/                        # 📊 Reports & aggregations
    └── reports/
        ├── quality_report_2025.csv
        └── quality_report_2026.csv
```

---

## 🔬 How It Works

### Step 1: Data Collection
Downloaded **6.5 million+** daily mandi transaction records from the Government of India's **Agmarknet** portal covering all states, 300+ commodities, and 2 years (2025–2026).

Each record contains: `State`, `District`, `Market`, `Commodity`, `Variety`, `Grade`, `Arrival_Date`, `Min_Price`, `Max_Price`, `Modal_Price`.

### Step 2: Data Cleaning (`src/data_processing.py`)
- Processed data in **chunks of 100,000 rows** (59 chunks for 2025, 7 for 2026).
- Validated price logic: `Min ≤ Modal ≤ Max`.
- Removed zero/negative prices and invalid dates.
- Generated quality reports for every chunk.

### Step 3: Feature Engineering (`src/mining/prepare.py`)
Aggregated millions of atomic transactions into commodity-month level features:
- **Monthly average prices** per commodity.
- **Volatility measures**: variance, standard deviation, coefficient of variation.
- **Momentum**: month-over-month percentage change.
- **Market breadth**: number of states trading each commodity.
- **Seasonal benchmarks**: average prices per season (Winter, Summer, Monsoon, Post-Monsoon).

### Step 4: Model Training

#### 🎯 Task 1 — Price Direction Classification
- **Algorithm**: Random Forest (200 decision trees)
- **Target**: Increase (>+5%), Stable (±5%), Decrease (<-5%)
- **Split**: Chronological (first 80% months → train, last 20% → test) — no future leakage
- **Result**: **98.0% accuracy**, ROC-AUC: 0.9985
- **Top feature**: Month-over-month % change (69.3% importance)

#### 💰 Task 2 — Price Regression
- **Algorithm**: Linear Regression + StandardScaler
- **Target**: Next month's modal price in ₹/quintal
- **Result**: MAE = ₹7,377 | R² = -0.16
- **Key insight**: Negative R² proves agricultural prices are non-linear — validating the need for ensemble methods like Random Forest.

#### 🧩 Task 3 — Commodity Clustering
- **Algorithm**: K-Means
- **Tested**: k = 2 to 10, optimized via Elbow Method + Silhouette Score
- **Result**: k = 3 (Silhouette: 0.9675)
  - **Cluster 0** (326 crops): Everyday staples — Potato, Onion, Wheat, Rice
  - **Cluster 1** (2 crops): Extreme volatility — Hippe Seed, Jaffri
  - **Cluster 2** (1 crop): Ultra-luxury — **Saffron** at ₹6.61 Lakhs/quintal

### Step 5: Interactive Web App (`app.py`)
A Streamlit dashboard with 5 pages for live inference, visual exploration, and model diagnostics.

---

## 🧠 Key Findings

1. **Price momentum is the #1 predictor** — Month-over-month % change alone drives 69.3% of the classification model's decisions.
2. **Agricultural prices are non-linear** — Linear regression's negative R² score mathematically proves that simple straight-line models fail on mandi data.
3. **The market has 3 natural tiers** — K-Means discovered that 99% of crops are "regular staples", <1% are extreme-volatility anomalies, and Saffron is in a league of its own.
4. **Chronological validation is essential** — Using a time-aware train-test split prevented data leakage and gave honest accuracy estimates.

---

## 📓 Notebook Walkthrough

The `Notebooks/` folder contains the complete research journey:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Initial data exploration and understanding the raw schema |
| `02_data_clean_test.ipynb` | Testing cleaning functions on a small sample |
| `03_cleaning_process_full_2025.ipynb` | Full chunk-based cleaning of 5.9M records (2025) |
| `04_cleaning_process_full_2026.ipynb` | Full chunk-based cleaning of 600K records (2026) |
| `05_commodity_monthly_analysis.ipynb` | Monthly price aggregation per commodity |
| `06_state_market_seasonal_combined_analysis.ipynb` | State, market, and seasonal aggregations |
| `07_commodity_statistics.ipynb` | Variance, std dev, CV calculations per commodity |
| `08_trend_analysis.ipynb` | Month-over-month trends and momentum features |
| `09_business_insights.ipynb` | Top/bottom commodities, best/worst seasons, key markets |
| `10_data_mining_models.ipynb` | Training & evaluation of all 3 ML models |
| `csv_to_database.ipynb` | Uploading processed CSVs to MySQL database |

---

## ⚙️ Environment Variables

Create a `.env` file in the project root (only needed for database features):

```
DB_HOST=localhost
DB_PORT=3306
DB_NAME=agri_market_analysis
DB_USER=root
DB_PASSWORD=your_password
```

---

## 🛣️ Future Roadmap

- [ ] Replace Linear Regression with **XGBoost / LSTM** for better price estimation
- [ ] Add **live data sync** with the Agmarknet API
- [ ] Deploy on **Streamlit Cloud / AWS / Hugging Face Spaces**
- [ ] Build a **WhatsApp bot** for farmer price alerts in Hindi
- [ ] Add **district-level GIS maps** for geographic price visualization
- [ ] Create an **agri-lending risk score API** for rural banks

---

## 🙏 Acknowledgements

- **Data Source**: [Agmarknet](https://agmarknet.gov.in/) — Ministry of Agriculture & Farmers Welfare, Government of India
- **Built with**: Python, Streamlit, Scikit-Learn, Pandas, Plotly, NumPy

---

## 📄 License

This project was developed as a **Summer Internship academic project**. All data is sourced from publicly available government portals.

---

<p align="center">
  <strong>Built with ❤️ for Indian Farmers</strong><br>
  <em>MandiLense — Because every farmer deserves market intelligence.</em>
</p>

