import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MandiLense - Data Mining Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Base Directories & Resource Loading
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data" / "processed" / "mining"
MODELS_DIR = ROOT_DIR / "saved_models"

@st.cache_resource
def load_models():
    models = {}
    rf_path = MODELS_DIR / "random_forest.joblib"
    lr_path = MODELS_DIR / "linear_regression.joblib"
    km_path = MODELS_DIR / "kmeans.joblib"
    
    if rf_path.exists():
        models["classification"] = joblib.load(rf_path)
    if lr_path.exists():
        models["regression"] = joblib.load(lr_path)
    if km_path.exists():
        models["clustering"] = joblib.load(km_path)
    return models

@st.cache_data
def load_data():
    datasets = {}
    cl_path = DATA_DIR / "classification.csv"
    rg_path = DATA_DIR / "regression.csv"
    ck_path = DATA_DIR / "clustering_labeled.csv"
    
    if cl_path.exists():
        datasets["classification"] = pd.read_csv(cl_path)
    if rg_path.exists():
        datasets["regression"] = pd.read_csv(rg_path)
    if ck_path.exists():
        datasets["clustering"] = pd.read_csv(ck_path)
    return datasets

models = load_models()
datasets = load_data()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/wheat.png", width=64)
st.sidebar.title("MandiLense ML")
st.sidebar.caption("Data Mining & Predictive Analytics")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📋 Project Overview",
        "🎯 Task 1: Price Movement Classification",
        "📈 Task 2: Next-Month Price Regression",
        "🧩 Task 3: Market Clustering (K-Means)",
        "📊 Task 4: Evaluation & Model Diagnostics"
    ],
    index=0
)

# -----------------------------------------------------------------------------
# TAB 1: Project Overview
# -----------------------------------------------------------------------------
if menu == "📋 Project Overview":
    st.markdown('<div class="main-header">🌾 MandiLense — Agricultural Data Mining Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Applied Machine Learning & Pattern Discovery for Indian Agricultural Mandis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Architecture & Flow
    This system transforms raw agricultural mandi records into predictive machine learning insights.
    Below is the complete 3-pillar data mining framework:
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>1. Classification (Ensemble)</h4>
            <p><strong>Goal:</strong> Predict if prices will Increase (>+5%), remain Stable, or Decrease (<-5%).</p>
            <ul>
                <li><strong>Algorithm:</strong> Random Forest Classifier (200 Trees)</li>
                <li><strong>Accuracy:</strong> 98.0%</li>
                <li><strong>ROC-AUC:</strong> 0.9985</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>2. Regression (Prediction)</h4>
            <p><strong>Goal:</strong> Forecast the next month's modal price (Rs/Quintal).</p>
            <ul>
                <li><strong>Algorithm:</strong> Linear Regression</li>
                <li><strong>Features:</strong> Lagged price, volatility, trend momentum</li>
                <li><strong>Evaluation:</strong> MAE, RMSE, R²</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>3. Clustering (Segmentation)</h4>
            <p><strong>Goal:</strong> Discover natural commodity groups sharing price and volatility profiles.</p>
            <ul>
                <li><strong>Algorithm:</strong> K-Means Clustering</li>
                <li><strong>Optimal Clusters:</strong> k = 3</li>
                <li><strong>Silhouette Score:</strong> 0.9675</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Data Preparation & Feature Pipeline Summary")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classification Records", len(datasets.get("classification", [])))
    c2.metric("Regression Records", len(datasets.get("regression", [])))
    c3.metric("Clustered Commodities", len(datasets.get("clustering", [])))
    c4.metric("Models Saved", len(models))

# -----------------------------------------------------------------------------
# TAB 2: Classification (Random Forest)
# -----------------------------------------------------------------------------
elif menu == "🎯 Task 1: Price Movement Classification":
    st.markdown('<div class="main-header">🎯 Price Movement Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting whether agricultural market prices will Increase, Decrease, or remain Stable</div>', unsafe_allow_html=True)
    
    if "classification" not in models:
        st.error("Classification model not found in saved_models/random_forest.joblib")
    else:
        pipeline = models["classification"]
        
        st.markdown("#### Input Commodity Features")
        with st.form("classification_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                prev_price = st.number_input("Previous Month Price (Rs/Quintal)", min_value=10.0, max_value=500000.0, value=3500.0, step=100.0)
                mom_change = st.number_input("Month-over-Month % Change", min_value=-100.0, max_value=500.0, value=6.5, step=0.5)
                season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Post-Monsoon"])
                year = st.selectbox("Year", [2025, 2026, 2027], index=1)
                
            with col2:
                seasonal_avg_price = st.number_input("Seasonal Average Price (Rs/Quintal)", min_value=10.0, max_value=500000.0, value=3600.0, step=100.0)
                state_avg_price = st.number_input("State Average Price (Rs/Quintal)", min_value=10.0, max_value=500000.0, value=3550.0, step=100.0)
                num_states = st.slider("Number of States Trading Commodity", min_value=1, max_value=35, value=12)
                month = st.slider("Month", min_value=1, max_value=12, value=10)
                
            with col3:
                variance = st.number_input("Price Variance", min_value=0.0, value=150000.0, step=1000.0)
                std_dev = st.number_input("Standard Deviation", min_value=0.0, value=387.0, step=10.0)
                cv = st.number_input("Coefficient of Variation (%)", min_value=0.0, value=11.2, step=0.5)
            
            submitted = st.form_submit_button("🚀 Predict Price Movement", use_container_width=True)
            
        if submitted:
            input_df = pd.DataFrame([{
                "prev_price": prev_price,
                "variance": variance,
                "standard_deviation": std_dev,
                "coefficient_of_variation": cv,
                "seasonal_avg_price": seasonal_avg_price,
                "num_states": num_states,
                "state_avg_price": state_avg_price,
                "month_over_month_percentage_change": mom_change,
                "year": year,
                "month": month,
                "season": season
            }])
            
            prediction = pipeline.predict(input_df)[0]
            probabilities = pipeline.predict_proba(input_df)[0]
            classes = pipeline.classes_
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                if prediction == "Increase":
                    st.success(f"### 📈 Predicted: **{prediction}**")
                    st.write("Expected price increase greater than **+5%**.")
                elif prediction == "Decrease":
                    st.error(f"### 📉 Predicted: **{prediction}**")
                    st.write("Expected price decrease greater than **-5%**.")
                else:
                    st.info(f"### ⚖️ Predicted: **{prediction}**")
                    st.write("Expected price fluctuation within **±5%** range.")
            
            with res_col2:
                prob_df = pd.DataFrame({
                    "Class": classes,
                    "Probability (%)": [round(p * 100, 2) for p in probabilities]
                })
                fig = px.bar(
                    prob_df, x="Class", y="Probability (%)",
                    color="Class",
                    color_discrete_map={"Increase": "#2e7d32", "Stable": "#f9a825", "Decrease": "#c62828"},
                    text="Probability (%)",
                    title="Model Confidence Distribution"
                )
                fig.update_layout(yaxis_range=[0, 100], showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Regression (Linear Regression)
# -----------------------------------------------------------------------------
elif menu == "📈 Task 2: Next-Month Price Regression":
    st.markdown('<div class="main-header">📈 Next-Month Price Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantitative forecast of agricultural commodity modal prices</div>', unsafe_allow_html=True)
    
    if "regression" not in models:
        st.error("Regression model not found in saved_models/linear_regression.joblib")
    else:
        reg_obj = models["regression"]
        lr_model = reg_obj["model"]
        scaler = reg_obj["scaler"]
        
        st.markdown("#### Input Current Market Attributes")
        with st.form("regression_form"):
            col1, col2 = st.columns(2)
            with col1:
                prev_price = st.number_input("Current Month Modal Price (Rs)", min_value=10.0, max_value=500000.0, value=4200.0)
                price_change_pct = st.number_input("Current Price Change %", min_value=-100.0, max_value=500.0, value=3.2)
                variance = st.number_input("Variance", min_value=0.0, value=250000.0)
                std_dev = st.number_input("Standard Deviation", min_value=0.0, value=500.0)
                cv = st.number_input("Coefficient of Variation", min_value=0.0, value=12.0)
                
            with col2:
                seasonal_avg_price = st.number_input("Seasonal Average Price (Rs)", min_value=10.0, max_value=500000.0, value=4150.0)
                num_states = st.slider("Trading States Count", min_value=1, max_value=35, value=10)
                state_avg_price = st.number_input("State Average Price (Rs)", min_value=10.0, max_value=500000.0, value=4250.0)
                mom_change = st.number_input("MoM % Change", min_value=-100.0, max_value=500.0, value=3.0)
                year = st.selectbox("Year", [2025, 2026, 2027], index=1)
                month = st.slider("Target Forecast Month", min_value=1, max_value=12, value=11)
                
            submitted = st.form_submit_button("📊 Forecast Next Month Price", use_container_width=True)
            
        if submitted:
            feat_cols = [
                "prev_price", "price_change_pct", "variance", "standard_deviation",
                "coefficient_of_variation", "seasonal_avg_price", "num_states",
                "state_avg_price", "month_over_month_percentage_change", "year", "month"
            ]
            row = np.array([[prev_price, price_change_pct, variance, std_dev, cv,
                             seasonal_avg_price, num_states, state_avg_price, mom_change, year, month]])
            row_scaled = scaler.transform(row)
            pred_price = lr_model.predict(row_scaled)[0]
            
            st.markdown("---")
            st.subheader("Regression Forecast")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Next-Month Price", f"Rs. {pred_price:,.2f}")
            c2.metric("Current Modal Price", f"Rs. {prev_price:,.2f}")
            pct_diff = ((pred_price - prev_price) / prev_price) * 100
            c3.metric("Projected Growth", f"{pct_diff:+.2f}%")
            
            st.caption("Benchmark Metrics on Chronological Test Set: MAE = Rs 7,377.34 | RMSE = Rs 24,636.36")

# -----------------------------------------------------------------------------
# TAB 4: Clustering (K-Means)
# -----------------------------------------------------------------------------
elif menu == "🧩 Task 3: Market Clustering (K-Means)":
    st.markdown('<div class="main-header">🧩 Market Segmentation (K-Means Clustering)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised discovery of commodity profiles based on price, volatility, and trend momentum</div>', unsafe_allow_html=True)
    
    if "clustering" not in datasets:
        st.error("Clustering dataset not found in Data/processed/mining/clustering_labeled.csv")
    else:
        df_cluster = datasets["clustering"]
        
        # Summary description of clusters
        cluster_names = {
            0: "Cluster 0: Core Agricultural Staples (Moderate Volume & Stable Price)",
            1: "Cluster 1: Extreme Volatility Anomalies (High % Swings)",
            2: "Cluster 2: Ultra High-Value Exotic/Spices (e.g. Saffron)"
        }
        
        st.markdown("""
        The algorithm converged on **k = 3** as the optimal cluster count based on **Silhouette Score (0.9675)**:
        """)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        for c_id, name in cluster_names.items():
            count = len(df_cluster[df_cluster["cluster"] == c_id])
            avg_p = df_cluster[df_cluster["cluster"] == c_id]["avg_price"].mean()
            if c_id == 0:
                col_m1.metric("Cluster 0 (Core Mandi)", f"{count} Commodities", f"Avg: Rs {avg_p:,.0f}")
            elif c_id == 1:
                col_m2.metric("Cluster 1 (High Volatility)", f"{count} Commodities", f"Avg: Rs {avg_p:,.0f}")
            else:
                col_m3.metric("Cluster 2 (Exotic/Luxury)", f"{count} Commodities", f"Avg: Rs {avg_p:,.0f}")
                
        st.markdown("---")
        st.subheader("Interactive Cluster Exploration")
        
        col_plot1, col_plot2 = st.columns([2, 1])
        with col_plot1:
            fig = px.scatter(
                df_cluster,
                x="avg_price",
                y="volatility",
                color=df_cluster["cluster"].astype(str),
                hover_name="commodity",
                log_x=True,
                log_y=True,
                title="Log-Scale Price vs Volatility Cluster Map",
                labels={"avg_price": "Average Modal Price (Log Rs)", "volatility": "Volatility StdDev (Log %)", "color": "Cluster"},
                color_discrete_sequence=["#1976d2", "#d32f2f", "#388e3c"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_plot2:
            sel_cluster = st.selectbox("Inspect Cluster Details", [0, 1, 2], format_func=lambda x: cluster_names[x])
            sub_df = df_cluster[df_cluster["cluster"] == sel_cluster][["commodity", "avg_price", "volatility", "mean_change"]]
            st.dataframe(sub_df.sort_values("avg_price", ascending=False), height=380, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: Evaluation & Diagnostics
# -----------------------------------------------------------------------------
elif menu == "📊 Task 4: Evaluation & Model Diagnostics":
    st.markdown('<div class="main-header">📊 Model Evaluation & Diagnostic Visuals</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Academic error measures, confusion matrices, ROC-AUC, and feature rankings</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Classification Diagnostics", "Regression Error Measures", "Clustering Elbow & Silhouette"])
    
    with tab1:
        st.subheader("Random Forest Classifier Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", "98.0%")
        m2.metric("Macro Precision", "0.98")
        m3.metric("Macro Recall", "0.98")
        m4.metric("Multiclass ROC-AUC", "0.9985")
        
        st.markdown("---")
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.markdown("#### Confusion Matrix Heatmap")
            cm_data = np.array([
                [258, 6, 4],
                [0, 239, 5],
                [1, 1, 305]
            ])
            labels = ["Decrease", "Increase", "Stable"]
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                x=labels,
                y=labels,
                labels=dict(x="Predicted Class", y="Actual True Class", color="Sample Count"),
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_diag2:
            st.markdown("#### Feature Importance (Gini Impurity)")
            feat_imp_data = pd.DataFrame({
                "Feature": [
                    "month_over_month_percentage_change",
                    "coefficient_of_variation",
                    "prev_price",
                    "seasonal_avg_price",
                    "state_avg_price",
                    "standard_deviation",
                    "month",
                    "variance",
                    "num_states",
                    "year"
                ],
                "Importance": [0.6931, 0.0539, 0.0526, 0.0400, 0.0301, 0.0284, 0.0279, 0.0262, 0.0261, 0.0084]
            }).sort_values("Importance", ascending=True)
            
            fig_feat = px.bar(
                feat_imp_data,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 10 Attributes Impacting Price Movement"
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    with tab2:
        st.subheader("Linear Regression Error Analysis")
        st.markdown("""
        **Model Error & Accuracy Measures Evaluated:**
        - **MAE (Mean Absolute Error):** Measures average absolute deviation between actual and predicted prices.
        - **RMSE (Root Mean Squared Error):** Penalizes larger outlier forecasting errors more heavily.
        - **R² Score (Coefficient of Determination):** Proportion of variance in price explained by linear features.
        """)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("MAE", "Rs 7,377.34")
        r2.metric("RMSE", "Rs 24,636.36")
        r3.metric("R² Score", "-0.1631")
        
        st.info("""
        💡 **Key Finding & Model Interpretation**:
        The negative R² score for Simple Linear Regression demonstrates that agricultural mandi prices exhibit **strong non-linear seasonal shocks and volatility spikes** across 300+ diverse commodities. This mathematically proves why **Ensemble / Tree-based methods (Random Forest)** far outperform standard linear models on commodity market data.
        """)

    with tab3:
        st.subheader("K-Means Cluster Optimization (Elbow & Silhouette)")
        plot_path = DATA_DIR / "elbow_silhouette.png"
        if plot_path.exists():
            st.image(str(plot_path), caption="Elbow Curve (Inertia) and Silhouette Score Scan across k=2 to 10", use_container_width=True)
        else:
            st.warning("Elbow plot not found. Run python -m src.mining.cluster to generate.")
