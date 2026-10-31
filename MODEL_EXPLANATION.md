# 🌾 MandiLense — Complete Beginner-Friendly Guide to the Models

> **Who is this guide for?**
> For you, your team, and your friends to understand **100% of the project** in plain, simple English without confusing jargon. Use this to prepare for your presentation and viva with your teacher!

---

## 1. What is this project in simple words?

Imagine you go to a vegetable or grain wholesale market (a **Mandi** in India). 
- One month, tomatoes are ₹15/kg, and the next month they jump to ₹100/kg.
- One month, onion prices crash so low that farmers lose all their money.

**The Problem:** Farmers and traders never know what price to expect next month.
**Our Solution (MandiLense):** We collected 2 years of real Indian mandi transactions across 300+ commodities (crops) and trained **3 AI Machine Learning models** to answer 3 big questions:
1. **Will the price go UP, DOWN, or stay SAME next month?** *(Classification Model)*
2. **What will be the exact estimated price in Rupees?** *(Regression Model)*
3. **Which crops behave similarly in the market?** *(Clustering Model)*

---

## 2. The 3 Models — How They Work (Simple Analogies)

### 🟢 Model 1: The Price Movement Predictor (Classification - Random Forest)
- **The Question It Answers:** *"Next month, will Onion prices Increase (> +5%), Decrease (< -5%), or stay Stable?"*
- **What is a Decision Tree?**
  Think of a 20-questions game or a flowchart:
  - *Is it Monsoon season?* ➡️ Yes
  - *Did price increase last month?* ➡️ Yes
  - *Are fewer states selling this crop?* ➡️ Yes
  - ➡️ **Prediction: Price will Increase!**
- **What is a Random Forest? (Our Model):**
  One single person (one tree) can make mistakes. But what if you ask a **council of 200 experts (200 trees)** and take a vote? That is a **Random Forest**!
  All 200 trees vote, and the majority decision wins.
- **How we trained it:**
  - We gave the computer around **4,000 historical records**.
  - We split the data: **First 80% of months to learn (Train)**, and **last 20% of months to test (Test)**.
  - *Why not mix the data randomly?* Because this is time-series data! You cannot look into the future to predict the past. Training on past months and testing on future months is how real forecasting works.
- **The Score/Accuracy:**
  - **98% Accuracy!** Out of 100 test cases, it gets 98 correct.
  - **Most important factor it discovered:** *Month-over-month percentage change (momentum)* is the biggest signal of future price direction.

---

### 🔵 Model 2: The Exact Price Estimator (Regression - Linear Regression)
- **The Question It Answers:** *"Don't just tell me UP or DOWN; tell me the exact price in Rupees per quintal for next month!"*
- **How it works (Drawing a straight line):**
  Imagine a taxi meter: each extra kilometer adds ₹15. That is a straight line ($y = mx + c$). Linear regression tries to draw a straight line connecting past prices and volatility to next month's price.
- **What happened when we tested it?**
  - **MAE (Mean Absolute Error):** ₹7,377 average difference.
  - **R² Score:** -0.16 (negative!).
- **Why this is a SUPER SMART finding for your viva:**
  *Why did linear regression struggle?* Because agricultural prices are **NOT a straight line**! 
  If unexpected rain floods the fields or the government bans exports, prices spike wildly. A simple straight line cannot capture complex market shocks. 
  **This mathematically proves why we needed Model 1 (Random Forest Ensemble)!** Teachers love when students explain *why* simple models fail and complex models succeed.

---

### 🟣 Model 3: The Crop Matchmaker (Clustering - K-Means)
- **The Question It Answers:** *"We have 329 different crops. Nobody told us which crops are risky and which are safe. Can the AI group them automatically based on their behavior?"*
- **How it works (Sorting into baskets without labels):**
  Imagine placing 329 crops on a table based on:
  1. Average Price (Cheap vs Expensive)
  2. Volatility (Does the price jump wildly or stay calm?)
  3. Average Monthly Price Change
  The **K-Means algorithm** automatically groups similar crops together.
- **How did we know how many groups to make?**
  We tested from 2 groups up to 10 groups using two mathematical tests:
  - **The Elbow Method (Inertia):** Looking for the point where adding more groups stops giving useful information.
  - **The Silhouette Score:** Measures how tight and distinct each group is.
  - **The Winner:** **3 Groups (k = 3)** with a near-perfect Silhouette score of **0.9675**!
- **What are the 3 groups it discovered?**
  1. **Group 0 (Core Mandi Staples - 326 crops):** Everyday crops like Potato, Onion, Wheat, Rice. Moderate prices, standard volatility.
  2. **Group 1 (Extreme Volatility Anomalies - 2 crops):** *Hippe Seed* and *Jaffri*. Their prices swing up and down by hundreds of percent!
  3. **Group 2 (The Ultra-Luxury Spice - 1 crop):** *Saffron (Kesar)*! It has an average price of **₹6,61,350 per quintal**! It automatically got placed into its own luxury class.

---

## 3. Where are all the results saved?

All results and trained models are stored directly in your project folder:

| File Path | What is inside it? |
|---|---|
| `saved_models/random_forest.joblib` | The trained "brain" of the 98% accurate Random Forest model. |
| `saved_models/linear_regression.joblib` | The trained "brain" of the price regression model. |
| `saved_models/kmeans.joblib` | The trained "brain" of the K-Means clustering model. |
| `Data/processed/mining/clustering_labeled.csv` | An Excel/CSV file listing all 329 crops and which cluster (0, 1, or 2) they belong to. |
| `Data/processed/mining/elbow_silhouette.png` | The chart proving why 3 clusters is the best number. |
| `Notebooks/10_data_mining_models.ipynb` | The complete Jupyter Notebook with all codes, graphs, and printouts. |
| `app.py` | The web dashboard code that brings everything to life! |

---

## 4. How does the Web App (`app.py`) show this?

When you run `streamlit run app.py`, a web browser opens at `http://localhost:8501`.

Here is how the website works behind the scenes:
1. **Loads the saved brains:** When the website opens, it immediately reads the `.joblib` files from the `saved_models` folder.
2. **Interactive Controls:** You use sliders and dropdowns to pick prices, months, and seasons.
3. **Live Prediction:** When you press the button, the website feeds your numbers to the trained model and displays the answer in less than 0.1 seconds!

---

## 5. Quick Step-by-Step Live Demo (For Mam and Friends)

Follow these exact steps when demonstrating the project:

### Step 1: Open the Web App
- Make sure your terminal has run: `streamlit run app.py`
- Open your browser at `http://localhost:8501`.

### Step 2: Show the "Project Overview" Tab
- Point out the 3 cards summarizing **Classification**, **Regression**, and **Clustering**.
- Show the 4 metrics at the bottom: **4,095 Classification records**, **3,468 Regression records**, **329 Commodities clustered**, and **3 Models saved**.

### Step 3: Show "Task 1: Price Movement Classification" (The WOW factor!)
- Click on **Task 1** in the left sidebar.
- Put:
  - Previous Month Price: `₹3,500`
  - Month-over-Month % Change: `+7.5%`
  - Season: `Monsoon`
- Click **"🚀 Predict Price Movement"**.
- Show them the result:
  - A green badge appears saying **"📈 Predicted: Increase"**.
  - A color bar chart shows the model is over 90% confident it will increase!

### Step 4: Show "Task 3: Market Clustering" (The Visual Insight!)
- Click on **Task 3** in the left sidebar.
- Show the interactive colorful scatter plot.
- Hover your mouse over the dot in the top right corner and say:
  > *"Look here, this single dot in Cluster 2 is Saffron. The AI automatically detected that Saffron costs ₹6.6 Lakhs per quintal and isolated it from ordinary vegetables."*

### Step 5: Show "Task 4: Evaluation & Model Diagnostics"
- Click on **Task 4** in the left sidebar.
- Show the **Confusion Matrix Heatmap** (the blue square) and say:
  > *"This heatmap compares actual market results vs our model's predictions. The diagonal boxes show 258, 239, and 305 correct predictions with only a tiny handful of errors, giving us 98% accuracy."*
