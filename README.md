# 📦 Vendor Invoice Intelligence System 💡

An end-to-end machine learning system that automates two critical finance operations: predicting freight costs for vendor invoices 🚛 and flagging anomalous invoices for manual review 🚩. Built using Python, Scikit-learn, SQLite, and deployed via a Streamlit web application.

---

## 🧩 Problem Statement

Finance and procurement teams spend significant time manually reviewing vendor invoices for two recurring problems:

1. **Unpredictable freight costs** 🚛 — making it difficult to budget, forecast, and negotiate with vendors.
2. **Invoice anomalies** 🚩 — pricing deviations, freight overcharges, and delivery delays that indicate errors, disputes, or compliance risks. Reviewing every invoice manually does not scale.

This system solves both problems with ML models that automate cost forecasting and risk flagging, allowing finance teams to focus human attention only where it matters most. 🎯

---

## 🗂️ Project Structure

```
Vendor_Invoice_Intelligence/
│
├── data/
│   └── inventory.db                     ← SQLite database (purchases + vendor_invoice tables)
│
├── Freight_Cost_Prediction/
│   ├── notebooks/
│   │   └── Predicting_Freight_Cost.ipynb   ← EDA, modelling, evaluation
│   ├── data_preprocessing.py               ← Data loading, feature prep, train-test split
│   ├── modeling_evaluation.py              ← Train Linear, Decision Tree, Random Forest regressors
│   ├── train.py                            ← Pipeline: load → train → evaluate → save best model
│   └── models/
│       └── predict_freight_model.pkl       ← Saved best model (Linear Regression)
│
├── Invoice_Flagging/
│   ├── notebooks/
│   │   └── Invoice_Flagging.ipynb          ← EDA, feature engineering, model comparison
│   ├── data_preprocessing.py               ← SQL join query, risk label creation, scaling
│   ├── modeling_evaluation.py              ← Random Forest with GridSearchCV (5-fold CV)
│   ├── train.py                            ← Pipeline: load → label → scale → train → save
│   └── models/
│       ├── predict_flag_invoice.pkl        ← Saved best classifier
│       └── scaler.pkl                      ← StandardScaler for inference
│
├── Inference/
│   ├── freight_predict.py                  ← Load model + predict freight cost
│   └── invoice_flagging_prediction.py      ← Load model + predict invoice flag
│
└── app.py                                  ← Streamlit web application 🌐
```

---

## 🚛 Module 1 — Freight Cost Prediction

### 🎯 Objective

Predict the freight cost for a vendor invoice given the invoice dollar value, to improve cost forecasting and vendor negotiation.

### 📊 Data

* Source: `vendor_invoice` table from SQLite database
* Feature used: `Dollars` (invoice value)
* Target: `Freight` (actual freight cost charged)

### ⚙️ Approach

Three regression models were trained and evaluated:

| Model                   | MAE   | RMSE   | R²         |
| ----------------------- | ----- | ------ | ---------- |
| Linear Regression       | 24.11 | 124.72 | **96.99%** |
| Decision Tree Regressor | 32.65 | 163.74 | 94.81%     |
| Random Forest Regressor | 28.27 | 142.21 | 96.08%     |

**🏆 Best model: Linear Regression** — selected automatically based on lowest MAE and saved via `joblib`.

### 📁 Key files

* `data_preprocessing.py` — connects to SQLite, extracts features and target, splits 80/20
* `modeling_evaluation.py` — trains all three models, returns MAE/RMSE/R² metrics
* `train.py` — orchestrates the full pipeline and saves the best model

---

## 🚩 Module 2 — Invoice Anomaly Flagging

### 🎯 Objective

Classify each vendor invoice as requiring manual approval (flag = 1) or safe for auto-processing (flag = 0), based on financial and delivery pattern anomalies.

### 📊 Data

* Source: SQL join between `vendor_invoice` and `purchases` tables
* Features engineered via aggregation:

  * `invoice_quantity`, `invoice_dollars`, `Freight`
  * `total_quantity`, `total_dollars` (PO-level aggregates)
  * `avg_receiving_delay` (average days between PO date and receiving date)

### 🏷️ Labelling Logic

Invoices were labelled as high-risk (flag = 1) using the following business rules:

* Invoice dollar amount deviates from PO total dollars by more than $5 💰
* Average receiving delay for that PO exceeds 10 days ⏳

### ⚙️ Approach

* StandardScaler applied to all features before training
* Multiple classifiers benchmarked: Logistic Regression (74% accuracy), Decision Tree (66%), Random Forest (65%)
* Final model: **Random Forest Classifier with GridSearchCV** — 5-fold cross-validation over 216 hyperparameter combinations, optimized for F1-score to handle class imbalance

| Metric              | Value |
| ------------------- | ----- |
| Accuracy            | 64%   |
| Precision (flagged) | 0.49  |
| Recall (flagged)    | 0.83  |
| F1-score (flagged)  | 0.62  |

> ⚠️ The model is optimized for high recall on flagged invoices — it is better to over-flag and review than to miss a risky invoice.

### 📁 Key files

* `data_preprocessing.py` — SQL join query with PO-level aggregations, label creation, StandardScaler
* `modeling_evaluation.py` — GridSearchCV with Random Forest, classification report
* `train.py` — end-to-end pipeline, saves model and scaler

---

## 🌐 Streamlit Application

A two-module web portal that allows finance teams to get predictions in real time without writing any code. ⚡

**🚛 Module 1 — Freight Cost Prediction**

* Input: Invoice dollar amount
* Output: Predicted freight cost in USD 💰

**🚩 Module 2 — Invoice Risk Flagging**

* Input: Invoice quantity, invoice dollars, freight, total item quantity, total item dollars
* Output: Safe for auto-approval ✅ OR requires manual review 🚨

### ▶️ Running the app

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn joblib plotly

# Run
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Layer             | Tools                                                                        |
| ----------------- | ---------------------------------------------------------------------------- |
| Data storage      | SQLite                                                                       |
| Data processing   | Pandas, NumPy                                                                |
| Machine learning  | Scikit-learn (Linear Regression, Decision Tree, Random Forest, GridSearchCV) |
| Model persistence | Joblib                                                                       |
| Web application   | Streamlit                                                                    |
| Visualization     | Matplotlib, Seaborn, Plotly                                                  |
| Language          | Python 3.10                                                                  |

---

## 💼 Business Impact

* Automates freight cost estimation for procurement budgeting and vendor negotiations 📊
* Reduces time spent on manual invoice review by routing only high-risk invoices to the finance team ⚡
* Provides a scalable, real-time decision support tool for finance operations via a no-code Streamlit interface 🌐

---

## 👨‍💻 Author

Built as part of a data science portfolio project focused on real-world finance and procurement automation using classical ML and Python 🚀
