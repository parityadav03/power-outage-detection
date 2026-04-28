# ⚡ Power Outage Detection System

> End-to-end Data Mining & Machine Learning pipeline for detecting 
> power outages from smart meter time-series data.  
> **XGBoost Champion — F1: 0.9935 | ROC-AUC: 1.0000**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion-green)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Overview

This project builds a complete power outage detection system on the 
**Tetouan City Power Consumption dataset** (52,416 records, 3 distribution 
zones, January–December 2017). It covers the full Data Mining pipeline:
EDA → Feature Engineering → Model Training → Evaluation → Deployment.

### Key Results

| Model | Type | F1 Score | ROC-AUC |
|---|---|---|---|
| **XGBoost** ⭐ | Supervised | **0.9935** | **1.0000** |
| Neural Network | Supervised | 0.9212 | 0.9999 |
| LSTM Autoencoder | Unsupervised | 0.0406 | N/A |
| Isolation Forest | Unsupervised | 0.0007 | N/A |

- ✅ **381/381** outages detected correctly
- ✅ Only **5 false alarms** across 52,416 records  
- ✅ **39 features** engineered from 8 raw columns
- ✅ **20 training runs** across 4 neural network architectures
- ✅ **SHAP explainability** for individual predictions
- ✅ **Live Streamlit dashboard** with real-time prediction

---

## 🗂️ Project Structure
power-outage-detection/
│
├── 📓 Notebooks
│   ├── day1_eda.ipynb                  # EDA & visualizations
│   ├── day2_features.ipynb             # Feature engineering
│   ├── day3_heavy.ipynb                # Heavy model training
│   └── day4_isolation_forest.ipynb     # Unsupervised + SHAP
│
├── 🌐 Dashboard
│   └── app.py                          # Streamlit dashboard
│
├── 📊 Plots (300 DPI)
│   ├── day4_plot1_contamination_tuning.png
│   ├── day4_plot3_supervised_vs_unsupervised.png
│   ├── day4_plot5_shap_bar.png
│   └── ...                             # 10 total plots
│
├── 💾 Models (saved artifacts)
│   ├── xgb_model_native.json           # XGBoost champion
│   ├── model_isolation_forest.pkl      # Isolation Forest
│   ├── lstm_ae_results.pkl             # LSTM AE results
│   └── scaler_hard.pkl                 # Feature scaler
│
├── 📋 Data
│   └── features_day2.csv               # Engineered feature matrix
│
├── .gitignore
├── requirements.txt
└── README.md

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/power-outage-detection.git
cd power-outage-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run notebooks in order
day1_eda.ipynb → day2_features.ipynb → day3_heavy.ipynb → day4_isolation_forest.ipynb

---

## 🔬 Methodology

### Feature Engineering
39 features engineered from 8 raw columns across 6 categories:

| Category | Features | Count |
|---|---|---|
| Rolling Statistics | Mean/std/min/max over 2hr, 24hr, 48hr windows | 12 |
| Lag & Diff | Lag 1/6/144 steps, rate-of-change | 6 |
| Z-Score | 24hr z-score, z-score momentum | 2 |
| Time Features | Hour, day, month, is_weekend, is_night | 5 |
| Interaction | Zone ratios, weather×power products | 9 |
| Raw Features | Zone 1/2/3, temperature, humidity, wind | 5 |

### Training Pipeline
- **Architecture Tournament**: 4 NN architectures × 5-fold CV = 20 runs
- **XGBoost Search**: RandomizedSearchCV (30 combos × 5-fold = 150 fits)
- **XGBoost Final**: 2000-tree ceiling with early stopping at round 164
- **LSTM Autoencoder**: 60 epochs, 6.5 minutes, best val loss: 0.002963
- **Isolation Forest**: 5 contamination values tuned (0.001 → 0.05)

### SHAP Analysis
Top features by mean |SHAP| value:
1. `z1_zscore` — 8.456 (primary anomaly signal)
2. `z1_rolling_std_w144` — 0.551 (24hr volatility)
3. `z1_diff_144` — 0.464 (rate of change vs yesterday)
4. `z1_diff_6` — 0.435 (1-hour rate of change)

---

## 📊 EDA Findings

- **Peak outage hour**: 6:00 AM (early-morning load switching)
- **Peak outage day**: Thursday
- **Peak outage month**: February (winter demand surge)
- **Outage rate**: 0.727% — severe class imbalance (137:1)
- **Zone correlation**: Zone 1 & Zone 2 negative (r = −0.460)

---

## 🖥️ Dashboard Features

- 📈 Live time-series with detected outages highlighted
- 🎚️ Adjustable detection threshold slider
- 🔀 Switch between XGBoost, NN, Isolation Forest, LSTM AE
- 🔮 Live prediction form — input sensor readings, get probability + gauge
- 📅 Date range filter for time-window inspection
- 📊 Detection breakdown: confusion pie, hourly bar, probability distribution

---

## 🧠 Key Finding

The performance gap between supervised (F1: 0.9935) and unsupervised 
(F1: 0.0406) models reveals that **label availability is the single most 
important factor** in outage detection performance on severely imbalanced 
time-series data. Unsupervised models serve as practical first-pass 
screening tools when labelling is expensive or infeasible.

---

## 📦 Requirements
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
tensorflow>=2.13
streamlit>=1.28
plotly>=5.17
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
imbalanced-learn>=0.11
scipy>=1.11