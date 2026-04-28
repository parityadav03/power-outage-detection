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