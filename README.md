# Benchmarking Long-Term Cognitive Networks for Football Prediction ⚽📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Thesis%20Research-orange)

**Repository Name:** `LTCN-Football-Prediction-Odds-XAI`

## 📖 Project Overview

This repository contains the source code and methodology for the Master's Thesis: **"Benchmarking Long-Term Cognitive Networks for Predicting Football Match Outcomes with Perspectives from Market Odds and Model Explainability"**.

The primary objective of this research is to benchmark the performance of **Long-Term Cognitive Networks (LTCN)** a biologically inspired recurrent neural network adapted for tabular classification against state of the art Gradient Boosting Machine (GBM) models in the domain of football match forecasting (Home/Draw/Away).

Unlike traditional approaches, this project implements an **"Odds-Plus"** strategy, integrating technical match statistics with betting market implied probabilities. Furthermore, it places a heavy emphasis on **Explainable AI (XAI)**, utilizing extensive ablation studies and genetic algorithms to decode model decision-making processes.

## 🧠 The Core Model: Long-Term Cognitive Network (LTCN)

This repository introduces the **Long-Term Cognitive Network (LTCN)** model applied to structured pattern classification problems. Unlike traditional black-box models, LTCN offers a transparent approach to decision-making.

> **Key Architectural Innovations:**
>
> * **Quasi-Nonlinear Reasoning:** Incorporates a specialized rule that allows precise control over the amount of non-linearity in the reasoning mechanism.
> * **Recurrence-Aware Decision Model:** Utilizes a decision model that effectively evades issues posed by unique fixed points common in standard RNNs.
> * **Deterministic Learning:** Introduces a deterministic algorithm to efficiently compute tunable parameters.

*The simulations in this study demonstrate that the LTCN classifier achieves competitive results when compared to state-of-the-art white-box and black-box models.*

## 🤖 Models Used

The study benchmarks a diverse set of machine learning algorithms, ranging from traditional statistical methods to advanced boosting frameworks and custom neural networks.

| Category | Model | Description |
| :--- | :--- | :--- |
| **Deep Learning** | **LTCN** | *Custom Implementation.* A Liquid Time-Constant Network adapted with distinct temporal weights ($W1, W2$) for non-linear tabular relationships. |
| **Boosting** | **XGBoost** | Optimized distributed gradient boosting library. |
| **Boosting** | **LightGBM** | Gradient boosting framework that uses tree based learning algorithms. |
| **Boosting** | **CatBoost** | Gradient boosting with categorical feature support. |
| **Boosting** | **AdaBoost** | Adaptive Boosting with Decision Trees. |
| **Boosting** | **HistGradientBoosting** | Scikit-learn's histogram-based gradient boosting classifier. |
| **Ensemble** | **Random Forest** | An ensemble learning method for classification using multitude of decision trees. |
| **Linear/Stat** | **Logistic Regression** | Baseline statistical model (One-vs-Rest). |
| **Kernel** | **SVM** | Support Vector Machines with RBF kernel. |

*Note: All models are tuned using **Optuna** with Time-Series Cross-Validation to prevent data leakage.*

## 📏 Evaluation Metrics

To ensure a holistic assessment of model performance—covering both classification accuracy and probabilistic calibration—the following metrics are employed:

### Classification Metrics (Hard Predictions)
* **Accuracy:** Overall correctness of the model.
* **F1-Score (Weighted):** Harmonic mean of precision and recall, accounting for class imbalance (Draws are rarer).
* **Cohen's Kappa:** Measures inter-rater agreement, accounting for the possibility of the agreement occurring by chance.

### Probabilistic Metrics (Soft Probabilities)
* **AUC-ROC (One-vs-Rest):** Measures the ability of the classifier to distinguish between classes.
* **Log Loss (Cross-Entropy):** Measures the performance of a classification model where the prediction input is a probability value between 0 and 1.
* **Brier Score:** Measures the mean squared difference between the predicted probability and the actual outcome (Lower is better).
* **Ranked Probability Score (RPS):** A strictly proper scoring rule that validates the distance between the cumulative probability distribution of the forecast and the observation. Critical for ordinal nature of football results (Home > Draw > Away).
* **Expected Calibration Error (ECE):** Measures how well the predicted probabilities correspond to the actual observed frequencies.

## 🚀 Key Features

* **Multi-Source Data Integration:** Merges In-game stats, ELO Ratings, and Transfermarkt financial data using advanced fuzzy matching.
* **Advanced Feature Engineering:** Leakage-free lag feature generation and log-transformations for skewed financial data.
* **Smart Feature Selection:** Hybrid approach using Correlation analysis and Mutual Information (MI) scores.
* **Rigorous Validation:** Pre-SMOTE temporal splitting ensures validation sets contain zero synthetic samples, preventing overfitting.
* **Explainable AI (XAI) Suite:** Comprehensive comparison of **SHAP**, **Permutation Feature Importance (PFI)**, **Predictive Mutual Information (PMI)**, and **SOFI (Sensitivity-based via Genetic Algorithms)**.

## 📂 Data Sources

This research integrates data from distinct high-quality sources:

1.  **Match Results, Statistics & Odds:** Historical match data, technical statistics, and betting odds were sourced from the **[Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)** repository by *xgabora*.
2.  **Market Values & Financials:** Player market values and club financial data were generously provided by the **Transfermarkt Data Science Team**.

## 🛠️ Methodology & Pipeline

The system (`FOOTBALL MATCH PREDICTION SYSTEM v18.0`) operates in sequential stages:

1.  **Data Ingestion & Fuzzy Matching:** Aligning team names across different sources.
2.  **Feature Engineering:** Rolling averages, differential features (Home vs Away), and lag creation.
3.  **Preprocessing:** Feature selection and targeted SMOTE application within the CV loop.
4.  **Training:** Hyperparameter optimization via Optuna.
5.  **Analysis:** Generation of thesis-compliant vector graphics (PDF) and performance tables.

## 📈 Results & Visualization

The system automatically generates comprehensive visualizations suitable for academic publishing:

* **G1-G12:** Comparative charts for Accuracy, F1, AUC, and Kappa.
* **G18:** Feature Importance plots (SHAP summary, Aggregated Top-10).
* **G20:** Radar charts for probabilistic performance.
* **G28:** XAI Method Comparison (Ablation Curves).
* **G29:** Cumulative Importance Distribution.

## 🙏 Acknowledgments

* **Prof. Gonzalo Nápoles** for supervision and guidance.
* **Transfermarkt Data Science Team** for providing the market value data.
* **xgabora** for maintaining the open-source match database.

---

### ⚠️ Disclaimer
This repository is for academic research purposes. The betting odds data and market values are used for predictive modeling analysis and do not constitute financial advice.
