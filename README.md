 Benchmarking Long-Term Cognitive Networks for Football Prediction ⚽📊

**Repository Name:** `LTCN-Football-Prediction-Odds-XAI`

This repository contains the source code and methodology for the Master's Thesis: **"Benchmarking Long-Term Cognitive Networks for Predicting Football Match Outcomes with Perspectives from Market Odds and Model Explainability"**.

This project implements a robust machine learning pipeline to predict football match results (Home/Draw/Away) across top 5 European leagues, featuring a custom **Long-Term Cognitive Network (LTCN)** model alongside state-of-the-art gradient boosting algorithms.

## 🚀 Key Features

  * **Custom LTCN Implementation:** A liquid time-constant network adapted for tabular classification tasks.
  * **Multi-Source Data Integration:**
      * Matches & Technical Stats (In-game statistics)
      * ELO Ratings (Historical team strength)
      * **Transfermarkt Data:** Advanced fuzzy matching algorithm to integrate player market values and financial data.
  * **Advanced Feature Engineering:**
      * "Odds-Plus" strategy combining betting market implied probabilities with technical stats.
      * Leakage-free lag feature generation (Time-series safe).
      * EDA-guided feature selection (Mutual Information & Diff-features).
  * **Rigorous Validation:**
      * Time-Series Cross-Validation to prevent look-ahead bias.
      * Pre-SMOTE temporal splitting for validation sets.
  * **Explainable AI (XAI) Suite:**
      * Comparison of 6 importance methods: **SHAP**, **Permutation Feature Importance (PFI)**, **Predictive Mutual Information (PMI)**, **SOFI (Sensitivity-based, via Genetic Algorithms)**, and model-specific methods.
      * Ablation studies to verify feature importance rankings.

## 🛠️ Methodology & Pipeline

The system (`FOOTBALL MATCH PREDICTION SYSTEM v18.0`) operates in sequential stages:

1.  **Data Ingestion & Fuzzy Matching:** Merges dataset using `rapidfuzz` to align team names across different sources.
2.  **Feature Engineering:** Calculates rolling averages, log-transforms skewed financial data, and creates differential features (Home vs Away).
3.  **Preprocessing:**
      * **Smart Feature Selection:** Hybrid approach using Correlation and Mutual Information.
      * **Class Balancing:** Targeted SMOTE application within the Cross-Validation loop (avoiding data leakage).
4.  **Model Training & Tuning:**
      * Hyperparameter optimization using **Optuna**.
      * Models: **LTCN**, XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, SVM, AdaBoost.
5.  **Evaluation:**
      * Metrics: F1-Score (Weighted), Accuracy, AUC-ROC, Brier Score, Ranked Probability Score (RPS), Cohen's Kappa.
      * Graphics: Thesis-compliant vector plots (PDF) generated via `matplotlib` and `seaborn`.

## 📦 Installation

Cloning the repository and installing dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/LTCN-Football-Prediction-Odds-XAI.git
cd LTCN-Football-Prediction-Odds-XAI

# It is recommended to use a virtual environment
pip install -r requirements.txt
```

*Note: The script automatically checks and installs required libraries (pandas, numpy, scikit-learn, optuna, shap, xgboost, lightgbm, catboost, rapidfuzz, pygad, etc.) upon execution if running in a Colab/Notebook environment.*

## 📂 Project Structure

```
├── data/
│   ├── Matches.csv             # Historical match data
│   ├── EloRatings.csv          # Team ELO ratings
│   └── data.xlsx               # Transfermarkt market values
├── outputs/
│   ├── graphics/               # Generated plots (PDF/PNG)
│   ├── tables/                 # Performance tables (CSV/PNG)
│   └── results.json            # Final metrics
├── src/
│   └── main_model_v18.py       # Main execution script
├── .gitignore
└── README.md
```

## 📊 Models & Configuration

The system is highly configurable via the `CONFIG` dictionary within the script:

```python
CONFIG = {
    "filters": {"leagues": ["E0", "D1", "I1", "SP1", "F1"], "min_date": "2015-07-01"},
    "optuna": {"n_trials": 20, "timeout": 2000},
    "class_balancing": {"use_smote": True, "strategy": "targeted"},
    "xai": {"use_shap": True, "pfi_n_repeats": 5},
    # ...
}
```

### The LTCN Model

This project benchmarks a **Long-Term Cognitive Network**, a type of recurrent neural network simplified for tabular tasks, defined in the `LTCN` class. It utilizes a sigmoid/tanh activation function with distinct temporal weights ($W1, W2$) to model non-linear relationships in football data.

## 📈 Results & Visualization

The system automatically generates comprehensive visualizations suitable for academic publishing:

  * **G1-G12:** Comparative charts for Accuracy, F1, AUC, and Kappa.
  * **G18:** Feature Importance plots (SHAP summary, Aggregated Top-10).
  * **G20:** Radar charts for probabilistic performance (Brier, RPS, ECE).
  * **G28:** XAI Method Comparison (Ablation Curves).
  * **G29:** Cumulative Importance Distribution.

## 🙏 Acknowledgments

  * **Prof. Gonzalo Nápoles** for supervision and guidance.
  * **Transfermarkt Data Science Team** for providing the market value data.
  * Open-source libraries: `scikit-learn`, `shap`, `optuna`, `pygad`, and gradient boosting frameworks.

-----

### ⚠️ Disclaimer

This repository is for academic research purposes. The betting odds data and market values are used for predictive modeling analysis and do not constitute financial advice.

```
```
