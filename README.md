# 💳 CreditGuard — Credit Scoring & Default Risk Prediction

### Smarter Credit Risk Analysis with Machine Learning

CreditGuard is a Machine Learning project that predicts the likelihood of credit default based on customer financial information, credit limits, repayment history, bill amounts and payment behavior.

The project uses **Python, Pandas, Scikit-learn and Streamlit** to build an interactive credit risk prediction application. It compares multiple classification algorithms and evaluates their performance using important machine learning metrics.

> ⚠️ **Disclaimer:** This is an educational machine learning project. Predictions are estimates based on historical data and must not be used as the sole basis for real lending or credit approval decisions.

---

## ✨ Key Features

- 💳 **Credit Default Prediction:** Predicts whether a customer belongs to the default or non-default class.
- 🤖 **Multiple ML Models:** Compares Logistic Regression and Random Forest.
- 📊 **Interactive Dashboard:** Provides a user-friendly Streamlit interface.
- 📈 **Probability Visualization:** Displays estimated probabilities for both prediction classes.
- 🧹 **Data Preprocessing:** Handles missing values, categorical features and numerical scaling.
- 📉 **Model Evaluation:** Uses Accuracy, Precision, Recall, F1-score and ROC-AUC.
- 📋 **Confusion Matrix:** Helps analyze correct and incorrect predictions.
- 💾 **Saved Model:** Reuses the trained model for future predictions.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data analysis and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Plotly | Interactive probability charts |
| Streamlit | Web application interface |
| Joblib | Saving and loading the trained model |
| UCI Machine Learning Repository | Dataset source |

## 📂 Dataset

This project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

The dataset contains historical customer information, including:

- Credit limit
- Age and demographic category codes
- Repayment status across multiple months
- Bill statement amounts
- Previous payment amounts
- Credit-default outcome for the following month

**Dataset details:**

- Records: 30,000
- Predictive features: 23
- Target: Default payment next month
- Class `0`: No default
- Class `1`: Default

Dataset source: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## 🧠 Machine Learning Workflow

1. **Data Collection:** Download the dataset from UCI.
2. **Exploratory Data Analysis:** Examine dataset structure, missing values, duplicates and target distribution.
3. **Data Preprocessing:** Handle missing values, scale numerical features and encode categorical features.
4. **Train-Test Split:** Split the data into training and testing sets using stratification.
5. **Model Training:** Train Logistic Regression and Random Forest classifiers.
6. **Model Comparison:** Compare classification metrics and ROC-AUC.
7. **Model Saving:** Save the selected pipeline with Joblib.
8. **Prediction:** Use the saved model through the Streamlit application.

## 🌳 Models Used

### 1. Logistic Regression

A classification algorithm used as a baseline model for estimating credit-default risk.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to identify