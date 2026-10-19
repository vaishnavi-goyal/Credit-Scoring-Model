# 💳 CreditGuard — Credit Scoring & Default Risk Prediction

### Smarter Credit Risk Analysis with Machine Learning


CreditGuard is a Machine Learning project that predicts the likelihood of credit default based on customer financial information, credit limits, repayment history, bill amounts, and payment behavior.

The project uses **Python, Pandas, Scikit-learn, and Streamlit** to build an interactive credit risk prediction application. It compares multiple classification algorithms and evaluates their performance using important machine learning metrics.

> ⚠️ **Disclaimer:** This is an educational machine learning project. Predictions are estimates based on historical data and must not be used as the sole basis for real lending or credit approval decisions.

---

## 🔗 Project Links

- **Live Streamlit Application:** [Launch CreditGuard](https://creditguard-vaishnavi.streamlit.app/)
- **GitHub Repository:** [View Source Code](https://github.com/vaishnavi-goyal/Credit-Scoring-Model)
- **Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## ✨ Key Features

- 💳 **Credit Default Prediction:** Predicts whether a customer belongs to the default or non-default class.
- 🤖 **Multiple ML Models:** Compares Logistic Regression and Random Forest.
- 📊 **Interactive Dashboard:** Provides a user-friendly Streamlit interface.
- 📈 **Probability Visualization:** Displays estimated probabilities for both prediction classes.
- 🧹 **Data Preprocessing:** Handles missing values, categorical features, and numerical scaling where configured in the training pipeline.
- 📉 **Model Evaluation:** Uses Accuracy, Precision, Recall, F1-score, and ROC-AUC.
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

### Dataset Details

- **Records:** 30,000
- **Predictive features:** 23
- **Target:** Default payment next month
- **Class 0:** No default
- **Class 1:** Default

**Dataset source:** [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## 🧠 Machine Learning Workflow

1. **Data Collection:** Download the dataset from UCI.
2. **Exploratory Data Analysis:** Examine dataset structure, missing values, duplicates, and target distribution.
3. **Data Preprocessing:** Handle missing values, scale numerical features, and encode categorical features as configured.
4. **Train-Test Split:** Split the data into training and testing sets using stratification.
5. **Model Training:** Train Logistic Regression and Random Forest classifiers.
6. **Model Comparison:** Compare classification metrics and ROC-AUC.
7. **Model Saving:** Save the selected model pipeline with Joblib.
8. **Prediction:** Use the saved model through the Streamlit application.

## 🤖 Models Used

### 1. Logistic Regression

Logistic Regression is a classification algorithm used as a baseline model for estimating the likelihood of credit default. It models the relationship between customer features and the probability of belonging to a particular class.

### 2. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It can capture complex relationships between customer financial features and credit-default outcomes.

### Model Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many predicted default cases are actual default cases.
- **Recall:** Measures how many actual default cases are correctly identified.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes.
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.

The best-performing model should be selected based on the evaluation results and the project's risk priorities, rather than accuracy alone.

## 📁 Project Structure

```text
Credit-Scoring-Model/
│
├── data/
│   ├── credit_default.csv
│   └── target_distribution.png
│
├── models/
│   ├── credit_scoring_model.joblib
│   └── model_info.json
│
├── results/
│   ├── logistic_regression_confusion_matrix.png
│   ├── model_comparison.csv
│   └── random_forest_confusion_matrix.png
│
├── analyze_data.py
├── app.py
├── download_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: This represents the intended project organization. Your actual GitHub repository may have some files stored directly in the root directory.*

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/vaishnavi-goyal/Credit-Scoring-Model.git
cd Credit-Scoring-Model
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

```bash
python download_dataset.py
```

### Step 5: Analyze the Dataset

```bash
python analyze_data.py
```

### Step 6: Train the Models

```bash
python train_model.py
```

This step trains the models, evaluates their performance, and saves the selected model and evaluation outputs according to the training script's configuration.

### Step 7: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 📊 Results and Visualizations

The project generates evaluation outputs that can be used to compare model performance and understand prediction errors.

- **Target Distribution:** Shows the distribution of default and non-default cases.
- **Model Comparison:** Stores evaluation metrics for the trained classifiers.
- **Confusion Matrices:** Visualize correct and incorrect predictions for each model.
- **Probability Chart:** Displays estimated class probabilities in the Streamlit application.

Actual performance values should be taken from the generated `model_comparison.csv` file rather than assumed in advance.

## 🔮 Future Enhancements

- Hyperparameter tuning to improve model performance.
- Cross-validation for more reliable evaluation.
- Explainable AI using SHAP or other feature-importance techniques.
- Improved handling of class imbalance.
- Enhanced dashboard with additional risk insights.
- Model monitoring and periodic retraining.
- More detailed prediction explanations for users.

## ⚠️ Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Class imbalance may affect the model's ability to identify default cases.
- Historical patterns may not reflect future customer behavior.
- The model may produce false positives and false negatives.
- The application is intended for educational purposes, not as a production-ready lending decision system.

## 👩‍💻 Author

**Vaishnavi Goyal**

BCA Student | Machine Learning & Data Science Enthusiast

- **GitHub:** [vaishnavi-goyal](https://github.com/vaishnavi-goyal)
- **Project Repository:** [Credit-Scoring-Model](https://github.com/vaishnavi-goyal/Credit-Scoring-Model)
- **Live Demo:** [CreditGuard Application](https://creditguard-vaishnavi.streamlit.app/)

---

⭐ If you find this project useful, consider giving the repository a star!
