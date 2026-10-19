
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("data/credit_default.csv")

target = "default_payment_next_month"

# Remove rows without a known target
df = df.dropna(subset=[target]).copy()

X = df.drop(columns=[target])
y = df[target].astype(int)

# Remove customer ID if present; it is not a useful predictor
X = X.drop(columns=["ID"], errors="ignore")

# Identify categorical columns
categorical_cols = [
    col for col in [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]
    if col in X.columns
]

numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

# 2. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 4. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []
trained_pipelines = {}

# 5. Train and evaluate both models
for name, classifier in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    results.append(metrics)
    trained_pipelines[name] = pipeline

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No Default", "Default"],
        cmap="Blues"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(
        f"results/{safe_name}_confusion_matrix.png",
        dpi=150
    )
    plt.close()

# 6. Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="ROC-AUC", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("results/model_comparison.csv", index=False)

# 7. Save the model with the highest ROC-AUC
best_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_name]

joblib.dump(
    best_pipeline,
    "models/credit_scoring_model.joblib",
    compress=3
)

with open("models/model_info.json", "w") as file:
    json.dump({
        "model_name": best_name,
        "target": target,
        "features": list(X.columns)
    }, file, indent=4)

print(f"\nBest model by ROC-AUC: {best_name}")
print("Saved model: models/credit_scoring_model.joblib")
print("Results saved in the results folder.")
