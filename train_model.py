
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path("data/customer_churn.csv.csv")
MODEL_PATH = Path("customer_churn_model.pkl")
METRICS_PATH = Path("metrics.json")

TARGET = "Churn"

NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = FEATURES + [TARGET]


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)
    data.columns = data.columns.str.strip()

    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove completely blank rows without modifying the source CSV.
    data = data.dropna(how="all").copy()

    # Ensure the target contains only 0 or 1.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1.")

    data[TARGET] = data[TARGET].astype(int)

    if data[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both Churn classes: 0 and 1.")

    return data


def build_model():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=500)),
    ])


def train_model():
    print("Loading customer churn dataset...")
    data = load_dataset()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Records after cleaning:", len(data))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training customer churn model...")

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "LogisticRegression",
        "dataset": str(DATASET_PATH),
        "records_after_cleaning": int(len(data)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "features": FEATURES,
        "target": TARGET,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()
