
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("customer_churn_model.pkl")

FEATURES = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "customer_churn_model.pkl was not found. "
            "Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "customer-churn-prediction",
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature] for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        return jsonify({
            "prediction": (
                "CHURN" if prediction_code == 1 else "NO_CHURN"
            ),
            "prediction_code": prediction_code,
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": f"Invalid prediction input: {exc}"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
