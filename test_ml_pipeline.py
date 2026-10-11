
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "customer_churn.csv.csv"
MODEL = ROOT / "customer_churn_model.pkl"
METRICS = ROOT / "metrics.json"

FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Gender",
    "Subscription Type",
    "Contract Length",
]


class TestMLPipeline(unittest.TestCase):

    def test_source_dataset_exists(self):
        self.assertTrue(DATASET.is_file())

    def test_model_was_saved(self):
        self.assertTrue(MODEL.is_file())

    def test_metrics_were_saved(self):
        self.assertTrue(METRICS.is_file())

    def test_metrics_are_valid(self):
        with METRICS.open(encoding="utf-8") as file:
            metrics = json.load(file)

        self.assertGreater(metrics["records_after_cleaning"], 0)
        self.assertGreater(metrics["training_records"], 0)
        self.assertGreater(metrics["testing_records"], 0)

        for metric in ("accuracy", "precision", "recall", "f1_score"):
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)

        self.assertEqual(
            metrics["training_records"] + metrics["testing_records"],
            metrics["records_after_cleaning"],
        )

    def test_saved_model_predicts_valid_classes(self):
        model = joblib.load(MODEL)

        sample = pd.DataFrame([{
            "Age": 35,
            "Tenure": 24,
            "Usage Frequency": 15,
            "Support Calls": 2,
            "Payment Delay": 5,
            "Total Spend": 500,
            "Last Interaction": 10,
            "Gender": "Female",
            "Subscription Type": "Standard",
            "Contract Length": "Annual",
        }])

        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [2])


if __name__ == "__main__":
    unittest.main()
