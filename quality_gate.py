
import json
import sys
from pathlib import Path

MINIMUM_ACCURACY = 0.85
METRICS_PATH = Path("metrics.json")


def main():
    print("Reading customer churn model metrics...")

    if not METRICS_PATH.is_file():
        print("QUALITY GATE FAILED")
        print("metrics.json was not found.")
        return 1

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        accuracy = float(metrics["accuracy"])

        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("Accuracy must be between 0 and 1.")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print("QUALITY GATE FAILED")
        print(f"Invalid metrics: {error}")
        return 1

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Required Accuracy: {MINIMUM_ACCURACY:.2f}")

    if accuracy < MINIMUM_ACCURACY:
        print("QUALITY GATE FAILED")
        print("Model accuracy is below the required threshold.")
        return 1

    print("QUALITY GATE PASSED")
    print("Customer churn model meets the accuracy requirement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
