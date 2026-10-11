import unittest
from pathlib import Path

from churn_data import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "customer_churn.csv.csv"


class TestChurnDataset(unittest.TestCase):

    def test_dataset_is_valid(self):
        errors = validate_dataset(DATASET)

        self.assertEqual(
            errors,
            [],
            f"Dataset validation errors: {errors}"
        )

    def test_missing_file_is_detected(self):
        missing_file = PROJECT_ROOT / "data" / "missing.csv"

        errors = validate_dataset(missing_file)

        self.assertTrue(
            any("File not found" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
