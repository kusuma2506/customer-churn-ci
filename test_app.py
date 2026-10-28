
import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        root_response = self.client.get("/")
    
        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(root_response.get_json()["status"], "ok")
        self.assertEqual(
            root_response.get_json()["service"],
            "customer-churn-prediction",
        )
    
        health_response = self.client.get("/health")
    
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(
            health_response.get_json()["status"],
            "healthy",
        )

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Age": 35,
                "Gender": "Female",
                "Tenure": 12,
                "Usage Frequency": 20,
                "Support Calls": 2,
                "Payment Delay": 5,
                "Subscription Type": "Standard",
                "Contract Length": "Annual",
                "Total Spend": 500.0,
                "Last Interaction": 10,
            },
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()
        self.assertIn(result["prediction_code"], [0, 1])
        self.assertIn(result["prediction"], ["CHURN", "NO_CHURN"])

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"Age": 35, "Gender": "Female"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_invalid_json_request(self):
        response = self.client.post(
            "/predict",
            data="not valid json",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
