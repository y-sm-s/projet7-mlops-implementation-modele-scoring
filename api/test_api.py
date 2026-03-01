import unittest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


class TestCreditAPI(unittest.TestCase):
    def test_health(self):
        """Test que l'API répond sur /"""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict_endpoint(self):
        """Test que l'endpoint /predict répond correctement"""
        payload = {"data": {"AMT_CREDIT": 100000}}

        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        self.assertIn("decision", result)
        self.assertIn("probability", result)
        self.assertIn("threshold", result)

    def test_predict_missing_data(self):
        """Test que l'API gère les données manquantes"""
        payload = {"data": {}}

        response = client.post("/predict", json=payload)
        self.assertNotEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
