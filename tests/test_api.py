import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health(self):
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    @patch("app.api.model_a")
    @patch("app.api.predict_model")
    def test_predict_mock(self, mock_predict, mock_model_a):
        # Подготовка мока
        mock_model_a.return_value = "MockModel"

        # Мокаем ответ от PyCaret predict_model
        import pandas as pd

        mock_df = pd.DataFrame({"prediction_label": [1], "prediction_score": [0.95]})
        mock_predict.return_value = mock_df

        # Имитируем, что модель загружена
        with patch("app.api.model_a", "MockObject"), patch(
            "app.api.model_b", "MockObject"
        ):
            payload = {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
            }
            response = self.app.post(
                "/predict", data=json.dumps(payload), content_type="application/json"
            )

            self.assertEqual(response.status_code, 200)
            data = response.json
            self.assertEqual(data["prediction_label"], 1)
            self.assertIn("model_version", data)


if __name__ == "__main__":
    unittest.main()
