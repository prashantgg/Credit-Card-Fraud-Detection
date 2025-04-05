import io
import pandas as pd
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from unittest.mock import patch, MagicMock

class TestSetup(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse("upload_file")  # Make sure your urls.py uses 'upload_file' as name
        self.valid_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # Example features
        self.test_data = pd.DataFrame([[0]*len(self.valid_columns)], columns=self.valid_columns)

class TestGETRequest(TestSetup):
    def test_get_upload_form(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<form")

class TestPOSTRequestValidFile(TestSetup):
    @patch("fraud_app.views.logistic_model")
    @patch("fraud_app.views.random_forest_model")
    def test_valid_csv_upload(self, mock_rf_model, mock_log_model):
        mock_log_model.predict.return_value = [0]
        mock_log_model.feature_names_in_ = self.valid_columns
        mock_rf_model.predict.return_value = [1]

        csv_buffer = io.StringIO()
        self.test_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        file = SimpleUploadedFile("test.csv", csv_buffer.read().encode(), content_type="text/csv")

        response = self.client.post(self.url, {"file": file}, format="multipart")

        self.assertEqual(response.status_code, 200)
        self.assertIn("table_html", response.context)
        self.assertIn("Fraud", response.context["table_html"])
        self.assertIn("Not Fraud", response.context["table_html"])

class TestPOSTRequestInvalidCases(TestSetup):
    @patch("fraud_app.views.logistic_model")
    def test_file_with_missing_columns(self, mock_log_model):
        mock_log_model.feature_names_in_ = self.valid_columns

        # Drop a column to simulate missing
        bad_data = self.test_data.drop(columns=[self.valid_columns[0]])
        csv_buffer = io.StringIO()
        bad_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        file = SimpleUploadedFile("missing.csv", csv_buffer.read().encode(), content_type="text/csv")

        response = self.client.post(self.url, {"file": file}, format="multipart")
        self.assertContains(response, "Missing columns")

    def test_large_file_rejection(self):
        large_content = b"x" * (151 * 1024 * 1024)  # 151MB
        file = SimpleUploadedFile("large.csv", large_content, content_type="text/csv")
        response = self.client.post(self.url, {"file": file}, format="multipart")
        self.assertContains(response, "File size exceeds")

    @patch("fraud_app.views.predict_fraud", return_value=(None, None))
    @patch("fraud_app.views.logistic_model")
    def test_prediction_failure(self, mock_log_model, mock_predict):
        mock_log_model.feature_names_in_ = self.valid_columns

        csv_buffer = io.StringIO()
        self.test_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        file = SimpleUploadedFile("test.csv", csv_buffer.read().encode(), content_type="text/csv")

        response = self.client.post(self.url, {"file": file}, format="multipart")
        self.assertContains(response, "Prediction failed")

