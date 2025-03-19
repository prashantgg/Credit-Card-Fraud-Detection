import pandas as pd
import joblib
from django.shortcuts import render
from .forms import UploadFileForm
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Load models
logistic_model = joblib.load("models/logistic_model.pkl")
random_forest_model = joblib.load("models/random_forest_model.pkl")

def predict_fraud(data):
    """Predict fraud using both models and return results"""
    try:
        logistic_pred = logistic_model.predict(data)
        random_forest_pred = random_forest_model.predict(data)
        return logistic_pred, random_forest_pred
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None, None

def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                file = request.FILES["file"]

                # Check file size (limit to 143MB)
                if file.size > 150 * 1024 * 1024:  # 143MB in bytes
                    return render(request, "fraud_app/upload.html", {
                        "form": form,
                        "error": "File size exceeds the 150MB limit.",
                    })

                df = pd.read_csv(file)

                # Drop 'Time' column if it exists (to match training data)
                if "Time" in df.columns:
                    df = df.drop(columns=["Time"])

                # Ensure the dataset has only the features used during training
                expected_features = logistic_model.feature_names_in_
                missing_features = [col for col in expected_features if col not in df.columns]

                if missing_features:
                    return render(request, "fraud_app/upload.html", {
                        "form": form,
                        "error": f"Missing columns in uploaded file: {missing_features}",
                    })

                # Select only expected features for prediction
                X = df[expected_features]

                # Predict using both models
                logistic_pred, random_forest_pred = predict_fraud(X)

                if logistic_pred is None or random_forest_pred is None:
                    return render(request, "fraud_app/upload.html", {
                        "form": form,
                        "error": "Prediction failed. Please check the uploaded file.",
                    })

                # Store predictions in DataFrame
                df["Logistic_Prediction"] = logistic_pred
                df["RandomForest_Prediction"] = random_forest_pred

                # Convert results to HTML table for display
                table_html = df.to_html(classes="table table-striped", index=False)

                return render(request, "fraud_app/upload.html", {
                    "form": form,
                    "table_html": table_html,
                })

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                return render(request, "fraud_app/upload.html", {
                    "form": form,
                    "error": f"An error occurred: {e}",
                })

    else:
        form = UploadFileForm()
    return render(request, "fraud_app/upload.html", {"form": form})

