
import os
import joblib
import pandas as pd

from src.features_v3 import extract_features_v3


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "v4", "random_forest_v4.joblib"
)

FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "v4", "feature_names_v4.csv"
)


class PhishingPredictorV4:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        self.feature_names = pd.read_csv(
            FEATURES_PATH
        )["feature"].tolist()

    def predict(self, url: str) -> dict:
        features = extract_features_v3(url)

        features_df = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names
        )

        prediction = int(self.model.predict(features_df)[0])
        probabilities = self.model.predict_proba(features_df)[0]

        classes = list(self.model.classes_)
        phishing_probability = float(
            probabilities[classes.index(0)]
        )
        safe_probability = float(
            probabilities[classes.index(1)]
        )

        if prediction == 0:
            result = "Phishing"
            probability = phishing_probability
            risk = (
                "HIGH" if probability >= 0.90
                else "MEDIUM" if probability >= 0.60
                else "LOW"
            )
        else:
            result = "Safe"
            probability = safe_probability
            risk = (
                "LOW" if probability >= 0.90
                else "MEDIUM" if probability >= 0.60
                else "HIGH"
            )

        return {
            "url": url,
            "result": result,
            "probability": probability,
            "phishing_probability": phishing_probability,
            "safe_probability": safe_probability,
            "risk": risk,
            "features": features
        }

