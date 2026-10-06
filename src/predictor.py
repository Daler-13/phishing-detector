import os
import sys

import joblib
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from src.features import extract_features


MODEL_PATH = "models/gradient_boosting.joblib"


class PhishingPredictor:

    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, url: str) -> dict:

        features = extract_features(url)

        features_df = pd.DataFrame([features])

        prediction = self.model.predict(features_df)[0]

        probabilities = self.model.predict_proba(
            features_df
        )[0]

        phishing_probability = probabilities[0]
        safe_probability = probabilities[1]

        if prediction == 0:
            result = "Phishing"
            probability = phishing_probability
        else:
            result = "Safe"
            probability = safe_probability

        if result == "Phishing":

            if probability >= 0.90:
                risk = "HIGH"
            elif probability >= 0.60:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        else:

            if probability >= 0.90:
                risk = "LOW"
            elif probability >= 0.60:
                risk = "MEDIUM"
            else:
                risk = "HIGH"

        return {
            "url": url,
            "result": result,
            "probability": float(probability),
            "risk": risk,
            "features": features
        }


def main():

    print("=" * 70)
    print(" PHISHING DETECTOR — ТЕСТ ПРЕДСКАЗАНИЯ")
    print("=" * 70)

    predictor = PhishingPredictor()

    test_urls = [
        "https://www.google.com",
        "https://www.youtube.com",
        "http://192.168.1.1/login.php",
        "https://secure-login-account-verification.com/update",
    ]

    for url in test_urls:

        result = predictor.predict(url)

        print("\nURL:")
        print(result["url"])

        print("\nРезультат:")
        print(result["result"])

        print("\nВероятность:")
        print(
            f"{result['probability'] * 100:.2f}%"
        )

        print("\nУровень риска:")
        print(result["risk"])

        print("-" * 70)


if __name__ == "__main__":
    main()