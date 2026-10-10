from flask import Flask, render_template, request, jsonify

from src.predictor_v4 import PhishingPredictorV4

from database.database import (
    init_database,
    save_prediction,
    get_predictions,
    get_statistics
)

import pandas as pd


app = Flask(__name__)

predictor = PhishingPredictorV4()

init_database()


MAX_URL_LENGTH = 2048

MODEL_COMPARISON_PATH = "models/model_comparison.csv"


def get_model_comparison():
    try:
        df = pd.read_csv(MODEL_COMPARISON_PATH)

        return df.to_dict(orient="records")

    except Exception:
        return []


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            error = "URL is required"

        elif len(url) > MAX_URL_LENGTH:
            error = f"URL is too long. Maximum length is {MAX_URL_LENGTH} characters."

        else:
            result = predictor.predict(url)

            save_prediction(
                url=result["url"],
                result=result["result"],
                probability=result["probability"],
                risk=result["risk"]
            )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


@app.route("/history")
def history():

    predictions = get_predictions()

    return render_template(
        "history.html",
        predictions=predictions
    )


@app.route("/dashboard")
def dashboard():
    statistics = get_statistics()
    model_comparison = get_model_comparison()

    return render_template(
        "dashboard.html",
        statistics=statistics,
        model_comparison=model_comparison
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "error": "JSON data is required"
        }), 400

    url = data.get("url")

    if not url or not isinstance(url, str):
        return jsonify({
            "success": False,
            "error": "URL is required"
        }), 400

    url = url.strip()

    if not url:
        return jsonify({
            "success": False,
            "error": "URL is required"
        }), 400

    if len(url) > MAX_URL_LENGTH:
        return jsonify({
            "success": False,
            "error": f"URL is too long. Maximum length is {MAX_URL_LENGTH} characters."
        }), 400

    try:
        result = predictor.predict(url)

        save_prediction(
            url=result["url"],
            result=result["result"],
            probability=result["probability"],
            risk=result["risk"]
        )

        return jsonify({
            "success": True,
            "data": {
                "url": result["url"],
                "result": result["result"],
                "probability": result["probability"],
                "risk": result["risk"]
            }
        })

    except Exception:
        return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)