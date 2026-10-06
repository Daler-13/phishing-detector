import requests


BASE_URL = "http://127.0.0.1:5000"


def test_safe_url():
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json={"url": "https://www.google.com"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["result"] == "Safe"

    print("✅ test_safe_url — PASSED")


def test_phishing_url():
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json={"url": "http://192.168.1.1/login.php"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["result"] == "Phishing"

    print("✅ test_phishing_url — PASSED")


def test_empty_json():
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json={}
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"] == "JSON data is required"

    print("✅ test_empty_json — PASSED")


def test_empty_url():
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json={"url": ""}
    )

    assert response.status_code == 400

    data = response.json()

    assert data["success"] is False
    assert data["error"] == "URL is required"

    print("✅ test_empty_url — PASSED")


if __name__ == "__main__":

    print("================================")
    print("PHISHING DETECTOR API TESTS")
    print("================================")

    test_safe_url()
    test_phishing_url()
    test_empty_json()
    test_empty_url()

    print("================================")
    print("ALL TESTS PASSED ✅")
    print("================================")