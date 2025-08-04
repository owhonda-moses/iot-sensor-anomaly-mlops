import os
import requests
import pytest


SERVICE_URL = os.getenv("SERVICE_URL")


@pytest.mark.skipif(not SERVICE_URL, reason="SERVICE_URL not set")
def test_prediction_service():
    """Tests the live prediction endpoint."""
    payload = {
        "data": [
            {
                "device": "b8:27:eb:bf:9d:51",
                "ts": 1593590400,
                "co": 0.004,
                "humidity": 75.4,
                "light": True,
                "lpg": 0.007,
                "motion": False,
                "smoke": 0.019,
                "temp": 21.4,
            }
        ]
    }
    response = requests.post(f"{SERVICE_URL}/predict", json=payload)

    # check request was successful
    assert response.status_code == 200

    # check response structure
    response_json = response.json()
    assert "results" in response_json

    results = response_json["results"]
    assert isinstance(results, list)
    assert len(results) > 0

    # check structure of first result object
    first_result = results[0]
    assert "label" in first_result
    assert "confidence" in first_result
    assert first_result["label"] in ["normal", "anomaly"]
    assert isinstance(first_result["confidence"], float)
