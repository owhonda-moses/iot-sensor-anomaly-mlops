import os
import requests
import pytest

# Get the service URL from an environment variable
SERVICE_URL = os.getenv("SERVICE_URL")

@pytest.mark.skipif(not SERVICE_URL, reason="SERVICE_URL not set")
def test_prediction_service():
    """Tests the live prediction endpoint."""
    payload = {
        "data": [
            {
                "device": "b8:27:eb:bf:9d:51", "ts": 1593590400, "co": 0.004,
                "humidity": 75.4, "light": True, "lpg": 0.007, "motion": False,
                "smoke": 0.019, "temp": 21.4
            }
        ]
    }
    response = requests.post(f"{SERVICE_URL}/predict", json=payload)

    
    assert response.status_code == 200

    # Check the response structure
    response_json = response.json()
    assert "predictions" in response_json
    assert "scores" in response_json
    assert isinstance(response_json["predictions"], list)
    assert isinstance(response_json["scores"], list)