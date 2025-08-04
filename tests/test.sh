#!/usr/bin/env bash

curl -X POST "https://prediction-service-243279652112.europe-west2.run.app/predict" \
-H "Content-Type: application/json" \
-d '{
      "data": [
        {
          "device": "b8:27:eb:bf:9d:51",
          "ts": 1593590400,
          "co": 0.004,
          "humidity": 75.5,
          "light": true,
          "lpg": 0.007,
          "motion": false,
          "smoke": 0.019,
          "temp": 25.4
        }
      ]
    }' | jq
