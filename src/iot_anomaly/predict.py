#!/usr/bin/env python

import os
import pickle
import joblib
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify
from google.cloud import storage
from iot_anomaly.core import engineer_flags, preprocess


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "iot-anomaly-model"
MODEL_ALIAS = "prod"
ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET")

if not MLFLOW_URI:
    raise EnvironmentError("MLFLOW_TRACKING_URI must be set")

mlflow.set_tracking_uri(MLFLOW_URI)
storage_client = storage.Client()

# global cache for artifacts
_artifacts = None


def load_artifacts_mlflow():
    """
    Connects to MLflow and downloads all necessary artifacts.
    """
    print("Loading model artifacts from MLflow...")
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    client = mlflow.tracking.MlflowClient()

    model_version_meta = client.get_model_version_by_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS
    )
    run_id = model_version_meta.run_id

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler/scaler.pkl"
    )
    scaler = joblib.load(scaler_path)

    meta_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="meta/sl_meta.pkl"
    )
    meta = pickle.load(open(meta_path, "rb"))
    threshold = meta["threshold"]

    xcols_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="meta/X_cols.pkl"
    )
    X_cols = pickle.load(open(xcols_path, "rb"))

    model = mlflow.pyfunc.load_model(model_uri)
    print("Artifacts loaded successfully.")
    return scaler, model._model_impl, threshold, X_cols


def get_artifacts():
    """
    Manages the global cache, loading artifacts only on the first call.
    """
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts_mlflow()
    return _artifacts


# prediction
def predict_iot(raw_df: pd.DataFrame):
    # Get the scaler, model, etc. from our caching function
    scaler, model, threshold, x_cols = get_artifacts()

    df = raw_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = engineer_flags(df)
    _, X_tr, *rest = preprocess(df, test_size=0, reference_columns=x_cols)
    X_scaled = scaler.transform(X_tr)
    probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    return y_pred, probs


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(force=True)
    if ARTIFACT_BUCKET:
        try:
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H%M%S_%f")
            blob_name = f"prediction_logs/{timestamp}.json"
            bucket = storage_client.bucket(ARTIFACT_BUCKET)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(payload), content_type="application/json"
            )
        except Exception as e:
            app.logger.error(f"Failed to log prediction payload: {e}")

    df = pd.DataFrame(payload["data"])
    y_pred, scores = predict_iot(df)
    return jsonify({"predictions": y_pred.tolist(), "scores": scores.tolist()})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# cli
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch-predict IoT anomalies CLI")
    p.add_argument("--csv", required=True, help="Path to raw IoT CSV")
    p.add_argument("--out", default="preds.npy", help="Where to write predictions")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    y, _ = predict_iot(df)
    np.save(args.out, y)
    print(f"Saved {len(y)} predictions to {args.out}")
