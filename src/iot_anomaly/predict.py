#!/usr/bin/env python

import os, pickle, joblib, argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from flask import Flask, request, jsonify
from iot_anomaly.core import engineer_flags, preprocess

# Load from registry
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "iot-anomaly-model"
MODEL_ALIAS = "prod"

if not MLFLOW_URI:
    raise EnvironmentError("MLFLOW_TRACKING_URI must be set")

mlflow.set_tracking_uri(MLFLOW_URI)

def load_artifacts_mlflow():
    """
    Loads the production model from the registry and its associated artifacts
    (scaler, metadata, feature columns) from the same MLflow run.
    """
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    client = mlflow.tracking.MlflowClient()
    model_version_meta = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
    run_id = model_version_meta.run_id
    
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler/scaler.pkl")
    scaler = joblib.load(scaler_path)

    meta_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="meta/sl_meta.pkl")
    meta = pickle.load(open(meta_path, "rb"))
    threshold = meta["threshold"]

    xcols_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="meta/X_cols.pkl")
    X_cols = pickle.load(open(xcols_path, "rb"))

    model = mlflow.pyfunc.load_model(model_uri)

    return scaler, model, threshold, X_cols

# cache artifacts
SCALER, MODEL, THRESHOLD, X_COLS = load_artifacts_mlflow()

def predict_iot(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df = engineer_flags(df)

    # use cache
    _, X_tr, *rest = preprocess(df, test_size=0, reference_columns=X_COLS)

    X_scaled = SCALER.transform(X_tr)
    probs = MODEL.predict_proba(X_scaled)[:, 1]
    y_pred = (probs >= THRESHOLD).astype(int)

    return y_pred, probs


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(force=True)
    df = pd.DataFrame(payload["data"])
    y_pred, scores = predict_iot(df)
    return jsonify({
        "predictions": y_pred.tolist(),
        "scores": scores.tolist()
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# cli
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Batch-predict IoT anomalies CLI"
    )
    p.add_argument("--csv", required=True,
                   help="Path to raw IoT CSV")
    p.add_argument("--out", default="preds.npy",
                   help="Where to write predictions")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    y, _ = predict_iot(df)
    np.save(args.out, y)
    print(f"Saved {len(y)} predictions to {args.out}")