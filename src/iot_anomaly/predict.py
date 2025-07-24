#!/usr/bin/env python

import os, pickle, joblib, json
import numpy as np
import pandas as pd

from iot_anomaly.core import engineer_flags, preprocess

MODEL_DIR = ".../iot_models/optimized"

def load_artifacts():
    """
    Load trained scaler, LR and threshold.
    """
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "sl_model.pkl"))
    meta = pickle.load(open(os.path.join(MODEL_DIR, "sl_meta.pkl"), "rb"))
    threshold = meta["threshold"]
    X_cols = pickle.load(open(os.path.join(MODEL_DIR, "X_cols.pkl"), "rb"))
    return scaler, model, threshold, X_cols

def predict_iot(raw_df: pd.DataFrame):
    """
    Enhanced prediction function that ensures feature consistency
    """
    raw_df = raw_df.copy()
    raw_df['ts'] = pd.to_datetime(raw_df['ts'], unit='s', utc=True)
    df = engineer_flags(raw_df)
    
    # Load artifacts
    scaler, model, threshold, X_cols = load_artifacts()
    
    _, X_tr, _, _, _, _ = preprocess(df, test_size=0, reference_columns=X_cols)
    
    # Scale and predict
    X_scaled = scaler.transform(X_tr)
    probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    return y_pred, probs

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="Path to raw IoT CSV with columns device,ts,co,...")
    p.add_argument("--out", default="preds.npy",
                   help="Where to write predictions")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    y, scores = predict_iot(df)
    np.save(args.out, y)
    print(f"Saved {len(y)} predictions to {args.out}")