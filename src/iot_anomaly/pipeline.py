import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

import time
import argparse
import pickle
import joblib
import tempfile
import json
from datetime import timedelta

import pandas as pd
import mlflow
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.blocks.system import String

# core logic
from iot_anomaly.core import engineer_flags, preprocess, train_usl, train_sl


# DRIVE_FILE_ID = "1fGQ_gJ8_1tpkaCN7SUc3bi6a81ZfPnVo"

baseline_params = {
    "if_n_estimators": 100,
    "if_max_samples": "auto",
    "contamination": 0.01,
    "ae_encoding_dim": 16,
    "ae_lr": 1e-3,
    "ae_epochs": 30,
    "ae_batch_size": 128,
    "w_if": 1,
    "w_ae": 0,
    "thr_q": 0.95,
}
best_usl = {
    "if_n_estimators": 148,
    "if_max_samples": 1.0,
    "contamination": 0.07513614867821206,
    "ae_encoding_dim": 8,
    "ae_lr": 0.0004484570984975867,
    "ae_epochs": 43,
    "ae_batch_size": 64,
    "w_if": 0.6582947205466332,
    "thr_q": 0.9691031375838758,
}
best_usl["w_ae"] = 1.0 - best_usl["w_if"]
for k, v in baseline_params.items():
    best_usl.setdefault(k, v)
sl_base_params = {"C": 1.0, "tol": 1e-4, "penalty": "l2"}
best_sl = {"penalty": "l2", "C": 5.232594029557045, "tol": 0.002263077789430857}
SEED = 42


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_and_preprocess_task(gcs_uri: str):
    """
    Streams the CSV from GCS, engineers features,
    splits, scales, and returns everything needed for modeling.
    """
    logger = get_run_logger()
    logger.info(f"Loading data from {gcs_uri}...")
    df = pd.read_csv(gcs_uri)

    # load from gDrive
    # import requests
    # from io import BytesIO
    # url = f"https://docs.google.com/uc?export=download&id={DRIVE_FILE_ID}"
    # resp = requests.get(url)
    # resp.raise_for_status()
    # df = pd.read_csv(BytesIO(resp.content))

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    df = engineer_flags(df)
    scaler, Xtr, Xte, ytr, yte, X_cols = preprocess(df, random_state=SEED)
    Xtr_norm = Xtr[ytr == 0]
    return scaler, Xtr, Xte, ytr, yte, Xtr_norm, X_cols


@task
def usl_task(
    scaler, Xtr_norm, Xte, yte, X_cols, params: dict, label: str, dry_run: bool
):
    """
    Runs Unsupervised training, logs params/metrics/artifacts to MLflow.
    """
    run_name = f"USL_{label}"
    logger = get_run_logger()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", label)
        mlflow.log_params(params)
        if dry_run:
            logger.info("Dry run: skipping USL")
            return {}

        res = train_usl(Xtr_norm, Xte, yte, params)
        mlflow.log_metrics(res["metrics"])

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = f"usl/{label}"
            # scaler
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, artifact_path=prefix)
            # feature columns
            cols_path = os.path.join(tmpdir, "X_cols.pkl")
            with open(cols_path, "wb") as f:
                pickle.dump(X_cols, f)
            mlflow.log_artifact(cols_path, artifact_path=prefix)
            # IsolationForest
            iso_path = os.path.join(tmpdir, "iso_model.pkl")
            joblib.dump(res["iso_model"], iso_path)
            mlflow.log_artifact(iso_path, artifact_path=prefix)
            # Autoencoder (if present)
            if res.get("ae_model"):
                ae_path = os.path.join(tmpdir, "autoencoder.keras")
                res["ae_model"].save(ae_path, include_optimizer=False)
                mlflow.log_artifact(ae_path, artifact_path=prefix)
            # Ensemble config
            cfg = {"threshold": float(res["threshold"]), "weights": res["weights"]}
            cfg_path = os.path.join(tmpdir, "ensemble_config.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            mlflow.log_artifact(cfg_path, artifact_path=prefix)

        return res


@task
def sl_task(
    scaler, Xtr, ytr, Xte, yte, X_cols, params: dict, label: str, dry_run: bool
):
    """
    Runs Supervised training, logs all necessary artifacts, and registers the model.
    """
    run_name = f"SL_{label}"
    logger = get_run_logger()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", label)
        mlflow.log_params(params)
        if dry_run:
            logger.info("Dry run: skipping SL")
            return {}

        res = train_sl(
            X_tr=Xtr,
            y_tr=ytr,
            X_te=Xte,
            y_te=yte,
            **params,
            random_state=SEED,
            n_splits=5,
        )
        mlflow.log_metrics(res["test_metrics"])

        # log all artifacts and register it
        if label == "optimized":
            logger.info("Logging and registering production model and artifacts...")

            with tempfile.TemporaryDirectory() as tmpdir:
                # scaler
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                joblib.dump(scaler, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="scaler")
                # feature columns list
                xcols_path = os.path.join(tmpdir, "X_cols.pkl")
                with open(xcols_path, "wb") as f:
                    pickle.dump(X_cols, f)
                mlflow.log_artifact(xcols_path, artifact_path="meta")
                # metadata (threshold, etc.) with the correct name
                meta = {"threshold": float(res["avg_thr"])}
                meta_path = os.path.join(tmpdir, "sl_meta.pkl")
                with open(meta_path, "wb") as f:
                    pickle.dump(meta, f)
                mlflow.log_artifact(meta_path, artifact_path="meta")

            # log and register the model
            mlflow.sklearn.log_model(
                sk_model=res["model"], artifact_path="sklearn-model"
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/sklearn-model"
            mlflow.register_model(model_uri=model_uri, name="iot-anomaly-model")
    return res


@flow(name="iot_training_pipeline")
def iot_training_pipeline(
    run_usl: bool = True, run_sl: bool = True, dry_run: bool = False
):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns"))
    mlflow.set_experiment("iot_fault_detection")

    bucket_block = String.load("gcs-bucket-name")
    bucket = bucket_block.value.strip()
    gcs_data_path = f"gs://{bucket}/data/iot_telemetry_data.csv"

    scaler, Xtr, Xte, ytr, yte, Xtr_norm, X_cols = load_and_preprocess_task(
        gcs_data_path
    )

    outputs = {}
    if run_usl:
        outputs["usl_baseline"] = usl_task(
            scaler, Xtr_norm, Xte, yte, X_cols, baseline_params, "baseline", dry_run
        )
        outputs["usl_optimized"] = usl_task(
            scaler, Xtr_norm, Xte, yte, X_cols, best_usl, "optimized", dry_run
        )

    if run_sl:
        outputs["sl_baseline"] = sl_task(
            scaler, Xtr, ytr, Xte, yte, X_cols, sl_base_params, "baseline", dry_run
        )

        outputs["sl_optimized"] = sl_task(
            scaler, Xtr, ytr, Xte, yte, X_cols, best_sl, "optimized", dry_run
        )

    return outputs


# cli
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_usl", action="store_true", help="Run unsupervised tasks")
    parser.add_argument("--run_sl", action="store_true", help="Run supervised tasks")
    parser.add_argument(
        "--dry_run", action="store_true", help="Log runs without training"
    )
    args = parser.parse_args()

    run_usl = args.run_usl or (not args.run_sl and not args.dry_run)
    run_sl = args.run_sl or (not args.run_usl and not args.dry_run)

    start = time.time()
    iot_training_pipeline(run_usl=run_usl, run_sl=run_sl, dry_run=args.dry_run)
    runtime_min = (time.time() - start) / 60
    print(f"Pipeline completed in {runtime_min:.2f} minutes.")
