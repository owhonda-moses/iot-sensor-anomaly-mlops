import os
import json
import pandas as pd
import mlflow
from prefect import flow, task
from prefect.blocks.system import String
from scipy.stats import ks_2samp
from google.cloud import storage


@task
def load_reference_data(gcs_uri: str) -> pd.DataFrame:
    """Loads the original training data to use as a reference."""
    return pd.read_csv(gcs_uri)


@task
def load_production_data(bucket_name: str) -> pd.DataFrame:
    """Loads all prediction logs from the last 24 hours from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket, prefix="prediction_logs/")

    all_data = []
    for blob in blobs:
        data = json.loads(blob.download_as_string())
        all_data.extend(data.get("data", []))

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


@task
def check_data_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame):
    """Checks for drift in key features and logs p-values to MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("model_monitoring")

    features_to_check = ["co", "humidity", "lpg", "smoke", "temp"]

    with mlflow.start_run():
        mlflow.log_param("num_production_samples", len(production_df))
        for feature in features_to_check:
            if feature in reference_df and feature in production_df:
                stat, p_value = ks_2samp(reference_df[feature], production_df[feature])
                mlflow.log_metric(f"p_value_{feature}", p_value)

                # if p-value is low theres drift
                if p_value < 0.05:
                    print(
                        f"Drift detected in feature '{feature}' (p-value: {p_value:.3f})"
                    )


@flow
def model_monitoring_flow():
    # load GCS bucket name
    bucket_block = String.load("gcs-bucket-name")
    bucket = bucket_block.value

    gcs_data_path = f"gs://{bucket}/data/iot_telemetry_data.csv"

    reference_data = load_reference_data(gcs_data_path)
    production_data = load_production_data(bucket)

    if not production_data.empty:
        check_data_drift(reference_data, production_data)
    else:
        print("No production data to check for drift.")


if __name__ == "__main__":
    model_monitoring_flow()
