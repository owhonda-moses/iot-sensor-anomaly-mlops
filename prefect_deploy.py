from prefect.deployments import Deployment
from prefect.blocks.system import String
from src.iot_anomaly.pipeline import iot_training_pipeline


gcs_bucket_name = "mlops-461322-iot-artifacts"
string_block = String(value=gcs_bucket_name)
string_block.save(name="gcs-bucket-name", overwrite=True)
print(f"Saved String block with GCS bucket name.")


deployment = Deployment.build_from_flow(
    flow=iot_training_pipeline,
    name="IoT Training",
    work_pool_name="mlops-pool",
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment successful.")