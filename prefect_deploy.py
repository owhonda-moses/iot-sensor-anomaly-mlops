from prefect.deployments import Deployment
from prefect.blocks.system import String
from prefect.server.schemas.schedules import CronSchedule
from src.iot_anomaly.pipeline import iot_training_pipeline
from src.iot_anomaly.monitoring import model_monitoring_flow

# string block to hold GCS bucket name
string_block = String(value="mlops-461322-iot-artifacts")
string_block.save(name="gcs-bucket-name", overwrite=True)
print("Saved String block 'gcs-bucket-name'.")


# training flow deployment
training_deployment = Deployment.build_from_flow(
    flow=iot_training_pipeline,
    name="IoT Training",
    work_pool_name="mlops-pool",
)

# monitoring flow deployment
monitoring_deployment = Deployment.build_from_flow(
    flow=model_monitoring_flow,
    name="IoT Model Monitoring",
    work_pool_name="mlops-pool",
    schedule=(
        CronSchedule(cron="0 6 * * *", timezone="UTC")
    ),  # run flow daily at 6AM UTC
)

# apply deployments to the server
if __name__ == "__main__":
    training_deployment.apply()
    monitoring_deployment.apply()
    print("Deployments successful.")
