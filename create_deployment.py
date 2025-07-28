from prefect.deployments import Deployment
from prefect.infrastructure.docker import DockerContainer
from src.iot_anomaly.pipeline import iot_training_pipeline

docker_image_uri = "europe-west2-docker.pkg.dev/mlops-461322/mlops-docker/mlops-app:latest"

# docker container
docker_block = DockerContainer(
    image=docker_image_uri,
    auto_remove=True,
)

deployment = Deployment.build_from_flow(
    flow=iot_training_pipeline,
    name="IoT Training Deployment",
    work_pool_name="mlops-pool",
    infrastructure=docker_block,
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment successful.")