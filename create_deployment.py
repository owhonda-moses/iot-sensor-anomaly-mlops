import sys
from prefect.deployments import Deployment
from prefect.infrastructure.container import DockerContainer
from prefect_github import GitHubRepository
from src.iot_anomaly.pipeline import iot_training_pipeline

if len(sys.argv) < 2:
    print("Provide full image URI as an argument")
    sys.exit(1)

docker_image_uri = sys.argv[1]
print(f"Using image: {docker_image_uri}")

# storage and infrastructure blocks
github_block = GitHubRepository.load("github-repo")
docker_container_block = DockerContainer(
    image=docker_image_uri,
    image_pull_policy="ALWAYS",
    auto_remove=True,
)

deployment = Deployment.build_from_flow(
    flow=iot_training_pipeline,
    name="IoT Pipeline Deployment",
    storage=github_block,
    infrastructure=docker_container_block,
    work_pool_name="mlops-pool",
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment successful.")