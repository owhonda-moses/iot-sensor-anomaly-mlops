from prefect.deployments import Deployment
from src.iot_anomaly.pipeline import iot_training_pipeline

deployment = Deployment.build_from_flow(
    flow=iot_training_pipeline,
    name="IoT Training",
    work_pool_name="mlops-pool",
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment successful.")















# import sys
# from prefect.deployments import Deployment
# from prefect.infrastructure.container import DockerContainer
# from prefect_github import GitHubRepository
# from src.iot_anomaly.pipeline import iot_training_pipeline


# github_block = GitHubRepository.load("github-repo")


# docker_container_block = DockerContainer(
#     image=docker_image_uri,
#     image_pull_policy="ALWAYS",
#     auto_remove=True,
# )


# deployment = Deployment.build_from_flow(
#     flow=iot_training_pipeline,
#     name="IoT Training Push Deployment",
#     storage=github_block,
#     infrastructure=docker_container_block,
#     work_pool_name="mlops-pool",
# )

# if __name__ == "__main__":
#     deployment.apply()
#     print("Push deployment successful.")