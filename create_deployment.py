from prefect.deployments import Deployment
from prefect_github import GitHubRepository
from src.iot_anomaly.pipeline import iot_training_pipeline

github_block = GitHubRepository.load("github-repo")

with open("requirements.txt", "r") as f:
    pip_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Found {len(pip_packages)} packages to install.")

# deployment
deployment = Deployment.build_from_flow(
    flow=iot_training_pipeline,
    name="IoT Training Deployment",
    storage=github_block,
    work_pool_name="mlops-pool",
    job_variables={"pip_packages": pip_packages},
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment with pip packages successfully created.")