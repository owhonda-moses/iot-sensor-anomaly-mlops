curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | grep login
github-actions-sa@mlops-461322.iam.gserviceaccount.com

mv .github github_temp #unhide
mv github_temp .github #rehide

source "$(poetry env info --path)/bin/activate"

gcloud auth application-default login
gcloud auth application-default set-quota-project mlops-461322

prefect profile create gce
prefect profile use gce
prefect config set PREFECT_API_URL="http://34.39.116.245:4200/api"
prefect work-pool create 'mlops-pool' --type process

export MLFLOW_TRACKING_URI="https://mlflow-server-243279652112.europe-west2.run.app"
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="mlflow_pass8080"

prefect worker start --pool 'mlops-pool'
"run deployment script"


python test.py --single --device 2 --co 0.03 --humidity 45.2 --lpg 0.01 --smoke 0.02 --temp 22.5
python test.py --csv ../data/iot_telemetry_data.csv

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

poetry export -f requirements.txt \
  --without-hashes --output requirements.txt


prefect profile create mlops --from-url http://https://d74feea798a164d668cd907e5cf1c8d87.clg07azjl.paperspacegradient.com/:4200/api
postgresql+asyncpg://prefect_user:prefect_pass8080@/prefect_db?host=/cloudsql/mlops-461322:europe-west2:mlflow-pg






# core
pydantic==1.10.13
griffe==0.30.1
anyio==3.7.1
joblib==1.3.2
numpy==2.0.0
pandas==2.3.0
scikit-learn==1.6.1
imbalanced-learn==0.13.0
ipykernel==6.29.0
optuna==4.4.0
tqdm==4.66.1
prefect==2.11.5
prefect-github==0.2.5
mlflow==3.1.1
tensorflow==2.19.0
flask==2.2.5
gunicorn==20.1.0
