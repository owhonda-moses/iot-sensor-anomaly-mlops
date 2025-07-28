chmod +x setup.sh
source ./setup.sh


curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | grep login
github-actions-sa@mlops-461322.iam.gserviceaccount.com

mv .github github_temp #unhide
mv github_temp .github #rehide

source "$(poetry env info --path)/bin/activate"

python create_deployment.py europe-west2-docker.pkg.dev/mlops-461322/mlops-docker/mlops-app:a1b2c3d

prefect deployment run 'iot_training_pipeline/IoT Training Deployment'

sudo apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli


python test.py --single --device 2 --co 0.03 --humidity 45.2 --lpg 0.01 --smoke 0.02 --temp 22.5
python test.py --csv ../data/iot_telemetry_data.csv

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

poetry export -f requirements.txt \
  --without-hashes --output requirements.txt



prefect server start
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

# In your Gradient terminal
gradient jobs logs --follow pr8nvtrddyu

https://nov7lh7kvs.clg07azjl.paperspacegradient.com:4200 #prefect
https://nov7lh7kvs.clg07azjl.paperspacegradient.com:5000 #mlflow










































# setup.sh with pip
#!/usr/bin/env bash
set -e
echo "Upgrading pip…"
pip install --upgrade pip > /dev/null 2>&1
echo "Installing Python packages…"
pip install --upgrade --ignore-installed blinker  > /dev/null 2>&1
pip uninstall greenlet -y > /dev/null 2>&1
pip install --no-binary :all: greenlet > /dev/null 2>&1
pip install optuna tqdm imbalanced-learn prefect mlflow poetry  > /dev/null 2>&1
pip install --upgrade tensorflow[and-cuda] matplotlib scikit-learn imbalanced-learn jupyter notebook numpy pyarrow pyyaml > /dev/null 2>&1
echo "Updating apt caches…"
apt-get update -y  > /dev/null 2>&1
echo "Dependencies installed."




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
