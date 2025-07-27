chmod +x setup.sh
source ./setup.sh


curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | grep login
github-actions-sa@mlops-461322.iam.gserviceaccount.com

mv .github github_temp #unhide
mv github_temp .github #rehide

source "$(poetry env info --path)/bin/activate"

prefect deployment build src/iot_anomaly/pipeline.py:iot_training_pipeline --name "IoT Training Deployment" --output iot-deployment.yaml
prefect deployment apply iot-deployment.yaml

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




# FROM python:3.11-slim AS builder

# # install system deps & Poetry
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends curl build-essential git \
#  && curl -sSL https://install.python-poetry.org | python3 - \
#  && mv /root/.local/bin/poetry /usr/local/bin/poetry \
#  && apt-get purge -y --auto-remove curl build-essential git \
#  && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # bring in dependency specs
# COPY pyproject.toml poetry.lock ./

# # install Python deps
# RUN pip install --upgrade pip \
#  && poetry config virtualenvs.create false \
#  && poetry install --no-interaction --no-ansi --without dev \
#  && pip install psycopg2-binary

# # copy app code  
# COPY src/ ./src/


# FROM python:3.11-slim

# WORKDIR /app

# # copy installed packages & executables
# COPY --from=builder /usr/local/lib/python3.11/site-packages/ \
#                     /usr/local/lib/python3.11/site-packages/
# COPY --from=builder /usr/local/bin/ \
#                     /usr/local/bin/

# # copy app code
# COPY --from=builder /app/src/ ./src/

# # Cloud Run 
# EXPOSE 8080

# ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# ENTRYPOINT ["mlflow", "server"]
# CMD ["--host","0.0.0.0","--port","8080","--backend-store-uri","$MLFLOW_TRACKING_URI","--default-artifact-root","gs://${ARTIFACT_BUCKET}/mlruns"]

# Dockerfile