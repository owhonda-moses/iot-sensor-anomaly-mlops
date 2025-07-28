#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/iot-mlops

# --- 1. Load Secrets ---
if [ -f pat.env ]; then
  chmod 600 pat.env
  source ./pat.env
  echo "PAT loaded."
else
  echo "pat.env not found." >&2
  exit 1
fi

# --- 2. Install System Dependencies ---
echo "Updating apt caches & sys dependencies…"
apt-get update -y >/dev/null 2&>1
apt-get install -y \
  git tmux curl apt-transport-https ca-certificates gnupg \
  python3.11 python3.11-venv python3.11-distutils python3-pip \
  >/dev/null 2>&1

# --- 3. Install Google Cloud CLI ---
echo "Installing Google Cloud CLI..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install -y google-cloud-sdk >/dev/null 2>&1

# --- 4. Install Terraform ---
echo "Installing Terraform CLI…"
curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add -
apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
apt-get update && apt-get install -y terraform >/dev/null 2>&1

# --- 5. Install ngrok ---
echo "Installing ngrok..."
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list
apt-get update && apt-get install -y ngrok >/dev/null 2>&1
# Authenticate ngrok - you must create an ngrok.env file with your token
if [ -f ngrok.env ]; then
  source ./ngrok.env
  ngrok config add-authtoken $NGROK_AUTHTOKEN
  echo "ngrok configured."
else
  echo "WARNING: ngrok.env not found. You will need to configure ngrok manually." >&2
fi

# --- 6. Set up Python Environment ---
echo "Setting up pkg manager & dependencies…"
python3.11 -m pip install --upgrade pip poetry >/dev/null 2>&1
poetry env use python3.11 >/dev/null 2>&1
poetry install --no-interaction --no-ansi >/dev/null 2>&1
echo "Python env ready: $(poetry run python -V)"

# --- 7. Start Prefect Server ---
echo "Starting Prefect server in the background..."
tmux new -d -s prefect 'poetry run prefect server start --host 0.0.0.0'
sleep 5 

# --- 8. Configure Local Prefect ---
echo "Configuring local Prefect profile..."
poetry run prefect profile create local --from http://127.0.0.1:4200/api || true
poetry run prefect profile use local
poetry run prefect work-pool create 'local-pool' --type process || true
echo "Prefect is configured to use the local server."

# --- 9. Configure Git ---
# (Your existing git config here)

echo "--------------------------------"
echo "Setup complete."
echo "IMPORTANT: Now run the gcloud login and set-quota-project commands."
echo "--------------------------------"