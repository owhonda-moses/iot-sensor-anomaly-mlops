#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/iot-mlops


if [ -f pat.env ]; then
  chmod 600 pat.env
  source ./pat.env
  echo "PAT loaded."
else
  echo "pat.env not found." >&2
  exit 1
fi

# sys dependencies
echo "Updating apt caches & sys dependencies…"
apt-get update -y >/dev/null 2>&1
apt-get install -y \
  git tmux curl apt-transport-https ca-certificates gnupg \
  python3.11 python3.11-venv python3.11-distutils python3-pip \
  >/dev/null 2>&1

# gCloud CLI & terraform
echo "Installing gCloud CLI and Terraform..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add -
apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
apt-get update
apt-get install -y google-cloud-sdk terraform


echo "Setting up pkg manager & dependencies…"
python3.11 -m pip install --upgrade pip poetry >/dev/null 2>&1
poetry env use python3.11 >/dev/null 2>&1
poetry install --no-interaction --no-ansi >/dev/null 2>&1
echo "Python env ready: $(poetry run python -V)"

# prefect server
echo "Starting prefect.."
if ! tmux has-session -t prefect 2>/dev/null; then
  tmux new -d -s prefect 'poetry run prefect server start --host 0.0.0.0'
  echo "Prefect server launched"
else
  echo "tmux session already exists. Skipping launch."
fi
sleep 5 


if ! poetry run prefect profile inspect mlops >/dev/null 2>&1; then
  poetry run prefect profile create mlops
fi


poetry run prefect profile use mlops
poetry run prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

if ! poetry run prefect work-pool inspect 'mlops-pool' >/dev/null 2>&1; then
  poetry run prefect work-pool create 'mlops-pool' --type process
  echo "Work pool created."
else
  echo "Work pool already exists."
fi

poetry run prefect work-pool create 'mlops-pool' --type process || true
echo "Prefect is configured."


# git config
echo "Configuring git"
git config --global user.name  "owhonda-moses"
git config --global user.email "owhondamoses7@gmail.com"

# ~/.netrc for https auth
cat > "$HOME/.netrc" <<EOF
machine github.com
  login x-access-token
  password $GITHUB_TOKEN
EOF
chmod 600 "$HOME/.netrc"

# bootstrap
echo "Bootstrapping git repo…"
git remote remove origin 2>/dev/null || true
if [ ! -d .git ]; then
  git init
fi
git remote add origin \
  https://x-access-token:${GITHUB_TOKEN}@github.com/owhonda-moses/iot-sensor-anomaly-mlops.git
git fetch origin main --depth=1 2>/dev/null || true
git checkout main 2>/dev/null || git checkout -b main

# git fetch origin main --depth=1

# commit changes
if [ -n "$(git status --porcelain)" ]; then
  echo "#…there are un-committed changes"
fi

echo "Setup complete! run gcloud login."