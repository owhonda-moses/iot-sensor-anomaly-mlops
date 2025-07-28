#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/iot-mlops

# PAT
if [ -f pat.env ]; then
  chmod 600 pat.env
  source ./pat.env
  echo "PAT loaded."
else
  echo "pat.env not found." >&2
  exit 1
fi

# apt + python
echo "Updating apt caches & sys dependencies…"
apt-get update -y >/dev/null 2>&1
apt-get install -y \
  git \
  tmux \
  python3.11 python3.11-venv python3.11-distutils \
  python3-pip \
  >/dev/null 2>&1

echo "Setting up pkg manager & dependencies…"
# pip + poetry
python3.11 -m pip install --upgrade pip poetry >/dev/null 2>&1
poetry env use python3.11 >/dev/null 2>&1
poetry install --no-interaction --no-ansi >/dev/null 2>&1

echo "Python env ready:$(poetry run python -V)"
echo "Venv path: $(poetry env info --path)"

# terraform
if ! command -v terraform >/dev/null 2>&1; then
  echo "Installing Terraform CLI…"
  curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add -
  apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
  apt-get install -y terraform
fi

# config
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


if [ -n "$(git status --porcelain)" ]; then
  echo "#…there are un-committed changes"
fi

export MLFLOW_TRACKING_URI="https://mlflow-server-243279652112.europe-west2.run.app"

echo "Starting prefect server..."
tmux new -d -s prefect 'poetry run prefect server start --host 0.0.0.0'

echo "Setup complete."