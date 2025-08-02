#!/usr/bin/env bash
set -euo pipefail

# system dependencies
echo "Updating apt caches & sys dependencies…"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
  git curl apt-transport-https ca-certificates gnupg \
  python3.12 python3.12-venv python3-pip \
  python3-poetry

# google Cloud CLI & Terraform
echo "Installing Google Cloud CLI and Terraform..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add -
apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
apt-get update
apt-get install -y google-cloud-sdk terraform


# user-level
echo "Setting up poetry..."
sudo -u $SUDO_USER bash << 'EOF'
  set -euo pipefail
  
  cd "$(pwd)"
  poetry env use python3.12
  poetry install --no-interaction --no-ansi
  echo "Python env ready: $(poetry run python -V)"