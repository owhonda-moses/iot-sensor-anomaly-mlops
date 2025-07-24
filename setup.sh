#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/iot-mlops

# PAT
if [ -f pat.env ]; then
  # chmod 600 pat.env          
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
  python3.11 python3.11-venv python3.11-distutils \
  >/dev/null 2>&1

echo "Setting up pkg manager & dependencies…"
# pip + poetry
pip install --upgrade pip >/dev/null 2>&1
pip install poetry >/dev/null 2>&1
poetry env use python3.11 >/dev/null 2>&1
poetry install --no-interaction --no-ansi >/dev/null 2>&1

# activate venv
source "$(poetry env info --path)/bin/activate"
echo "Python env ready:$(poetry run python -V)"
echo "Venv path: $(poetry env info --path)"

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

echo "Setup complete."

# commit changes
if [ -n "$(git status --porcelain)" ]; then
  echo "#…there are un-committed changes"
fi

