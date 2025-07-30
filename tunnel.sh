#!/usr/bin/env bash
set -euo pipefail

echo "Installing cloudflared..."
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared.deb

echo "Launching cloudflare tunnel.."
cloudflared tunnel --url http://localhost:4200