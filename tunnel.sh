#!/usr/bin/env bash

# ngrok
echo "Installing ngrok..."
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null 2>&1
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list
apt-get update && apt-get install -y ngrok >/dev/null 2>&1

if [ -f ngrok.env ]; then
  source ./ngrok.env
  ngrok config add-authtoken $NGROK_AUTHTOKEN
  echo "ngrok configured."
else
  echo "ngrok.env not found." >&2
fi

echo "Launching ngrok tunnel..."
ngrok http 4200